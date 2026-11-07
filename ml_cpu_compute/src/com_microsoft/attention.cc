// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "ml_cpu_compute/com_microsoft/attention.h"
#include "ml_cpu_compute/checked_arithmetic.h"
#include "ml_cpu_compute/execution.h"
#include "ml_cpu_compute/math/gemm/gemm_plan.h"

#include <algorithm>
#include <stdexcept>

namespace ml_cpu_compute {
namespace {
std::size_t Width(const AttentionPlan &plan) {
  return CheckedMultiply(
      plan.q_num_heads,
      CheckedAdd(CheckedMultiply(2, plan.head_dim, "MicrosoftAttention", "QK width"),
                 plan.v_head_dim, "MicrosoftAttention", "QKV width"),
      "MicrosoftAttention", "QKV width");
}
GemmPlan<float> ProjectionPlan(const AttentionPlan &plan, std::size_t input_hidden) {
  GemmPlanOptions<float> options;
  options.m = CheckedMultiply(plan.batch, plan.q_length, "MicrosoftAttention", "projection rows");
  options.n = Width(plan);
  options.k = input_hidden;
  return GemmPlan<float>(options);
}
AttentionPlan ProjectedPlan(const AttentionPlan &source) {
  auto plan = source;
  const auto width = Width(plan);
  const auto batch_stride =
      CheckedMultiply(plan.q_length, width, "MicrosoftAttention", "projection batch stride");
  plan.q_strides.batch = plan.k_strides.batch = plan.v_strides.batch =
      CheckedStride(batch_stride, "MicrosoftAttention", "projection batch stride");
  plan.q_strides.sequence = plan.k_strides.sequence = plan.v_strides.sequence =
      CheckedStride(width, "MicrosoftAttention", "projection row stride");
  return plan;
}
} // namespace

std::size_t MicrosoftAttentionPeakMemoryBytes(const AttentionPlan &plan, std::size_t input_hidden) {
  const auto width = Width(plan);
  const auto projection = CheckedProduct({plan.batch, plan.q_length, width}, "MicrosoftAttention",
                                         "projection elements");
  const auto panels = ProjectionPlan(plan, input_hidden).ScratchBytes();
  return CheckedAdd(CheckedAdd(CheckedByteSize(projection, sizeof(float), "MicrosoftAttention",
                                               "projection bytes"),
                               panels, "MicrosoftAttention", "projection scratch"),
                    AttentionPeakMemoryBytes(ProjectedPlan(plan)), "MicrosoftAttention",
                    "peak scratch");
}

void ComputeMicrosoftAttentionFloat32(const AttentionPlan &source_plan, std::size_t input_hidden,
                                      const float *input, const float *weights, const float *bias,
                                      const float *mask, const float *past, float *output,
                                      float *present) {
  auto plan = ProjectedPlan(source_plan);
  if (plan.layout != AttentionLayout::kRank3 || plan.q_num_heads != plan.kv_num_heads ||
      plan.q_length != plan.kv_length || input_hidden == 0 ||
      ((past || present) && plan.head_dim != plan.v_head_dim)) {
    throw std::invalid_argument("MicrosoftAttention: invalid projection/cache plan.");
  }
  const auto width = Width(plan);
  const auto rows = CheckedMultiply(plan.batch, plan.q_length, "MicrosoftAttention", "rows");
  const auto count = CheckedMultiply(rows, width, "MicrosoftAttention", "projection elements");
  CheckedStride(count, "MicrosoftAttention", "projection elements");
  CheckedStride(CheckedMultiply(rows, input_hidden, "MicrosoftAttention", "input elements"),
                "MicrosoftAttention", "input elements");
  CheckedStride(CheckedMultiply(input_hidden, width, "MicrosoftAttention", "weight elements"),
                "MicrosoftAttention", "weight elements");
  if (rows && (!input || !weights || !output))
    throw std::invalid_argument("MicrosoftAttention: missing projection input/output buffer.");
  const auto q_width =
      CheckedMultiply(plan.q_num_heads, plan.head_dim, "MicrosoftAttention", "Q width");
  const auto past_count =
      CheckedProduct({plan.batch, plan.kv_num_heads, plan.past_length, plan.head_dim},
                     "MicrosoftAttention", "past elements");
  if (past_count && !past)
    throw std::invalid_argument("MicrosoftAttention: missing past buffer.");
  const auto present_count =
      CheckedProduct({plan.batch, plan.kv_num_heads, plan.total_kv_length, plan.head_dim},
                     "MicrosoftAttention", "present elements");
  const auto checked_stride = [](std::size_t value) {
    return CheckedStride(value, "MicrosoftAttention", "stride");
  };
  const auto work = checked_stride(count);
  const auto present_work = checked_stride(present_count);
  const auto gemm = ProjectionPlan(plan, input_hidden);
  ExecutionBuffer projection(
      CheckedByteSize(count, sizeof(float), "MicrosoftAttention", "projection allocation"));
  auto *qkv = static_cast<float *>(projection.data());
  if (rows) {
    gemm.Execute(input, weights, nullptr, qkv);
    if (bias) {
      ExecuteRanges(work, 1., [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i)
          qkv[i] += bias[static_cast<std::size_t>(i) % width];
      });
    }
  }
  if (present && present_count) {
    ExecuteRanges(present_work, 2., [&](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        const auto feature = i % plan.head_dim;
        const auto pos = (i / plan.head_dim) % plan.total_kv_length;
        const auto head = (i / (plan.head_dim * plan.total_kv_length)) % plan.kv_num_heads;
        const auto batch = i / (plan.head_dim * plan.total_kv_length * plan.kv_num_heads);
        if (pos < plan.past_length) {
          const auto offset =
              ((batch * plan.kv_num_heads + head) * plan.past_length + pos) * plan.head_dim +
              feature;
          present[i] = past[offset];
          present[present_count + i] = past[past_count + offset];
        } else {
          const auto offset = (batch * plan.q_length + pos - plan.past_length) * width +
                              head * plan.head_dim + feature;
          present[i] = qkv[offset + q_width];
          present[present_count + i] = qkv[offset + 2 * q_width];
        }
      }
    });
  }
  if (rows)
    ComputeAttentionFloat32(plan, qkv, qkv + q_width, qkv + 2 * q_width, mask, output, past,
                            past_count ? past + past_count : nullptr);
}
} // namespace ml_cpu_compute
