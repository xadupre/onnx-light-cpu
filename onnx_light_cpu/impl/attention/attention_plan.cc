// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/attention/attention_plan.h"

#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
#include "onnx_light_cpu/impl/attention/avx512/attention_kernel_avx512.h"
#endif
#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/impl/math/math_kernels.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

namespace onnx_light_cpu {

namespace {

[[noreturn]] void Fail(const char *message) { throw std::invalid_argument(message); }

std::size_t ToSize(std::int64_t dimension, const char *what) {
  if (dimension < 0) {
    Fail(what);
  }
  return static_cast<std::size_t>(dimension);
}

std::int64_t SaturatingAdd(std::int64_t left, std::int64_t right) noexcept {
  if (right > 0 && left > std::numeric_limits<std::int64_t>::max() - right) {
    return std::numeric_limits<std::int64_t>::max();
  }
  if (right < 0 && left < std::numeric_limits<std::int64_t>::min() - right) {
    return std::numeric_limits<std::int64_t>::min();
  }
  return left + right;
}

std::int64_t WindowCenter(std::size_t query, std::int64_t causal_offset) noexcept {
  return SaturatingAdd(static_cast<std::int64_t>(query), causal_offset);
}

} // namespace

void AttentionDescriptor::Validate() const {
  if (opset != 23 && opset != 24 && opset != 25) {
    Fail("AttentionDescriptor: only opset 23, 24, and 25 are recognized.");
  }
  if (opset < 24 && has_nonpad_kv_seqlen) {
    Fail("AttentionDescriptor: nonpad_kv_seqlen requires opset >= 24.");
  }
  if (opset < 25 && (left_window_size >= 0 || right_window_size >= 0)) {
    Fail("AttentionDescriptor: left_window_size / right_window_size require opset >= 25.");
  }
  if (qk_matmul_output_mode < 0 || qk_matmul_output_mode > 3) {
    Fail("AttentionDescriptor: qk_matmul_output_mode must be in [0, 3].");
  }
  if (has_past_key != has_past_value) {
    Fail("AttentionDescriptor: past_key and past_value must be used together.");
  }
  if (has_present_key != has_present_value) {
    Fail("AttentionDescriptor: present_key and present_value must be used together.");
  }
  if (has_nonpad_kv_seqlen && (has_past_key || has_present_key)) {
    Fail("AttentionDescriptor: nonpad_kv_seqlen cannot be combined with a tensor "
         "past/present cache.");
  }
}

void AttentionDescriptor::ValidateSupportedByMaterializedPath() const {
  Validate();
  if (softmax_precision.has_value()) {
    // FLOAT == 1 and DOUBLE == 11 in onnx::TensorProto::DataType;
    // UNDEFINED == 0 means "use input precision" (the default).
    constexpr std::int64_t kUndefined = 0;
    constexpr std::int64_t kFloat = 1;
    constexpr std::int64_t kDouble = 11;
    if (*softmax_precision != kFloat && *softmax_precision != kDouble &&
        *softmax_precision != kUndefined) {
      Fail("AttentionDescriptor: only the default, FP32, or FP64 softmax_precision is supported "
           "by the CPU materialized path.");
    }
  }
}

AttentionPlan::AttentionPlan(const AttentionDescriptor &descriptor, AttentionLayout layout,
                             std::span<const std::int64_t> q_shape,
                             std::span<const std::int64_t> k_shape,
                             std::span<const std::int64_t> v_shape,
                             std::span<const std::int64_t> mask_shape, AttentionMaskKind mask_kind,
                             std::span<const std::int64_t> past_k_shape,
                             std::span<const std::int64_t> past_v_shape)
    : layout(layout), mask_kind(mask_kind) {
  descriptor.Validate();
  if (past_k_shape.empty() != past_v_shape.empty()) {
    Fail("AttentionPlan: past_key and past_value must be used together.");
  }

  if (layout == AttentionLayout::kRank4) {
    if (q_shape.size() != 4 || k_shape.size() != 4 || v_shape.size() != 4) {
      Fail("AttentionPlan: rank-4 Q/K/V must each have 4 dimensions.");
    }
    batch = ToSize(q_shape[0], "AttentionPlan: batch_size must be non-negative.");
    q_num_heads = ToSize(q_shape[1], "AttentionPlan: q_num_heads must be non-negative.");
    q_length = ToSize(q_shape[2], "AttentionPlan: q_sequence_length must be non-negative.");
    head_dim = ToSize(q_shape[3], "AttentionPlan: head_size must be non-negative.");
    kv_num_heads = ToSize(k_shape[1], "AttentionPlan: kv_num_heads must be non-negative.");
    kv_length = ToSize(k_shape[2], "AttentionPlan: kv_sequence_length must be non-negative.");
    const std::size_t k_head_dim =
        ToSize(k_shape[3], "AttentionPlan: head_size must be non-negative.");
    v_head_dim = ToSize(v_shape[3], "AttentionPlan: v_head_size must be non-negative.");
    if (ToSize(k_shape[0], "AttentionPlan: batch_size must be non-negative.") != batch ||
        ToSize(v_shape[0], "AttentionPlan: batch_size must be non-negative.") != batch) {
      Fail("AttentionPlan: Q, K and V must share the same batch_size.");
    }
    if (ToSize(v_shape[1], "AttentionPlan: kv_num_heads must be non-negative.") != kv_num_heads) {
      Fail("AttentionPlan: K and V must share the same kv_num_heads.");
    }
    if (ToSize(v_shape[2], "AttentionPlan: kv_sequence_length must be non-negative.") !=
        kv_length) {
      Fail("AttentionPlan: K and V must share the same kv_sequence_length.");
    }
    if (k_head_dim != head_dim) {
      Fail("AttentionPlan: Q and K must share the same head_size.");
    }
    q_strides = {static_cast<std::ptrdiff_t>(q_num_heads * q_length * head_dim),
                 static_cast<std::ptrdiff_t>(q_length * head_dim),
                 static_cast<std::ptrdiff_t>(head_dim)};
    k_strides = {static_cast<std::ptrdiff_t>(kv_num_heads * kv_length * head_dim),
                 static_cast<std::ptrdiff_t>(kv_length * head_dim),
                 static_cast<std::ptrdiff_t>(head_dim)};
    v_strides = {static_cast<std::ptrdiff_t>(kv_num_heads * kv_length * v_head_dim),
                 static_cast<std::ptrdiff_t>(kv_length * v_head_dim),
                 static_cast<std::ptrdiff_t>(v_head_dim)};
    y_strides = {static_cast<std::ptrdiff_t>(q_num_heads * q_length * v_head_dim),
                 static_cast<std::ptrdiff_t>(q_length * v_head_dim),
                 static_cast<std::ptrdiff_t>(v_head_dim)};
  } else {
    if (q_shape.size() != 3 || k_shape.size() != 3 || v_shape.size() != 3) {
      Fail("AttentionPlan: rank-3 Q/K/V must each have 3 dimensions.");
    }
    if (!descriptor.q_num_heads.has_value() || !descriptor.kv_num_heads.has_value()) {
      Fail("AttentionPlan: q_num_heads and kv_num_heads are required for rank-3 Q/K/V.");
    }
    q_num_heads =
        ToSize(*descriptor.q_num_heads, "AttentionPlan: q_num_heads must be non-negative.");
    kv_num_heads =
        ToSize(*descriptor.kv_num_heads, "AttentionPlan: kv_num_heads must be non-negative.");
    if (q_num_heads == 0 || kv_num_heads == 0) {
      Fail("AttentionPlan: q_num_heads and kv_num_heads must be positive.");
    }
    batch = ToSize(q_shape[0], "AttentionPlan: batch_size must be non-negative.");
    q_length = ToSize(q_shape[1], "AttentionPlan: q_sequence_length must be non-negative.");
    kv_length = ToSize(k_shape[1], "AttentionPlan: kv_sequence_length must be non-negative.");
    const std::size_t q_hidden =
        ToSize(q_shape[2], "AttentionPlan: q_hidden_size must be non-negative.");
    const std::size_t k_hidden =
        ToSize(k_shape[2], "AttentionPlan: k_hidden_size must be non-negative.");
    const std::size_t v_hidden =
        ToSize(v_shape[2], "AttentionPlan: v_hidden_size must be non-negative.");
    if (ToSize(k_shape[0], "AttentionPlan: batch_size must be non-negative.") != batch ||
        ToSize(v_shape[0], "AttentionPlan: batch_size must be non-negative.") != batch) {
      Fail("AttentionPlan: Q, K and V must share the same batch_size.");
    }
    if (ToSize(v_shape[1], "AttentionPlan: kv_sequence_length must be non-negative.") !=
        kv_length) {
      Fail("AttentionPlan: K and V must share the same kv_sequence_length.");
    }
    if (q_hidden % q_num_heads != 0) {
      Fail("AttentionPlan: q_hidden_size must be a multiple of q_num_heads.");
    }
    if (k_hidden % kv_num_heads != 0) {
      Fail("AttentionPlan: k_hidden_size must be a multiple of kv_num_heads.");
    }
    if (v_hidden % kv_num_heads != 0) {
      Fail("AttentionPlan: v_hidden_size must be a multiple of kv_num_heads.");
    }
    head_dim = q_hidden / q_num_heads;
    if (k_hidden / kv_num_heads != head_dim) {
      Fail("AttentionPlan: Q and K must share the same head_size.");
    }
    v_head_dim = v_hidden / kv_num_heads;
    q_strides = {static_cast<std::ptrdiff_t>(q_length * q_hidden),
                 static_cast<std::ptrdiff_t>(head_dim), static_cast<std::ptrdiff_t>(q_hidden)};
    k_strides = {static_cast<std::ptrdiff_t>(kv_length * k_hidden),
                 static_cast<std::ptrdiff_t>(head_dim), static_cast<std::ptrdiff_t>(k_hidden)};
    v_strides = {static_cast<std::ptrdiff_t>(kv_length * v_hidden),
                 static_cast<std::ptrdiff_t>(v_head_dim), static_cast<std::ptrdiff_t>(v_hidden)};
    const std::size_t y_hidden = q_num_heads * v_head_dim;
    y_strides = {static_cast<std::ptrdiff_t>(q_length * y_hidden),
                 static_cast<std::ptrdiff_t>(v_head_dim), static_cast<std::ptrdiff_t>(y_hidden)};
  }

  if (kv_num_heads == 0 || q_num_heads % kv_num_heads != 0) {
    Fail("AttentionPlan: q_num_heads must be a positive multiple of kv_num_heads (MHA/GQA/MQA).");
  }
  group_size = q_num_heads / kv_num_heads;

  if (!past_k_shape.empty()) {
    if (past_k_shape.size() != 4 || past_v_shape.size() != 4) {
      Fail("AttentionPlan: past_key/past_value must each have 4 dimensions.");
    }
    if (ToSize(past_k_shape[0], "AttentionPlan: batch_size must be non-negative.") != batch ||
        ToSize(past_v_shape[0], "AttentionPlan: batch_size must be non-negative.") != batch) {
      Fail("AttentionPlan: past_key/past_value must share Q/K/V's batch_size.");
    }
    if (ToSize(past_k_shape[1], "AttentionPlan: kv_num_heads must be non-negative.") !=
            kv_num_heads ||
        ToSize(past_v_shape[1], "AttentionPlan: kv_num_heads must be non-negative.") !=
            kv_num_heads) {
      Fail("AttentionPlan: past_key/past_value must share K/V's kv_num_heads.");
    }
    past_length =
        ToSize(past_k_shape[2], "AttentionPlan: past_sequence_length must be non-negative.");
    if (ToSize(past_v_shape[2], "AttentionPlan: past_sequence_length must be non-negative.") !=
        past_length) {
      Fail("AttentionPlan: past_key and past_value must share the same past_sequence_length.");
    }
    if (ToSize(past_k_shape[3], "AttentionPlan: head_size must be non-negative.") != head_dim) {
      Fail("AttentionPlan: past_key must share Q/K's head_size.");
    }
    if (ToSize(past_v_shape[3], "AttentionPlan: v_head_size must be non-negative.") != v_head_dim) {
      Fail("AttentionPlan: past_value must share V's v_head_size.");
    }
    past_k_strides = {static_cast<std::ptrdiff_t>(kv_num_heads * past_length * head_dim),
                      static_cast<std::ptrdiff_t>(past_length * head_dim),
                      static_cast<std::ptrdiff_t>(head_dim)};
    past_v_strides = {static_cast<std::ptrdiff_t>(kv_num_heads * past_length * v_head_dim),
                      static_cast<std::ptrdiff_t>(past_length * v_head_dim),
                      static_cast<std::ptrdiff_t>(v_head_dim)};
  }
  total_kv_length = past_length + kv_length;

  scale = descriptor.scale.has_value()
              ? *descriptor.scale
              : (head_dim > 0 ? static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)))
                              : 1.0f);
  // Bottom-right, offset-aware causal masking: query in-block index `i`
  // attends key `j` iff `j <= i + causal_offset`. `causal_offset` is the
  // number of valid keys preceding this query block: `past_length` when a
  // tensor `past_key` cache is present, `0` otherwise (a per-batch offset
  // derived from `nonpad_kv_seqlen` at compute time overrides this scalar).
  causal = descriptor.is_causal;
  causal_offset = static_cast<std::int64_t>(past_length);
  softcap = descriptor.softcap;
  softmax_fp64 = descriptor.softmax_precision == 11;
  qk_matmul_output_mode = descriptor.qk_matmul_output_mode;
  has_qk_matmul_output = descriptor.has_qk_matmul_output;
  has_present_output = descriptor.has_present_key || descriptor.has_present_value;
  left_window_size = descriptor.left_window_size;
  right_window_size = descriptor.right_window_size;

  if (mask_kind != AttentionMaskKind::kNone) {
    // Right-justify mask_shape to (batch, q_num_heads, q_length,
    // total_kv_length) following ONNX broadcasting: missing leading
    // dimensions and dimensions of size 1 broadcast (stride 0).
    if (mask_shape.empty() || mask_shape.size() > 4) {
      Fail("AttentionPlan: attn_mask must have between 1 and 4 dimensions.");
    }
    const std::array<std::size_t, 4> target = {batch, q_num_heads, q_length, total_kv_length};
    std::array<std::size_t, 4> aligned = {1, 1, 1, 1};
    const std::size_t offset = 4 - mask_shape.size();
    for (std::size_t i = 0; i < mask_shape.size(); ++i) {
      aligned[offset + i] =
          ToSize(mask_shape[i], "AttentionPlan: attn_mask dimensions must be non-negative.");
    }
    for (std::size_t i = 0; i < 4; ++i) {
      if (aligned[i] != 1 && aligned[i] != target[i]) {
        Fail("AttentionPlan: attn_mask is not broadcastable to (batch_size, q_num_heads, "
             "q_sequence_length, total_sequence_length).");
      }
    }
    std::array<std::ptrdiff_t, 4> contiguous_stride = {0, 0, 0, 0};
    std::ptrdiff_t running = 1;
    for (std::size_t i = 4; i-- > 0;) {
      contiguous_stride[i] = running;
      running *= static_cast<std::ptrdiff_t>(aligned[i]);
    }
    mask_strides.batch = aligned[0] == 1 ? 0 : contiguous_stride[0];
    mask_strides.head = aligned[1] == 1 ? 0 : contiguous_stride[1];
    mask_strides.q = aligned[2] == 1 ? 0 : contiguous_stride[2];
    mask_strides.kv = aligned[3] == 1 ? 0 : contiguous_stride[3];
  }
}

std::vector<std::int64_t> AttentionPlan::output_shape() const {
  if (layout == AttentionLayout::kRank4) {
    return {static_cast<std::int64_t>(batch), static_cast<std::int64_t>(q_num_heads),
            static_cast<std::int64_t>(q_length), static_cast<std::int64_t>(v_head_dim)};
  }
  return {static_cast<std::int64_t>(batch), static_cast<std::int64_t>(q_length),
          static_cast<std::int64_t>(q_num_heads * v_head_dim)};
}

std::vector<std::int64_t> AttentionPlan::present_key_shape() const {
  return {static_cast<std::int64_t>(batch), static_cast<std::int64_t>(kv_num_heads),
          static_cast<std::int64_t>(total_kv_length), static_cast<std::int64_t>(head_dim)};
}

std::vector<std::int64_t> AttentionPlan::present_value_shape() const {
  return {static_cast<std::int64_t>(batch), static_cast<std::int64_t>(kv_num_heads),
          static_cast<std::int64_t>(total_kv_length), static_cast<std::int64_t>(v_head_dim)};
}

std::vector<std::int64_t> AttentionPlan::qk_matmul_output_shape() const {
  return {static_cast<std::int64_t>(batch), static_cast<std::int64_t>(q_num_heads),
          static_cast<std::int64_t>(q_length), static_cast<std::int64_t>(total_kv_length)};
}

namespace {

constexpr float kNegativeInfinity = -std::numeric_limits<float>::infinity();

bool IsMaskFilterValue(float value) noexcept {
  return value == kNegativeInfinity || value == std::numeric_limits<float>::lowest();
}

} // namespace

void ComputeAttentionFloat32Materialized(const AttentionPlan &plan, const float *q, const float *k,
                                         const float *v, const void *mask, float *y,
                                         const float *past_k, const float *past_v,
                                         const std::int64_t *nonpad_kv_seqlen,
                                         float *qk_matmul_output) {
  const auto *mask_bool = static_cast<const std::uint8_t *>(mask);
  const auto *mask_float = static_cast<const float *>(mask);
  const std::size_t total_kv_length = plan.total_kv_length;
  const bool has_softcap = plan.softcap != 0.0f;

  std::vector<float> scores(total_kv_length);
  std::vector<float> qk_row(plan.has_qk_matmul_output ? total_kv_length : 0);
  std::vector<double> softmax_fp64(plan.softmax_fp64 ? total_kv_length : 0);
  for (std::size_t b = 0; b < plan.batch; ++b) {
    // Bottom-right, offset-aware causal frontier: `nonpad_kv_seqlen`, when
    // supplied, overrides the plan's scalar `causal_offset` with a per-batch
    // value (`nonpad_kv_seqlen[b] - q_length`) and additionally masks KV
    // positions beyond `nonpad_kv_seqlen[b]` regardless of `is_causal`.
    const std::int64_t causal_offset =
        nonpad_kv_seqlen != nullptr ? nonpad_kv_seqlen[b] - static_cast<std::int64_t>(plan.q_length)
                                    : plan.causal_offset;
    const std::int64_t nonpad_length = nonpad_kv_seqlen != nullptr ? nonpad_kv_seqlen[b] : -1;

    for (std::size_t h = 0; h < plan.q_num_heads; ++h) {
      const std::size_t kv_h = h / plan.group_size;
      const float *q_head = q + b * plan.q_strides.batch + h * plan.q_strides.head;
      const float *k_head = k + b * plan.k_strides.batch + kv_h * plan.k_strides.head;
      const float *v_head = v + b * plan.v_strides.batch + kv_h * plan.v_strides.head;
      const float *past_k_head = past_k != nullptr ? past_k + b * plan.past_k_strides.batch +
                                                         kv_h * plan.past_k_strides.head
                                                   : nullptr;
      const float *past_v_head = past_v != nullptr ? past_v + b * plan.past_v_strides.batch +
                                                         kv_h * plan.past_v_strides.head
                                                   : nullptr;
      float *y_head = y + b * plan.y_strides.batch + h * plan.y_strides.head;
      float *qk_head =
          plan.has_qk_matmul_output
              ? qk_matmul_output + (b * plan.q_num_heads + h) * plan.q_length * total_kv_length
              : nullptr;
      const std::ptrdiff_t mask_base =
          b * plan.mask_strides.batch + static_cast<std::ptrdiff_t>(h) * plan.mask_strides.head;

      for (std::size_t i = 0; i < plan.q_length; ++i) {
        const float *q_row = q_head + i * plan.q_strides.sequence;
        const std::ptrdiff_t mask_row =
            mask_base + static_cast<std::ptrdiff_t>(i) * plan.mask_strides.q;
        float row_max = kNegativeInfinity;
        // Tracks whether every KV position of this row is disallowed by the
        // combined causal/padding/attn_mask bias (independent of the raw QK
        // score), matching the spec's fully-masked-row guard.
        bool row_has_unmasked_key = false;
        for (std::size_t j = 0; j < total_kv_length; ++j) {
          const std::int64_t window_center = WindowCenter(i, causal_offset);
          bool allowed = true;
          if (plan.causal) {
            allowed = static_cast<std::int64_t>(j) <= window_center;
          }
          if (nonpad_length >= 0) {
            allowed = allowed && static_cast<std::int64_t>(j) < nonpad_length;
          }
          if (allowed) {
            if (plan.left_window_size >= 0 &&
                static_cast<std::int64_t>(j) <
                    SaturatingAdd(window_center, -plan.left_window_size)) {
              allowed = false;
            }
            if (plan.right_window_size >= 0 &&
                static_cast<std::int64_t>(j) >
                    SaturatingAdd(window_center, plan.right_window_size)) {
              allowed = false;
            }
          }
          float additive_bias = 0.0f;
          if (plan.mask_kind == AttentionMaskKind::kBoolean) {
            const std::ptrdiff_t index =
                mask_row + static_cast<std::ptrdiff_t>(j) * plan.mask_strides.kv;
            allowed = allowed && mask_bool[index] != 0;
          } else if (plan.mask_kind == AttentionMaskKind::kAdditive) {
            const std::ptrdiff_t index =
                mask_row + static_cast<std::ptrdiff_t>(j) * plan.mask_strides.kv;
            additive_bias = mask_float[index];
          }
          const float bias = (allowed ? 0.0f : kNegativeInfinity) + additive_bias;
          row_has_unmasked_key |= !IsMaskFilterValue(bias);

          const float *k_row;
          if (j < plan.past_length) {
            k_row = past_k_head + j * plan.past_k_strides.sequence;
          } else {
            k_row = k_head + (j - plan.past_length) * plan.k_strides.sequence;
          }
          float dot = 0.0f;
          for (std::size_t d = 0; d < plan.head_dim; ++d) {
            dot += q_row[d] * k_row[d];
          }
          const float raw = plan.scale * dot;
          const float capped = has_softcap ? plan.softcap * std::tanh(raw / plan.softcap) : raw;
          const float with_bias = capped + bias;
          scores[j] = with_bias;
          row_max = std::max(row_max, with_bias);
          if (plan.has_qk_matmul_output) {
            switch (plan.qk_matmul_output_mode) {
            case 0:
              qk_row[j] = raw;
              break;
            case 1:
              qk_row[j] = capped;
              break;
            default:
              qk_row[j] = with_bias;
              break;
            }
          }
        }

        float *y_row = y_head + i * plan.y_strides.sequence;
        float *qk_out_row = plan.has_qk_matmul_output ? qk_head + i * total_kv_length : nullptr;
        const bool row_all_masked = !row_has_unmasked_key;
        if (row_all_masked) {
          // Fully-masked query row: zero output, not NaN, for both Y and the
          // mode-3 qk_matmul_output. Decided on the additive bias (not the
          // possibly -inf-but-finite-looking QK score) so it matches every
          // combination of causal/padding/attn_mask.
          std::fill(y_row, y_row + plan.v_head_dim, 0.0f);
          if (plan.has_qk_matmul_output) {
            if (plan.qk_matmul_output_mode == 3) {
              std::fill(qk_out_row, qk_out_row + total_kv_length, 0.0f);
            } else {
              std::copy(qk_row.begin(), qk_row.end(), qk_out_row);
            }
          }
          continue;
        }
        double sum_fp64 = 0.0;
        float sum_fp32 = 0.0f;
        for (std::size_t j = 0; j < total_kv_length; ++j) {
          const double shifted = static_cast<double>(scores[j]) - static_cast<double>(row_max);
          const double probability =
              scores[j] == kNegativeInfinity
                  ? 0.0
                  : (plan.softmax_fp64
                         ? std::exp(shifted)
                         : static_cast<double>(std::exp(static_cast<float>(shifted))));
          if (plan.softmax_fp64) {
            softmax_fp64[j] = probability;
            sum_fp64 += probability;
          } else {
            scores[j] = static_cast<float>(probability);
            sum_fp32 += scores[j];
          }
        }
        const double inv_sum =
            plan.softmax_fp64 ? 1.0 / sum_fp64 : static_cast<double>(1.0f / sum_fp32);
        std::fill(y_row, y_row + plan.v_head_dim, 0.0f);
        for (std::size_t j = 0; j < total_kv_length; ++j) {
          const double probability =
              plan.softmax_fp64 ? softmax_fp64[j] : static_cast<double>(scores[j]);
          const float p = static_cast<float>(probability * inv_sum);
          if (plan.has_qk_matmul_output) {
            qk_out_row[j] = plan.qk_matmul_output_mode == 3 ? p : qk_row[j];
          }
          if (p == 0.0f) {
            continue;
          }
          const float *v_row;
          if (j < plan.past_length) {
            v_row = past_v_head + j * plan.past_v_strides.sequence;
          } else {
            v_row = v_head + (j - plan.past_length) * plan.v_strides.sequence;
          }
          for (std::size_t d = 0; d < plan.v_head_dim; ++d) {
            y_row[d] += p * v_row[d];
          }
        }
      }
    }
  }
}

namespace {

// KV block size for the streaming recurrence: bounds the temporary score
// buffer to a constant instead of `O(total_kv_length)`.
constexpr std::size_t kStreamingKvBlock = 128;

// Estimated FMA cost of one query row visiting every KV position it may
// attend to (`head_dim` for the QK dot product plus `v_head_dim` for the
// P @ V accumulation): only used to size the runtime-owned outer schedule
// (`ExecuteRanges`) below; it never changes the numeric result.
std::size_t StreamingParticipantCount(const AttentionPlan &plan, std::size_t total_rows) {
  constexpr std::size_t kTargetFmasPerParticipant = 2'000'000;
  constexpr std::size_t kMaximumParticipants = 16;
  const std::size_t fmas_per_row =
      (plan.head_dim + plan.v_head_dim) * std::max<std::size_t>(plan.total_kv_length, 1);
  const std::size_t total_fmas = total_rows * fmas_per_row;
  const std::size_t participants =
      (total_fmas + kTargetFmasPerParticipant - 1) / kTargetFmasPerParticipant;
  return std::clamp<std::size_t>(participants, 1, kMaximumParticipants);
}

void AttentionExp(float *values, std::size_t count) {
  static const SimdLevel simd = DetectSimdLevel();
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
  if (simd == SimdLevel::kAVX512) {
    ExpFloat32_AVX512(values, values, count);
    return;
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
  if (simd >= SimdLevel::kAVX2) {
    ExpFloat32_AVX2_FMA(values, values, count);
    return;
  }
#endif
  for (std::size_t index = 0; index < count; ++index) {
    values[index] = std::exp(values[index]);
  }
}

// FP32-storage codec: streaming Q/K/V/Y elements are already FP32, so
// loading/storing is a no-op copy.
struct Float32Codec {
  using Storage = float;
  static float Load(Storage value) noexcept { return value; }
  static Storage Store(float value) noexcept { return value; }
};

// FP16-storage codec: elements are IEEE-754 binary16 stored as raw
// `uint16_t` bit patterns; every arithmetic operation still happens in FP32.
struct Float16Codec {
  using Storage = std::uint16_t;
  static float Load(Storage value) noexcept { return detail::Float16BitsToFloat(value); }
  static Storage Store(float value) noexcept { return detail::FloatToFloat16Bits(value); }
};

// BF16-storage codec: elements are `bfloat16` stored as raw `uint16_t` bit
// patterns; every arithmetic operation still happens in FP32.
struct BFloat16Codec {
  using Storage = std::uint16_t;
  static float Load(Storage value) noexcept { return detail::Bfloat16BitsToFloat(value); }
  static Storage Store(float value) noexcept { return detail::FloatToBFloat16Bits(value); }
};

void ConvertFloat16Rank3ToRank4(const std::uint16_t *source, float *destination, std::size_t batch,
                                std::size_t heads, std::size_t length, std::size_t dimension) {
  const std::size_t rows = batch * heads * length;
  ExecuteRanges(static_cast<std::int64_t>(rows), static_cast<double>(dimension) * 0.25, 1,
                [&](std::int64_t begin, std::int64_t end) {
                  for (std::size_t row = static_cast<std::size_t>(begin);
                       row < static_cast<std::size_t>(end); ++row) {
                    const std::size_t b = row / (heads * length);
                    const std::size_t remainder = row % (heads * length);
                    const std::size_t h = remainder / length;
                    const std::size_t sequence = remainder % length;
                    const std::uint16_t *source_row =
                        source + ((b * length + sequence) * heads + h) * dimension;
                    detail::ConvertFloat16ToFloat32(source_row, destination + row * dimension,
                                                    dimension);
                  }
                });
}

void ConvertFloat32Rank4ToFloat16Rank3(const float *source, std::uint16_t *destination,
                                       std::size_t batch, std::size_t heads, std::size_t length,
                                       std::size_t dimension) {
  const std::size_t rows = batch * heads * length;
  ExecuteRanges(static_cast<std::int64_t>(rows), static_cast<double>(dimension) * 0.25, 1,
                [&](std::int64_t begin, std::int64_t end) {
                  for (std::size_t row = static_cast<std::size_t>(begin);
                       row < static_cast<std::size_t>(end); ++row) {
                    const std::size_t b = row / (heads * length);
                    const std::size_t remainder = row % (heads * length);
                    const std::size_t h = remainder / length;
                    const std::size_t sequence = remainder % length;
                    std::uint16_t *destination_row =
                        destination + ((b * length + sequence) * heads + h) * dimension;
                    detail::ConvertFloat32ToFloat16(source + row * dimension, destination_row,
                                                    dimension);
                  }
                });
}

// Shared streaming online-softmax recurrence, templated on `Codec` so FP32,
// FP16, and BF16 Q/K/V/Y share one implementation while every intermediate
// (score, softmax denominator, and P @ V accumulator) stays FP32. Consumes an
// internal tensor `past_key`/`past_value` cache and/or `nonpad_kv_seqlen`
// block by block: neither is materialized. Callers (the type-specific public
// entry points) must not invoke this when `plan.has_qk_matmul_output` or
// `plan.has_present_output` is set: those require the materialized path.
// `past_k`/`past_v` must be non-null whenever `plan.past_length != 0` (and
// may be null otherwise); this mirrors the same precondition already
// documented on `ComputeAttentionFloat32Materialized`.
template <typename Codec>
void ComputeAttentionStreamingGeneric(const AttentionPlan &plan, const typename Codec::Storage *q,
                                      const typename Codec::Storage *k,
                                      const typename Codec::Storage *v, const void *mask,
                                      typename Codec::Storage *y,
                                      const typename Codec::Storage *past_k,
                                      const typename Codec::Storage *past_v,
                                      const std::int64_t *nonpad_kv_seqlen) {
  const auto *mask_bool = static_cast<const std::uint8_t *>(mask);
  const auto *mask_float = static_cast<const float *>(mask);
  const std::size_t total_kv_length = plan.total_kv_length;
  const bool has_softcap = plan.softcap != 0.0f;
  const std::size_t block =
      total_kv_length == 0 ? std::size_t{0} : std::min(total_kv_length, kStreamingKvBlock);
  const std::size_t rows_per_batch = plan.q_num_heads * plan.q_length;
  const std::size_t total_rows = plan.batch * rows_per_batch;
  const std::size_t participants = StreamingParticipantCount(plan, total_rows);
  const ExecutionSchedule schedule{
      1, static_cast<std::int64_t>((total_rows + participants - 1) / participants),
      static_cast<std::int64_t>(participants), static_cast<std::int64_t>(participants)};

  // Runtime-owned outer schedule: one task per (batch, head, query-row)
  // triple. `ExecuteRanges` decides how many workers to admit from the total
  // estimated cost; it never fixes a thread count, and collapses to a single
  // worker (no parallel overhead) when there is little work -- always the
  // case for short-query/decode invocations -- while a prefill invocation
  // (many rows) exposes enough independent outer work to scale.
  ExecuteRanges(
      static_cast<std::int64_t>(total_rows), schedule, [&](std::int64_t begin, std::int64_t end) {
        // Peak temporary storage: one `Bc`-sized score tile, one
        // `head_dim`-sized FP32 query cache, and one `v_head_dim`
        // accumulator per worker (`Br == 1`); never the full
        // `[q_length, total_kv_length]` score or probability tensor.
        thread_local std::vector<float> scores;
        thread_local std::vector<float> q_fp32;
        thread_local std::vector<float> accumulator;
        scores.resize(block);
        q_fp32.resize(plan.head_dim);
        accumulator.resize(plan.v_head_dim);

        for (std::int64_t row = begin; row < end; ++row) {
          const std::size_t b = static_cast<std::size_t>(row) / rows_per_batch;
          const std::size_t rem = static_cast<std::size_t>(row) % rows_per_batch;
          const std::size_t h = rem / plan.q_length;
          const std::size_t i = rem % plan.q_length;
          const std::size_t kv_h = h / plan.group_size;

          const typename Codec::Storage *q_head =
              q + b * plan.q_strides.batch + h * plan.q_strides.head;
          const typename Codec::Storage *k_head =
              k + b * plan.k_strides.batch + kv_h * plan.k_strides.head;
          const typename Codec::Storage *v_head =
              v + b * plan.v_strides.batch + kv_h * plan.v_strides.head;
          const typename Codec::Storage *past_k_head =
              past_k != nullptr
                  ? past_k + b * plan.past_k_strides.batch + kv_h * plan.past_k_strides.head
                  : nullptr;
          const typename Codec::Storage *past_v_head =
              past_v != nullptr
                  ? past_v + b * plan.past_v_strides.batch + kv_h * plan.past_v_strides.head
                  : nullptr;
          typename Codec::Storage *y_head = y + b * plan.y_strides.batch + h * plan.y_strides.head;
          const std::ptrdiff_t mask_base =
              b * plan.mask_strides.batch + static_cast<std::ptrdiff_t>(h) * plan.mask_strides.head;

          // Bottom-right, offset-aware causal frontier: `nonpad_kv_seqlen`,
          // when supplied, overrides the plan's scalar `causal_offset` with a
          // per-batch value and additionally bounds the KV axis to a
          // contiguous prefix, exactly like the materialized path.
          const std::int64_t causal_offset =
              nonpad_kv_seqlen != nullptr
                  ? nonpad_kv_seqlen[b] - static_cast<std::int64_t>(plan.q_length)
                  : plan.causal_offset;
          const std::int64_t nonpad_length = nonpad_kv_seqlen != nullptr ? nonpad_kv_seqlen[b] : -1;
          const typename Codec::Storage *q_row = q_head + i * plan.q_strides.sequence;
          const std::ptrdiff_t mask_row =
              mask_base + static_cast<std::ptrdiff_t>(i) * plan.mask_strides.q;
          typename Codec::Storage *y_row = y_head + i * plan.y_strides.sequence;

          // Fold Q into FP32 once per row rather than once per KV element.
          for (std::size_t d = 0; d < plan.head_dim; ++d) {
            q_fp32[d] = Codec::Load(q_row[d]);
          }

          // Combined causal/`nonpad_kv_seqlen` bound: the last KV index
          // (inclusive) this row may ever attend to. Positions beyond it are
          // never visited -- a safe, always-correct tile skip, since both
          // are contiguous suffixes of the KV axis.
          std::int64_t bound = static_cast<std::int64_t>(total_kv_length) - 1;
          if (plan.causal) {
            bound = std::min(bound, WindowCenter(i, causal_offset));
          }
          if (nonpad_length >= 0) {
            bound = std::min(bound, nonpad_length - 1);
          }
          // Window upper bound: right_window_size limits how far *ahead*
          // each query can look (diff = i + offset - j, disallow -diff > right_window_size,
          // i.e. j > i + offset + right_window_size).
          if (plan.right_window_size >= 0) {
            bound = std::min(bound,
                             SaturatingAdd(WindowCenter(i, causal_offset), plan.right_window_size));
          }
          const std::size_t kv_limit =
              bound < 0 ? std::size_t{0}
                        : std::min(total_kv_length, static_cast<std::size_t>(bound) + 1);

          // Window lower bound: left_window_size limits how far *back* each
          // query can look (diff > left_window_size ↔ j < i + offset - left_window_size).
          std::size_t kv_start = 0;
          if (plan.left_window_size >= 0) {
            const std::int64_t lo =
                SaturatingAdd(WindowCenter(i, causal_offset), -plan.left_window_size);
            if (lo > 0) {
              kv_start = static_cast<std::size_t>(lo);
            }
          }

          float m = kNegativeInfinity;
          float l = 0.0f;
          std::fill(accumulator.begin(), accumulator.end(), 0.0f);
          bool any_valid = false;

          for (std::size_t j0 = kv_start; j0 < kv_limit; j0 += block) {
            const std::size_t j1 = std::min(j0 + block, kv_limit);
            const std::size_t count = j1 - j0;

            // Safe inferable tile skip: a boolean `attn_mask` that
            // disallows every position in this block contributes nothing,
            // so the QK dot products for the whole block can be skipped
            // outright. Additive masks are not inspected here: an arbitrary
            // real bias cannot be assumed to ever equal -infinity, so no
            // skip is inferred for them and they remain exactly correct.
            if (plan.mask_kind == AttentionMaskKind::kBoolean) {
              bool any_allowed = false;
              for (std::size_t jj = 0; jj < count && !any_allowed; ++jj) {
                const std::size_t j = j0 + jj;
                const std::ptrdiff_t index =
                    mask_row + static_cast<std::ptrdiff_t>(j) * plan.mask_strides.kv;
                any_allowed = mask_bool[index] != 0;
              }
              if (!any_allowed) {
                continue;
              }
            }

            float block_max = kNegativeInfinity;
            for (std::size_t jj = 0; jj < count; ++jj) {
              const std::size_t j = j0 + jj;
              bool allowed = true;
              float additive_bias = 0.0f;
              if (plan.mask_kind == AttentionMaskKind::kBoolean) {
                const std::ptrdiff_t index =
                    mask_row + static_cast<std::ptrdiff_t>(j) * plan.mask_strides.kv;
                allowed = mask_bool[index] != 0;
              } else if (plan.mask_kind == AttentionMaskKind::kAdditive) {
                const std::ptrdiff_t index =
                    mask_row + static_cast<std::ptrdiff_t>(j) * plan.mask_strides.kv;
                additive_bias = mask_float[index];
              }
              const float bias = (allowed ? 0.0f : kNegativeInfinity) + additive_bias;

              const typename Codec::Storage *k_row;
              if (j < plan.past_length) {
                k_row = past_k_head + j * plan.past_k_strides.sequence;
              } else {
                k_row = k_head + (j - plan.past_length) * plan.k_strides.sequence;
              }
              float dot = 0.0f;
              if constexpr (std::is_same_v<Codec, Float32Codec>) {
                constexpr std::size_t kDotLanes = 8;
                float partial[kDotLanes] = {};
                std::size_t d = 0;
                for (; d + kDotLanes <= plan.head_dim; d += kDotLanes) {
                  for (std::size_t lane = 0; lane < kDotLanes; ++lane) {
                    partial[lane] += q_fp32[d + lane] * k_row[d + lane];
                  }
                }
                dot = ((partial[0] + partial[1]) + (partial[2] + partial[3])) +
                      ((partial[4] + partial[5]) + (partial[6] + partial[7]));
                for (; d < plan.head_dim; ++d) {
                  dot += q_fp32[d] * k_row[d];
                }
              } else {
                for (std::size_t d = 0; d < plan.head_dim; ++d) {
                  dot += q_fp32[d] * Codec::Load(k_row[d]);
                }
              }
              const float raw = plan.scale * dot;
              const float capped = has_softcap ? plan.softcap * std::tanh(raw / plan.softcap) : raw;
              const float with_bias = capped + bias;
              scores[jj] = with_bias;
              if (!IsMaskFilterValue(bias)) {
                any_valid = true;
              }
              block_max = std::max(block_max, with_bias);
            }

            // Online softmax recurrence: rescale the running denominator and
            // accumulator to the new running maximum before folding in this
            // block's contribution.
            const float m_new = std::max(m, block_max);
            float correction = 1.0f;
            if (m_new != kNegativeInfinity) {
              correction = m == kNegativeInfinity ? 0.0f : std::exp(m - m_new);
            }
            l *= correction;
            for (std::size_t d = 0; d < plan.v_head_dim; ++d) {
              accumulator[d] *= correction;
            }

            if (m_new == kNegativeInfinity) {
              std::fill_n(scores.begin(), count, 0.0f);
            } else {
              for (std::size_t jj = 0; jj < count; ++jj) {
                const float score = scores[jj];
                scores[jj] = score == kNegativeInfinity ? 0.0f : std::exp(score - m_new);
              }
            }
            for (std::size_t jj = 0; jj < count; ++jj) {
              const float p = scores[jj];
              if (p == 0.0f) {
                continue;
              }
              l += p;
              const std::size_t j = j0 + jj;
              const typename Codec::Storage *v_row;
              if (j < plan.past_length) {
                v_row = past_v_head + j * plan.past_v_strides.sequence;
              } else {
                v_row = v_head + (j - plan.past_length) * plan.v_strides.sequence;
              }
              for (std::size_t d = 0; d < plan.v_head_dim; ++d) {
                accumulator[d] += p * Codec::Load(v_row[d]);
              }
            }
            m = m_new;
          }

          if (!any_valid || l == 0.0f) {
            // Fully-masked query row (or an empty KV length): zero output
            // rather than NaN, matching the materialized path.
            for (std::size_t d = 0; d < plan.v_head_dim; ++d) {
              y_row[d] = Codec::Store(0.0f);
            }
            continue;
          }
          const float inv_l = 1.0f / l;
          for (std::size_t d = 0; d < plan.v_head_dim; ++d) {
            y_row[d] = Codec::Store(accumulator[d] * inv_l);
          }
        }
      });
}

void ComputeAttentionFloat32Tiled(const AttentionPlan &plan, const float *q, const float *k,
                                  const float *v, const void *mask, float *y,
                                  const std::int64_t *nonpad_kv_seqlen) {
  constexpr std::size_t kQueryBlock = 16;
  const std::size_t kv_block = std::min(plan.total_kv_length, kStreamingKvBlock);
  const std::size_t query_block = std::min(plan.q_length, kQueryBlock);
  const std::size_t query_blocks = (plan.q_length + query_block - 1) / query_block;
  const std::size_t tasks_per_batch = plan.q_num_heads * query_blocks;
  const std::size_t total_tasks = plan.batch * tasks_per_batch;
  const auto *mask_bool = static_cast<const std::uint8_t *>(mask);
  const auto *mask_float = static_cast<const float *>(mask);
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
  const bool use_bounded_avx512 =
      DetectSimdLevel() == SimdLevel::kAVX512 && plan.mask_kind == AttentionMaskKind::kNone &&
      plan.softcap == 0.0f && plan.left_window_size < 0 && plan.right_window_size < 0;
#endif
  const std::size_t participants =
      plan.q_length <= kQueryBlock
          ? 1
          : StreamingParticipantCount(plan, plan.batch * plan.q_num_heads * plan.q_length);
  const ExecutionSchedule schedule{
      1, static_cast<std::int64_t>((total_tasks + participants - 1) / participants),
      static_cast<std::int64_t>(participants), static_cast<std::int64_t>(participants)};
  ExecuteRanges(
      static_cast<std::int64_t>(total_tasks), schedule, [&](std::int64_t begin, std::int64_t end) {
        thread_local std::vector<float> scores;
        thread_local std::vector<float> accumulator;
        thread_local std::vector<float> product;
        thread_local std::vector<float> maxima;
        thread_local std::vector<float> denominators;
        thread_local std::vector<std::uint8_t> valid;
        scores.resize(query_block * kv_block);
        accumulator.resize(query_block * plan.v_head_dim);
        product.resize(query_block * plan.v_head_dim);
        maxima.resize(query_block);
        denominators.resize(query_block);
        valid.resize(query_block);

        for (std::int64_t task = begin; task < end; ++task) {
          const std::size_t b = static_cast<std::size_t>(task) / tasks_per_batch;
          const std::size_t rem = static_cast<std::size_t>(task) % tasks_per_batch;
          const std::size_t h = rem / query_blocks;
          const std::size_t qb = rem % query_blocks;
          const std::size_t q0 = qb * query_block;
          const std::size_t rows = std::min(query_block, plan.q_length - q0);
          const std::size_t kv_h = h / plan.group_size;
          const float *q_block =
              q + b * plan.q_strides.batch + h * plan.q_strides.head + q0 * plan.q_strides.sequence;
          const float *k_head = k + b * plan.k_strides.batch + kv_h * plan.k_strides.head;
          const float *v_head = v + b * plan.v_strides.batch + kv_h * plan.v_strides.head;
          float *y_block =
              y + b * plan.y_strides.batch + h * plan.y_strides.head + q0 * plan.y_strides.sequence;
          const std::ptrdiff_t mask_base =
              b * plan.mask_strides.batch + static_cast<std::ptrdiff_t>(h) * plan.mask_strides.head;
          const std::int64_t causal_offset =
              nonpad_kv_seqlen != nullptr
                  ? nonpad_kv_seqlen[b] - static_cast<std::int64_t>(plan.q_length)
                  : plan.causal_offset;
          const std::int64_t nonpad_length = nonpad_kv_seqlen != nullptr ? nonpad_kv_seqlen[b] : -1;
          std::int64_t task_bound = static_cast<std::int64_t>(plan.total_kv_length) - 1;
          if (plan.causal) {
            task_bound = std::min(task_bound, WindowCenter(q0 + rows - 1, causal_offset));
          }
          if (nonpad_length >= 0) {
            task_bound = std::min(task_bound, nonpad_length - 1);
          }
          if (plan.right_window_size >= 0) {
            task_bound =
                std::min(task_bound, SaturatingAdd(WindowCenter(q0 + rows - 1, causal_offset),
                                                   plan.right_window_size));
          }
          const std::size_t task_kv_limit =
              task_bound < 0
                  ? 0
                  : std::min(plan.total_kv_length, static_cast<std::size_t>(task_bound) + 1);
          // Left-window lower bound for the entire query block: the most
          // restrictive (earliest query's) left window start.
          std::size_t task_kv_start = 0;
          if (plan.left_window_size >= 0) {
            const std::int64_t lo =
                SaturatingAdd(WindowCenter(q0, causal_offset), -plan.left_window_size);
            if (lo > 0) {
              task_kv_start = static_cast<std::size_t>(lo);
            }
          }

          std::fill_n(accumulator.begin(), rows * plan.v_head_dim, 0.0f);
          std::fill_n(maxima.begin(), rows, kNegativeInfinity);
          std::fill_n(denominators.begin(), rows, 0.0f);
          std::fill_n(valid.begin(), rows, std::uint8_t{0});

          for (std::size_t j0 = task_kv_start; j0 < task_kv_limit; j0 += kv_block) {
            const std::size_t columns = std::min(kv_block, task_kv_limit - j0);
            GemmFloat32(false, true, rows, columns, plan.head_dim, plan.scale, q_block,
                        k_head + j0 * plan.k_strides.sequence, 0.0f, nullptr, scores.data());

            for (std::size_t i = 0; i < rows; ++i) {
              float *score_row = scores.data() + i * columns;
              const std::size_t query = q0 + i;
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
              if (use_bounded_avx512) {
                std::int64_t bound = static_cast<std::int64_t>(plan.total_kv_length) - 1;
                if (plan.causal) {
                  bound = std::min(bound, WindowCenter(query, causal_offset));
                }
                if (nonpad_length >= 0) {
                  bound = std::min(bound, nonpad_length - 1);
                }
                const std::size_t valid_columns =
                    bound < static_cast<std::int64_t>(j0)
                        ? 0
                        : std::min(columns, static_cast<std::size_t>(bound) - j0 + 1);
                if (valid_columns == 0) {
                  std::fill_n(score_row, columns, 0.0f);
                  continue;
                }
                std::fill(score_row + valid_columns, score_row + columns, kNegativeInfinity);
                const AttentionSoftmaxBlockResult result = AttentionSoftmaxBlockFloat32_AVX512(
                    score_row, columns, maxima[i], denominators[i]);
                maxima[i] = result.maximum;
                valid[i] = 1;
                float *accumulator_row = accumulator.data() + i * plan.v_head_dim;
                for (std::size_t d = 0; d < plan.v_head_dim; ++d) {
                  accumulator_row[d] *= result.correction;
                }
                continue;
              }
#endif
              const std::ptrdiff_t mask_row =
                  mask_base + static_cast<std::ptrdiff_t>(query) * plan.mask_strides.q;
              float block_max = kNegativeInfinity;
              for (std::size_t jj = 0; jj < columns; ++jj) {
                const std::size_t j = j0 + jj;
                const std::int64_t window_center = WindowCenter(query, causal_offset);
                bool allowed = (!plan.causal || static_cast<std::int64_t>(j) <= window_center) &&
                               (nonpad_length < 0 || static_cast<std::int64_t>(j) < nonpad_length);
                if (allowed) {
                  if (plan.left_window_size >= 0 &&
                      static_cast<std::int64_t>(j) <
                          SaturatingAdd(window_center, -plan.left_window_size)) {
                    allowed = false;
                  }
                  if (plan.right_window_size >= 0 &&
                      static_cast<std::int64_t>(j) >
                          SaturatingAdd(window_center, plan.right_window_size)) {
                    allowed = false;
                  }
                }
                float additive_bias = 0.0f;
                if (plan.mask_kind == AttentionMaskKind::kBoolean) {
                  const std::ptrdiff_t index =
                      mask_row + static_cast<std::ptrdiff_t>(j) * plan.mask_strides.kv;
                  allowed = allowed && mask_bool[index] != 0;
                } else if (plan.mask_kind == AttentionMaskKind::kAdditive) {
                  const std::ptrdiff_t index =
                      mask_row + static_cast<std::ptrdiff_t>(j) * plan.mask_strides.kv;
                  additive_bias = mask_float[index];
                }
                valid[i] |= static_cast<std::uint8_t>(allowed && !IsMaskFilterValue(additive_bias));
                float score = score_row[jj];
                if (plan.softcap != 0.0f) {
                  score = plan.softcap * std::tanh(score / plan.softcap);
                }
                score = allowed ? score + additive_bias : kNegativeInfinity;
                score_row[jj] = score;
                block_max = std::max(block_max, score);
              }

              const float new_max = std::max(maxima[i], block_max);
              const float correction =
                  new_max == kNegativeInfinity
                      ? 0.0f
                      : (maxima[i] == kNegativeInfinity ? 0.0f : std::exp(maxima[i] - new_max));
              denominators[i] *= correction;
              float *accumulator_row = accumulator.data() + i * plan.v_head_dim;
              for (std::size_t d = 0; d < plan.v_head_dim; ++d) {
                accumulator_row[d] *= correction;
              }
              if (new_max == kNegativeInfinity) {
                std::fill_n(score_row, columns, 0.0f);
              } else {
                for (std::size_t jj = 0; jj < columns; ++jj) {
                  score_row[jj] -= new_max;
                }
                AttentionExp(score_row, columns);
                for (std::size_t jj = 0; jj < columns; ++jj) {
                  denominators[i] += score_row[jj];
                }
              }
              maxima[i] = new_max;
            }

            GemmFloat32(false, false, rows, plan.v_head_dim, columns, 1.0f, scores.data(),
                        v_head + j0 * plan.v_strides.sequence, 0.0f, nullptr, product.data());
            for (std::size_t index = 0; index < rows * plan.v_head_dim; ++index) {
              accumulator[index] += product[index];
            }
          }

          for (std::size_t i = 0; i < rows; ++i) {
            float *y_row = y_block + i * plan.y_strides.sequence;
            const float scale = valid[i] && denominators[i] != 0.0f ? 1.0f / denominators[i] : 0.0f;
            const float *accumulator_row = accumulator.data() + i * plan.v_head_dim;
            for (std::size_t d = 0; d < plan.v_head_dim; ++d) {
              y_row[d] = accumulator_row[d] * scale;
            }
          }
        }
      });
}

} // namespace

void ComputeAttentionFloat32Streaming(const AttentionPlan &plan, const float *q, const float *k,
                                      const float *v, const void *mask, float *y,
                                      const float *past_k, const float *past_v,
                                      const std::int64_t *nonpad_kv_seqlen) {
  if (plan.layout == AttentionLayout::kRank4 && plan.past_length == 0 && plan.q_length > 8 &&
      plan.total_kv_length != 0) {
    ComputeAttentionFloat32Tiled(plan, q, k, v, mask, y, nonpad_kv_seqlen);
    return;
  }
  ComputeAttentionStreamingGeneric<Float32Codec>(plan, q, k, v, mask, y, past_k, past_v,
                                                 nonpad_kv_seqlen);
}

void ComputeAttentionFloat16Streaming(const AttentionPlan &plan, const std::uint16_t *q,
                                      const std::uint16_t *k, const std::uint16_t *v,
                                      const void *mask, std::uint16_t *y,
                                      const std::uint16_t *past_k, const std::uint16_t *past_v,
                                      const std::int64_t *nonpad_kv_seqlen) {
  const std::size_t q_count = plan.batch * plan.q_num_heads * plan.q_length * plan.head_dim;
  const std::size_t k_count = plan.batch * plan.kv_num_heads * plan.kv_length * plan.head_dim;
  const std::size_t v_count = plan.batch * plan.kv_num_heads * plan.kv_length * plan.v_head_dim;
  const std::size_t y_count = plan.batch * plan.q_num_heads * plan.q_length * plan.v_head_dim;
  const std::size_t past_k_count =
      plan.batch * plan.kv_num_heads * plan.past_length * plan.head_dim;
  const std::size_t past_v_count =
      plan.batch * plan.kv_num_heads * plan.past_length * plan.v_head_dim;
  if (y_count == 0) {
    return;
  }

  constexpr std::size_t kMaxRetainedWorkspaceElements = 16 * 1024 * 1024;
  const std::size_t workspace_count =
      q_count + k_count + v_count + y_count + past_k_count + past_v_count;
  thread_local std::vector<float> retained_workspace;
  std::unique_ptr<float[]> oversized_workspace;
  float *workspace;
  if (workspace_count <= kMaxRetainedWorkspaceElements) {
    retained_workspace.resize(workspace_count);
    workspace = retained_workspace.data();
  } else {
    oversized_workspace = std::make_unique_for_overwrite<float[]>(workspace_count);
    workspace = oversized_workspace.get();
  }
  float *q_fp32 = workspace;
  float *k_fp32 = q_fp32 + q_count;
  float *v_fp32 = k_fp32 + k_count;
  float *y_fp32 = v_fp32 + v_count;
  float *past_k_fp32 = y_fp32 + y_count;
  float *past_v_fp32 = past_k_fp32 + past_k_count;
  if (plan.layout == AttentionLayout::kRank3) {
    ConvertFloat16Rank3ToRank4(q, q_fp32, plan.batch, plan.q_num_heads, plan.q_length,
                               plan.head_dim);
    ConvertFloat16Rank3ToRank4(k, k_fp32, plan.batch, plan.kv_num_heads, plan.kv_length,
                               plan.head_dim);
    ConvertFloat16Rank3ToRank4(v, v_fp32, plan.batch, plan.kv_num_heads, plan.kv_length,
                               plan.v_head_dim);
    if (past_k_count != 0) {
      detail::ConvertFloat16ToFloat32(past_k, past_k_fp32, past_k_count);
      detail::ConvertFloat16ToFloat32(past_v, past_v_fp32, past_v_count);
    }

    AttentionPlan rank4_plan = plan;
    rank4_plan.layout = AttentionLayout::kRank4;
    rank4_plan.q_strides = {
        static_cast<std::ptrdiff_t>(plan.q_num_heads * plan.q_length * plan.head_dim),
        static_cast<std::ptrdiff_t>(plan.q_length * plan.head_dim),
        static_cast<std::ptrdiff_t>(plan.head_dim)};
    rank4_plan.k_strides = {
        static_cast<std::ptrdiff_t>(plan.kv_num_heads * plan.kv_length * plan.head_dim),
        static_cast<std::ptrdiff_t>(plan.kv_length * plan.head_dim),
        static_cast<std::ptrdiff_t>(plan.head_dim)};
    rank4_plan.v_strides = {
        static_cast<std::ptrdiff_t>(plan.kv_num_heads * plan.kv_length * plan.v_head_dim),
        static_cast<std::ptrdiff_t>(plan.kv_length * plan.v_head_dim),
        static_cast<std::ptrdiff_t>(plan.v_head_dim)};
    rank4_plan.y_strides = {
        static_cast<std::ptrdiff_t>(plan.q_num_heads * plan.q_length * plan.v_head_dim),
        static_cast<std::ptrdiff_t>(plan.q_length * plan.v_head_dim),
        static_cast<std::ptrdiff_t>(plan.v_head_dim)};
    ComputeAttentionFloat32Streaming(rank4_plan, q_fp32, k_fp32, v_fp32, mask, y_fp32,
                                     past_k_count != 0 ? past_k_fp32 : nullptr,
                                     past_v_count != 0 ? past_v_fp32 : nullptr, nonpad_kv_seqlen);
    ConvertFloat32Rank4ToFloat16Rank3(y_fp32, y, plan.batch, plan.q_num_heads, plan.q_length,
                                      plan.v_head_dim);
    return;
  }

  const std::array input_buffers = {std::tuple{q, q_fp32, q_count}, std::tuple{k, k_fp32, k_count},
                                    std::tuple{v, v_fp32, v_count},
                                    std::tuple{past_k, past_k_fp32, past_k_count},
                                    std::tuple{past_v, past_v_fp32, past_v_count}};
  const std::size_t input_count = q_count + k_count + v_count + past_k_count + past_v_count;
  ExecuteRanges(
      static_cast<std::int64_t>(input_count), 0.25, 8, [&](std::int64_t begin, std::int64_t end) {
        std::size_t offset = 0;
        for (const auto &[source, destination, count] : input_buffers) {
          const std::size_t range_begin = std::max(offset, static_cast<std::size_t>(begin));
          const std::size_t range_end = std::min(offset + count, static_cast<std::size_t>(end));
          if (range_begin < range_end) {
            detail::ConvertFloat16ToFloat32(source + range_begin - offset,
                                            destination + range_begin - offset,
                                            range_end - range_begin);
          }
          offset += count;
        }
      });

  ComputeAttentionFloat32Streaming(plan, q_fp32, k_fp32, v_fp32, mask, y_fp32,
                                   past_k_count != 0 ? past_k_fp32 : nullptr,
                                   past_v_count != 0 ? past_v_fp32 : nullptr, nonpad_kv_seqlen);
  ExecuteRanges(static_cast<std::int64_t>(y_count), 0.25, 8,
                [&](std::int64_t begin, std::int64_t end) {
                  detail::ConvertFloat32ToFloat16(y_fp32 + begin, y + begin,
                                                  static_cast<std::size_t>(end - begin));
                });
}

void ComputeAttentionBFloat16Streaming(const AttentionPlan &plan, const std::uint16_t *q,
                                       const std::uint16_t *k, const std::uint16_t *v,
                                       const void *mask, std::uint16_t *y,
                                       const std::uint16_t *past_k, const std::uint16_t *past_v,
                                       const std::int64_t *nonpad_kv_seqlen) {
  ComputeAttentionStreamingGeneric<BFloat16Codec>(plan, q, k, v, mask, y, past_k, past_v,
                                                  nonpad_kv_seqlen);
}

void ComputeAttentionFloat32(const AttentionPlan &plan, const float *q, const float *k,
                             const float *v, const void *mask, float *y, const float *past_k,
                             const float *past_v, const std::int64_t *nonpad_kv_seqlen,
                             float *qk_matmul_output) {
  // Streaming preconditions: no observable `qk_matmul_output` and no
  // requested `present` output. Either necessarily materializes a full
  // observable tensor, so the materialized path is selected instead. An
  // internal tensor `past_key`/`past_value` cache and `nonpad_kv_seqlen` are
  // consumed block by block by the streaming path itself, so neither
  // excludes it.
  const bool can_stream = qk_matmul_output == nullptr && !plan.has_qk_matmul_output &&
                          !plan.has_present_output && !plan.softmax_fp64;
  if (can_stream) {
    ComputeAttentionFloat32Streaming(plan, q, k, v, mask, y, past_k, past_v, nonpad_kv_seqlen);
    return;
  }
  ComputeAttentionFloat32Materialized(plan, q, k, v, mask, y, past_k, past_v, nonpad_kv_seqlen,
                                      qk_matmul_output);
}

} // namespace onnx_light_cpu
