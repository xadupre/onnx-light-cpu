// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/attention/attention_plan.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
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

} // namespace

void AttentionDescriptor::Validate() const {
  if (opset != 23 && opset != 24) {
    Fail("AttentionDescriptor: only opset 23 and 24 are recognized.");
  }
  if (opset < 24 && has_nonpad_kv_seqlen) {
    Fail("AttentionDescriptor: nonpad_kv_seqlen requires opset >= 24.");
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
    // FLOAT == 1 in onnx::TensorProto::DataType; only the default (input)
    // precision or an explicit FP32 accumulation are supported.
    constexpr std::int64_t kFloat = 1;
    if (*softmax_precision != kFloat) {
      Fail("AttentionDescriptor: only the default or explicit FP32 softmax_precision is "
           "supported by the CPU materialized path.");
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
    if (ToSize(past_v_shape[3], "AttentionPlan: v_head_size must be non-negative.") !=
        v_head_dim) {
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
  qk_matmul_output_mode = descriptor.qk_matmul_output_mode;
  has_qk_matmul_output = descriptor.has_qk_matmul_output;

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

}

void ComputeAttentionFloat32(const AttentionPlan &plan, const float *q, const float *k,
                             const float *v, const void *mask, float *y, const float *past_k,
                             const float *past_v, const std::int64_t *nonpad_kv_seqlen,
                             float *qk_matmul_output) {
  const auto *mask_bool = static_cast<const std::uint8_t *>(mask);
  const auto *mask_float = static_cast<const float *>(mask);
  const std::size_t total_kv_length = plan.total_kv_length;
  const bool has_softcap = plan.softcap != 0.0f;

  std::vector<float> scores(total_kv_length);
  std::vector<float> qk_row(plan.has_qk_matmul_output ? total_kv_length : 0);
  for (std::size_t b = 0; b < plan.batch; ++b) {
    // Bottom-right, offset-aware causal frontier: `nonpad_kv_seqlen`, when
    // supplied, overrides the plan's scalar `causal_offset` with a per-batch
    // value (`nonpad_kv_seqlen[b] - q_length`) and additionally masks KV
    // positions beyond `nonpad_kv_seqlen[b]` regardless of `is_causal`.
    const std::int64_t causal_offset =
        nonpad_kv_seqlen != nullptr
            ? nonpad_kv_seqlen[b] - static_cast<std::int64_t>(plan.q_length)
            : plan.causal_offset;
    const std::int64_t nonpad_length = nonpad_kv_seqlen != nullptr ? nonpad_kv_seqlen[b] : -1;

    for (std::size_t h = 0; h < plan.q_num_heads; ++h) {
      const std::size_t kv_h = h / plan.group_size;
      const float *q_head = q + b * plan.q_strides.batch + h * plan.q_strides.head;
      const float *k_head = k + b * plan.k_strides.batch + kv_h * plan.k_strides.head;
      const float *v_head = v + b * plan.v_strides.batch + kv_h * plan.v_strides.head;
      const float *past_k_head =
          past_k != nullptr ? past_k + b * plan.past_k_strides.batch +
                                  kv_h * plan.past_k_strides.head
                            : nullptr;
      const float *past_v_head =
          past_v != nullptr ? past_v + b * plan.past_v_strides.batch +
                                  kv_h * plan.past_v_strides.head
                            : nullptr;
      float *y_head = y + b * plan.y_strides.batch + h * plan.y_strides.head;
      float *qk_head = plan.has_qk_matmul_output
                            ? qk_matmul_output +
                                  (b * plan.q_num_heads + h) * plan.q_length * total_kv_length
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
        float row_bias_max = kNegativeInfinity;
        for (std::size_t j = 0; j < total_kv_length; ++j) {
          bool allowed = true;
          if (plan.causal) {
            allowed = static_cast<std::int64_t>(j) <= static_cast<std::int64_t>(i) + causal_offset;
          }
          if (nonpad_length >= 0) {
            allowed = allowed && static_cast<std::int64_t>(j) < nonpad_length;
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
          row_bias_max = std::max(row_bias_max, bias);

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
        const bool row_all_masked = row_bias_max == kNegativeInfinity;
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
        float sum = 0.0f;
        for (std::size_t j = 0; j < total_kv_length; ++j) {
          const float p = scores[j] == kNegativeInfinity ? 0.0f : std::exp(scores[j] - row_max);
          scores[j] = p;
          sum += p;
        }
        const float inv_sum = 1.0f / sum;
        std::fill(y_row, y_row + plan.v_head_dim, 0.0f);
        for (std::size_t j = 0; j < total_kv_length; ++j) {
          const float p = scores[j] * inv_sum;
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

} // namespace onnx_light_cpu
