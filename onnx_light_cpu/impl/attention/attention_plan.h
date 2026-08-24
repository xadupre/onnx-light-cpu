// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace onnx_light_cpu {

/// Layout of the ``Q``/``K``/``V``/``Y`` tensors of an ``ai.onnx::Attention``
/// invocation.
enum class AttentionLayout {
  /// ``(batch_size, num_heads, sequence_length, head_size)``.
  kRank4,
  /// ``(batch_size, sequence_length, num_heads * head_size)``.
  kRank3,
};

/// Representation of the resolved ``attn_mask`` input, if any.
enum class AttentionMaskKind {
  /// No ``attn_mask`` input.
  kNone,
  /// Boolean mask: ``True`` means the position takes part in attention.
  kBoolean,
  /// Additive float mask, added to the attention score before softmax.
  kAdditive,
};

/// Immutable, node-level description of one ``ai.onnx::Attention`` (v23/v24)
/// instance: the opset, its attributes, and which optional inputs/outputs are
/// wired to this node. It is built once, without any concrete tensor, when
/// the node is initialized (mirrors ``GemmPlanOptions``: a plain, dependency
/// free options struct populated by the onnx-light kernel from the
/// ``NodeProto``).
struct AttentionDescriptor {
  /// ``ai.onnx`` opset version this node was defined against; only 23 and 24
  /// are recognized.
  int opset = 23;

  /// ``scale`` attribute. Defaults to ``1 / sqrt(head_size)`` when unset.
  std::optional<float> scale;
  /// ``is_causal`` attribute.
  bool is_causal = false;
  /// ``softcap`` attribute; ``0`` disables softcapping.
  float softcap = 0.0f;
  /// ``qk_matmul_output_mode`` attribute (``0``-``3``).
  int64_t qk_matmul_output_mode = 0;
  /// ``softmax_precision`` attribute, an ``onnx::TensorProto::DataType``
  /// value when set.
  std::optional<int64_t> softmax_precision;
  /// ``q_num_heads`` attribute; required for :cpp:enumerator:`AttentionLayout::kRank3`.
  std::optional<int64_t> q_num_heads;
  /// ``kv_num_heads`` attribute; required for :cpp:enumerator:`AttentionLayout::kRank3`.
  std::optional<int64_t> kv_num_heads;

  /// Whether the optional ``attn_mask`` input is wired.
  bool has_attn_mask = false;
  /// Whether the optional ``past_key`` input is wired.
  bool has_past_key = false;
  /// Whether the optional ``past_value`` input is wired.
  bool has_past_value = false;
  /// Whether the optional ``nonpad_kv_seqlen`` input is wired (opset >= 24
  /// only).
  bool has_nonpad_kv_seqlen = false;

  /// Whether the optional ``present_key`` output is wired.
  bool has_present_key = false;
  /// Whether the optional ``present_value`` output is wired.
  bool has_present_value = false;
  /// Whether the optional ``qk_matmul_output`` output is wired.
  bool has_qk_matmul_output = false;

  /// Validates the attribute/IO wiring independently of any concrete tensor
  /// shape (opset range, attribute ranges, ``nonpad_kv_seqlen`` only on
  /// opset >= 24, ``past``/``present`` used in pairs). Throws
  /// ``std::invalid_argument`` on an invalid configuration.
  void Validate() const;

  /// Throws ``std::invalid_argument`` when this instance uses a feature
  /// outside the CPU materialized path's advertised boundary. Roadmap PR12
  /// implements tensor ``past``/``present`` cache, ``nonpad_kv_seqlen``,
  /// ``softcap``, every ``qk_matmul_output_mode``, and the observable
  /// ``qk_matmul_output`` for equal-type FP32/FP16/BF16 Q/K/V. Only the
  /// default (input-precision) or explicit FP32 ``softmax_precision`` are
  /// supported; any other value is rejected here.
  void ValidateSupportedByMaterializedPath() const;
};

/// Lightweight, per-invocation execution plan built from the concrete
/// ``Q``/``K``/``V``/``attn_mask`` shapes of one ``Attention`` call. It is
/// cheap to construct (no allocation beyond the mask strides) and is not
/// retained across invocations: shapes and strides may change every call.
struct AttentionPlan {
  /// Per-tensor element strides indexed as ``[batch, head, sequence]``; the
  /// innermost (head-size) dimension is always contiguous (stride 1). Rank-3
  /// and rank-4 layouts only differ in the relative order of ``head`` and
  /// ``sequence`` strides, so the materialized kernel is layout-agnostic.
  struct TensorStrides {
    std::ptrdiff_t batch = 0;
    std::ptrdiff_t head = 0;
    std::ptrdiff_t sequence = 0;
  };

  /// Builds the plan and validates ``q_shape``/``k_shape``/``v_shape`` (and
  /// ``mask_shape`` when ``mask_kind != kNone``) against ``descriptor`` and
  /// ``layout``. Throws ``std::invalid_argument`` on any shape mismatch (rank,
  /// head-count divisibility, incompatible head/mask dimensions).
  ///
  /// ``past_k_shape``/``past_v_shape`` are the (rank-4, always
  /// ``(batch, kv_num_heads, past_sequence_length, head_size)``/
  /// ``(batch, kv_num_heads, past_sequence_length, v_head_size)``) shapes of
  /// the optional tensor cache; pass empty spans when there is no ``past_key``
  /// / ``past_value`` input. ``attn_mask`` (and every mask broadcast target)
  /// is resolved against the *total* KV length, i.e.
  /// ``past_sequence_length + kv_sequence_length``.
  AttentionPlan(const AttentionDescriptor &descriptor, AttentionLayout layout,
                std::span<const std::int64_t> q_shape, std::span<const std::int64_t> k_shape,
                std::span<const std::int64_t> v_shape, std::span<const std::int64_t> mask_shape,
                AttentionMaskKind mask_kind, std::span<const std::int64_t> past_k_shape = {},
                std::span<const std::int64_t> past_v_shape = {});

  AttentionLayout layout = AttentionLayout::kRank4;

  std::size_t batch = 0;
  std::size_t q_num_heads = 0;
  std::size_t kv_num_heads = 0;
  /// ``Q``/``K`` head size.
  std::size_t head_dim = 0;
  /// ``V`` head size (``Y`` head size); may differ from :cpp:member:`head_dim`.
  std::size_t v_head_dim = 0;
  std::size_t q_length = 0;
  /// Length of the ``K``/``V`` inputs of *this* invocation (excludes any
  /// tensor ``past_key``/``past_value`` cache).
  std::size_t kv_length = 0;
  /// Length of the ``past_key``/``past_value`` cache; ``0`` when there is
  /// none.
  std::size_t past_length = 0;
  /// ``past_length + kv_length``: the effective (total) KV length attended
  /// to, and the length ``attn_mask``/``present_key``/``present_value`` are
  /// resolved against.
  std::size_t total_kv_length = 0;
  /// ``q_num_heads / kv_num_heads``; ``1`` for MHA, ``> 1`` for GQA/MQA.
  std::size_t group_size = 1;

  float scale = 1.0f;
  bool causal = false;
  /// Bottom-right causal offset used when no per-batch offset (derived from
  /// ``nonpad_kv_seqlen``) is supplied at compute time: query in-block index
  /// ``i`` attends key ``j`` iff ``j <= i + causal_offset``. Equals
  /// :cpp:member:`past_length` when a tensor ``past_key`` cache is present,
  /// ``0`` otherwise.
  std::int64_t causal_offset = 0;
  AttentionMaskKind mask_kind = AttentionMaskKind::kNone;
  /// ``softcap`` attribute, copied from the descriptor; ``0`` disables it.
  float softcap = 0.0f;
  /// ``qk_matmul_output_mode`` attribute, copied from the descriptor.
  std::int64_t qk_matmul_output_mode = 0;
  /// Whether the optional ``qk_matmul_output`` output is wired.
  bool has_qk_matmul_output = false;

  TensorStrides q_strides;
  TensorStrides k_strides;
  TensorStrides v_strides;
  TensorStrides y_strides;
  /// Strides of the tensor ``past_key``/``past_value`` cache; only valid when
  /// :cpp:member:`past_length` is non-zero.
  TensorStrides past_k_strides;
  TensorStrides past_v_strides;

  /// ``attn_mask`` element strides aligned (right-justified, ONNX broadcast
  /// rules) to ``(batch, q_num_heads, q_length, total_kv_length)``; a
  /// broadcast dimension has stride ``0``. Unused when
  /// :cpp:member:`mask_kind` is :cpp:enumerator:`AttentionMaskKind::kNone`.
  struct MaskStrides {
    std::ptrdiff_t batch = 0;
    std::ptrdiff_t head = 0;
    std::ptrdiff_t q = 0;
    std::ptrdiff_t kv = 0;
  };
  MaskStrides mask_strides;

  /// ``Y`` output shape, in :cpp:member:`layout`.
  std::vector<std::int64_t> output_shape() const;
  /// ``present_key`` output shape: always rank-4
  /// ``(batch, kv_num_heads, total_kv_length, head_size)``.
  std::vector<std::int64_t> present_key_shape() const;
  /// ``present_value`` output shape: always rank-4
  /// ``(batch, kv_num_heads, total_kv_length, v_head_size)``.
  std::vector<std::int64_t> present_value_shape() const;
  /// ``qk_matmul_output`` output shape (when requested): always rank-4
  /// ``(batch, q_num_heads, q_length, total_kv_length)``.
  std::vector<std::int64_t> qk_matmul_output_shape() const;
};

/// Executes the stateless-or-cached FP32 materialized baseline:
/// ``S = scale * Q @ transpose(K)``; applies ``softcap`` when configured;
/// composes causal and ``attn_mask`` biases additively (bottom-right,
/// offset-aware causal per :cpp:member:`AttentionPlan::causal_offset` or, when
/// ``nonpad_kv_seqlen`` is supplied, a per-batch offset and padding mask);
/// ``P = softmax(S)`` (a fully-masked row produces an all-zero output row
/// rather than ``NaN``); ``Y = P @ V``.
///
/// ``mask`` points to ``bool``-sized (``uint8_t``, non-zero is ``true``)
/// elements when :cpp:member:`AttentionPlan::mask_kind` is
/// :cpp:enumerator:`AttentionMaskKind::kBoolean`, or ``float`` elements when
/// it is :cpp:enumerator:`AttentionMaskKind::kAdditive`; it is unused (may be
/// ``nullptr``) when :cpp:enumerator:`AttentionMaskKind::kNone`.
///
/// ``past_k``/``past_v`` (may be ``nullptr`` when
/// :cpp:member:`AttentionPlan::past_length` is ``0``) are read through
/// :cpp:member:`AttentionPlan::past_k_strides`/:cpp:member:`AttentionPlan::past_v_strides`
/// and logically precede ``k``/``v`` along the KV axis.
///
/// ``nonpad_kv_seqlen`` (may be ``nullptr``), when supplied, is a per-batch
/// ``int64`` array of length :cpp:member:`AttentionPlan::batch`: it overrides
/// the causal offset with ``nonpad_kv_seqlen[b] - q_length`` and additionally
/// masks KV positions ``>= nonpad_kv_seqlen[b]``.
///
/// ``qk_matmul_output`` (may be ``nullptr``), when supplied, is filled with
/// the ``(batch, q_num_heads, q_length, total_kv_length)`` tensor selected by
/// :cpp:member:`AttentionPlan::qk_matmul_output_mode`.
void ComputeAttentionFloat32(const AttentionPlan &plan, const float *q, const float *k,
                             const float *v, const void *mask, float *y,
                             const float *past_k = nullptr, const float *past_v = nullptr,
                             const std::int64_t *nonpad_kv_seqlen = nullptr,
                             float *qk_matmul_output = nullptr);

} // namespace onnx_light_cpu
