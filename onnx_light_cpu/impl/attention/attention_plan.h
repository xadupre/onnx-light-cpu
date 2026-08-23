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

  /// Throws ``std::invalid_argument`` when this instance uses a feature not
  /// yet implemented by the CPU materialized path: tensor ``past``/``present``
  /// cache, ``nonpad_kv_seqlen``, a non-zero ``softcap``, an observable
  /// ``qk_matmul_output``, or a ``softmax_precision`` other than the default.
  /// Roadmap PR11 only delivers the stateless FP32 materialized baseline;
  /// later PRs lift these restrictions.
  void ValidateSupportedByMaterializedPath() const;
};

/// Lightweight, per-invocation execution plan built from the concrete
/// ``Q``/``K``/``V``/``attn_mask`` shapes of one ``Attention`` call. It is
/// cheap to construct (no allocation beyond the mask strides) and is not
/// retained across invocations: shapes and strides may change every call.
struct AttentionPlan {
  /// Builds the plan and validates ``q_shape``/``k_shape``/``v_shape`` (and
  /// ``mask_shape`` when ``mask_kind != kNone``) against ``descriptor`` and
  /// ``layout``. Throws ``std::invalid_argument`` on any shape mismatch (rank,
  /// head-count divisibility, incompatible head/mask dimensions).
  AttentionPlan(const AttentionDescriptor &descriptor, AttentionLayout layout,
                std::span<const std::int64_t> q_shape, std::span<const std::int64_t> k_shape,
                std::span<const std::int64_t> v_shape, std::span<const std::int64_t> mask_shape,
                AttentionMaskKind mask_kind);

  AttentionLayout layout = AttentionLayout::kRank4;

  std::size_t batch = 0;
  std::size_t q_num_heads = 0;
  std::size_t kv_num_heads = 0;
  /// ``Q``/``K`` head size.
  std::size_t head_dim = 0;
  /// ``V`` head size (``Y`` head size); may differ from :cpp:member:`head_dim`.
  std::size_t v_head_dim = 0;
  std::size_t q_length = 0;
  std::size_t kv_length = 0;
  /// ``q_num_heads / kv_num_heads``; ``1`` for MHA, ``> 1`` for GQA/MQA.
  std::size_t group_size = 1;

  float scale = 1.0f;
  bool causal = false;
  AttentionMaskKind mask_kind = AttentionMaskKind::kNone;

  /// Per-tensor element strides indexed as ``[batch, head, sequence]``; the
  /// innermost (head-size) dimension is always contiguous (stride 1). Rank-3
  /// and rank-4 layouts only differ in the relative order of ``head`` and
  /// ``sequence`` strides, so the materialized kernel is layout-agnostic.
  struct TensorStrides {
    std::ptrdiff_t batch = 0;
    std::ptrdiff_t head = 0;
    std::ptrdiff_t sequence = 0;
  };
  TensorStrides q_strides;
  TensorStrides k_strides;
  TensorStrides v_strides;
  TensorStrides y_strides;

  /// ``attn_mask`` element strides aligned (right-justified, ONNX broadcast
  /// rules) to ``(batch, q_num_heads, q_length, kv_length)``; a broadcast
  /// dimension has stride ``0``. Unused when :cpp:member:`mask_kind` is
  /// :cpp:enumerator:`AttentionMaskKind::kNone`.
  struct MaskStrides {
    std::ptrdiff_t batch = 0;
    std::ptrdiff_t head = 0;
    std::ptrdiff_t q = 0;
    std::ptrdiff_t kv = 0;
  };
  MaskStrides mask_strides;

  /// ``Y`` output shape, in :cpp:member:`layout`.
  std::vector<std::int64_t> output_shape() const;
};

/// Executes the stateless FP32 materialized baseline:
/// ``S = scale * Q @ K^T``; composes causal and ``attn_mask`` biases
/// additively; ``P = softmax(S)`` (a fully-masked row produces an all-zero
/// output row rather than ``NaN``); ``Y = P @ V``.
///
/// ``mask`` points to ``bool``-sized (``uint8_t``, non-zero is ``true``)
/// elements when :cpp:member:`AttentionPlan::mask_kind` is
/// :cpp:enumerator:`AttentionMaskKind::kBoolean`, or ``float`` elements when
/// it is :cpp:enumerator:`AttentionMaskKind::kAdditive`; it is unused (may be
/// ``nullptr``) when :cpp:enumerator:`AttentionMaskKind::kNone`.
void ComputeAttentionFloat32(const AttentionPlan &plan, const float *q, const float *k,
                            const float *v, const void *mask, float *y);

} // namespace onnx_light_cpu
