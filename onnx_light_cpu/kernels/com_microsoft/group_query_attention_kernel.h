// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu/kernels/attention/attention_kernel.h"

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

#ifndef ONNX_LIGHT_NAMESPACE
#define ONNX_LIGHT_NAMESPACE onnx_light
#endif

namespace onnx_light_cpu {

/// CPU kernel for ``com.microsoft::GroupQueryAttention``.
///
/// Implements the exact contract used by KV-cached decoder-only LLM export
/// graphs (e.g. Qwen3): rank-3 ``query``/``key``/``value``, an optional
/// rank-4 ``(batch, kv_num_heads, past_sequence_length, head_size)``
/// ``past_key``/``past_value`` tensor cache, ``do_rotary``/
/// ``rotary_interleaved=0`` split-half RoPE applied to ``query``/``key`` at
/// the absolute position derived from ``seqlens_k`` (or an explicit
/// ``position_ids``), and observable ``present_key``/``present_value``
/// outputs holding the concatenation of the past cache with the (rotated)
/// current step. The attention score/softmax/value reduction itself is
/// delegated to :cpp:class:`AttentionKernel` (this kernel never duplicates
/// that math); only the RoPE application and the ``present`` cache
/// materialization are implemented here.
class GroupQueryAttentionKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  GroupQueryAttentionKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                            const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx);

  static constexpr const char *kName = "onnx_light_cpu::GroupQueryAttention";

  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;

  /// Direct invocation used by backend test cases and unit tests. ``node``
  /// carries the operator attributes and is also used to decide, from its
  /// declared inputs/outputs, which optional tensors participate (matching
  /// :cpp:func:`Run`'s reading of ``rt.tensors()``): a non-null optional
  /// tensor pointer is only honored when the corresponding ``node`` input is
  /// wired (non-empty name), and ``present_key``/``present_value`` are only
  /// populated when ``node`` declares the corresponding output.
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &query,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &key,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &value,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &seqlens_k,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &total_sequence_length,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *past_key = nullptr,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *past_value = nullptr,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *cos_cache = nullptr,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *sin_cache = nullptr,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *position_ids = nullptr,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *attention_bias = nullptr,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr,
             ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *present_key = nullptr,
             ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *present_value = nullptr) const;

private:
  AttentionKernel attention_;
};

void RegisterGroupQueryAttentionKernel();

} // namespace onnx_light_cpu
