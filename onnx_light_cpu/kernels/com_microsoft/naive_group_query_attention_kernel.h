// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

#ifndef ONNX_LIGHT_NAMESPACE
#define ONNX_LIGHT_NAMESPACE onnx_light
#endif

namespace onnx_light_cpu {

/// Independent, scalar reference CPU kernel for ``com.microsoft::GroupQueryAttention``.
///
/// Implements the exact same contract as :cpp:class:`GroupQueryAttentionKernel`
/// (rank-3 ``query``/``key``/``value``, an optional rank-4
/// ``(batch, kv_num_heads, past_sequence_length, head_size)``
/// ``past_key``/``past_value`` tensor cache, ``do_rotary``/
/// ``rotary_interleaved=0`` split-half RoPE applied to ``query``/``key`` at
/// the absolute position derived from ``seqlens_k`` (or an explicit
/// ``position_ids``), an optional ``attention_bias`` additive mask, and
/// observable ``present_key``/``present_value`` outputs) but computes every
/// step -- RoPE, the score/softmax/value reduction, and the KV-cache
/// concatenation -- with plain, readable scalar loops. It never calls
/// :cpp:class:`AttentionKernel` or any other optimized attention compute
/// helper, so it is a self-contained reference implementation intended for
/// differential testing against the optimized kernel.
class NaiveGroupQueryAttentionKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  NaiveGroupQueryAttentionKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                                 const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx);

  static constexpr const char *kName = "onnx_light_cpu::NaiveGroupQueryAttention";

  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;

  /// Direct invocation used by unit tests. ``node`` carries the operator
  /// attributes and is also used to decide, from its declared
  /// inputs/outputs, which optional tensors participate (matching
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
};

/// Registers :cpp:class:`NaiveGroupQueryAttentionKernel` for
/// ``com.microsoft::GroupQueryAttention``. Mirrors the sibling
/// ``RegisterNaive*Kernel`` functions (e.g. ``RegisterNaiveBiasGeluKernel``,
/// ``RegisterNaiveCDistKernel``); not called by ``RegisterAllKernels`` yet --
/// wiring it into the ``MicrosoftKernelImplementation::NAIVE`` family is left
/// to ``register_kernels.cc``.
void RegisterNaiveGroupQueryAttentionKernel();

} // namespace onnx_light_cpu
