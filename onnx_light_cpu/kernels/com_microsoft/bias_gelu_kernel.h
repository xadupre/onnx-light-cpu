// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu/impl/com_microsoft/bias_gelu.h"

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

#include <cstdint>

#ifndef ONNX_LIGHT_NAMESPACE
#define ONNX_LIGHT_NAMESPACE onnx_light
#endif

namespace onnx_light_cpu {

/// Scalar onnx-light-cpu kernel for the ``com.microsoft`` ``BiasGelu`` contrib
/// operator.
///
/// ``BiasGeluKernel`` derives from onnx-light's
/// :cpp:class:`onnx_light::core::runtime::KernelBase` so it plugs into the
/// runtime exactly like a built-in kernel: the dispatch table constructs it
/// once per node and calls :cpp:func:`Run` on every execution. Given an input
/// ``A`` of any positive rank and a rank-1 bias ``B`` whose length matches
/// ``A``'s last dimension, it broadcasts ``B`` over that last dimension and
/// applies the exact (erf-based) Gaussian Error Linear Unit
/// ``Gelu(z) = 0.5 * z * (1 + erf(z / sqrt(2)))`` to ``A + B``. The
/// computation is delegated to the ``BiasGelu*`` routines declared in
/// ``onnx_light_cpu/impl/com_microsoft/bias_gelu.h`` and scheduled row-by-row.
/// Float32 uses runtime-dispatched AVX2/FMA or AVX-512 kernels when available;
/// float16 and bfloat16 values are decoded to float32 scalar-by-scalar and
/// re-encoded without temporary allocations.
class BiasGeluKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  using ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase::KernelBase;
  static constexpr const char *kName = "onnx_light_cpu::BiasGelu";
  static constexpr std::uint32_t kTuningAbi = 2;

  BiasGeluKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                 const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx);

  static void RegisterTuningSchemas();
  ONNX_LIGHT_NAMESPACE::core::runtime::KernelTuningKey
  TuningKey(int32_t element_type) const override;
  void
  Configure(const ONNX_LIGHT_NAMESPACE::core::runtime::KernelTuningParameters &parameters) override;

  /// Reads the node's ``A`` and ``B`` inputs, computes ``Gelu(A + B)`` and
  /// stores the single output back into ``rt``.
  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;

  /// Allocates a fresh output tensor with ``A``'s shape and writes
  /// ``Gelu(A + B)`` into it, broadcasting ``b`` over ``a``'s last dimension.
  /// When ``rt`` is non-null its allocator backs the output buffer.
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &a,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &b,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;

  /// Writes ``Gelu(A + B)`` into the caller-supplied ``output`` tensor, whose
  /// ``data_type``, ``shape`` and buffer size must already match ``a``.
  void operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &a,
                  const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &b,
                  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &output) const;

private:
  BiasGeluExecutionTuning tuning_ = kDefaultBiasGeluFloat32ExecutionTuning;
  bool tuning_configured_ = false;
};

/// Registers the onnx-light-cpu ``BiasGelu`` kernel into onnx-light's shared
/// ``KernelDispatchTable`` for the CPU device, under the ``com.microsoft``
/// domain (opset version 1).
void RegisterBiasGeluKernel();

} // namespace onnx_light_cpu
