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

/// SIMD-accelerated onnx-light kernel for the ONNX ``Not`` operator.
///
/// ``NotKernel`` derives from onnx-light's
/// :cpp:class:`onnx_light::core::runtime::KernelBase` so it plugs into the
/// runtime exactly like a built-in kernel: the dispatch table constructs it
/// once per node and calls :cpp:func:`Run` on every execution. The computation
/// is delegated to the SIMD ``NotBool`` routine declared in
/// ``onnx_light_cpu/impl/logical/logical_kernels.h`` (runtime AVX-512/AVX2/SSE2 dispatch). ONNX
/// ``Not`` only accepts ``bool`` inputs, so the kernel is a full drop-in
/// replacement for the built-in one.
class NotKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  using ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase::KernelBase;

  /// Library-qualified name identifying this kernel, recorded through
  /// :cpp:func:`RecordKernelUsage` on every :cpp:func:`Run` so callers can
  /// tell the onnx-light-cpu kernel apart from onnx-light's built-in ``Not``.
  static constexpr const char *kName = "onnx_light_cpu::Not";

  /// Reads the node's single input, computes the elementwise logical negation
  /// and stores the single output back into ``rt``.
  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;

  /// Allocates a fresh output tensor and writes ``!x`` into it. When ``rt`` is
  /// non-null its allocator backs the output buffer.
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &x,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;

  /// Writes ``!x`` into the caller-supplied ``output`` tensor, whose
  /// ``data_type``, ``shape`` and buffer size must already match ``x``.
  void operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &x,
                  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &output) const;
};

/// Registers the onnx-light-cpu ``Not`` kernel into onnx-light's shared
/// ``KernelDispatchTable`` for the CPU device.
///
/// After this call, every ``Not`` node dispatched by onnx-light's runtime
/// (``RunNode`` / ``RuntimeSession``, and therefore any model executed through
/// ``ReferenceEvaluator``) resolves to :cpp:class:`NotKernel`. Registering
/// under the default ONNX domain replaces the built-in ``Not`` entry.
void RegisterNotKernel();

} // namespace onnx_light_cpu
