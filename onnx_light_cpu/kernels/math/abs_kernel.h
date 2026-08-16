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

/// SIMD-accelerated onnx-light kernel for the ONNX ``Abs`` operator.
///
/// ``AbsKernel`` derives from onnx-light's
/// :cpp:class:`onnx_light::core::runtime::KernelBase` so it plugs into the
/// runtime exactly like a built-in kernel: the dispatch table constructs it
/// once per node and calls :cpp:func:`Run` on every execution. The
/// computation is delegated to the SIMD ``Abs*`` routines declared in
/// ``onnx_light_cpu/impl/math/math_kernels.h`` (runtime AVX-512/AVX2/AVX/SSE2 dispatch)
/// for ``float32``/``float64``/``float16``/``int8``/``int32``/``int64``; the
/// remaining element types supported by ONNX ``Abs`` (``bfloat16`` and
/// ``int16``) fall back to a scalar implementation so the kernel is a full
/// drop-in replacement for the built-in one.
class AbsKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  using ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase::KernelBase;

  /// Library-qualified name identifying this kernel, recorded through
  /// :cpp:func:`RecordKernelUsage` on every :cpp:func:`Run` so callers can
  /// tell the onnx-light-cpu kernel apart from onnx-light's built-in ``Abs``.
  static constexpr const char *kName = "onnx_light_cpu::Abs";

  /// Reads the node's single input, computes the elementwise absolute value
  /// and stores the single output back into ``rt``.
  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;

  /// Allocates a fresh output tensor and writes ``|x|`` into it. When ``rt``
  /// is non-null its allocator backs the output buffer.
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &x,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;

  /// Writes ``|x|`` into the caller-supplied ``output`` tensor, whose
  /// ``data_type``, ``shape`` and buffer size must already match ``x``.
  void operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &x,
                  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &output) const;
};

/// Registers the onnx-light-cpu ``Abs`` kernel into onnx-light's shared
/// ``KernelDispatchTable`` for the CPU device.
///
/// After this call, every ``Abs`` node dispatched by onnx-light's runtime
/// (``RunNode`` / ``RuntimeSession``, and therefore any model executed through
/// ``ReferenceEvaluator``) resolves to :cpp:class:`AbsKernel`, so any ONNX
/// model using ``Abs`` runs the SIMD-accelerated kernel. Registering under the
/// default ONNX domain replaces the built-in ``Abs`` entry.
void RegisterAbsKernel();

} // namespace onnx_light_cpu
