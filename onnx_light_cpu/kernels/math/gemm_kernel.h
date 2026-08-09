// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernel_context.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/simple_tensor.h"

#ifndef ONNX_LIGHT_NAMESPACE
#define ONNX_LIGHT_NAMESPACE onnx_light
#endif

namespace onnx_light_cpu {

/// SIMD-accelerated onnx-light kernel for the ONNX ``Gemm`` operator.
///
/// ``GemmKernel`` derives from onnx-light's
/// :cpp:class:`onnx_light::core::runtime::KernelBase` so it plugs into the
/// runtime exactly like a built-in kernel: the dispatch table constructs it
/// once per node and calls :cpp:func:`Run` on every execution. The computation
///
///     Y = alpha * op(A) @ op(B) + beta * C
///
/// (where ``op(X)`` transposes ``X`` when the corresponding ``transA`` /
/// ``transB`` attribute is set) is delegated to the register-blocked SIMD
/// ``Gemm*`` routines declared in
/// ``onnx_light_cpu/impl/math/math_kernels.h`` (runtime AVX/SSE2 dispatch) for
/// ``float32`` and ``float64``. ``float16`` and ``bfloat16`` inputs are widened
/// to ``float32``, computed through the same SIMD ``GemmFloat32`` routine, and
/// rounded back down for the output (the reduction happens in ``float32``
/// precision, matching common fp16/bf16 GEMM backend conventions). The
/// optional bias ``C`` is unidirectionally broadcast to the ``M x N`` output
/// shape, matching the ONNX ``Gemm`` specification.
class GemmKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  using ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase::KernelBase;

  /// Library-qualified name identifying this kernel, recorded through
  /// :cpp:func:`RecordKernelUsage` on every :cpp:func:`Run` so callers can
  /// tell the onnx-light-cpu kernel apart from onnx-light's built-in ``Gemm``.
  static constexpr const char *kName = "onnx_light_cpu::Gemm";

  /// Reads the node's ``A``, ``B`` and optional ``C`` inputs together with the
  /// ``alpha``, ``beta``, ``transA`` and ``transB`` attributes, computes the
  /// general matrix multiplication and stores the single output back into
  /// ``rt``.
  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;

  /// Allocates a fresh output tensor and writes
  /// ``alpha * op(A) @ op(B) + beta * C`` into it, broadcasting the bias ``c``
  /// to the ``M x N`` output shape. When ``rt`` is non-null its allocator backs
  /// the output buffer.
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &a,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &b,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &c, float alpha, float beta,
             bool trans_a, bool trans_b,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;

  /// Same as above but for a ``Gemm`` node without a bias input: allocates a
  /// fresh output tensor and writes ``alpha * op(A) @ op(B)`` into it.
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &a,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &b, float alpha, bool trans_a,
             bool trans_b, ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;
};

/// Registers the onnx-light-cpu ``Gemm`` kernel into onnx-light's shared
/// ``KernelDispatchTable`` for the CPU device.
///
/// After this call, every ``Gemm`` node dispatched by onnx-light's runtime
/// (``RunNode`` / ``RuntimeSession``, and therefore any model executed through
/// ``ReferenceEvaluator``) resolves to :cpp:class:`GemmKernel`, so any ONNX
/// model using ``Gemm`` runs the SIMD-accelerated kernel. Registering under the
/// default ONNX domain replaces the built-in ``Gemm`` entry.
void RegisterGemmKernel();

} // namespace onnx_light_cpu
