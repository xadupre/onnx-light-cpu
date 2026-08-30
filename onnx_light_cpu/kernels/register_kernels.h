// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace onnx_light_cpu {

/// Selects the complete ``com.microsoft`` kernel implementation family.
enum class MicrosoftKernelImplementation {
  /// Readable scalar correctness-oracle implementations.
  NAIVE,
  /// Production implementations with tuning, SIMD, and parallel execution.
  OPTIMIZED,
};

/// Registers exactly one implementation for every ``com.microsoft`` operator.
void RegisterMicrosoftKernels(MicrosoftKernelImplementation implementation);

/// Registers every onnx-light-cpu kernel class into onnx-light's shared
/// ``KernelDispatchTable`` for the CPU device.
///
/// This is a convenience wrapper that calls each per-operator registration
/// function (:cpp:func:`RegisterAbsKernel`, :cpp:func:`RegisterAttentionKernel`,
/// :cpp:func:`RegisterBiasGeluKernel`, :cpp:func:`RegisterBinaryKernels`,
/// :cpp:func:`RegisterCDistKernel`, :cpp:func:`RegisterExpKernel`,
/// :cpp:func:`RegisterLogKernel`, :cpp:func:`RegisterGemmKernel`,
/// :cpp:func:`RegisterMatMulKernel`, :cpp:func:`RegisterIntegerMatMulKernels`
/// :cpp:func:`RegisterNotKernel`, :cpp:func:`RegisterTreeEnsembleKernel`, and
/// :cpp:func:`RegisterVariadicElementwiseKernels`), so a single call installs
/// the accelerated elementwise/GEMM kernels, the stateless materialized
/// ``Attention`` baseline, the prepared ``TreeEnsemble`` kernel, the portable
/// integer matrix-multiplication kernels, and the ``com.microsoft``
/// ``BiasGelu``/``CDist`` contrib kernels.
/// After this call every such node dispatched by onnx-light's
/// runtime (``RunNode`` / ``RuntimeSession``, and therefore any model executed
/// through ``ReferenceEvaluator``) resolves to the onnx-light-cpu kernel,
/// replacing the corresponding built-in entries for the default ONNX domain.
void RegisterAllKernels();

/// Registers all kernels, selecting the complete ``com.microsoft`` family
/// explicitly. The no-argument overload is equivalent to ``OPTIMIZED``.
void RegisterAllKernels(MicrosoftKernelImplementation implementation);

} // namespace onnx_light_cpu
