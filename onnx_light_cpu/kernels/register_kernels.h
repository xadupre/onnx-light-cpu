// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/runtime_context.h"

#include <cstddef>
#include <string>

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

/// Registers one shipped kernel in the process-wide dispatch table.
///
/// The empty domain and ``"ai.onnx"`` are equivalent. If `replace` is false,
/// an existing registration is retained and this function returns false.
/// Unknown domain/operator pairs throw ``std::invalid_argument``.
bool RegisterKernelGlobal(
    const std::string &domain, const std::string &op_type, bool replace = true,
    MicrosoftKernelImplementation implementation = MicrosoftKernelImplementation::OPTIMIZED);

/// Registers every shipped kernel in the process-wide dispatch table.
///
/// Returns the number of factories installed. With `replace` false, repeated
/// calls are idempotent and return zero once all entries exist.
std::size_t RegisterAllKernelsGlobal(
    bool replace = true,
    MicrosoftKernelImplementation implementation = MicrosoftKernelImplementation::OPTIMIZED);

/// Registers one shipped kernel only on `session`.
///
/// The session's ``RuntimeContext`` owns the registration. It takes precedence
/// over global and built-in kernels for that session and is destroyed with the
/// context. If `replace` is false, an existing session-local registration is
/// retained and this function returns false.
bool RegisterKernelForSession(
    ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &session, const std::string &domain,
    const std::string &op_type, bool replace = true,
    MicrosoftKernelImplementation implementation = MicrosoftKernelImplementation::OPTIMIZED);

/// Registers every shipped kernel only on `session`.
///
/// Returns the number of session-local registrations installed.
std::size_t RegisterAllKernelsForSession(
    ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &session, bool replace = true,
    MicrosoftKernelImplementation implementation = MicrosoftKernelImplementation::OPTIMIZED);

} // namespace onnx_light_cpu
