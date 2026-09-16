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

/// Registers every shipped operator into onnx-light's process-wide
/// ``KernelDispatchTable`` for the CPU device.
///
/// See :ref:`l-register-all-kernels-inventory` for the complete family,
/// domain, and operator table, validated against ``CollectRegisteredKernels``.
/// The inventory covers ``ai.onnx`` (also accepted as the empty domain),
/// ``ai.onnx.ml``, and ``com.microsoft``.
///
/// Existing global entries are replaced. Register before nodes are resolved:
/// already-resolved sessions retain cached factories, and session-local
/// overrides take precedence. Use ``RegisterAllKernelsForSession`` to install
/// the same operator set on one ``RuntimeContext`` without changing global state,
/// or ``RegisterAllKernelsGlobal`` for explicit replacement control.
///
/// This overload selects ``MicrosoftKernelImplementation::OPTIMIZED``.
/// The policy overload can instead select the complete ``NAIVE`` Microsoft
/// reference family; it does not change the ONNX or ONNX-ML registrations.
/// Optimized kernels may still use scalar fallbacks depending on types,
/// shapes, build options, and available CPU instructions.
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
/// Callback copies share prepared kernels keyed by node contents, input
/// presence/type/shape, opset, device, and allocator. Each instance is constructed
/// lazily once and its runs are synchronized; changing tensor values alone does
/// not rebuild it. Resolved callbacks retain their cache after replacement.
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
