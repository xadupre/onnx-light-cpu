// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace onnx_light_cpu {

/// Registers every onnx-light-cpu kernel class into onnx-light's shared
/// ``KernelDispatchTable`` for the CPU device.
///
/// It drives the self-registration registry (:cpp:func:`KernelRegistrations`):
/// every kernel translation unit registers its installer through
/// ``ONNX_LIGHT_CPU_REGISTER_KERNEL``, and this iterates that registry, so a
/// single call installs the accelerated elementwise/GEMM kernels and the
/// portable integer matrix-multiplication kernels without any hand-maintained
/// list. After this call every such node dispatched by onnx-light's
/// runtime (``RunNode`` / ``RuntimeSession``, and therefore any model executed
/// through ``ReferenceEvaluator``) resolves to the onnx-light-cpu kernel,
/// replacing the corresponding built-in entries for the default ONNX domain.
void RegisterAllKernels();

} // namespace onnx_light_cpu
