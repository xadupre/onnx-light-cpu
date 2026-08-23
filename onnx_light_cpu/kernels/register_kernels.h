// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace onnx_light_cpu {

/// Registers every onnx-light-cpu kernel class into onnx-light's shared
/// ``KernelDispatchTable`` for the CPU device.
///
/// This is a convenience wrapper that calls each per-operator registration
/// function (:cpp:func:`RegisterAbsKernel`, :cpp:func:`RegisterBinaryKernels`,
/// :cpp:func:`RegisterExpKernel`, :cpp:func:`RegisterLogKernel`,
/// :cpp:func:`RegisterGemmKernel`, :cpp:func:`RegisterIntegerMatMulKernels`
/// and :cpp:func:`RegisterNotKernel`), so a single call installs the
/// accelerated elementwise/GEMM kernels and the portable integer
/// matrix-multiplication kernels.
/// After this call every such node dispatched by onnx-light's
/// runtime (``RunNode`` / ``RuntimeSession``, and therefore any model executed
/// through ``ReferenceEvaluator``) resolves to the onnx-light-cpu kernel,
/// replacing the corresponding built-in entries for the default ONNX domain.
void RegisterAllKernels();

} // namespace onnx_light_cpu
