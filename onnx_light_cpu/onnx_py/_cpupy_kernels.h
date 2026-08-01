// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace onnx_light_cpu {

// Registers the math kernel bindings (Abs/Exp/Log and SIMD detection) into the
// nanobind module. Defined in ``_cpupy_math_kernels.cc``.
void RegisterMathKernels(nanobind::module_ &m);

// Registers the logical kernel bindings (logical_not) into the nanobind module.
// Defined in ``_cpupy_logical_kernels.cc``.
void RegisterLogicalKernels(nanobind::module_ &m);

} // namespace onnx_light_cpu
