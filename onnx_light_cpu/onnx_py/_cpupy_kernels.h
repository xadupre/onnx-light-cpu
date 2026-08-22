// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace onnx_light_cpu {

// Registers the SIMD detection helpers (detect_simd_level, has_cpu_kernels)
// into the nanobind module. Defined in ``_cpupy_math_kernels.cc``.
void RegisterMathKernels(nanobind::module_ &m);

// Registers ``benchmark_processor_performance_raw`` (the versioned
// processor performance profile: memory bandwidth/latency plus
// register-resident compute throughput) into the nanobind module. Defined in
// ``_cpupy_processor_profile.cc``.
void RegisterProcessorPerformanceProfile(nanobind::module_ &m);

} // namespace onnx_light_cpu
