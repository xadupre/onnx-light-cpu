// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Declarations for the native AVX-512F register-resident FP32/FP64
// arithmetic throughput kernels (Processor Profile PR03). This translation
// unit is only compiled into lib_onnx_light_cpu when CMake's
// check_cxx_compiler_flag confirms the compiler accepts -mavx512f -- see
// CMakeLists.txt. Gate every reference behind ``#ifdef
// ONNX_LIGHT_CPU_HAVE_AVX512`` and only call once ``DetectSimdLevel()``
// reports AVX-512 at runtime.

#pragma once

#include <cstddef>

namespace onnx_light_cpu {

/// Independent scalar-equivalent accumulators (4 ZMM registers x 16 float32
/// lanes) advanced per pass by ``ComputeArithmeticAvx512Float32Round``.
constexpr std::size_t kComputeAvx512Float32AccumulatorCount = 4 * 16;
/// Independent scalar-equivalent accumulators (4 ZMM registers x 8 float64
/// lanes) advanced per pass by ``ComputeArithmeticAvx512Float64Round``.
constexpr std::size_t kComputeAvx512Float64AccumulatorCount = 4 * 8;

/// Runs ``passes`` rounds of the shared arithmetic-profile FMA chain length
/// over register-resident AVX-512 float32 accumulators and returns a
/// checksum (the caller consumes it outside the timed interval so the
/// compiler cannot remove the work).
double ComputeArithmeticAvx512Float32Round(std::size_t passes, double seed);

/// Float64 counterpart of ``ComputeArithmeticAvx512Float32Round``.
double ComputeArithmeticAvx512Float64Round(std::size_t passes, double seed);

} // namespace onnx_light_cpu
