// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace onnx_light_cpu {

/// Supported SIMD instruction sets for CPU kernel dispatch.
enum class SimdLevel : int {
  kNone = 0,   ///< Scalar fallback (no SIMD).
  kSSE2 = 1,   ///< SSE2 (128-bit).
  kAVX = 2,    ///< AVX (256-bit).
  kAVX2 = 3,   ///< AVX2 (256-bit with FMA, integer ops).
  kAVX512 = 4, ///< AVX-512F (512-bit).
};

/// Detects the highest SIMD level supported by the current CPU at runtime.
SimdLevel DetectSimdLevel();

/// Computes elementwise absolute value: out[i] = |input[i]| for float32.
/// Dispatches to the best available SIMD path at runtime.
void AbsFloat32(const float *input, float *output, std::size_t count);

/// Computes elementwise absolute value: out[i] = |input[i]| for float64.
/// Dispatches to the best available SIMD path at runtime.
void AbsFloat64(const double *input, double *output, std::size_t count);

/// Computes elementwise absolute value: out[i] = |input[i]| for int32.
/// Dispatches to the best available SIMD path at runtime.
void AbsInt32(const int32_t *input, int32_t *output, std::size_t count);

/// Computes elementwise absolute value: out[i] = |input[i]| for int64.
/// Dispatches to the best available SIMD path at runtime.
void AbsInt64(const int64_t *input, int64_t *output, std::size_t count);

} // namespace onnx_light_cpu
