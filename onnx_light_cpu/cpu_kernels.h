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

/// Computes elementwise absolute value: out[i] = |input[i]| for float16.
/// The input and output are the raw IEEE 754 half-precision bit patterns
/// (as ``uint16_t``); the absolute value simply clears the sign bit.
/// Dispatches to the best available SIMD path at runtime.
void AbsFloat16(const uint16_t *input, uint16_t *output, std::size_t count);

/// Computes elementwise absolute value: out[i] = |input[i]| for int8.
/// Dispatches to the best available SIMD path at runtime.
void AbsInt8(const int8_t *input, int8_t *output, std::size_t count);

/// Computes elementwise absolute value: out[i] = |input[i]| for int32.
/// Dispatches to the best available SIMD path at runtime.
void AbsInt32(const int32_t *input, int32_t *output, std::size_t count);

/// Computes elementwise absolute value: out[i] = |input[i]| for int64.
/// Dispatches to the best available SIMD path at runtime.
void AbsInt64(const int64_t *input, int64_t *output, std::size_t count);

/// Computes elementwise natural exponential: out[i] = exp(input[i]) for float32.
/// Uses a vectorized minimax polynomial approximation with runtime AVX-512/
/// AVX/SSE2 dispatch and a ``std::exp`` scalar fallback. The result is accurate
/// to within a few ULPs and special values (+/-inf, NaN, overflow, underflow)
/// match ``std::exp``.
void ExpFloat32(const float *input, float *output, std::size_t count);

/// Computes elementwise natural exponential: out[i] = exp(input[i]) for float64.
/// Dispatches to the best available SIMD path at runtime.
void ExpFloat64(const double *input, double *output, std::size_t count);

/// Computes elementwise natural exponential: out[i] = exp(input[i]) for float16.
/// The input and output are the raw IEEE 754 half-precision bit patterns (as
/// ``uint16_t``); each value is widened to float32, exponentiated and rounded
/// back to float16.
void ExpFloat16(const uint16_t *input, uint16_t *output, std::size_t count);

/// Computes elementwise natural logarithm: out[i] = log(input[i]) for float32.
/// Uses a vectorized minimax polynomial approximation with runtime AVX-512/
/// AVX/SSE2 dispatch and a ``std::log`` scalar fallback. The result is accurate
/// to within a few ULPs and special values (0 -> -inf, negative -> NaN, +inf,
/// NaN) match ``std::log``.
void LogFloat32(const float *input, float *output, std::size_t count);

/// Computes elementwise natural logarithm: out[i] = log(input[i]) for float64.
/// Dispatches to the best available SIMD path at runtime.
void LogFloat64(const double *input, double *output, std::size_t count);

/// Computes elementwise natural logarithm: out[i] = log(input[i]) for float16.
/// The input and output are the raw IEEE 754 half-precision bit patterns (as
/// ``uint16_t``); each value is widened to float32, its logarithm computed and
/// rounded back to float16.
void LogFloat16(const uint16_t *input, uint16_t *output, std::size_t count);

/// Computes the elementwise logical negation: out[i] = (input[i] == 0) for
/// ``bool``. ONNX ``bool`` tensors are stored as one byte per element, so the
/// input and output are the raw byte patterns (as ``uint8_t``): every zero byte
/// maps to ``1`` and every non-zero byte maps to ``0``, matching
/// ``numpy.logical_not``. Dispatches to the best available SIMD path at
/// runtime.
void NotBool(const uint8_t *input, uint8_t *output, std::size_t count);

} // namespace onnx_light_cpu
