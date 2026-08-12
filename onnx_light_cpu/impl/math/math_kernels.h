// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

// The math kernels dispatch on the runtime SIMD level shared across all kernel
// families; pull in the neutral ``SimdLevel`` enum and ``DetectSimdLevel``.
#include "onnx_light_cpu/impl/simd_level.h"

namespace onnx_light_cpu {

/// Computes elementwise absolute value: out[i] = |input[i]| for float32.
/// Dispatches to the best available SIMD path at runtime.
void AbsFloat32(const float *input, float *output, std::size_t count);

#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
void AbsFloat32_AVX512(const float *input, float *output, std::size_t count);
#endif

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

/// Computes the ONNX ``Gemm`` general matrix multiplication for float32:
///
///     Y = alpha * op(A) @ op(B) + beta * C
///
/// where ``op(A)`` is an ``M x K`` matrix (``A`` transposed when ``trans_a`` is
/// true) and ``op(B)`` is a ``K x N`` matrix (``B`` transposed when ``trans_b``
/// is true). All matrices are C-contiguous (row-major). ``C`` is an optional
/// ``M x N`` bias matrix; pass ``nullptr`` (or ``beta == 0``) to skip the bias
/// term. ``Y`` is the ``M x N`` output and must not alias any input.
///
/// The hot loop is a register-blocked micro-kernel that keeps a tile of output
/// rows by one SIMD vector of columns in registers while it accumulates the
/// whole ``k`` reduction (reusing each ``B`` row across the tile), with runtime
/// AVX-512/AVX/SSE2 dispatch and a scalar fallback; the work is split across
/// rows of ``Y`` using the shared ``ParallelFor`` thread pool.
void GemmFloat32(bool trans_a, bool trans_b, std::size_t M, std::size_t N, std::size_t K,
                 float alpha, const float *A, const float *B, float beta, const float *C, float *Y);

/// Computes the ONNX ``Gemm`` general matrix multiplication for float64.
/// Same semantics as :cpp:func:`GemmFloat32` with ``double`` operands.
void GemmFloat64(bool trans_a, bool trans_b, std::size_t M, std::size_t N, std::size_t K,
                 double alpha, const double *A, const double *B, double beta, const double *C,
                 double *Y);

} // namespace onnx_light_cpu
