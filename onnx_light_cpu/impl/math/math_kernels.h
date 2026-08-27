// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include "onnx_light_cpu/impl/math/unary_execution_tuning.h"

// The math kernels dispatch on the runtime SIMD level shared across all kernel
// families; pull in the neutral ``SimdLevel`` enum and ``DetectSimdLevel``.
#include "onnx_light_cpu/impl/simd_level.h"

namespace onnx_light_cpu {

/// Logical layout of a tensor broadcast over an ``M x N`` GEMM output.
enum class GemmBroadcast {
  kNone,
  kScalar,
  kRow,
  kColumn,
  kMatrix,
};

/// Activation fused into a GEMM epilogue.
enum class GemmActivation {
  kNone,
  kRelu,
};

/// Optional narrowing performed by a GEMM epilogue.
enum class GemmOutputConversion {
  kNone,
  kFloat16,
  kBFloat16,
};

/// Typed operations applied after the matrix-product accumulation.
///
/// A row tensor contains ``N`` values and broadcasts over M. A column tensor
/// contains ``M`` values and broadcasts over N. Matrix tensors contain
/// ``M * N`` row-major values. When output conversion is enabled, ``Y`` remains
/// the accumulation workspace and the final values are written to
/// ``converted_output``.
template <typename T> struct GemmEpilogue {
  const T *bias = nullptr;
  GemmBroadcast bias_layout = GemmBroadcast::kNone;
  T beta = T(0);
  const T *residual = nullptr;
  GemmBroadcast residual_layout = GemmBroadcast::kNone;
  T residual_scale = T(1);
  GemmActivation activation = GemmActivation::kNone;
  GemmOutputConversion output_conversion = GemmOutputConversion::kNone;
  std::uint16_t *converted_output = nullptr;
};

/// Computes elementwise absolute value: out[i] = |input[i]| for float32.
/// Dispatches to the best available SIMD path at runtime.
void AbsFloat32(const float *input, float *output, std::size_t count);
void AbsFloat32WithTuning(const float *input, float *output, std::size_t count,
                          const UnaryExecutionTuning &tuning);

#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
void AbsFloat32_AVX512(const float *input, float *output, std::size_t count);
void AbsFloat32_AVX512Streaming(const float *input, float *output, std::size_t count);
#endif

/// Computes elementwise absolute value: out[i] = |input[i]| for float64.
/// Dispatches to the best available SIMD path at runtime.
void AbsFloat64(const double *input, double *output, std::size_t count);
void AbsFloat64WithTuning(const double *input, double *output, std::size_t count,
                          const UnaryExecutionTuning &tuning);

/// Computes elementwise absolute value: out[i] = |input[i]| for float16.
/// The input and output are the raw IEEE 754 half-precision bit patterns
/// (as ``uint16_t``); the absolute value simply clears the sign bit.
/// Dispatches to the best available SIMD path at runtime.
void AbsFloat16(const uint16_t *input, uint16_t *output, std::size_t count);
void AbsFloat16WithTuning(const uint16_t *input, uint16_t *output, std::size_t count,
                          const UnaryExecutionTuning &tuning);

/// Computes elementwise absolute value: out[i] = |input[i]| for int8.
/// Dispatches to the best available SIMD path at runtime.
void AbsInt8(const int8_t *input, int8_t *output, std::size_t count);
void AbsInt8WithTuning(const int8_t *input, int8_t *output, std::size_t count,
                       const UnaryExecutionTuning &tuning);

/// Computes elementwise absolute value for int16.
void AbsInt16(const int16_t *input, int16_t *output, std::size_t count);
void AbsInt16WithTuning(const int16_t *input, int16_t *output, std::size_t count,
                        const UnaryExecutionTuning &tuning);

/// Computes elementwise absolute value: out[i] = |input[i]| for int32.
/// Dispatches to the best available SIMD path at runtime.
void AbsInt32(const int32_t *input, int32_t *output, std::size_t count);
void AbsInt32WithTuning(const int32_t *input, int32_t *output, std::size_t count,
                        const UnaryExecutionTuning &tuning);

/// Computes elementwise absolute value: out[i] = |input[i]| for int64.
/// Dispatches to the best available SIMD path at runtime.
void AbsInt64(const int64_t *input, int64_t *output, std::size_t count);
void AbsInt64WithTuning(const int64_t *input, int64_t *output, std::size_t count,
                        const UnaryExecutionTuning &tuning);

/// Computes elementwise natural exponential: out[i] = exp(input[i]) for float32.
/// Uses a vectorized minimax polynomial approximation with runtime AVX-512/
/// AVX/SSE2 dispatch and a ``std::exp`` scalar fallback. The result is accurate
/// to within a few ULPs and special values (+/-inf, NaN, overflow, underflow)
/// match ``std::exp``.
void ExpFloat32(const float *input, float *output, std::size_t count);
void ExpFloat32WithTuning(const float *input, float *output, std::size_t count,
                          const UnaryExecutionTuning &tuning);

#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
void ExpFloat32_AVX512(const float *input, float *output, std::size_t count);
#endif

#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
void ExpFloat32_AVX2_FMA(const float *input, float *output, std::size_t count);
#endif

/// Computes elementwise natural exponential: out[i] = exp(input[i]) for float64.
/// Dispatches to the best available SIMD path at runtime.
void ExpFloat64(const double *input, double *output, std::size_t count);
void ExpFloat64WithTuning(const double *input, double *output, std::size_t count,
                          const UnaryExecutionTuning &tuning);

#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
void ExpFloat64_AVX512(const double *input, double *output, std::size_t count);
#endif

/// Computes elementwise natural exponential: out[i] = exp(input[i]) for float16.
/// The input and output are the raw IEEE 754 half-precision bit patterns (as
/// ``uint16_t``); blocks are widened with vector conversion, evaluated by the
/// float32 SIMD approximation and narrowed once.
void ExpFloat16(const uint16_t *input, uint16_t *output, std::size_t count);
void ExpFloat16WithTuning(const uint16_t *input, uint16_t *output, std::size_t count,
                          const UnaryExecutionTuning &tuning);
void ExpBFloat16WithTuning(const uint16_t *input, uint16_t *output, std::size_t count,
                           const UnaryExecutionTuning &tuning);

void SwiGLUFloat32(const float *gate, const float *value, float *output, std::size_t count,
                   float alpha);
void SwiGLUFloat64(const double *gate, const double *value, double *output, std::size_t count,
                   double alpha);
void SwiGLUFloat16(const std::uint16_t *gate, const std::uint16_t *value, std::uint16_t *output,
                   std::size_t count, float alpha);
void SwiGLUBFloat16(const std::uint16_t *gate, const std::uint16_t *value, std::uint16_t *output,
                    std::size_t count, float alpha);

/// Computes elementwise natural logarithm: out[i] = log(input[i]) for float32.
/// Uses a vectorized minimax polynomial approximation with runtime AVX-512/
/// AVX/SSE2 dispatch and a ``std::log`` scalar fallback. The result is accurate
/// to within a few ULPs and special values (0 -> -inf, negative -> NaN, +inf,
/// NaN) match ``std::log``.
void LogFloat32(const float *input, float *output, std::size_t count);
void LogFloat32WithTuning(const float *input, float *output, std::size_t count,
                          const UnaryExecutionTuning &tuning);

#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
void LogFloat32_AVX2_FMA(const float *input, float *output, std::size_t count);
#endif

#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
void LogFloat32_AVX512(const float *input, float *output, std::size_t count);
void PowFloat32_AVX512(const float *base, const float *exponent, float *output, std::size_t count);
void PowFloat32LeftScalar_AVX512(float base, const float *exponent, float *output,
                                 std::size_t count);
#endif

/// Computes elementwise natural logarithm: out[i] = log(input[i]) for float64.
/// Dispatches to the best available SIMD path at runtime.
void LogFloat64(const double *input, double *output, std::size_t count);
void LogFloat64WithTuning(const double *input, double *output, std::size_t count,
                          const UnaryExecutionTuning &tuning);

#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
void LogFloat64_AVX512(const double *input, double *output, std::size_t count);
#endif

/// Computes elementwise natural logarithm: out[i] = log(input[i]) for float16.
/// The input and output are the raw IEEE 754 half-precision bit patterns (as
/// ``uint16_t``); each value is widened, evaluated with the standard library
/// and narrowed with exact half-precision rounding.
void LogFloat16(const uint16_t *input, uint16_t *output, std::size_t count);
void LogFloat16WithTuning(const uint16_t *input, uint16_t *output, std::size_t count,
                          const UnaryExecutionTuning &tuning);
void LogBFloat16WithTuning(const uint16_t *input, uint16_t *output, std::size_t count,
                           const UnaryExecutionTuning &tuning);

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
/// rows of ``Y`` using the executor supplied by the embedding runtime.
void GemmFloat32(bool trans_a, bool trans_b, std::size_t M, std::size_t N, std::size_t K,
                 float alpha, const float *A, const float *B, float beta, const float *C, float *Y);

/// Computes float32 GEMM and applies broadcast bias, residual, and activation
/// through one typed epilogue. Full-matrix bias without other post-operations
/// stays fused in the SIMD micro-kernel. FP16/BF16 conversion writes through
/// ``epilogue.converted_output``.
void GemmFloat32WithEpilogue(bool trans_a, bool trans_b, std::size_t M, std::size_t N,
                             std::size_t K, float alpha, const float *A, const float *B,
                             const GemmEpilogue<float> &epilogue, float *Y);

/// Computes the ONNX ``Gemm`` general matrix multiplication for float64.
/// Same semantics as :cpp:func:`GemmFloat32` with ``double`` operands.
void GemmFloat64(bool trans_a, bool trans_b, std::size_t M, std::size_t N, std::size_t K,
                 double alpha, const double *A, const double *B, double beta, const double *C,
                 double *Y);

/// Float64 counterpart of :cpp:func:`GemmFloat32WithEpilogue`.
void GemmFloat64WithEpilogue(bool trans_a, bool trans_b, std::size_t M, std::size_t N,
                             std::size_t K, double alpha, const double *A, const double *B,
                             const GemmEpilogue<double> &epilogue, double *Y);

/// Computes an FP16 or BF16 GEMM and applies the epilogue. ``A`` and ``B`` are
/// the raw 16-bit input patterns; ``is_bfloat16`` selects BF16 (``true``) or
/// FP16 (``false``). Both operands are converted to ``float32`` while they are
/// packed into the micro-kernel panels -- there is no separate full-tensor
/// widening pass -- and the reduction accumulates in ``float32``. ``Y`` is the
/// ``M x N`` float32 accumulation workspace; the epilogue narrows the result
/// back to FP16/BF16 through ``epilogue.converted_output``.
void GemmHalfWithEpilogue(bool is_bfloat16, bool trans_a, bool trans_b, std::size_t M,
                          std::size_t N, std::size_t K, float alpha, const std::uint16_t *A,
                          const std::uint16_t *B, const GemmEpilogue<float> &epilogue, float *Y);

/// The four ONNX Float8 storage formats accepted by
/// :cpp:func:`GemmFloat8WithEpilogue`.
enum class GemmFloat8Format {
  kE4M3FN,   ///< 1-4-3, bias 7, finite, NaN at 0x7f / 0xff.
  kE4M3FNUZ, ///< 1-4-3, bias 8, finite, single zero, NaN at 0x80.
  kE5M2,     ///< 1-5-2, bias 15, IEEE-like with infinities and NaNs.
  kE5M2FNUZ, ///< 1-5-2, bias 16, finite, single zero, NaN at 0x80.
};

/// Computes a Float8 GEMM and applies the epilogue (Roadmap PR09.5). ``A`` and
/// ``B`` are raw one-byte Float8 patterns of ``format`` (both operands share the
/// format). Each element is decoded to ``float32`` while it is packed into the
/// micro-kernel panels -- there is no separate full-tensor conversion pass, and
/// each format is handled as a separate packing format rather than a branch in
/// the FP32 inner loop -- and the reduction accumulates in ``float32``. ``Y`` is
/// the ``M x N`` float32 accumulation workspace on which the epilogue is applied.
void GemmFloat8WithEpilogue(GemmFloat8Format format, bool trans_a, bool trans_b, std::size_t M,
                            std::size_t N, std::size_t K, float alpha, const std::uint8_t *A,
                            const std::uint8_t *B, const GemmEpilogue<float> &epilogue, float *Y);

/// Validates that an :cpp:class:`GemmEpilogue` is internally consistent for an
/// ``M x N`` output, throwing ``std::invalid_argument`` otherwise. Exposed so a
/// prepared :cpp:class:`GemmPlan` can apply the same epilogue as the ONNX
/// operator kernel without re-deriving the plan on every run.
template <typename T>
void ValidateGemmEpilogue(std::size_t M, std::size_t N, const GemmEpilogue<T> &epilogue);

/// Applies the broadcast bias, residual, activation, and optional output
/// narrowing described by ``epilogue`` to the ``M x N`` accumulation buffer
/// ``Y`` in place. The matrix product must already be stored in ``Y``. Exposed
/// alongside :cpp:func:`ValidateGemmEpilogue` for the prepared-plan operator
/// path.
template <typename T>
void ApplyGemmEpilogue(std::size_t M, std::size_t N, const GemmEpilogue<T> &epilogue, T *Y);

} // namespace onnx_light_cpu
