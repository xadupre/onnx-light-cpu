// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Fast elementwise natural exponential (Exp) and natural logarithm (Log)
// kernels for float32, float64 and float16.
//
// The float32 and float64 paths use vectorized minimax polynomial
// approximations (the classic Cephes / avx_mathfun reductions) with runtime
// SSE2/AVX2 dispatch and an accurate ``std::exp``/``std::log`` scalar fallback.
// Special values (+/-inf, NaN, overflow, underflow, non-positive logarithm
// arguments) are patched with explicit vector masks so the results match the
// standard library. The float16 path widens each half-precision value to
// float32, evaluates the scalar standard-library function and rounds the
// result back to float16.

#include "onnx_light_cpu/impl/math/math_kernels.h"

#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/exp_log_constants.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define ONNX_LIGHT_CPU_X86 1
#include <immintrin.h>
#else
#define ONNX_LIGHT_CPU_X86 0
#endif

namespace onnx_light_cpu {

#ifdef ONNX_LIGHT_CPU_HAVE_AVX2
void ExpFloat32_AVX2(const float *input, float *output, std::size_t count);
void LogFloat32_AVX2(const float *input, float *output, std::size_t count);
void ExpFloat64_AVX2(const double *input, double *output, std::size_t count);
void LogFloat64_AVX2(const double *input, double *output, std::size_t count);
#endif

namespace {

void ExpFloat32_Scalar(const float *input, float *output, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    output[i] = std::exp(input[i]);
  }
}

void LogFloat32_Scalar(const float *input, float *output, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    output[i] = std::log(input[i]);
  }
}

void ExpFloat64_Scalar(const double *input, double *output, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    output[i] = std::exp(input[i]);
  }
}

void LogFloat64_Scalar(const double *input, double *output, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    output[i] = std::log(input[i]);
  }
}

const std::array<std::uint16_t, 65536> &LogFloat16Table() {
  static const std::array<std::uint16_t, 65536> table = [] {
    std::array<std::uint16_t, 65536> values{};
    for (std::size_t i = 0; i < values.size(); ++i) {
      const auto bits = static_cast<std::uint16_t>(i);
      values[i] = detail::FloatToFloat16Bits(std::log(detail::Float16BitsToFloat(bits)));
    }
    return values;
  }();
  return table;
}

#if ONNX_LIGHT_CPU_X86

// ---------------------------------------------------------------------------
// float32 constants (Cephes / avx_mathfun).
// ---------------------------------------------------------------------------

using namespace detail;

// ---------------------------------------------------------------------------
// SSE2 helpers.
// ---------------------------------------------------------------------------

inline __m128 Select(__m128 mask, __m128 a, __m128 b) {
  return _mm_or_ps(_mm_and_ps(mask, a), _mm_andnot_ps(mask, b));
}

inline __m128d Select(__m128d mask, __m128d a, __m128d b) {
  return _mm_or_pd(_mm_and_pd(mask, a), _mm_andnot_pd(mask, b));
}

__m128 ExpPs(__m128 x) {
  const __m128 one = _mm_set1_ps(1.0f);
  const __m128 hi = _mm_set1_ps(kExpHi32);
  const __m128 lo = _mm_set1_ps(kExpLo32);

  const __m128 is_nan = _mm_cmpunord_ps(x, x);
  const __m128 over = _mm_cmpgt_ps(x, hi);  // includes +inf
  const __m128 under = _mm_cmplt_ps(x, lo); // includes -inf

  __m128 xc = _mm_min_ps(_mm_max_ps(x, lo), hi);
  __m128 fx = _mm_add_ps(_mm_mul_ps(xc, _mm_set1_ps(kLog2ef)), _mm_set1_ps(0.5f));
  __m128i emm0 = _mm_cvttps_epi32(fx);
  __m128 tmp = _mm_cvtepi32_ps(emm0);
  const __m128 floor_fix = _mm_and_ps(_mm_cmpgt_ps(tmp, fx), one);
  fx = _mm_sub_ps(tmp, floor_fix);

  xc = _mm_sub_ps(xc, _mm_mul_ps(fx, _mm_set1_ps(kExpC1_32)));
  xc = _mm_sub_ps(xc, _mm_mul_ps(fx, _mm_set1_ps(kExpC2_32)));
  const __m128 z = _mm_mul_ps(xc, xc);

  __m128 y = _mm_set1_ps(kExpP0_32);
  y = _mm_add_ps(_mm_mul_ps(y, xc), _mm_set1_ps(kExpP1_32));
  y = _mm_add_ps(_mm_mul_ps(y, xc), _mm_set1_ps(kExpP2_32));
  y = _mm_add_ps(_mm_mul_ps(y, xc), _mm_set1_ps(kExpP3_32));
  y = _mm_add_ps(_mm_mul_ps(y, xc), _mm_set1_ps(kExpP4_32));
  y = _mm_add_ps(_mm_mul_ps(y, xc), _mm_set1_ps(kExpP5_32));
  y = _mm_add_ps(_mm_mul_ps(y, z), xc);
  y = _mm_add_ps(y, one);

  // Reconstruct 2^n as two normal-range factors so that n as low as -149
  // (the smallest float32 subnormal exponent) still produces a correctly
  // rounded subnormal result instead of an invalid bit pattern from a single
  // biased-exponent shift.
  emm0 = _mm_cvttps_epi32(fx);
  const __m128i n1 = _mm_srai_epi32(emm0, 1);
  const __m128i n2 = _mm_sub_epi32(emm0, n1);
  const __m128i pow1 = _mm_slli_epi32(_mm_add_epi32(n1, _mm_set1_epi32(0x7f)), 23);
  const __m128i pow2 = _mm_slli_epi32(_mm_add_epi32(n2, _mm_set1_epi32(0x7f)), 23);
  y = _mm_mul_ps(y, _mm_castsi128_ps(pow1));
  y = _mm_mul_ps(y, _mm_castsi128_ps(pow2));

  y = Select(under, _mm_setzero_ps(), y);
  y = Select(over, _mm_set1_ps(std::numeric_limits<float>::infinity()), y);
  y = Select(is_nan, _mm_set1_ps(std::numeric_limits<float>::quiet_NaN()), y);
  return y;
}

__m128 LogPs(__m128 x) {
  const __m128 one = _mm_set1_ps(1.0f);
  const __m128 zero = _mm_setzero_ps();
  const __m128 pos_inf = _mm_set1_ps(std::numeric_limits<float>::infinity());

  const __m128 is_nan = _mm_cmpunord_ps(x, x);
  const __m128 is_neg = _mm_cmplt_ps(x, zero);
  const __m128 is_zero = _mm_cmpeq_ps(x, zero);
  const __m128 is_inf = _mm_cmpeq_ps(x, pos_inf);

  // Normalize positive subnormals instead of clamping them to the smallest
  // normal float: multiplying by 2^23 is exact (no rounding) and moves the
  // value into the normal range, so every subnormal keeps its distinct
  // mantissa. The extracted exponent is corrected below by subtracting 23
  // for the lanes that were normalized.
  const __m128 smallest_normal = _mm_castsi128_ps(_mm_set1_epi32(0x00800000));
  const __m128 is_subnormal = _mm_and_ps(_mm_cmpgt_ps(x, zero), _mm_cmplt_ps(x, smallest_normal));
  const __m128 scaled = _mm_mul_ps(x, _mm_set1_ps(kSubnormalScale32));
  __m128 xw = Select(is_subnormal, scaled, x);
  __m128i emm0 = _mm_srli_epi32(_mm_castps_si128(xw), 23);
  xw = _mm_and_ps(xw, _mm_castsi128_ps(_mm_set1_epi32(static_cast<int>(~0x7f800000u))));
  xw = _mm_or_ps(xw, _mm_set1_ps(0.5f));
  emm0 = _mm_sub_epi32(emm0, _mm_set1_epi32(0x7f));
  __m128 e = _mm_add_ps(_mm_cvtepi32_ps(emm0), one);
  e = _mm_sub_ps(e, _mm_and_ps(is_subnormal, _mm_set1_ps(23.0f)));

  const __m128 mask = _mm_cmplt_ps(xw, _mm_set1_ps(kSqrtHf));
  const __m128 tmp = _mm_and_ps(xw, mask);
  xw = _mm_sub_ps(xw, one);
  e = _mm_sub_ps(e, _mm_and_ps(one, mask));
  xw = _mm_add_ps(xw, tmp);
  const __m128 z = _mm_mul_ps(xw, xw);

  __m128 y = _mm_set1_ps(kLogP0_32);
  y = _mm_add_ps(_mm_mul_ps(y, xw), _mm_set1_ps(kLogP1_32));
  y = _mm_add_ps(_mm_mul_ps(y, xw), _mm_set1_ps(kLogP2_32));
  y = _mm_add_ps(_mm_mul_ps(y, xw), _mm_set1_ps(kLogP3_32));
  y = _mm_add_ps(_mm_mul_ps(y, xw), _mm_set1_ps(kLogP4_32));
  y = _mm_add_ps(_mm_mul_ps(y, xw), _mm_set1_ps(kLogP5_32));
  y = _mm_add_ps(_mm_mul_ps(y, xw), _mm_set1_ps(kLogP6_32));
  y = _mm_add_ps(_mm_mul_ps(y, xw), _mm_set1_ps(kLogP7_32));
  y = _mm_add_ps(_mm_mul_ps(y, xw), _mm_set1_ps(kLogP8_32));
  y = _mm_mul_ps(y, xw);
  y = _mm_mul_ps(y, z);
  y = _mm_add_ps(y, _mm_mul_ps(e, _mm_set1_ps(kLogQ1_32)));
  y = _mm_sub_ps(y, _mm_mul_ps(z, _mm_set1_ps(0.5f)));
  xw = _mm_add_ps(xw, y);
  xw = _mm_add_ps(xw, _mm_mul_ps(e, _mm_set1_ps(kLogQ2_32)));

  xw = Select(is_neg, _mm_set1_ps(std::numeric_limits<float>::quiet_NaN()), xw);
  xw = Select(is_zero, _mm_set1_ps(-std::numeric_limits<float>::infinity()), xw);
  xw = Select(is_inf, pos_inf, xw);
  xw = Select(is_nan, _mm_set1_ps(std::numeric_limits<float>::quiet_NaN()), xw);
  return xw;
}

__m128d ExpPd(__m128d x) {
  const __m128d one = _mm_set1_pd(1.0);
  const __m128d hi = _mm_set1_pd(kExpHi64);
  const __m128d lo = _mm_set1_pd(kExpLo64);

  const __m128d is_nan = _mm_cmpunord_pd(x, x);
  const __m128d over = _mm_cmpgt_pd(x, hi);
  const __m128d under = _mm_cmplt_pd(x, lo);

  __m128d xc = _mm_min_pd(_mm_max_pd(x, lo), hi);
  __m128d fx = _mm_add_pd(_mm_mul_pd(xc, _mm_set1_pd(kLog2e)), _mm_set1_pd(0.5));
  __m128i n32 = _mm_cvttpd_epi32(fx);
  __m128d tmp = _mm_cvtepi32_pd(n32);
  const __m128d floor_fix = _mm_and_pd(_mm_cmpgt_pd(tmp, fx), one);
  fx = _mm_sub_pd(tmp, floor_fix);

  xc = _mm_sub_pd(xc, _mm_mul_pd(fx, _mm_set1_pd(kExpC1_64)));
  xc = _mm_sub_pd(xc, _mm_mul_pd(fx, _mm_set1_pd(kExpC2_64)));
  const __m128d xx = _mm_mul_pd(xc, xc);

  __m128d px = _mm_set1_pd(kExpP0_64);
  px = _mm_add_pd(_mm_mul_pd(px, xx), _mm_set1_pd(kExpP1_64));
  px = _mm_add_pd(_mm_mul_pd(px, xx), _mm_set1_pd(kExpP2_64));
  px = _mm_mul_pd(px, xc);

  __m128d qx = _mm_set1_pd(kExpQ0_64);
  qx = _mm_add_pd(_mm_mul_pd(qx, xx), _mm_set1_pd(kExpQ1_64));
  qx = _mm_add_pd(_mm_mul_pd(qx, xx), _mm_set1_pd(kExpQ2_64));
  qx = _mm_add_pd(_mm_mul_pd(qx, xx), _mm_set1_pd(kExpQ3_64));

  __m128d r = _mm_div_pd(px, _mm_sub_pd(qx, px));
  r = _mm_add_pd(one, _mm_add_pd(r, r));

  // 2^n via exponent bits: (n + 1023) << 52.
  n32 = _mm_cvttpd_epi32(fx);
  const __m128i sign = _mm_srai_epi32(n32, 31);
  __m128i n64 = _mm_unpacklo_epi32(n32, sign);
  n64 = _mm_add_epi64(n64, _mm_set1_epi64x(1023));
  n64 = _mm_slli_epi64(n64, 52);
  r = _mm_mul_pd(r, _mm_castsi128_pd(n64));

  r = Select(under, _mm_setzero_pd(), r);
  r = Select(over, _mm_set1_pd(std::numeric_limits<double>::infinity()), r);
  r = Select(is_nan, _mm_set1_pd(std::numeric_limits<double>::quiet_NaN()), r);
  return r;
}

__m128d LogPd(__m128d x) {
  const __m128d zero = _mm_setzero_pd();
  const __m128d pos_inf = _mm_set1_pd(std::numeric_limits<double>::infinity());

  const __m128d is_nan = _mm_cmpunord_pd(x, x);
  const __m128d is_neg = _mm_cmplt_pd(x, zero);
  const __m128d is_zero = _mm_cmpeq_pd(x, zero);
  const __m128d is_inf = _mm_cmpeq_pd(x, pos_inf);

  __m128d xw = _mm_max_pd(x, _mm_castsi128_pd(_mm_set1_epi64x(0x0010000000000000LL)));
  __m128i xi = _mm_castpd_si128(xw);
  __m128i ei = _mm_and_si128(_mm_srli_epi64(xi, 52), _mm_set1_epi64x(0x7ff));
  __m128d e = _mm_sub_pd(_mm_castsi128_pd(_mm_or_si128(ei, _mm_set1_epi64x(0x4330000000000000LL))),
                         _mm_set1_pd(4503599627370496.0 + 1022.0));
  __m128i mant = _mm_and_si128(xi, _mm_set1_epi64x(0x800fffffffffffffLL));
  mant = _mm_or_si128(mant, _mm_set1_epi64x(0x3fe0000000000000LL));
  xw = _mm_castsi128_pd(mant);

  const __m128d one = _mm_set1_pd(1.0);
  const __m128d mask = _mm_cmplt_pd(xw, _mm_set1_pd(kSqrtH64));
  const __m128d tmp = _mm_and_pd(xw, mask);
  xw = _mm_sub_pd(xw, one);
  e = _mm_sub_pd(e, _mm_and_pd(one, mask));
  xw = _mm_add_pd(xw, tmp);
  const __m128d z = _mm_mul_pd(xw, xw);

  __m128d px = _mm_set1_pd(kLogP0_64);
  px = _mm_add_pd(_mm_mul_pd(px, xw), _mm_set1_pd(kLogP1_64));
  px = _mm_add_pd(_mm_mul_pd(px, xw), _mm_set1_pd(kLogP2_64));
  px = _mm_add_pd(_mm_mul_pd(px, xw), _mm_set1_pd(kLogP3_64));
  px = _mm_add_pd(_mm_mul_pd(px, xw), _mm_set1_pd(kLogP4_64));
  px = _mm_add_pd(_mm_mul_pd(px, xw), _mm_set1_pd(kLogP5_64));

  __m128d qx = _mm_add_pd(xw, _mm_set1_pd(kLogQ0_64));
  qx = _mm_add_pd(_mm_mul_pd(qx, xw), _mm_set1_pd(kLogQ1_64));
  qx = _mm_add_pd(_mm_mul_pd(qx, xw), _mm_set1_pd(kLogQ2_64));
  qx = _mm_add_pd(_mm_mul_pd(qx, xw), _mm_set1_pd(kLogQ3_64));
  qx = _mm_add_pd(_mm_mul_pd(qx, xw), _mm_set1_pd(kLogQ4_64));

  __m128d y = _mm_mul_pd(xw, _mm_mul_pd(z, _mm_div_pd(px, qx)));
  y = _mm_sub_pd(y, _mm_mul_pd(e, _mm_set1_pd(kLogC1_64)));
  y = _mm_sub_pd(y, _mm_mul_pd(z, _mm_set1_pd(0.5)));
  __m128d r = _mm_add_pd(xw, y);
  r = _mm_add_pd(r, _mm_mul_pd(e, _mm_set1_pd(kLogC2_64)));

  r = Select(is_neg, _mm_set1_pd(std::numeric_limits<double>::quiet_NaN()), r);
  r = Select(is_zero, _mm_set1_pd(-std::numeric_limits<double>::infinity()), r);
  r = Select(is_inf, pos_inf, r);
  r = Select(is_nan, _mm_set1_pd(std::numeric_limits<double>::quiet_NaN()), r);
  return r;
}

#endif // ONNX_LIGHT_CPU_X86
} // namespace

#if ONNX_LIGHT_CPU_X86
void ExpFloat32_SSE2(const float *input, float *output, std::size_t count) {
  std::size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    _mm_storeu_ps(output + i, ExpPs(_mm_loadu_ps(input + i)));
  }
  if (i < count) {
    ExpFloat32_Scalar(input + i, output + i, count - i);
  }
}

void LogFloat32_SSE2(const float *input, float *output, std::size_t count) {
  std::size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    _mm_storeu_ps(output + i, LogPs(_mm_loadu_ps(input + i)));
  }
  if (i < count) {
    LogFloat32_Scalar(input + i, output + i, count - i);
  }
}

void ExpFloat64_SSE2(const double *input, double *output, std::size_t count) {
  std::size_t i = 0;
  for (; i + 2 <= count; i += 2) {
    _mm_storeu_pd(output + i, ExpPd(_mm_loadu_pd(input + i)));
  }
  if (i < count) {
    ExpFloat64_Scalar(input + i, output + i, count - i);
  }
}

void LogFloat64_SSE2(const double *input, double *output, std::size_t count) {
  std::size_t i = 0;
  for (; i + 2 <= count; i += 2) {
    _mm_storeu_pd(output + i, LogPd(_mm_loadu_pd(input + i)));
  }
  if (i < count) {
    LogFloat64_Scalar(input + i, output + i, count - i);
  }
}

#endif // ONNX_LIGHT_CPU_X86

// ---------------------------------------------------------------------------
// Public dispatchers.
// ---------------------------------------------------------------------------

namespace {
using Float32UnaryFn = void (*)(const float *, float *, std::size_t);

struct Float32UnaryDispatch {
  Float32UnaryFn function;
  double compute_cycles;
};

constexpr double SimdComputeCycles(double avx2_cycles, std::size_t simd_lanes) {
  constexpr std::size_t kAvx2Lanes = 8;
  return avx2_cycles * static_cast<double>(kAvx2Lanes) / static_cast<double>(simd_lanes);
}

const Float32UnaryDispatch &GetExpFloat32Dispatch() {
  static const Float32UnaryDispatch dispatch = [] {
#if ONNX_LIGHT_CPU_X86
    const SimdLevel level = DetectSimdLevel();
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
    if (level >= SimdLevel::kAVX512) {
      return Float32UnaryDispatch{&ExpFloat32_AVX512, SimdComputeCycles(1.5, 16)};
    }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
    if (level >= SimdLevel::kAVX2 && CpuSupportsFma()) {
      return Float32UnaryDispatch{&ExpFloat32_AVX2_FMA, SimdComputeCycles(1.5, 8)};
    }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2
    if (level >= SimdLevel::kAVX2) {
      return Float32UnaryDispatch{&ExpFloat32_AVX2, SimdComputeCycles(1.5, 8)};
    }
#endif
    if (level >= SimdLevel::kSSE2) {
      return Float32UnaryDispatch{&ExpFloat32_SSE2, SimdComputeCycles(1.5, 4)};
    }
#endif
    return Float32UnaryDispatch{&ExpFloat32_Scalar, SimdComputeCycles(1.5, 1)};
  }();
  return dispatch;
}

void ExpFloat32_Dispatch(const float *input, float *output, std::size_t count) {
  GetExpFloat32Dispatch().function(input, output, count);
}
} // namespace

void ExpFloat32(const float *input, float *output, std::size_t count) {
  ExpFloat32WithTuning(input, output, count, kDefaultExpLogExecutionTuning);
}

void ExpFloat32WithTuning(const float *input, float *output, std::size_t count,
                          const UnaryExecutionTuning &tuning) {
  const Float32UnaryDispatch &dispatch = GetExpFloat32Dispatch();
  auto execute = [input, output, &dispatch](std::int64_t begin, std::int64_t end) {
    dispatch.function(input + begin, output + begin, static_cast<std::size_t>(end - begin));
  };
  if (tuning.use_cost_model) {
    ExecuteCostedUnaryRanges<float>(count, tuning, dispatch.compute_cycles, std::move(execute));
  } else {
    ExecuteUnaryRanges<float>(count, tuning, std::move(execute));
  }
}

namespace {
const Float32UnaryDispatch &GetLogFloat32Dispatch() {
  static const Float32UnaryDispatch dispatch = [] {
#if ONNX_LIGHT_CPU_X86
    const SimdLevel level = DetectSimdLevel();
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
    const bool use_avx2_fma = level >= SimdLevel::kAVX2 && CpuSupportsFma();
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
    if (level >= SimdLevel::kAVX512) {
      return Float32UnaryDispatch{&LogFloat32_AVX512, SimdComputeCycles(100.0, 16)};
    }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
    if (use_avx2_fma) {
      return Float32UnaryDispatch{&LogFloat32_AVX2_FMA, SimdComputeCycles(100.0, 8)};
    }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2
    if (level >= SimdLevel::kAVX2) {
      return Float32UnaryDispatch{&LogFloat32_AVX2, SimdComputeCycles(100.0, 8)};
    }
#endif
    if (level >= SimdLevel::kSSE2) {
      return Float32UnaryDispatch{&LogFloat32_SSE2, SimdComputeCycles(100.0, 4)};
    }
#endif
    return Float32UnaryDispatch{&LogFloat32_Scalar, SimdComputeCycles(100.0, 1)};
  }();
  return dispatch;
}

void LogFloat32_Dispatch(const float *input, float *output, std::size_t count) {
  GetLogFloat32Dispatch().function(input, output, count);
}
} // namespace

void LogFloat32(const float *input, float *output, std::size_t count) {
  LogFloat32WithTuning(input, output, count, kDefaultExpLogExecutionTuning);
}

void LogFloat32WithTuning(const float *input, float *output, std::size_t count,
                          const UnaryExecutionTuning &tuning) {
  const Float32UnaryDispatch &dispatch = GetLogFloat32Dispatch();
  auto execute = [input, output, &dispatch](std::int64_t begin, std::int64_t end) {
    dispatch.function(input + begin, output + begin, static_cast<std::size_t>(end - begin));
  };
  if (tuning.use_cost_model) {
    ExecuteCostedUnaryRanges<float>(count, tuning, dispatch.compute_cycles, std::move(execute));
  } else {
    ExecuteUnaryRanges<float>(count, tuning, std::move(execute));
  }
}

namespace {
void ExpFloat64_Dispatch(const double *input, double *output, std::size_t count) {
#if ONNX_LIGHT_CPU_X86
  static const SimdLevel level = DetectSimdLevel();
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
  if (level >= SimdLevel::kAVX512) {
    ExpFloat64_AVX512(input, output, count);
    return;
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2
  if (level >= SimdLevel::kAVX2) {
    ExpFloat64_AVX2(input, output, count);
    return;
  }
#endif
  if (level >= SimdLevel::kSSE2) {
    ExpFloat64_SSE2(input, output, count);
    return;
  }
#endif
  ExpFloat64_Scalar(input, output, count);
}
} // namespace

void ExpFloat64(const double *input, double *output, std::size_t count) {
  ExpFloat64WithTuning(input, output, count, kDefaultExpLogExecutionTuning);
}

void ExpFloat64WithTuning(const double *input, double *output, std::size_t count,
                          const UnaryExecutionTuning &tuning) {
  ExecuteUnaryRanges<double>(count, tuning, [input, output](std::int64_t begin, std::int64_t end) {
    ExpFloat64_Dispatch(input + begin, output + begin, static_cast<std::size_t>(end - begin));
  });
}

namespace {
void LogFloat64_Dispatch(const double *input, double *output, std::size_t count) {
#if ONNX_LIGHT_CPU_X86
  static const SimdLevel level = DetectSimdLevel();
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
  if (level >= SimdLevel::kAVX512) {
    LogFloat64_AVX512(input, output, count);
    return;
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2
  if (level >= SimdLevel::kAVX2) {
    LogFloat64_AVX2(input, output, count);
    return;
  }
#endif
  if (level >= SimdLevel::kSSE2) {
    LogFloat64_SSE2(input, output, count);
    return;
  }
#endif
  LogFloat64_Scalar(input, output, count);
}
} // namespace

void LogFloat64(const double *input, double *output, std::size_t count) {
  LogFloat64WithTuning(input, output, count, kDefaultExpLogExecutionTuning);
}

void LogFloat64WithTuning(const double *input, double *output, std::size_t count,
                          const UnaryExecutionTuning &tuning) {
  auto execute = [input, output](std::int64_t begin, std::int64_t end) {
    LogFloat64_Dispatch(input + begin, output + begin, static_cast<std::size_t>(end - begin));
  };
  if (tuning.use_cost_model) {
    ExecuteCostedUnaryRanges<double>(count, tuning, 100.0, std::move(execute));
  } else {
    ExecuteUnaryRanges<double>(count, tuning, std::move(execute));
  }
}

namespace {

constexpr std::size_t kHalfConversionBlock = 4096;

template <bool BFloat16, bool MaskLogNaN>
void TransformHalfRange(const std::uint16_t *input, std::uint16_t *output, std::int64_t begin,
                        std::int64_t end, void (*dispatch)(const float *, float *, std::size_t)) {
  alignas(64) float values[kHalfConversionBlock];
  alignas(64) std::uint8_t nan_lanes[kHalfConversionBlock];
  while (begin < end) {
    const std::size_t block = static_cast<std::size_t>(
        std::min<std::int64_t>(end - begin, static_cast<std::int64_t>(kHalfConversionBlock)));
    if constexpr (BFloat16) {
      detail::ConvertBFloat16ToFloat32(input + begin, values, block);
    } else {
      detail::ConvertFloat16ToFloat32(input + begin, values, block);
    }
    if constexpr (MaskLogNaN) {
      constexpr std::uint16_t magnitude_mask = 0x7FFFu;
      constexpr std::uint16_t infinity = BFloat16 ? 0x7F80u : 0x7C00u;
      for (std::size_t i = 0; i < block; ++i) {
        const std::uint16_t bits = input[begin + i];
        const std::uint16_t magnitude = bits & magnitude_mask;
        nan_lanes[i] = static_cast<std::uint8_t>((((bits & 0x8000u) != 0) && magnitude != 0) ||
                                                 magnitude > infinity);
        if (nan_lanes[i] != 0) {
          values[i] = 1.0f;
        }
      }
    }
    dispatch(values, values, block);
    if constexpr (BFloat16) {
      detail::ConvertFloat32ToBFloat16(values, output + begin, block);
    } else {
      detail::ConvertFloat32ToFloat16(values, output + begin, block);
    }
    if constexpr (MaskLogNaN) {
      constexpr std::uint16_t quiet_nan = BFloat16 ? 0x7FC0u : 0x7E00u;
      for (std::size_t i = 0; i < block; ++i) {
        if (nan_lanes[i] != 0) {
          output[begin + i] = quiet_nan;
        }
      }
    }
    begin += static_cast<std::int64_t>(block);
  }
}

template <bool BFloat16, bool MaskLogNaN = false>
void TransformHalf(const std::uint16_t *input, std::uint16_t *output, std::size_t count,
                   const UnaryExecutionTuning &tuning,
                   void (*dispatch)(const float *, float *, std::size_t),
                   double compute_cycles = 0.0) {
  auto execute = [input, output, dispatch](std::int64_t begin, std::int64_t end) {
    TransformHalfRange<BFloat16, MaskLogNaN>(input, output, begin, end, dispatch);
  };
  if (tuning.use_cost_model && compute_cycles > 0.0) {
    ExecuteCostedUnaryRanges<std::uint16_t>(count, tuning, compute_cycles, std::move(execute));
  } else {
    ExecuteUnaryRanges<std::uint16_t>(count, tuning, std::move(execute));
  }
}

} // namespace

void ExpFloat16(const uint16_t *input, uint16_t *output, std::size_t count) {
  ExpFloat16WithTuning(input, output, count, kDefaultExpLogHalfExecutionTuning);
}

void ExpFloat16WithTuning(const uint16_t *input, uint16_t *output, std::size_t count,
                          const UnaryExecutionTuning &tuning) {
  TransformHalf<false>(input, output, count, tuning, &ExpFloat32_Dispatch);
}

void ExpBFloat16WithTuning(const uint16_t *input, uint16_t *output, std::size_t count,
                           const UnaryExecutionTuning &tuning) {
  TransformHalf<true>(input, output, count, tuning, &ExpFloat32_Dispatch);
}

void LogFloat16(const uint16_t *input, uint16_t *output, std::size_t count) {
  LogFloat16WithTuning(input, output, count, kDefaultLogFloat16ExecutionTuning);
}

void LogFloat16WithTuning(const uint16_t *input, uint16_t *output, std::size_t count,
                          const UnaryExecutionTuning &tuning) {
  const auto &table = LogFloat16Table();
  auto execute = [input, output, &table](std::int64_t begin, std::int64_t end) {
    for (std::int64_t i = begin; i < end; ++i) {
      output[i] = table[input[i]];
    }
  };
  if (tuning.use_cost_model) {
    ExecuteCostedUnaryRanges<std::uint16_t>(count, tuning, 25.0, std::move(execute));
  } else {
    ExecuteUnaryRanges<std::uint16_t>(count, tuning, std::move(execute));
  }
}

void LogBFloat16WithTuning(const uint16_t *input, uint16_t *output, std::size_t count,
                           const UnaryExecutionTuning &tuning) {
  TransformHalf<true, true>(input, output, count, tuning, &LogFloat32_Dispatch, 20.0);
}

} // namespace onnx_light_cpu
