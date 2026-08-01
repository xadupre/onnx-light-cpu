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

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define ONNX_LIGHT_CPU_X86 1
#include <immintrin.h>
#else
#define ONNX_LIGHT_CPU_X86 0
#endif

namespace onnx_light_cpu {

namespace {

// ---------------------------------------------------------------------------
// float16 <-> float32 conversion (round to nearest, ties to even)
// ---------------------------------------------------------------------------

float HalfBitsToFloat(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
  std::uint32_t exp = (h >> 10) & 0x1fu;
  std::uint32_t mant = h & 0x3ffu;
  std::uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign; // +/-0
    } else {
      // Subnormal half: normalize into a float32 normal.
      exp = 127u - 15u + 1u;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x3ffu;
      f = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 0x1f) {
    f = sign | 0x7f800000u | (mant << 13); // Inf / NaN
  } else {
    f = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}

std::uint16_t FloatToHalfBits(float value) {
  std::uint32_t x;
  std::memcpy(&x, &value, sizeof(x));
  const std::uint16_t sign = static_cast<std::uint16_t>((x >> 16) & 0x8000u);
  const std::uint32_t biased = (x >> 23) & 0xffu;
  const std::uint32_t mant = x & 0x7fffffu;

  if (biased == 0xff) {
    // Inf or NaN. Preserve NaN-ness (non-zero mantissa) with a quiet NaN.
    return static_cast<std::uint16_t>(sign | (mant ? 0x7e00u : 0x7c00u));
  }

  const std::int32_t exp = static_cast<std::int32_t>(biased) - 127 + 15;
  if (exp >= 0x1f) {
    return static_cast<std::uint16_t>(sign | 0x7c00u); // overflow -> Inf
  }
  if (exp <= 0) {
    if (exp < -10) {
      return sign; // too small -> signed zero
    }
    // Subnormal half.
    std::uint32_t m = mant | 0x800000u; // restore implicit leading 1
    const int shift = 14 - exp;
    std::uint32_t half_mant = m >> shift;
    const std::uint32_t rem = m & ((1u << shift) - 1u);
    const std::uint32_t halfway = 1u << (shift - 1);
    if (rem > halfway || (rem == halfway && (half_mant & 1u))) {
      ++half_mant;
    }
    return static_cast<std::uint16_t>(sign | half_mant);
  }
  // Normal half.
  std::uint16_t half_mant = static_cast<std::uint16_t>(mant >> 13);
  const std::uint32_t rem = mant & 0x1fffu;
  std::uint16_t h =
      static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | half_mant);
  if (rem > 0x1000u || (rem == 0x1000u && (half_mant & 1u))) {
    ++h; // round up; carry propagates into the exponent field as required
  }
  return h;
}

// ---------------------------------------------------------------------------
// Scalar implementations (also the non-x86 fallback).
// ---------------------------------------------------------------------------

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

#if ONNX_LIGHT_CPU_X86

// ---------------------------------------------------------------------------
// float32 constants (Cephes / avx_mathfun).
// ---------------------------------------------------------------------------

constexpr float kExpHi32 = 88.3762626647949f;
constexpr float kExpLo32 = -88.3762626647949f;
constexpr float kLog2ef = 1.44269504088896341f;
constexpr float kExpC1_32 = 0.693359375f;
constexpr float kExpC2_32 = -2.12194440e-4f;
constexpr float kExpP0_32 = 1.9875691500e-4f;
constexpr float kExpP1_32 = 1.3981999507e-3f;
constexpr float kExpP2_32 = 8.3334519073e-3f;
constexpr float kExpP3_32 = 4.1665795894e-2f;
constexpr float kExpP4_32 = 1.6666665459e-1f;
constexpr float kExpP5_32 = 5.0000001201e-1f;

constexpr float kSqrtHf = 0.707106781186547524f;
constexpr float kLogP0_32 = 7.0376836292e-2f;
constexpr float kLogP1_32 = -1.1514610310e-1f;
constexpr float kLogP2_32 = 1.1676998740e-1f;
constexpr float kLogP3_32 = -1.2420140846e-1f;
constexpr float kLogP4_32 = 1.4249322787e-1f;
constexpr float kLogP5_32 = -1.6668057665e-1f;
constexpr float kLogP6_32 = 2.0000714765e-1f;
constexpr float kLogP7_32 = -2.4999993993e-1f;
constexpr float kLogP8_32 = 3.3333331174e-1f;
constexpr float kLogQ1_32 = -2.12194440e-4f;
constexpr float kLogQ2_32 = 0.693359375f;

// ---------------------------------------------------------------------------
// double constants (Cephes).
// ---------------------------------------------------------------------------

constexpr double kExpHi64 = 709.78271289338399673222;
constexpr double kExpLo64 = -708.39641853226410622441;
constexpr double kLog2e = 1.4426950408889634073599;
constexpr double kExpC1_64 = 6.93145751953125e-1;
constexpr double kExpC2_64 = 1.42860682030941723212e-6;
constexpr double kExpP0_64 = 1.26177193074810590878e-4;
constexpr double kExpP1_64 = 3.02994407707441961300e-2;
constexpr double kExpP2_64 = 9.99999999999999999910e-1;
constexpr double kExpQ0_64 = 3.00198505138664455042e-6;
constexpr double kExpQ1_64 = 2.52448340349684104192e-3;
constexpr double kExpQ2_64 = 2.27265548208155028766e-1;
constexpr double kExpQ3_64 = 2.00000000000000000005e0;

constexpr double kSqrtH64 = 0.70710678118654752440;
constexpr double kLogP0_64 = 1.01875663804580931796e-4;
constexpr double kLogP1_64 = 4.97494994976747001425e-1;
constexpr double kLogP2_64 = 4.70579119878881725854e0;
constexpr double kLogP3_64 = 1.44989225341610930846e1;
constexpr double kLogP4_64 = 1.79368678507819816313e1;
constexpr double kLogP5_64 = 7.70838733755885391666e0;
constexpr double kLogQ0_64 = 1.12873587189167450590e1;
constexpr double kLogQ1_64 = 4.52279145837532221105e1;
constexpr double kLogQ2_64 = 8.29875266912776603211e1;
constexpr double kLogQ3_64 = 7.11544750618563894466e1;
constexpr double kLogQ4_64 = 2.31251620126765340583e1;
constexpr double kLogC1_64 = 2.121944400546905827679e-4; // subtracted
constexpr double kLogC2_64 = 0.693359375;                // added

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

  emm0 = _mm_cvttps_epi32(fx);
  emm0 = _mm_add_epi32(emm0, _mm_set1_epi32(0x7f));
  emm0 = _mm_slli_epi32(emm0, 23);
  y = _mm_mul_ps(y, _mm_castsi128_ps(emm0));

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

  __m128 xw = _mm_max_ps(x, _mm_castsi128_ps(_mm_set1_epi32(0x00800000)));
  __m128i emm0 = _mm_srli_epi32(_mm_castps_si128(xw), 23);
  xw = _mm_and_ps(xw, _mm_castsi128_ps(_mm_set1_epi32(static_cast<int>(~0x7f800000u))));
  xw = _mm_or_ps(xw, _mm_set1_ps(0.5f));
  emm0 = _mm_sub_epi32(emm0, _mm_set1_epi32(0x7f));
  __m128 e = _mm_add_ps(_mm_cvtepi32_ps(emm0), one);

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

// ---------------------------------------------------------------------------
// AVX2 helpers (256-bit). The 256-bit reductions require AVX2 integer ops.
// ---------------------------------------------------------------------------

inline __m256 Select(__m256 mask, __m256 a, __m256 b) { return _mm256_blendv_ps(b, a, mask); }

inline __m256d Select(__m256d mask, __m256d a, __m256d b) { return _mm256_blendv_pd(b, a, mask); }

__m256 ExpPs256(__m256 x) {
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 hi = _mm256_set1_ps(kExpHi32);
  const __m256 lo = _mm256_set1_ps(kExpLo32);

  const __m256 is_nan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
  const __m256 over = _mm256_cmp_ps(x, hi, _CMP_GT_OQ);
  const __m256 under = _mm256_cmp_ps(x, lo, _CMP_LT_OQ);

  __m256 xc = _mm256_min_ps(_mm256_max_ps(x, lo), hi);
  __m256 fx = _mm256_add_ps(_mm256_mul_ps(xc, _mm256_set1_ps(kLog2ef)), _mm256_set1_ps(0.5f));
  __m256i emm0 = _mm256_cvttps_epi32(fx);
  __m256 tmp = _mm256_cvtepi32_ps(emm0);
  const __m256 floor_fix = _mm256_and_ps(_mm256_cmp_ps(tmp, fx, _CMP_GT_OQ), one);
  fx = _mm256_sub_ps(tmp, floor_fix);

  xc = _mm256_sub_ps(xc, _mm256_mul_ps(fx, _mm256_set1_ps(kExpC1_32)));
  xc = _mm256_sub_ps(xc, _mm256_mul_ps(fx, _mm256_set1_ps(kExpC2_32)));
  const __m256 z = _mm256_mul_ps(xc, xc);

  __m256 y = _mm256_set1_ps(kExpP0_32);
  y = _mm256_add_ps(_mm256_mul_ps(y, xc), _mm256_set1_ps(kExpP1_32));
  y = _mm256_add_ps(_mm256_mul_ps(y, xc), _mm256_set1_ps(kExpP2_32));
  y = _mm256_add_ps(_mm256_mul_ps(y, xc), _mm256_set1_ps(kExpP3_32));
  y = _mm256_add_ps(_mm256_mul_ps(y, xc), _mm256_set1_ps(kExpP4_32));
  y = _mm256_add_ps(_mm256_mul_ps(y, xc), _mm256_set1_ps(kExpP5_32));
  y = _mm256_add_ps(_mm256_mul_ps(y, z), xc);
  y = _mm256_add_ps(y, one);

  emm0 = _mm256_cvttps_epi32(fx);
  emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(0x7f));
  emm0 = _mm256_slli_epi32(emm0, 23);
  y = _mm256_mul_ps(y, _mm256_castsi256_ps(emm0));

  y = Select(under, _mm256_setzero_ps(), y);
  y = Select(over, _mm256_set1_ps(std::numeric_limits<float>::infinity()), y);
  y = Select(is_nan, _mm256_set1_ps(std::numeric_limits<float>::quiet_NaN()), y);
  return y;
}

__m256 LogPs256(__m256 x) {
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 zero = _mm256_setzero_ps();
  const __m256 pos_inf = _mm256_set1_ps(std::numeric_limits<float>::infinity());

  const __m256 is_nan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
  const __m256 is_neg = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);
  const __m256 is_zero = _mm256_cmp_ps(x, zero, _CMP_EQ_OQ);
  const __m256 is_inf = _mm256_cmp_ps(x, pos_inf, _CMP_EQ_OQ);

  __m256 xw = _mm256_max_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0x00800000)));
  __m256i emm0 = _mm256_srli_epi32(_mm256_castps_si256(xw), 23);
  xw = _mm256_and_ps(xw, _mm256_castsi256_ps(_mm256_set1_epi32(static_cast<int>(~0x7f800000u))));
  xw = _mm256_or_ps(xw, _mm256_set1_ps(0.5f));
  emm0 = _mm256_sub_epi32(emm0, _mm256_set1_epi32(0x7f));
  __m256 e = _mm256_add_ps(_mm256_cvtepi32_ps(emm0), one);

  const __m256 mask = _mm256_cmp_ps(xw, _mm256_set1_ps(kSqrtHf), _CMP_LT_OQ);
  const __m256 tmp = _mm256_and_ps(xw, mask);
  xw = _mm256_sub_ps(xw, one);
  e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
  xw = _mm256_add_ps(xw, tmp);
  const __m256 z = _mm256_mul_ps(xw, xw);

  __m256 y = _mm256_set1_ps(kLogP0_32);
  y = _mm256_add_ps(_mm256_mul_ps(y, xw), _mm256_set1_ps(kLogP1_32));
  y = _mm256_add_ps(_mm256_mul_ps(y, xw), _mm256_set1_ps(kLogP2_32));
  y = _mm256_add_ps(_mm256_mul_ps(y, xw), _mm256_set1_ps(kLogP3_32));
  y = _mm256_add_ps(_mm256_mul_ps(y, xw), _mm256_set1_ps(kLogP4_32));
  y = _mm256_add_ps(_mm256_mul_ps(y, xw), _mm256_set1_ps(kLogP5_32));
  y = _mm256_add_ps(_mm256_mul_ps(y, xw), _mm256_set1_ps(kLogP6_32));
  y = _mm256_add_ps(_mm256_mul_ps(y, xw), _mm256_set1_ps(kLogP7_32));
  y = _mm256_add_ps(_mm256_mul_ps(y, xw), _mm256_set1_ps(kLogP8_32));
  y = _mm256_mul_ps(y, xw);
  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, _mm256_mul_ps(e, _mm256_set1_ps(kLogQ1_32)));
  y = _mm256_sub_ps(y, _mm256_mul_ps(z, _mm256_set1_ps(0.5f)));
  xw = _mm256_add_ps(xw, y);
  xw = _mm256_add_ps(xw, _mm256_mul_ps(e, _mm256_set1_ps(kLogQ2_32)));

  xw = Select(is_neg, _mm256_set1_ps(std::numeric_limits<float>::quiet_NaN()), xw);
  xw = Select(is_zero, _mm256_set1_ps(-std::numeric_limits<float>::infinity()), xw);
  xw = Select(is_inf, pos_inf, xw);
  xw = Select(is_nan, _mm256_set1_ps(std::numeric_limits<float>::quiet_NaN()), xw);
  return xw;
}

__m256d ExpPd256(__m256d x) {
  const __m256d one = _mm256_set1_pd(1.0);
  const __m256d hi = _mm256_set1_pd(kExpHi64);
  const __m256d lo = _mm256_set1_pd(kExpLo64);

  const __m256d is_nan = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);
  const __m256d over = _mm256_cmp_pd(x, hi, _CMP_GT_OQ);
  const __m256d under = _mm256_cmp_pd(x, lo, _CMP_LT_OQ);

  __m256d xc = _mm256_min_pd(_mm256_max_pd(x, lo), hi);
  __m256d fx = _mm256_add_pd(_mm256_mul_pd(xc, _mm256_set1_pd(kLog2e)), _mm256_set1_pd(0.5));
  __m128i n32 = _mm256_cvttpd_epi32(fx);
  __m256d tmp = _mm256_cvtepi32_pd(n32);
  const __m256d floor_fix = _mm256_and_pd(_mm256_cmp_pd(tmp, fx, _CMP_GT_OQ), one);
  fx = _mm256_sub_pd(tmp, floor_fix);

  xc = _mm256_sub_pd(xc, _mm256_mul_pd(fx, _mm256_set1_pd(kExpC1_64)));
  xc = _mm256_sub_pd(xc, _mm256_mul_pd(fx, _mm256_set1_pd(kExpC2_64)));
  const __m256d xx = _mm256_mul_pd(xc, xc);

  __m256d px = _mm256_set1_pd(kExpP0_64);
  px = _mm256_add_pd(_mm256_mul_pd(px, xx), _mm256_set1_pd(kExpP1_64));
  px = _mm256_add_pd(_mm256_mul_pd(px, xx), _mm256_set1_pd(kExpP2_64));
  px = _mm256_mul_pd(px, xc);

  __m256d qx = _mm256_set1_pd(kExpQ0_64);
  qx = _mm256_add_pd(_mm256_mul_pd(qx, xx), _mm256_set1_pd(kExpQ1_64));
  qx = _mm256_add_pd(_mm256_mul_pd(qx, xx), _mm256_set1_pd(kExpQ2_64));
  qx = _mm256_add_pd(_mm256_mul_pd(qx, xx), _mm256_set1_pd(kExpQ3_64));

  __m256d r = _mm256_div_pd(px, _mm256_sub_pd(qx, px));
  r = _mm256_add_pd(one, _mm256_add_pd(r, r));

  n32 = _mm256_cvttpd_epi32(fx);
  __m256i n64 = _mm256_cvtepi32_epi64(n32);
  n64 = _mm256_add_epi64(n64, _mm256_set1_epi64x(1023));
  n64 = _mm256_slli_epi64(n64, 52);
  r = _mm256_mul_pd(r, _mm256_castsi256_pd(n64));

  r = Select(under, _mm256_setzero_pd(), r);
  r = Select(over, _mm256_set1_pd(std::numeric_limits<double>::infinity()), r);
  r = Select(is_nan, _mm256_set1_pd(std::numeric_limits<double>::quiet_NaN()), r);
  return r;
}

__m256d LogPd256(__m256d x) {
  const __m256d zero = _mm256_setzero_pd();
  const __m256d pos_inf = _mm256_set1_pd(std::numeric_limits<double>::infinity());

  const __m256d is_nan = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);
  const __m256d is_neg = _mm256_cmp_pd(x, zero, _CMP_LT_OQ);
  const __m256d is_zero = _mm256_cmp_pd(x, zero, _CMP_EQ_OQ);
  const __m256d is_inf = _mm256_cmp_pd(x, pos_inf, _CMP_EQ_OQ);

  __m256d xw = _mm256_max_pd(x, _mm256_castsi256_pd(_mm256_set1_epi64x(0x0010000000000000LL)));
  __m256i xi = _mm256_castpd_si256(xw);
  __m256i ei = _mm256_and_si256(_mm256_srli_epi64(xi, 52), _mm256_set1_epi64x(0x7ff));
  __m256d e = _mm256_sub_pd(
      _mm256_castsi256_pd(_mm256_or_si256(ei, _mm256_set1_epi64x(0x4330000000000000LL))),
      _mm256_set1_pd(4503599627370496.0 + 1022.0));
  __m256i mant = _mm256_and_si256(xi, _mm256_set1_epi64x(0x800fffffffffffffLL));
  mant = _mm256_or_si256(mant, _mm256_set1_epi64x(0x3fe0000000000000LL));
  xw = _mm256_castsi256_pd(mant);

  const __m256d one = _mm256_set1_pd(1.0);
  const __m256d mask = _mm256_cmp_pd(xw, _mm256_set1_pd(kSqrtH64), _CMP_LT_OQ);
  const __m256d tmp = _mm256_and_pd(xw, mask);
  xw = _mm256_sub_pd(xw, one);
  e = _mm256_sub_pd(e, _mm256_and_pd(one, mask));
  xw = _mm256_add_pd(xw, tmp);
  const __m256d z = _mm256_mul_pd(xw, xw);

  __m256d px = _mm256_set1_pd(kLogP0_64);
  px = _mm256_add_pd(_mm256_mul_pd(px, xw), _mm256_set1_pd(kLogP1_64));
  px = _mm256_add_pd(_mm256_mul_pd(px, xw), _mm256_set1_pd(kLogP2_64));
  px = _mm256_add_pd(_mm256_mul_pd(px, xw), _mm256_set1_pd(kLogP3_64));
  px = _mm256_add_pd(_mm256_mul_pd(px, xw), _mm256_set1_pd(kLogP4_64));
  px = _mm256_add_pd(_mm256_mul_pd(px, xw), _mm256_set1_pd(kLogP5_64));

  __m256d qx = _mm256_add_pd(xw, _mm256_set1_pd(kLogQ0_64));
  qx = _mm256_add_pd(_mm256_mul_pd(qx, xw), _mm256_set1_pd(kLogQ1_64));
  qx = _mm256_add_pd(_mm256_mul_pd(qx, xw), _mm256_set1_pd(kLogQ2_64));
  qx = _mm256_add_pd(_mm256_mul_pd(qx, xw), _mm256_set1_pd(kLogQ3_64));
  qx = _mm256_add_pd(_mm256_mul_pd(qx, xw), _mm256_set1_pd(kLogQ4_64));

  __m256d y = _mm256_mul_pd(xw, _mm256_mul_pd(z, _mm256_div_pd(px, qx)));
  y = _mm256_sub_pd(y, _mm256_mul_pd(e, _mm256_set1_pd(kLogC1_64)));
  y = _mm256_sub_pd(y, _mm256_mul_pd(z, _mm256_set1_pd(0.5)));
  __m256d r = _mm256_add_pd(xw, y);
  r = _mm256_add_pd(r, _mm256_mul_pd(e, _mm256_set1_pd(kLogC2_64)));

  r = Select(is_neg, _mm256_set1_pd(std::numeric_limits<double>::quiet_NaN()), r);
  r = Select(is_zero, _mm256_set1_pd(-std::numeric_limits<double>::infinity()), r);
  r = Select(is_inf, pos_inf, r);
  r = Select(is_nan, _mm256_set1_pd(std::numeric_limits<double>::quiet_NaN()), r);
  return r;
}

void ExpFloat32_AVX2(const float *input, float *output, std::size_t count) {
  std::size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    _mm256_storeu_ps(output + i, ExpPs256(_mm256_loadu_ps(input + i)));
  }
  if (i < count) {
    ExpFloat32_SSE2(input + i, output + i, count - i);
  }
}

void LogFloat32_AVX2(const float *input, float *output, std::size_t count) {
  std::size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    _mm256_storeu_ps(output + i, LogPs256(_mm256_loadu_ps(input + i)));
  }
  if (i < count) {
    LogFloat32_SSE2(input + i, output + i, count - i);
  }
}

void ExpFloat64_AVX2(const double *input, double *output, std::size_t count) {
  std::size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    _mm256_storeu_pd(output + i, ExpPd256(_mm256_loadu_pd(input + i)));
  }
  if (i < count) {
    ExpFloat64_SSE2(input + i, output + i, count - i);
  }
}

void LogFloat64_AVX2(const double *input, double *output, std::size_t count) {
  std::size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    _mm256_storeu_pd(output + i, LogPd256(_mm256_loadu_pd(input + i)));
  }
  if (i < count) {
    LogFloat64_SSE2(input + i, output + i, count - i);
  }
}

#endif // ONNX_LIGHT_CPU_X86

} // namespace

// ---------------------------------------------------------------------------
// Public dispatchers.
// ---------------------------------------------------------------------------

void ExpFloat32(const float *input, float *output, std::size_t count) {
  if (count == 0)
    return;
#if ONNX_LIGHT_CPU_X86
  static const SimdLevel level = DetectSimdLevel();
  if (level >= SimdLevel::kAVX2) {
    ExpFloat32_AVX2(input, output, count);
    return;
  }
  if (level >= SimdLevel::kSSE2) {
    ExpFloat32_SSE2(input, output, count);
    return;
  }
#endif
  ExpFloat32_Scalar(input, output, count);
}

void LogFloat32(const float *input, float *output, std::size_t count) {
  if (count == 0)
    return;
#if ONNX_LIGHT_CPU_X86
  static const SimdLevel level = DetectSimdLevel();
  if (level >= SimdLevel::kAVX2) {
    LogFloat32_AVX2(input, output, count);
    return;
  }
  if (level >= SimdLevel::kSSE2) {
    LogFloat32_SSE2(input, output, count);
    return;
  }
#endif
  LogFloat32_Scalar(input, output, count);
}

void ExpFloat64(const double *input, double *output, std::size_t count) {
  if (count == 0)
    return;
#if ONNX_LIGHT_CPU_X86
  static const SimdLevel level = DetectSimdLevel();
  if (level >= SimdLevel::kAVX2) {
    ExpFloat64_AVX2(input, output, count);
    return;
  }
  if (level >= SimdLevel::kSSE2) {
    ExpFloat64_SSE2(input, output, count);
    return;
  }
#endif
  ExpFloat64_Scalar(input, output, count);
}

void LogFloat64(const double *input, double *output, std::size_t count) {
  if (count == 0)
    return;
#if ONNX_LIGHT_CPU_X86
  static const SimdLevel level = DetectSimdLevel();
  if (level >= SimdLevel::kAVX2) {
    LogFloat64_AVX2(input, output, count);
    return;
  }
  if (level >= SimdLevel::kSSE2) {
    LogFloat64_SSE2(input, output, count);
    return;
  }
#endif
  LogFloat64_Scalar(input, output, count);
}

void ExpFloat16(const uint16_t *input, uint16_t *output, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    output[i] = FloatToHalfBits(std::exp(HalfBitsToFloat(input[i])));
  }
}

void LogFloat16(const uint16_t *input, uint16_t *output, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    output[i] = FloatToHalfBits(std::log(HalfBitsToFloat(input[i])));
  }
}

} // namespace onnx_light_cpu
