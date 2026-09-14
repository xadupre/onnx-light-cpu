// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/exp_log_constants.h"

#include <cstddef>
#include <immintrin.h>
#include <limits>

namespace onnx_light_cpu {

void ExpFloat32_SSE2(const float *input, float *output, std::size_t count);
void LogFloat32_SSE2(const float *input, float *output, std::size_t count);
void ExpFloat64_SSE2(const double *input, double *output, std::size_t count);
void LogFloat64_SSE2(const double *input, double *output, std::size_t count);

namespace {
using namespace detail;

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

  // Reconstruct 2^n as two normal-range factors so subnormal exponents
  // down to -149 round correctly instead of producing an invalid bit pattern.
  emm0 = _mm256_cvttps_epi32(fx);
  const __m256i n1 = _mm256_srai_epi32(emm0, 1);
  const __m256i n2 = _mm256_sub_epi32(emm0, n1);
  const __m256i pow1 = _mm256_slli_epi32(_mm256_add_epi32(n1, _mm256_set1_epi32(0x7f)), 23);
  const __m256i pow2 = _mm256_slli_epi32(_mm256_add_epi32(n2, _mm256_set1_epi32(0x7f)), 23);
  y = _mm256_mul_ps(y, _mm256_castsi256_ps(pow1));
  y = _mm256_mul_ps(y, _mm256_castsi256_ps(pow2));
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

  // Normalize positive subnormals instead of clamping them.
  const __m256 smallest_normal = _mm256_castsi256_ps(_mm256_set1_epi32(0x00800000));
  const __m256 is_subnormal = _mm256_and_ps(_mm256_cmp_ps(x, zero, _CMP_GT_OQ),
                                            _mm256_cmp_ps(x, smallest_normal, _CMP_LT_OQ));
  const __m256 scaled = _mm256_mul_ps(x, _mm256_set1_ps(kSubnormalScale32));
  __m256 xw = Select(is_subnormal, scaled, x);
  __m256i emm0 = _mm256_srli_epi32(_mm256_castps_si256(xw), 23);
  xw = _mm256_and_ps(xw, _mm256_castsi256_ps(_mm256_set1_epi32(static_cast<int>(~0x7f800000u))));
  xw = _mm256_or_ps(xw, _mm256_set1_ps(0.5f));
  emm0 = _mm256_sub_epi32(emm0, _mm256_set1_epi32(0x7f));
  __m256 e = _mm256_add_ps(_mm256_cvtepi32_ps(emm0), one);
  e = _mm256_sub_ps(e, _mm256_and_ps(is_subnormal, _mm256_set1_ps(23.0f)));
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

} // namespace

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

} // namespace onnx_light_cpu
