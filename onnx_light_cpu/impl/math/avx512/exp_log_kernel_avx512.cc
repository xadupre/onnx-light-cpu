// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/math_kernels.h"

#include <cstddef>
#include <immintrin.h>
#include <limits>

namespace onnx_light_cpu {

namespace {

constexpr float kExpHi = 88.7762626647950f;
// True float32 underflow boundary (half of the smallest subnormal, i.e.
// -150 * ln(2)); see exp_log_kernel.cc for the full rationale. Range
// reduction below this bound uses a split power-of-two reconstruction so
// subnormal results round correctly instead of flushing to zero early. The
// clamp keeps the reduced exponent n above -150, so the split halves
// (n1 = n>>1, n2 = n-n1, biased by 0x7f) always stay within the normal
// exponent range [1, 254].
constexpr float kExpLo = -103.97208f;

// Smallest positive normal float32 (bit pattern 0x00800000) and the exact
// 2^23 scale factor used to normalize positive Log subnormals without
// rounding error; see exp_log_kernel.cc for the full rationale.
constexpr float kSmallestNormal = 1.17549435e-38f;
constexpr float kSubnormalScale = 8388608.0f;
constexpr float kLog2ef = 1.44269504088896341f;
constexpr float kExpC1 = -6.93145752e-1f;
constexpr float kExpC2 = -1.42860677e-6f;
constexpr float kExpP0 = 0x1.694000p-10f;
constexpr float kExpP1 = 0x1.125edcp-7f;
constexpr float kExpP2 = 0x1.555b5ap-5f;
constexpr float kExpP3 = 0x1.555450p-3f;
constexpr float kExpP4 = 0x1.fffff6p-2f;

constexpr float kSqrtHalf = 0.707106781186547524f;
constexpr float kLogP0 = 7.0376836292e-2f;
constexpr float kLogP1 = -1.1514610310e-1f;
constexpr float kLogP2 = 1.1676998740e-1f;
constexpr float kLogP3 = -1.2420140846e-1f;
constexpr float kLogP4 = 1.4249322787e-1f;
constexpr float kLogP5 = -1.6668057665e-1f;
constexpr float kLogP6 = 2.0000714765e-1f;
constexpr float kLogP7 = -2.4999993993e-1f;
constexpr float kLogP8 = 3.3333331174e-1f;
constexpr float kLogQ1 = -2.12194440e-4f;
constexpr float kLogQ2 = 0.693359375f;

constexpr double kExpHi64 = 709.78271289338399673222;
constexpr double kExpLo64 = -708.39641853226410622441;
constexpr double kLog2e64 = 1.4426950408889634073599;
constexpr double kExpC1_64 = 6.93145751953125e-1;
constexpr double kExpC2_64 = 1.42860682030941723212e-6;
constexpr double kExpP0_64 = 1.26177193074810590878e-4;
constexpr double kExpP1_64 = 3.02994407707441961300e-2;
constexpr double kExpP2_64 = 9.99999999999999999910e-1;
constexpr double kExpQ0_64 = 3.00198505138664455042e-6;
constexpr double kExpQ1_64 = 2.52448340349684104192e-3;
constexpr double kExpQ2_64 = 2.27265548208155028766e-1;
constexpr double kExpQ3_64 = 2.00000000000000000005e0;

constexpr double kSqrtHalf64 = 0.70710678118654752440;
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
constexpr double kLogC1_64 = 2.121944400546905827679e-4;
constexpr double kLogC2_64 = 0.693359375;

__m512 Select(__mmask16 mask, __m512 selected, __m512 fallback) {
  return _mm512_mask_mov_ps(fallback, mask, selected);
}

__m512d Select(__mmask8 mask, __m512d selected, __m512d fallback) {
  return _mm512_mask_mov_pd(fallback, mask, selected);
}

#if defined(_MSC_VER)
#define ONNX_LIGHT_CPU_FORCE_INLINE __forceinline
#else
#define ONNX_LIGHT_CPU_FORCE_INLINE inline __attribute__((always_inline))
#endif

ONNX_LIGHT_CPU_FORCE_INLINE __m512 ExpPs(__m512 x) {
  const __m512 one = _mm512_set1_ps(1.0f);
  const __m512 hi = _mm512_set1_ps(kExpHi);
  const __m512 lo = _mm512_set1_ps(kExpLo);

  const __mmask16 is_nan = _mm512_cmp_ps_mask(x, x, _CMP_UNORD_Q);
  const __mmask16 over = _mm512_cmp_ps_mask(x, hi, _CMP_GT_OQ);
  const __mmask16 under = _mm512_cmp_ps_mask(x, lo, _CMP_LT_OQ);

  __m512 reduced = _mm512_min_ps(_mm512_max_ps(x, lo), hi);
  const __m512 magic = _mm512_set1_ps(12582912.0f);
  const __m512 scaled = _mm512_fmadd_ps(reduced, _mm512_set1_ps(kLog2ef), magic);
  const __m512i exponent_int =
      _mm512_sub_epi32(_mm512_castps_si512(scaled), _mm512_set1_epi32(0x4b400000));
  const __m512 exponent = _mm512_cvtepi32_ps(exponent_int);

  reduced = _mm512_fmadd_ps(exponent, _mm512_set1_ps(kExpC1), reduced);
  reduced = _mm512_fmadd_ps(exponent, _mm512_set1_ps(kExpC2), reduced);

  __m512 polynomial = _mm512_set1_ps(kExpP0);
  polynomial = _mm512_fmadd_ps(polynomial, reduced, _mm512_set1_ps(kExpP1));
  polynomial = _mm512_fmadd_ps(polynomial, reduced, _mm512_set1_ps(kExpP2));
  polynomial = _mm512_fmadd_ps(polynomial, reduced, _mm512_set1_ps(kExpP3));
  polynomial = _mm512_fmadd_ps(polynomial, reduced, _mm512_set1_ps(kExpP4));
  polynomial = _mm512_fmadd_ps(polynomial, reduced, one);
  polynomial = _mm512_fmadd_ps(polynomial, reduced, one);
  __m512 result = _mm512_scalef_ps(polynomial, exponent);

  result = Select(under, _mm512_setzero_ps(), result);
  result = Select(over, _mm512_set1_ps(std::numeric_limits<float>::infinity()), result);
  return Select(is_nan, _mm512_set1_ps(std::numeric_limits<float>::quiet_NaN()), result);
}

ONNX_LIGHT_CPU_FORCE_INLINE __m512 LogPs(__m512 x) {
  const __m512 one = _mm512_set1_ps(1.0f);
  const __m512 zero = _mm512_setzero_ps();
  const __m512 positive_infinity = _mm512_set1_ps(std::numeric_limits<float>::infinity());

  const __mmask16 is_nan = _mm512_cmp_ps_mask(x, x, _CMP_UNORD_Q);
  const __mmask16 is_negative = _mm512_cmp_ps_mask(x, zero, _CMP_LT_OQ);
  const __mmask16 is_zero = _mm512_cmp_ps_mask(x, zero, _CMP_EQ_OQ);
  const __mmask16 is_infinite = _mm512_cmp_ps_mask(x, positive_infinity, _CMP_EQ_OQ);

  // Normalize positive subnormals instead of clamping them to the smallest
  // normal float (see exp_log_kernel.cc) so distinct subnormals still yield
  // distinct, correct results.
  const __mmask16 is_subnormal = _mm512_cmp_ps_mask(x, zero, _CMP_GT_OQ) &
                                 _mm512_cmp_ps_mask(x, _mm512_set1_ps(kSmallestNormal), _CMP_LT_OQ);
  const __m512 scaled = _mm512_mul_ps(x, _mm512_set1_ps(kSubnormalScale));
  __m512 reduced = Select(is_subnormal, scaled, x);
  __m512i exponent_int = _mm512_srli_epi32(_mm512_castps_si512(reduced), 23);
  __m512i reduced_bits = _mm512_and_si512(_mm512_castps_si512(reduced),
                                          _mm512_set1_epi32(static_cast<int>(~0x7f800000u)));
  reduced_bits = _mm512_or_si512(reduced_bits, _mm512_castps_si512(_mm512_set1_ps(0.5f)));
  reduced = _mm512_castsi512_ps(reduced_bits);
  exponent_int = _mm512_sub_epi32(exponent_int, _mm512_set1_epi32(0x7f));
  __m512 exponent = _mm512_add_ps(_mm512_cvtepi32_ps(exponent_int), one);
  exponent = _mm512_sub_ps(exponent, _mm512_maskz_mov_ps(is_subnormal, _mm512_set1_ps(23.0f)));

  const __mmask16 normalize = _mm512_cmp_ps_mask(reduced, _mm512_set1_ps(kSqrtHalf), _CMP_LT_OQ);
  const __m512 normalized_part = _mm512_maskz_mov_ps(normalize, reduced);
  reduced = _mm512_sub_ps(reduced, one);
  exponent = _mm512_sub_ps(exponent, _mm512_maskz_mov_ps(normalize, one));
  reduced = _mm512_add_ps(reduced, normalized_part);
  const __m512 squared = _mm512_mul_ps(reduced, reduced);

  __m512 polynomial = _mm512_set1_ps(kLogP0);
  polynomial = _mm512_fmadd_ps(polynomial, reduced, _mm512_set1_ps(kLogP1));
  polynomial = _mm512_fmadd_ps(polynomial, reduced, _mm512_set1_ps(kLogP2));
  polynomial = _mm512_fmadd_ps(polynomial, reduced, _mm512_set1_ps(kLogP3));
  polynomial = _mm512_fmadd_ps(polynomial, reduced, _mm512_set1_ps(kLogP4));
  polynomial = _mm512_fmadd_ps(polynomial, reduced, _mm512_set1_ps(kLogP5));
  polynomial = _mm512_fmadd_ps(polynomial, reduced, _mm512_set1_ps(kLogP6));
  polynomial = _mm512_fmadd_ps(polynomial, reduced, _mm512_set1_ps(kLogP7));
  polynomial = _mm512_fmadd_ps(polynomial, reduced, _mm512_set1_ps(kLogP8));
  polynomial = _mm512_mul_ps(polynomial, reduced);
  polynomial = _mm512_mul_ps(polynomial, squared);
  polynomial = _mm512_fmadd_ps(exponent, _mm512_set1_ps(kLogQ1), polynomial);
  polynomial = _mm512_fnmadd_ps(squared, _mm512_set1_ps(0.5f), polynomial);
  __m512 result = _mm512_add_ps(reduced, polynomial);
  result = _mm512_fmadd_ps(exponent, _mm512_set1_ps(kLogQ2), result);

  result = Select(is_negative, _mm512_set1_ps(std::numeric_limits<float>::quiet_NaN()), result);
  result = Select(is_zero, _mm512_set1_ps(-std::numeric_limits<float>::infinity()), result);
  result = Select(is_infinite, positive_infinity, result);
  return Select(is_nan, _mm512_set1_ps(std::numeric_limits<float>::quiet_NaN()), result);
}

#undef ONNX_LIGHT_CPU_FORCE_INLINE

__m512d ExpPd(__m512d x) {
  const __m512d one = _mm512_set1_pd(1.0);
  const __m512d hi = _mm512_set1_pd(kExpHi64);
  const __m512d lo = _mm512_set1_pd(kExpLo64);
  const __mmask8 is_nan = _mm512_cmp_pd_mask(x, x, _CMP_UNORD_Q);
  const __mmask8 over = _mm512_cmp_pd_mask(x, hi, _CMP_GT_OQ);
  const __mmask8 under = _mm512_cmp_pd_mask(x, lo, _CMP_LT_OQ);

  __m512d reduced = _mm512_min_pd(_mm512_max_pd(x, lo), hi);
  __m512d exponent =
      _mm512_add_pd(_mm512_mul_pd(reduced, _mm512_set1_pd(kLog2e64)), _mm512_set1_pd(0.5));
  __m256i exponent_int32 = _mm512_cvttpd_epi32(exponent);
  const __m512d exponent_floor = _mm512_cvtepi32_pd(exponent_int32);
  const __mmask8 floor_mask = _mm512_cmp_pd_mask(exponent_floor, exponent, _CMP_GT_OQ);
  exponent = _mm512_sub_pd(exponent_floor, _mm512_maskz_mov_pd(floor_mask, one));

  reduced = _mm512_sub_pd(reduced, _mm512_mul_pd(exponent, _mm512_set1_pd(kExpC1_64)));
  reduced = _mm512_sub_pd(reduced, _mm512_mul_pd(exponent, _mm512_set1_pd(kExpC2_64)));
  const __m512d squared = _mm512_mul_pd(reduced, reduced);

  __m512d numerator = _mm512_set1_pd(kExpP0_64);
  numerator = _mm512_add_pd(_mm512_mul_pd(numerator, squared), _mm512_set1_pd(kExpP1_64));
  numerator = _mm512_add_pd(_mm512_mul_pd(numerator, squared), _mm512_set1_pd(kExpP2_64));
  numerator = _mm512_mul_pd(numerator, reduced);

  __m512d denominator = _mm512_set1_pd(kExpQ0_64);
  denominator = _mm512_add_pd(_mm512_mul_pd(denominator, squared), _mm512_set1_pd(kExpQ1_64));
  denominator = _mm512_add_pd(_mm512_mul_pd(denominator, squared), _mm512_set1_pd(kExpQ2_64));
  denominator = _mm512_add_pd(_mm512_mul_pd(denominator, squared), _mm512_set1_pd(kExpQ3_64));

  __m512d result = _mm512_div_pd(numerator, _mm512_sub_pd(denominator, numerator));
  result = _mm512_add_pd(one, _mm512_add_pd(result, result));

  exponent_int32 = _mm512_cvttpd_epi32(exponent);
  __m512i exponent_int64 = _mm512_cvtepi32_epi64(exponent_int32);
  exponent_int64 = _mm512_add_epi64(exponent_int64, _mm512_set1_epi64(1023));
  exponent_int64 = _mm512_slli_epi64(exponent_int64, 52);
  result = _mm512_mul_pd(result, _mm512_castsi512_pd(exponent_int64));

  result = Select(under, _mm512_setzero_pd(), result);
  result = Select(over, _mm512_set1_pd(std::numeric_limits<double>::infinity()), result);
  return Select(is_nan, _mm512_set1_pd(std::numeric_limits<double>::quiet_NaN()), result);
}

__m512d LogPd(__m512d x) {
  const __m512d zero = _mm512_setzero_pd();
  const __m512d one = _mm512_set1_pd(1.0);
  const __m512d positive_infinity = _mm512_set1_pd(std::numeric_limits<double>::infinity());
  const __mmask8 is_nan = _mm512_cmp_pd_mask(x, x, _CMP_UNORD_Q);
  const __mmask8 is_negative = _mm512_cmp_pd_mask(x, zero, _CMP_LT_OQ);
  const __mmask8 is_zero = _mm512_cmp_pd_mask(x, zero, _CMP_EQ_OQ);
  const __mmask8 is_infinite = _mm512_cmp_pd_mask(x, positive_infinity, _CMP_EQ_OQ);

  __m512d reduced = _mm512_max_pd(x, _mm512_castsi512_pd(_mm512_set1_epi64(0x0010000000000000LL)));
  const __m512i bits = _mm512_castpd_si512(reduced);
  const __m512i exponent_bits =
      _mm512_and_si512(_mm512_srli_epi64(bits, 52), _mm512_set1_epi64(0x7ff));
  __m512d exponent = _mm512_sub_pd(
      _mm512_castsi512_pd(_mm512_or_si512(exponent_bits, _mm512_set1_epi64(0x4330000000000000LL))),
      _mm512_set1_pd(4503599627370496.0 + 1022.0));
  __m512i mantissa = _mm512_and_si512(bits, _mm512_set1_epi64(0x800fffffffffffffLL));
  mantissa = _mm512_or_si512(mantissa, _mm512_set1_epi64(0x3fe0000000000000LL));
  reduced = _mm512_castsi512_pd(mantissa);

  const __mmask8 normalize = _mm512_cmp_pd_mask(reduced, _mm512_set1_pd(kSqrtHalf64), _CMP_LT_OQ);
  const __m512d normalized_part = _mm512_maskz_mov_pd(normalize, reduced);
  reduced = _mm512_sub_pd(reduced, one);
  exponent = _mm512_sub_pd(exponent, _mm512_maskz_mov_pd(normalize, one));
  reduced = _mm512_add_pd(reduced, normalized_part);
  const __m512d squared = _mm512_mul_pd(reduced, reduced);

  __m512d numerator = _mm512_set1_pd(kLogP0_64);
  numerator = _mm512_add_pd(_mm512_mul_pd(numerator, reduced), _mm512_set1_pd(kLogP1_64));
  numerator = _mm512_add_pd(_mm512_mul_pd(numerator, reduced), _mm512_set1_pd(kLogP2_64));
  numerator = _mm512_add_pd(_mm512_mul_pd(numerator, reduced), _mm512_set1_pd(kLogP3_64));
  numerator = _mm512_add_pd(_mm512_mul_pd(numerator, reduced), _mm512_set1_pd(kLogP4_64));
  numerator = _mm512_add_pd(_mm512_mul_pd(numerator, reduced), _mm512_set1_pd(kLogP5_64));

  __m512d denominator = _mm512_add_pd(reduced, _mm512_set1_pd(kLogQ0_64));
  denominator = _mm512_add_pd(_mm512_mul_pd(denominator, reduced), _mm512_set1_pd(kLogQ1_64));
  denominator = _mm512_add_pd(_mm512_mul_pd(denominator, reduced), _mm512_set1_pd(kLogQ2_64));
  denominator = _mm512_add_pd(_mm512_mul_pd(denominator, reduced), _mm512_set1_pd(kLogQ3_64));
  denominator = _mm512_add_pd(_mm512_mul_pd(denominator, reduced), _mm512_set1_pd(kLogQ4_64));

  __m512d correction =
      _mm512_mul_pd(reduced, _mm512_mul_pd(squared, _mm512_div_pd(numerator, denominator)));
  correction = _mm512_sub_pd(correction, _mm512_mul_pd(exponent, _mm512_set1_pd(kLogC1_64)));
  correction = _mm512_sub_pd(correction, _mm512_mul_pd(squared, _mm512_set1_pd(0.5)));
  __m512d result = _mm512_add_pd(reduced, correction);
  result = _mm512_add_pd(result, _mm512_mul_pd(exponent, _mm512_set1_pd(kLogC2_64)));

  result = Select(is_negative, _mm512_set1_pd(std::numeric_limits<double>::quiet_NaN()), result);
  result = Select(is_zero, _mm512_set1_pd(-std::numeric_limits<double>::infinity()), result);
  result = Select(is_infinite, positive_infinity, result);
  return Select(is_nan, _mm512_set1_pd(std::numeric_limits<double>::quiet_NaN()), result);
}

} // namespace

void ExpFloat32_AVX512(const float *input, float *output, std::size_t count) {
  std::size_t i = 0;
  for (; i + 32 <= count; i += 32) {
    _mm512_storeu_ps(output + i, ExpPs(_mm512_loadu_ps(input + i)));
    _mm512_storeu_ps(output + i + 16, ExpPs(_mm512_loadu_ps(input + i + 16)));
  }
  for (; i + 16 <= count; i += 16) {
    _mm512_storeu_ps(output + i, ExpPs(_mm512_loadu_ps(input + i)));
  }
  if (i < count) {
    const __mmask16 tail = static_cast<__mmask16>((1u << (count - i)) - 1u);
    const __m512 result = ExpPs(_mm512_maskz_loadu_ps(tail, input + i));
    _mm512_mask_storeu_ps(output + i, tail, result);
  }
}

void LogFloat32_AVX512(const float *input, float *output, std::size_t count) {
  std::size_t i = 0;
  for (; i + 32 <= count; i += 32) {
    _mm512_storeu_ps(output + i, LogPs(_mm512_loadu_ps(input + i)));
    _mm512_storeu_ps(output + i + 16, LogPs(_mm512_loadu_ps(input + i + 16)));
  }
  for (; i + 16 <= count; i += 16) {
    _mm512_storeu_ps(output + i, LogPs(_mm512_loadu_ps(input + i)));
  }
  if (i < count) {
    const __mmask16 tail = static_cast<__mmask16>((1u << (count - i)) - 1u);
    const __m512 result = LogPs(_mm512_maskz_loadu_ps(tail, input + i));
    _mm512_mask_storeu_ps(output + i, tail, result);
  }
}

void ExpFloat64_AVX512(const double *input, double *output, std::size_t count) {
  std::size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    _mm512_storeu_pd(output + i, ExpPd(_mm512_loadu_pd(input + i)));
  }
  if (i < count) {
    const __mmask8 tail = static_cast<__mmask8>((1u << (count - i)) - 1u);
    const __m512d result = ExpPd(_mm512_maskz_loadu_pd(tail, input + i));
    _mm512_mask_storeu_pd(output + i, tail, result);
  }
}

void LogFloat64_AVX512(const double *input, double *output, std::size_t count) {
  std::size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    _mm512_storeu_pd(output + i, LogPd(_mm512_loadu_pd(input + i)));
  }
  if (i < count) {
    const __mmask8 tail = static_cast<__mmask8>((1u << (count - i)) - 1u);
    const __m512d result = LogPd(_mm512_maskz_loadu_pd(tail, input + i));
    _mm512_mask_storeu_pd(output + i, tail, result);
  }
}

} // namespace onnx_light_cpu
