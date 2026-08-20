// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/math_kernels.h"

#include <immintrin.h>

#include <cmath>
#include <cstddef>
#include <limits>

namespace onnx_light_cpu {
namespace {

constexpr float kExpHi = 88.3762626647949f;
constexpr float kExpLo = -103.97208f;
constexpr float kLog2ef = 1.44269504088896341f;
constexpr float kExpC1 = 0.693359375f;
constexpr float kExpC2 = -2.12194440e-4f;
constexpr float kExpP0 = 1.9875691500e-4f;
constexpr float kExpP1 = 1.3981999507e-3f;
constexpr float kExpP2 = 8.3334519073e-3f;
constexpr float kExpP3 = 4.1665795894e-2f;
constexpr float kExpP4 = 1.6666665459e-1f;
constexpr float kExpP5 = 5.0000001201e-1f;

constexpr float kSmallestNormal = 1.17549435e-38f;
constexpr float kSubnormalScale = 8388608.0f;
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

inline __m256 Select(__m256 mask, __m256 selected, __m256 fallback) {
  return _mm256_blendv_ps(fallback, selected, mask);
}

__m256 ExpPs256Fma(__m256 x) {
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 hi = _mm256_set1_ps(kExpHi);
  const __m256 lo = _mm256_set1_ps(kExpLo);
  const __m256 is_nan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
  const __m256 over = _mm256_cmp_ps(x, hi, _CMP_GT_OQ);
  const __m256 under = _mm256_cmp_ps(x, lo, _CMP_LT_OQ);

  __m256 reduced = _mm256_min_ps(_mm256_max_ps(x, lo), hi);

  // 2^23 is exactly representable, so adding this magic bias rounds the
  // reduced exponent to an integer without a multiply/add dependency.
  const __m256 magic = _mm256_set1_ps(12582912.0f);
  const __m256 scaled = _mm256_fmadd_ps(reduced, _mm256_set1_ps(kLog2ef), magic);
  __m256i exponent_int =
      _mm256_sub_epi32(_mm256_castps_si256(scaled), _mm256_set1_epi32(0x4b400000));
  const __m256 exponent = _mm256_cvtepi32_ps(exponent_int);

  reduced = _mm256_fnmadd_ps(exponent, _mm256_set1_ps(kExpC1), reduced);
  reduced = _mm256_fnmadd_ps(exponent, _mm256_set1_ps(kExpC2), reduced);
  const __m256 squared = _mm256_mul_ps(reduced, reduced);

  __m256 polynomial = _mm256_set1_ps(kExpP0);
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(kExpP1));
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(kExpP2));
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(kExpP3));
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(kExpP4));
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(kExpP5));
  polynomial = _mm256_fmadd_ps(polynomial, squared, reduced);
  polynomial = _mm256_add_ps(polynomial, one);

  // Reconstruct 2^n from two normal factors, preserving valid subnormals.
  const __m256i exponent_n1 = _mm256_srai_epi32(exponent_int, 1);
  const __m256i exponent_n2 = _mm256_sub_epi32(exponent_int, exponent_n1);
  const __m256i pow1 =
      _mm256_slli_epi32(_mm256_add_epi32(exponent_n1, _mm256_set1_epi32(0x7f)), 23);
  const __m256i pow2 =
      _mm256_slli_epi32(_mm256_add_epi32(exponent_n2, _mm256_set1_epi32(0x7f)), 23);
  __m256 result = _mm256_mul_ps(polynomial, _mm256_castsi256_ps(pow1));
  result = _mm256_mul_ps(result, _mm256_castsi256_ps(pow2));

  result = Select(under, _mm256_setzero_ps(), result);
  result = Select(over, _mm256_set1_ps(std::numeric_limits<float>::infinity()), result);
  return Select(is_nan, _mm256_set1_ps(std::numeric_limits<float>::quiet_NaN()), result);
}

#if defined(_MSC_VER)
#define ONNX_LIGHT_CPU_FORCE_INLINE __forceinline
#else
#define ONNX_LIGHT_CPU_FORCE_INLINE inline __attribute__((always_inline))
#endif

ONNX_LIGHT_CPU_FORCE_INLINE __m256 LogPs256Fma(__m256 x) {
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 zero = _mm256_setzero_ps();
  const __m256 positive_infinity = _mm256_set1_ps(std::numeric_limits<float>::infinity());
  const __m256 is_nan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
  const __m256 is_negative = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);
  const __m256 is_zero = _mm256_cmp_ps(x, zero, _CMP_EQ_OQ);
  const __m256 is_infinite = _mm256_cmp_ps(x, positive_infinity, _CMP_EQ_OQ);

  const __m256 is_subnormal =
      _mm256_and_ps(_mm256_cmp_ps(x, zero, _CMP_GT_OQ),
                    _mm256_cmp_ps(x, _mm256_set1_ps(kSmallestNormal), _CMP_LT_OQ));
  const __m256 scaled = _mm256_mul_ps(x, _mm256_set1_ps(kSubnormalScale));
  __m256 reduced = Select(is_subnormal, scaled, x);
  __m256i exponent_int = _mm256_srli_epi32(_mm256_castps_si256(reduced), 23);
  reduced = _mm256_and_ps(reduced,
                          _mm256_castsi256_ps(_mm256_set1_epi32(static_cast<int>(~0x7f800000u))));
  reduced = _mm256_or_ps(reduced, _mm256_set1_ps(0.5f));
  exponent_int = _mm256_sub_epi32(exponent_int, _mm256_set1_epi32(0x7f));
  __m256 exponent = _mm256_add_ps(_mm256_cvtepi32_ps(exponent_int), one);
  exponent = _mm256_sub_ps(exponent, _mm256_and_ps(is_subnormal, _mm256_set1_ps(23.0f)));

  const __m256 normalize = _mm256_cmp_ps(reduced, _mm256_set1_ps(kSqrtHalf), _CMP_LT_OQ);
  const __m256 normalized_part = _mm256_and_ps(reduced, normalize);
  reduced = _mm256_sub_ps(reduced, one);
  exponent = _mm256_sub_ps(exponent, _mm256_and_ps(one, normalize));
  reduced = _mm256_add_ps(reduced, normalized_part);
  const __m256 squared = _mm256_mul_ps(reduced, reduced);

  __m256 polynomial = _mm256_set1_ps(kLogP0);
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(kLogP1));
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(kLogP2));
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(kLogP3));
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(kLogP4));
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(kLogP5));
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(kLogP6));
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(kLogP7));
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(kLogP8));
  polynomial = _mm256_mul_ps(polynomial, reduced);
  polynomial = _mm256_mul_ps(polynomial, squared);
  polynomial = _mm256_fmadd_ps(exponent, _mm256_set1_ps(kLogQ1), polynomial);
  polynomial = _mm256_fnmadd_ps(squared, _mm256_set1_ps(0.5f), polynomial);
  __m256 result = _mm256_add_ps(reduced, polynomial);
  result = _mm256_fmadd_ps(exponent, _mm256_set1_ps(kLogQ2), result);

  result = Select(is_negative, _mm256_set1_ps(std::numeric_limits<float>::quiet_NaN()), result);
  result = Select(is_zero, _mm256_set1_ps(-std::numeric_limits<float>::infinity()), result);
  result = Select(is_infinite, positive_infinity, result);
  return Select(is_nan, _mm256_set1_ps(std::numeric_limits<float>::quiet_NaN()), result);
}

#undef ONNX_LIGHT_CPU_FORCE_INLINE

} // namespace

void ExpFloat32_AVX2_FMA(const float *input, float *output, std::size_t count) {
  std::size_t i = 0;
  for (; i + 16 <= count; i += 16) {
    _mm256_storeu_ps(output + i, ExpPs256Fma(_mm256_loadu_ps(input + i)));
    _mm256_storeu_ps(output + i + 8, ExpPs256Fma(_mm256_loadu_ps(input + i + 8)));
  }
  for (; i + 8 <= count; i += 8) {
    _mm256_storeu_ps(output + i, ExpPs256Fma(_mm256_loadu_ps(input + i)));
  }
  for (; i < count; ++i) {
    output[i] = std::exp(input[i]);
  }
}

void LogFloat32_AVX2_FMA(const float *input, float *output, std::size_t count) {
  std::size_t i = 0;
  for (; i + 16 <= count; i += 16) {
    _mm256_storeu_ps(output + i, LogPs256Fma(_mm256_loadu_ps(input + i)));
    _mm256_storeu_ps(output + i + 8, LogPs256Fma(_mm256_loadu_ps(input + i + 8)));
  }
  for (; i + 8 <= count; i += 8) {
    _mm256_storeu_ps(output + i, LogPs256Fma(_mm256_loadu_ps(input + i)));
  }
  if (i < count) {
    const __m256i lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i tail = _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(count - i)), lanes);
    const __m256 result = LogPs256Fma(_mm256_maskload_ps(input + i, tail));
    _mm256_maskstore_ps(output + i, tail, result);
  }
}

} // namespace onnx_light_cpu
