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

constexpr float kExpHi = 88.7762626647950f;
constexpr float kExpLo = -103.9720840454f;
constexpr float kLog2ef = 1.44269504088896341f;
constexpr float kExpLog2High = -6.93145752e-1f;
constexpr float kExpLog2Low = -1.42860677e-6f;
constexpr float kExpP0 = 0x1.694000p-10f;
constexpr float kExpP1 = 0x1.125edcp-7f;
constexpr float kExpP2 = 0x1.555b5ap-5f;
constexpr float kExpP3 = 0x1.555450p-3f;
constexpr float kExpP4 = 0x1.fffff6p-2f;

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
  // Keeping the rounded exponent in its biased floating-point representation
  // avoids a float-to-integer conversion on the polynomial dependency chain.
  __m256 reduced = _mm256_min_ps(_mm256_set1_ps(kExpHi), _mm256_max_ps(_mm256_set1_ps(kExpLo), x));
  const __m256 magic = _mm256_set1_ps(12582912.0f);
  const __m256 biased = _mm256_fmadd_ps(reduced, _mm256_set1_ps(kLog2ef), magic);
  const __m256 exponent = _mm256_sub_ps(biased, magic);
  reduced = _mm256_fmadd_ps(exponent, _mm256_set1_ps(kExpLog2High), reduced);
  reduced = _mm256_fmadd_ps(exponent, _mm256_set1_ps(kExpLog2Low), reduced);

  __m256i exponent_bits = _mm256_slli_epi32(_mm256_castps_si256(biased), 23);
  __m256i normal_bits =
      _mm256_min_epi32(exponent_bits, _mm256_set1_epi32(static_cast<int>(0x3f800000u)));
  normal_bits = _mm256_max_epi32(normal_bits, _mm256_set1_epi32(static_cast<int>(0xc1000000u)));
  __m256i overflow_bits = _mm256_sub_epi32(exponent_bits, normal_bits);
  normal_bits = _mm256_add_epi32(normal_bits, _mm256_set1_epi32(0x3f800000));
  overflow_bits = _mm256_add_epi32(overflow_bits, _mm256_set1_epi32(0x3f800000));
  const __m256 normal = _mm256_castsi256_ps(normal_bits);
  const __m256 overflow = _mm256_castsi256_ps(overflow_bits);

  __m256 polynomial = _mm256_set1_ps(kExpP0);
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(kExpP1));
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(kExpP2));
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(kExpP3));
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(kExpP4));
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(1.0f));
  const __m256 scaled_reduced = _mm256_mul_ps(reduced, overflow);
  polynomial = _mm256_fmadd_ps(polynomial, scaled_reduced, overflow);
  return _mm256_mul_ps(polynomial, normal);
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
  for (; i + 8 <= count; i += 8) {
    _mm256_storeu_ps(output + i, ExpPs256Fma(_mm256_loadu_ps(input + i)));
  }
  if (i < count) {
    const __m256i lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i tail = _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(count - i)), lanes);
    const __m256 result = ExpPs256Fma(_mm256_maskload_ps(input + i, tail));
    _mm256_maskstore_ps(output + i, tail, result);
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
