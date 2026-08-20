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
  if (i < count) {
    for (; i < count; ++i) {
      output[i] = std::exp(input[i]);
    }
  }
}

} // namespace onnx_light_cpu
