// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstddef>
#include <immintrin.h>
#include <iterator>
#include <limits>

namespace onnx_light_cpu {
namespace {

__m256 GeluFloat32(__m256 z) {
  constexpr float coefficients[] = {
      1.8827368f,       1.32283127f,     -0.29206121f,   0.131107315f,    -0.0683164895f,
      0.0358457267f,    -0.0179936364f,  0.00847151037f, -0.00371518102f, 0.0015147439f,
      -0.000575297687f, 0.000203221469f, -6.7434863e-5f, 2.01439125e-5f,  -5.92767856e-6f,
  };
  const __m256 x =
      _mm256_fmadd_ps(z, _mm256_mul_ps(z, _mm256_set1_ps(1.0f / 18.0f)), _mm256_set1_ps(-1.0f));
  __m256 next_vector = _mm256_setzero_ps();
  __m256 next_next = _mm256_setzero_ps();
  for (std::size_t index = std::size(coefficients) - 1; index > 0; --index) {
    const __m256 current =
        _mm256_add_ps(_mm256_fmsub_ps(_mm256_add_ps(x, x), next_vector, next_next),
                      _mm256_set1_ps(coefficients[index]));
    next_next = next_vector;
    next_vector = current;
  }
  __m256 result = _mm256_add_ps(
      _mm256_mul_ps(_mm256_set1_ps(0.5f), z),
      _mm256_add_ps(_mm256_fmsub_ps(x, next_vector, next_next), _mm256_set1_ps(coefficients[0])));

  const __m256 positive = _mm256_cmp_ps(z, _mm256_set1_ps(6.0f), _CMP_GE_OQ);
  const __m256 negative = _mm256_cmp_ps(z, _mm256_set1_ps(-6.0f), _CMP_LE_OQ);
  result = _mm256_blendv_ps(result, z, positive);
  result = _mm256_blendv_ps(result, _mm256_setzero_ps(), negative);
  const __m256 is_zero = _mm256_cmp_ps(z, _mm256_setzero_ps(), _CMP_EQ_OQ);
  result = _mm256_blendv_ps(result, z, is_zero);
  const __m256 negative_infinity =
      _mm256_cmp_ps(z, _mm256_set1_ps(-std::numeric_limits<float>::infinity()), _CMP_EQ_OQ);
  return _mm256_blendv_ps(result, _mm256_set1_ps(std::numeric_limits<float>::quiet_NaN()),
                          negative_infinity);
}

float GeluFloat32Scalar(float z) {
  constexpr float kInvSqrtTwo = 0.7071067811865475244f;
  return 0.5f * z * (1.0f + std::erf(z * kInvSqrtTwo));
}

} // namespace

void BiasGeluFloat32_AVX2_FMA(const float *a, const float *bias, float *output, std::size_t count) {
  std::size_t index = 0;
  for (; index + 8 <= count; index += 8) {
    const __m256 z = _mm256_add_ps(_mm256_loadu_ps(a + index), _mm256_loadu_ps(bias + index));
    _mm256_storeu_ps(output + index, GeluFloat32(z));
  }
  for (; index < count; ++index) {
    output[index] = GeluFloat32Scalar(a[index] + bias[index]);
  }
}

} // namespace onnx_light_cpu
