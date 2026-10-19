// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <iterator>
#include <limits>

namespace onnx_light_cpu {
namespace {

__m256 GeluFloat32(__m256 z) {
  const __m256 x =
      _mm256_fmadd_ps(z, _mm256_mul_ps(z, _mm256_set1_ps(1.0f / 18.0f)), _mm256_set1_ps(-1.0f));
  const __m256 x2 = _mm256_mul_ps(x, x);

  // The same degree-14 Chebyshev approximation as the scalar path, converted
  // to the power basis. Two independent Horner chains halve the dependency
  // depth while preserving the approximation and its special-value handling.
  __m256 even = _mm256_set1_ps(-0.0485595427635f);
  __m256 odd = _mm256_set1_ps(0.0825094656f);
  even = _mm256_fmadd_ps(even, x2, _mm256_set1_ps(0.0318518002483f));
  odd = _mm256_fmadd_ps(odd, x2, _mm256_set1_ps(-0.060056978944f));
  even = _mm256_fmadd_ps(even, x2, _mm256_set1_ps(-0.113925417021f));
  odd = _mm256_fmadd_ps(odd, x2, _mm256_set1_ps(0.150697485696f));
  even = _mm256_fmadd_ps(even, x2, _mm256_set1_ps(-0.0459359045632f));
  odd = _mm256_fmadd_ps(odd, x2, _mm256_set1_ps(0.040839011584f));
  even = _mm256_fmadd_ps(even, x2, _mm256_set1_ps(-0.0831244840205f));
  odd = _mm256_fmadd_ps(odd, x2, _mm256_set1_ps(0.087382053952f));
  even = _mm256_fmadd_ps(even, x2, _mm256_set1_ps(-0.0944979421379f));
  odd = _mm256_fmadd_ps(odd, x2, _mm256_set1_ps(0.13752637775f));
  even = _mm256_fmadd_ps(even, x2, _mm256_set1_ps(-0.267080653273f));
  odd = _mm256_fmadd_ps(odd, x2, _mm256_set1_ps(1.06109651571f));
  even = _mm256_fmadd_ps(even, x2, _mm256_set1_ps(2.12127376638f));
  __m256 result = _mm256_fmadd_ps(x, odd, _mm256_fmadd_ps(_mm256_set1_ps(0.5f), z, even));

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

} // namespace

void BiasGeluFloat32_AVX2_FMA(const float *a, const float *bias, float *output, std::size_t count) {
  std::size_t index = 0;
  for (; index + 8 <= count; index += 8) {
    const __m256 z = _mm256_add_ps(_mm256_loadu_ps(a + index), _mm256_loadu_ps(bias + index));
    _mm256_storeu_ps(output + index, GeluFloat32(z));
  }
  if (index < count) {
    alignas(32) static constexpr std::int32_t masks[8][8] = {
        {-1, 0, 0, 0, 0, 0, 0, 0},       {-1, -1, 0, 0, 0, 0, 0, 0},
        {-1, -1, -1, 0, 0, 0, 0, 0},     {-1, -1, -1, -1, 0, 0, 0, 0},
        {-1, -1, -1, -1, -1, 0, 0, 0},   {-1, -1, -1, -1, -1, -1, 0, 0},
        {-1, -1, -1, -1, -1, -1, -1, 0}, {-1, -1, -1, -1, -1, -1, -1, -1},
    };
    const __m256i mask =
        _mm256_load_si256(reinterpret_cast<const __m256i *>(masks[count - index - 1]));
    const __m256 z =
        _mm256_add_ps(_mm256_maskload_ps(a + index, mask), _mm256_maskload_ps(bias + index, mask));
    _mm256_maskstore_ps(output + index, mask, GeluFloat32(z));
  }
}

} // namespace onnx_light_cpu
