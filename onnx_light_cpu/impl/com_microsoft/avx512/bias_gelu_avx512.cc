// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <immintrin.h>
#include <iterator>
#include <limits>

namespace onnx_light_cpu {
namespace {

__m512 GeluFloat32(__m512 z) {
  constexpr float coefficients[] = {
      1.8827368f,       1.32283127f,     -0.29206121f,   0.131107315f,    -0.0683164895f,
      0.0358457267f,    -0.0179936364f,  0.00847151037f, -0.00371518102f, 0.0015147439f,
      -0.000575297687f, 0.000203221469f, -6.7434863e-5f, 2.01439125e-5f,  -5.92767856e-6f,
  };
  const __m512 x =
      _mm512_fmadd_ps(z, _mm512_mul_ps(z, _mm512_set1_ps(1.0f / 18.0f)), _mm512_set1_ps(-1.0f));
  __m512 next = _mm512_setzero_ps();
  __m512 next_next = _mm512_setzero_ps();
  for (std::size_t index = std::size(coefficients) - 1; index > 0; --index) {
    const __m512 current = _mm512_add_ps(_mm512_fmsub_ps(_mm512_add_ps(x, x), next, next_next),
                                         _mm512_set1_ps(coefficients[index]));
    next_next = next;
    next = current;
  }
  __m512 result = _mm512_add_ps(
      _mm512_mul_ps(_mm512_set1_ps(0.5f), z),
      _mm512_add_ps(_mm512_fmsub_ps(x, next, next_next), _mm512_set1_ps(coefficients[0])));

  result = _mm512_mask_mov_ps(result, _mm512_cmp_ps_mask(z, _mm512_set1_ps(6.0f), _CMP_GE_OQ), z);
  result = _mm512_mask_mov_ps(result, _mm512_cmp_ps_mask(z, _mm512_set1_ps(-6.0f), _CMP_LE_OQ),
                              _mm512_setzero_ps());
  result = _mm512_mask_mov_ps(result, _mm512_cmp_ps_mask(z, _mm512_setzero_ps(), _CMP_EQ_OQ), z);
  return _mm512_mask_mov_ps(
      result,
      _mm512_cmp_ps_mask(z, _mm512_set1_ps(-std::numeric_limits<float>::infinity()), _CMP_EQ_OQ),
      _mm512_set1_ps(std::numeric_limits<float>::quiet_NaN()));
}

} // namespace

void BiasGeluFloat32_AVX512(const float *a, const float *bias, float *output, std::size_t count) {
  std::size_t index = 0;
  for (; index + 16 <= count; index += 16) {
    const __m512 z = _mm512_add_ps(_mm512_loadu_ps(a + index), _mm512_loadu_ps(bias + index));
    _mm512_storeu_ps(output + index, GeluFloat32(z));
  }
  if (index < count) {
    const __mmask16 mask = static_cast<__mmask16>((1u << (count - index)) - 1u);
    const __m512 z = _mm512_add_ps(_mm512_maskz_loadu_ps(mask, a + index),
                                   _mm512_maskz_loadu_ps(mask, bias + index));
    _mm512_mask_storeu_ps(output + index, mask, GeluFloat32(z));
  }
}

} // namespace onnx_light_cpu
