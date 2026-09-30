// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <iterator>
#include <limits>

namespace onnx_light_cpu {
namespace {

__m512 BFloat16ToFloat32(const std::uint16_t *values) {
  const __m256i halves = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(values));
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(halves), 16));
}

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

void StoreBFloat16(__m512 values, std::uint16_t *output) {
  const __m512i bits = _mm512_castps_si512(values);
  const __mmask16 is_nan =
      _mm512_cmp_epu32_mask(_mm512_and_si512(bits, _mm512_set1_epi32(0x7fffffff)),
                            _mm512_set1_epi32(0x7f800000), _MM_CMPINT_GT);
  const __m512 quiet_values =
      _mm512_castsi512_ps(_mm512_mask_or_epi32(bits, is_nan, bits, _mm512_set1_epi32(0x00400000)));
  const __m256bh converted = _mm512_cvtneps_pbh(quiet_values);
  std::memcpy(output, &converted, sizeof(converted));
}

} // namespace

void BiasGeluBFloat16_AVX512BF16(const std::uint16_t *a, const std::uint16_t *bias,
                                 std::uint16_t *output, std::size_t count) {
  std::size_t index = 0;
  for (; index + 16 <= count; index += 16) {
    const __m512 z = _mm512_add_ps(BFloat16ToFloat32(a + index), BFloat16ToFloat32(bias + index));
    StoreBFloat16(GeluFloat32(z), output + index);
  }
  if (index < count) {
    alignas(32) std::uint16_t a_tail[16]{};
    alignas(32) std::uint16_t bias_tail[16]{};
    alignas(32) std::uint16_t output_tail[16];
    const std::size_t tail_bytes = (count - index) * sizeof(std::uint16_t);
    std::memcpy(a_tail, a + index, tail_bytes);
    std::memcpy(bias_tail, bias + index, tail_bytes);
    const __m512 z = _mm512_add_ps(BFloat16ToFloat32(a_tail), BFloat16ToFloat32(bias_tail));
    StoreBFloat16(GeluFloat32(z), output_tail);
    std::memcpy(output + index, output_tail, tail_bytes);
  }
}

} // namespace onnx_light_cpu
