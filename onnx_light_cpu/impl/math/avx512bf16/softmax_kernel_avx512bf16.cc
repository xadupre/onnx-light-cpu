// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/avx512/exp_kernel_avx512.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <vector>

namespace onnx_light_cpu {
namespace {

__m512 BFloat16ToFloat32(const std::uint16_t *values) {
  const __m256i halves = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(values));
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(halves), 16));
}

void StoreBFloat16(__m512 values, std::uint16_t *output) {
  const __m512i bits = _mm512_castps_si512(values);
  const __m512i magnitude = _mm512_and_si512(bits, _mm512_set1_epi32(0x7fffffff));
  const __mmask16 is_nan =
      _mm512_cmp_epu32_mask(magnitude, _mm512_set1_epi32(0x7f800000), _MM_CMPINT_GT);
  const __m512 quiet_values =
      _mm512_castsi512_ps(_mm512_mask_or_epi32(bits, is_nan, bits, _mm512_set1_epi32(0x00400000)));
  const __m256bh converted = _mm512_cvtneps_pbh(quiet_values);
  std::memcpy(output, &converted, sizeof(converted));
  unsigned subnormal =
      _mm512_cmp_epu32_mask(magnitude, _mm512_setzero_si512(), _MM_CMPINT_NE) &
      _mm512_cmp_epu32_mask(magnitude, _mm512_set1_epi32(0x00800000), _MM_CMPINT_LT);
  if (subnormal != 0) {
    alignas(64) std::uint32_t value_bits[16];
    _mm512_store_si512(value_bits, bits);
    while (subnormal != 0) {
      const unsigned lane = std::countr_zero(subnormal);
      const std::uint32_t lane_bits = value_bits[lane];
      output[lane] =
          static_cast<std::uint16_t>((lane_bits + 0x7fffu + ((lane_bits >> 16) & 1u)) >> 16);
      subnormal &= subnormal - 1;
    }
  }
}

void SoftmaxRow(const std::uint16_t *input, std::uint16_t *output, std::size_t columns,
                float *buffer) {
  const __m512 negative_infinity = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
  __m512 maximum0 = negative_infinity;
  __m512 maximum1 = negative_infinity;
  std::size_t column = 0;
  for (; column + 32 <= columns; column += 32) {
    maximum0 = _mm512_max_ps(maximum0, BFloat16ToFloat32(input + column));
    maximum1 = _mm512_max_ps(maximum1, BFloat16ToFloat32(input + column + 16));
  }
  for (; column + 16 <= columns; column += 16) {
    maximum0 = _mm512_max_ps(maximum0, BFloat16ToFloat32(input + column));
  }
  if (column < columns) {
    alignas(32) std::uint16_t tail[16]{};
    std::memcpy(tail, input + column, (columns - column) * sizeof(std::uint16_t));
    const __mmask16 mask = static_cast<__mmask16>((1u << (columns - column)) - 1u);
    maximum0 = _mm512_max_ps(maximum0,
                             _mm512_mask_mov_ps(negative_infinity, mask, BFloat16ToFloat32(tail)));
  }
  const __m512 maximum = _mm512_set1_ps(_mm512_reduce_max_ps(_mm512_max_ps(maximum0, maximum1)));

  __m512 sum0 = _mm512_setzero_ps();
  __m512 sum1 = _mm512_setzero_ps();
  column = 0;
  for (; column + 32 <= columns; column += 32) {
    const __m512 exponent0 =
        detail::ExpFloat32Vector_AVX512(_mm512_sub_ps(BFloat16ToFloat32(input + column), maximum));
    const __m512 exponent1 = detail::ExpFloat32Vector_AVX512(
        _mm512_sub_ps(BFloat16ToFloat32(input + column + 16), maximum));
    _mm512_storeu_ps(buffer + column, exponent0);
    _mm512_storeu_ps(buffer + column + 16, exponent1);
    sum0 = _mm512_add_ps(sum0, exponent0);
    sum1 = _mm512_add_ps(sum1, exponent1);
  }
  for (; column + 16 <= columns; column += 16) {
    const __m512 exponent =
        detail::ExpFloat32Vector_AVX512(_mm512_sub_ps(BFloat16ToFloat32(input + column), maximum));
    _mm512_storeu_ps(buffer + column, exponent);
    sum0 = _mm512_add_ps(sum0, exponent);
  }
  if (column < columns) {
    alignas(32) std::uint16_t tail[16]{};
    std::memcpy(tail, input + column, (columns - column) * sizeof(std::uint16_t));
    const __mmask16 mask = static_cast<__mmask16>((1u << (columns - column)) - 1u);
    const __m512 exponent = detail::ExpFloat32Vector_AVX512(
        _mm512_sub_ps(_mm512_mask_mov_ps(maximum, mask, BFloat16ToFloat32(tail)), maximum));
    _mm512_mask_storeu_ps(buffer + column, mask, exponent);
    sum0 = _mm512_add_ps(sum0, _mm512_maskz_mov_ps(mask, exponent));
  }
  const __m512 inverse_sum = _mm512_set1_ps(1.0f / _mm512_reduce_add_ps(_mm512_add_ps(sum0, sum1)));
  column = 0;
  for (; column + 16 <= columns; column += 16) {
    StoreBFloat16(_mm512_mul_ps(_mm512_loadu_ps(buffer + column), inverse_sum), output + column);
  }
  if (column < columns) {
    alignas(64) float tail[16]{};
    alignas(32) std::uint16_t output_tail[16];
    std::memcpy(tail, buffer + column, (columns - column) * sizeof(float));
    StoreBFloat16(_mm512_mul_ps(_mm512_load_ps(tail), inverse_sum), output_tail);
    std::memcpy(output + column, output_tail, (columns - column) * sizeof(std::uint16_t));
  }
}

} // namespace

void SoftmaxBFloat16_AVX512BF16(const std::uint16_t *input, std::uint16_t *output, std::size_t rows,
                                std::size_t columns) {
  std::vector<float> buffer(columns);
  for (std::size_t row = 0; row < rows; ++row) {
    SoftmaxRow(input + row * columns, output + row * columns, columns, buffer.data());
  }
}

} // namespace onnx_light_cpu
