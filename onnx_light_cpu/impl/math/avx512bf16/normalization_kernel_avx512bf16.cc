// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/normalization_kernel.h"

#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <bit>
#include <cstring>
#include <immintrin.h>

namespace onnx_light_cpu {
namespace {

__m512 LoadBFloat16(const std::uint16_t *input) {
  const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input));
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(packed), 16));
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

} // namespace

float ComputeNormalizationMeanSquareBFloat16_AVX512BF16(const std::uint16_t *input,
                                                        std::size_t count) {
  __m512 sums[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                    _mm512_setzero_ps()};
  std::size_t index = 0;
  for (; index + 64 <= count; index += 64) {
    for (std::size_t lane = 0; lane < 4; ++lane) {
      const __m512 value = LoadBFloat16(input + index + lane * 16);
      sums[lane] = _mm512_fmadd_ps(value, value, sums[lane]);
    }
  }
  for (; index + 16 <= count; index += 16) {
    const __m512 value = LoadBFloat16(input + index);
    sums[0] = _mm512_fmadd_ps(value, value, sums[0]);
  }
  float sum = _mm512_reduce_add_ps(
      _mm512_add_ps(_mm512_add_ps(sums[0], sums[1]), _mm512_add_ps(sums[2], sums[3])));
  for (; index < count; ++index) {
    const float value = detail::Bfloat16BitsToFloat(input[index]);
    sum += value * value;
  }
  return sum / static_cast<float>(count);
}

void ApplyNormalizationAffineBFloat16_AVX512BF16(const std::uint16_t *input,
                                                 const std::uint16_t *scale, std::uint16_t *output,
                                                 std::size_t count, float multiplier) {
  const __m512 multiplier16 = _mm512_set1_ps(multiplier);
  std::size_t index = 0;
  for (; index + 16 <= count; index += 16) {
    const __m512 value = LoadBFloat16(input + index);
    const __m512 weight = LoadBFloat16(scale + index);
    StoreBFloat16(_mm512_mul_ps(_mm512_mul_ps(value, multiplier16), weight), output + index);
  }
  for (; index < count; ++index) {
    const float value = detail::Bfloat16BitsToFloat(input[index]);
    const float weight = detail::Bfloat16BitsToFloat(scale[index]);
    output[index] = detail::FloatToBFloat16Bits(value * multiplier * weight);
  }
}

} // namespace onnx_light_cpu
