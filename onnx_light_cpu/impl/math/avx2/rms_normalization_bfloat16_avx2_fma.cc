// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/rms_normalization.h"

#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <immintrin.h>

#include <cmath>

namespace onnx_light_cpu {
namespace {

__m256 LoadBFloat16(const std::uint16_t *source) {
  const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i *>(source));
  return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(packed), 16));
}

__m256i FloatToBFloat16(__m256 value) {
  const __m256i bits = _mm256_castps_si256(value);
  const __m256i upper = _mm256_srli_epi32(bits, 16);
  const __m256i rounding =
      _mm256_add_epi32(_mm256_set1_epi32(0x7fff), _mm256_and_si256(upper, _mm256_set1_epi32(1)));
  const __m256i rounded = _mm256_srli_epi32(_mm256_add_epi32(bits, rounding), 16);
  const __m256i is_nan = _mm256_cmpgt_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff)),
                                            _mm256_set1_epi32(0x7f800000));
  const __m256i canonical_nan = _mm256_or_si256(upper, _mm256_set1_epi32(0x40));
  return _mm256_blendv_epi8(rounded, canonical_nan, is_nan);
}

void StoreBFloat16(__m256i values, std::uint16_t *destination) {
  const __m128i packed =
      _mm_packus_epi32(_mm256_castsi256_si128(values), _mm256_extracti128_si256(values, 1));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(destination), packed);
}

float HorizontalSum(__m256 value) {
  __m128 sum = _mm_add_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1));
  sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
  sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
  return _mm_cvtss_f32(sum);
}

} // namespace

void RmsNormalizationBFloat16_AVX2(const std::uint16_t *input, const std::uint16_t *scale,
                                   std::uint16_t *output, std::size_t row_begin,
                                   std::size_t row_end, std::size_t width, float epsilon) {
  for (std::size_t row = row_begin; row < row_end; ++row) {
    const std::size_t offset = row * width;
    __m256 sum_squares = _mm256_setzero_ps();
    std::size_t column = 0;
    for (; column + 8 <= width; column += 8) {
      const __m256 value = LoadBFloat16(input + offset + column);
      sum_squares = _mm256_fmadd_ps(value, value, sum_squares);
    }
    float sum = HorizontalSum(sum_squares);
    for (; column < width; ++column) {
      const float value = detail::Bfloat16BitsToFloat(input[offset + column]);
      sum += value * value;
    }

    const __m256 inverse_rms =
        _mm256_set1_ps(1.0F / std::sqrt(sum / static_cast<float>(width) + epsilon));
    column = 0;
    for (; column + 8 <= width; column += 8) {
      const __m256 value = LoadBFloat16(input + offset + column);
      const __m256 normalized = _mm256_mul_ps(value, inverse_rms);
      const __m256 rounded_normalized =
          _mm256_castsi256_ps(_mm256_slli_epi32(FloatToBFloat16(normalized), 16));
      const __m256 scaled = _mm256_mul_ps(rounded_normalized, LoadBFloat16(scale + column));
      StoreBFloat16(FloatToBFloat16(scaled), output + offset + column);
    }
    const float scalar_inverse_rms = _mm_cvtss_f32(_mm256_castps256_ps128(inverse_rms));
    for (; column < width; ++column) {
      const float value = detail::Bfloat16BitsToFloat(input[offset + column]);
      const float weight = detail::Bfloat16BitsToFloat(scale[column]);
      const float normalized =
          detail::Bfloat16BitsToFloat(detail::FloatToBFloat16Bits(value * scalar_inverse_rms));
      output[offset + column] = detail::FloatToBFloat16Bits(normalized * weight);
    }
  }
}

} // namespace onnx_light_cpu
