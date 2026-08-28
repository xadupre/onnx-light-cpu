// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/rms_normalization.h"

#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <immintrin.h>

#include <cmath>

namespace onnx_light_cpu {
namespace {

float HorizontalSum(__m256 value) {
  __m128 sum = _mm_add_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1));
  sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
  sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
  return _mm_cvtss_f32(sum);
}

} // namespace

void RmsNormalizationFloat16_F16C(const std::uint16_t *input, const std::uint16_t *scale,
                                  std::uint16_t *output, std::size_t row_begin, std::size_t row_end,
                                  std::size_t width, float epsilon) {
  for (std::size_t row = row_begin; row < row_end; ++row) {
    const std::size_t offset = row * width;
    __m256 sum_squares8 = _mm256_setzero_ps();
    std::size_t column = 0;
    for (; column + 8 <= width; column += 8) {
      const __m128i packed =
          _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + offset + column));
      const __m256 value = _mm256_cvtph_ps(packed);
      sum_squares8 = _mm256_add_ps(sum_squares8, _mm256_mul_ps(value, value));
    }
    float sum_squares = HorizontalSum(sum_squares8);
    for (; column < width; ++column) {
      const float value = detail::Float16BitsToFloat(input[offset + column]);
      sum_squares += value * value;
    }

    const __m256 inverse_rms =
        _mm256_set1_ps(1.0F / std::sqrt(sum_squares / static_cast<float>(width) + epsilon));
    column = 0;
    for (; column + 8 <= width; column += 8) {
      const __m128i packed_input =
          _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + offset + column));
      const __m128i packed_scale =
          _mm_loadu_si128(reinterpret_cast<const __m128i *>(scale + column));
      const __m256 value = _mm256_cvtph_ps(packed_input);
      const __m256 weight = _mm256_cvtph_ps(packed_scale);
      const __m256 normalized = _mm256_mul_ps(value, inverse_rms);
      const __m128i packed_normalized =
          _mm256_cvtps_ph(normalized, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
      const __m256 rounded_normalized = _mm256_cvtph_ps(packed_normalized);
      const __m256 scaled = _mm256_mul_ps(rounded_normalized, weight);
      const __m128i packed_output =
          _mm256_cvtps_ph(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(output + offset + column), packed_output);
      const int nan_mask = _mm256_movemask_ps(_mm256_cmp_ps(scaled, scaled, _CMP_UNORD_Q));
      if (nan_mask != 0) {
        alignas(32) float lanes[8];
        _mm256_store_ps(lanes, scaled);
        for (int lane = 0; lane < 8; ++lane) {
          if ((nan_mask & (1 << lane)) != 0) {
            output[offset + column + static_cast<std::size_t>(lane)] =
                detail::FloatToFloat16Bits(lanes[lane]);
          }
        }
      }
    }
    const float scalar_inverse_rms = _mm_cvtss_f32(_mm256_castps256_ps128(inverse_rms));
    for (; column < width; ++column) {
      const float value = detail::Float16BitsToFloat(input[offset + column]);
      const float weight = detail::Float16BitsToFloat(scale[column]);
      const float normalized =
          detail::Float16BitsToFloat(detail::FloatToFloat16Bits(value * scalar_inverse_rms));
      output[offset + column] = detail::FloatToFloat16Bits(normalized * weight);
    }
  }
}

} // namespace onnx_light_cpu
