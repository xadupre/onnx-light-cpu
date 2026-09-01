// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/rms_normalization.h"

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

void RmsNormalizationFloat32_AVX2(const float *input, const float *scale, float *output,
                                  std::size_t row_begin, std::size_t row_end, std::size_t width,
                                  float epsilon) {
  for (std::size_t row = row_begin; row < row_end; ++row) {
    const std::size_t offset = row * width;
    __m256 sums[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                      _mm256_setzero_ps()};
    std::size_t column = 0;
    for (; column + 32 <= width; column += 32) {
      for (std::size_t lane = 0; lane < 4; ++lane) {
        const __m256 value = _mm256_loadu_ps(input + offset + column + lane * 8);
        sums[lane] = _mm256_fmadd_ps(value, value, sums[lane]);
      }
    }
    for (; column + 8 <= width; column += 8) {
      const __m256 value = _mm256_loadu_ps(input + offset + column);
      sums[0] = _mm256_fmadd_ps(value, value, sums[0]);
    }
    const __m256 sum_squares =
        _mm256_add_ps(_mm256_add_ps(sums[0], sums[1]), _mm256_add_ps(sums[2], sums[3]));
    float sum = HorizontalSum(sum_squares);
    for (; column < width; ++column) {
      const float value = input[offset + column];
      sum += value * value;
    }

    const __m256 inverse_rms =
        _mm256_set1_ps(1.0F / std::sqrt(sum / static_cast<float>(width) + epsilon));
    column = 0;
    for (; column + 8 <= width; column += 8) {
      const __m256 value = _mm256_loadu_ps(input + offset + column);
      const __m256 weight = _mm256_loadu_ps(scale + column);
      _mm256_storeu_ps(output + offset + column,
                       _mm256_mul_ps(_mm256_mul_ps(value, inverse_rms), weight));
    }
    const float scalar_inverse_rms = _mm_cvtss_f32(_mm256_castps256_ps128(inverse_rms));
    for (; column < width; ++column) {
      output[offset + column] = input[offset + column] * scalar_inverse_rms * scale[column];
    }
  }
}

} // namespace onnx_light_cpu
