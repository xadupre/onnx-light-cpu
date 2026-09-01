// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/normalization_kernel.h"

#include <immintrin.h>

namespace onnx_light_cpu {
namespace {

float HorizontalSum(__m256 value) {
  __m128 sum = _mm_add_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1));
  sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
  sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
  return _mm_cvtss_f32(sum);
}

} // namespace

float ComputeNormalizationMeanSquareFloat32_AVX2(const float *input, std::size_t count) {
  __m256 sums[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                    _mm256_setzero_ps()};
  std::size_t index = 0;
  for (; index + 32 <= count; index += 32) {
    for (std::size_t lane = 0; lane < 4; ++lane) {
      const __m256 value = _mm256_loadu_ps(input + index + lane * 8);
      sums[lane] = _mm256_fmadd_ps(value, value, sums[lane]);
    }
  }
  for (; index + 8 <= count; index += 8) {
    const __m256 value = _mm256_loadu_ps(input + index);
    sums[0] = _mm256_fmadd_ps(value, value, sums[0]);
  }
  float sum = HorizontalSum(
      _mm256_add_ps(_mm256_add_ps(sums[0], sums[1]), _mm256_add_ps(sums[2], sums[3])));
  for (; index < count; ++index) {
    sum += input[index] * input[index];
  }
  return sum / static_cast<float>(count);
}

Float32NormalizationMoments ComputeNormalizationMomentsFloat32_AVX2(const float *input,
                                                                    std::size_t count) {
  __m256 sums[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                    _mm256_setzero_ps()};
  __m256 square_sums[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                           _mm256_setzero_ps()};
  std::size_t index = 0;
  for (; index + 32 <= count; index += 32) {
    for (std::size_t lane = 0; lane < 4; ++lane) {
      const __m256 value = _mm256_loadu_ps(input + index + lane * 8);
      sums[lane] = _mm256_add_ps(sums[lane], value);
      square_sums[lane] = _mm256_fmadd_ps(value, value, square_sums[lane]);
    }
  }
  for (; index + 8 <= count; index += 8) {
    const __m256 value = _mm256_loadu_ps(input + index);
    sums[0] = _mm256_add_ps(sums[0], value);
    square_sums[0] = _mm256_fmadd_ps(value, value, square_sums[0]);
  }
  float sum = HorizontalSum(
      _mm256_add_ps(_mm256_add_ps(sums[0], sums[1]), _mm256_add_ps(sums[2], sums[3])));
  float sum_squares = HorizontalSum(_mm256_add_ps(_mm256_add_ps(square_sums[0], square_sums[1]),
                                                  _mm256_add_ps(square_sums[2], square_sums[3])));
  for (; index < count; ++index) {
    const float value = input[index];
    sum += value;
    sum_squares += value * value;
  }
  const float mean = sum / static_cast<float>(count);
  return {mean, sum_squares / static_cast<float>(count) - mean * mean};
}

void ApplyNormalizationAffineFloat32_AVX2(const float *input, const float *scale, const float *bias,
                                          float *output, std::size_t count, float center,
                                          float multiplier) {
  const __m256 center8 = _mm256_set1_ps(center);
  const __m256 multiplier8 = _mm256_set1_ps(multiplier);
  std::size_t index = 0;
  for (; index + 8 <= count; index += 8) {
    const __m256 value = _mm256_loadu_ps(input + index);
    const __m256 weight = _mm256_loadu_ps(scale + index);
    __m256 result =
        _mm256_mul_ps(_mm256_mul_ps(_mm256_sub_ps(value, center8), multiplier8), weight);
    if (bias != nullptr) {
      result = _mm256_add_ps(result, _mm256_loadu_ps(bias + index));
    }
    _mm256_storeu_ps(output + index, result);
  }
  for (; index < count; ++index) {
    float value = (input[index] - center) * multiplier * scale[index];
    if (bias != nullptr) {
      value += bias[index];
    }
    output[index] = value;
  }
}

} // namespace onnx_light_cpu
