// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/normalization_kernel.h"

#include <immintrin.h>

namespace onnx_light_cpu {

float ComputeNormalizationMeanSquareFloat32_AVX512(const float *input, std::size_t count) {
  __m512 sums[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                    _mm512_setzero_ps()};
  std::size_t index = 0;
  for (; index + 64 <= count; index += 64) {
    for (std::size_t lane = 0; lane < 4; ++lane) {
      const __m512 value = _mm512_loadu_ps(input + index + lane * 16);
      sums[lane] = _mm512_fmadd_ps(value, value, sums[lane]);
    }
  }
  for (; index + 16 <= count; index += 16) {
    const __m512 value = _mm512_loadu_ps(input + index);
    sums[0] = _mm512_fmadd_ps(value, value, sums[0]);
  }
  float sum = _mm512_reduce_add_ps(
      _mm512_add_ps(_mm512_add_ps(sums[0], sums[1]), _mm512_add_ps(sums[2], sums[3])));
  for (; index < count; ++index) {
    sum += input[index] * input[index];
  }
  return sum / static_cast<float>(count);
}

Float32NormalizationMoments ComputeNormalizationMomentsFloat32_AVX512(const float *input,
                                                                      std::size_t count) {
  __m512 sums[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                    _mm512_setzero_ps()};
  __m512 square_sums[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                           _mm512_setzero_ps()};
  std::size_t index = 0;
  for (; index + 64 <= count; index += 64) {
    for (std::size_t lane = 0; lane < 4; ++lane) {
      const __m512 value = _mm512_loadu_ps(input + index + lane * 16);
      sums[lane] = _mm512_add_ps(sums[lane], value);
      square_sums[lane] = _mm512_fmadd_ps(value, value, square_sums[lane]);
    }
  }
  for (; index + 16 <= count; index += 16) {
    const __m512 value = _mm512_loadu_ps(input + index);
    sums[0] = _mm512_add_ps(sums[0], value);
    square_sums[0] = _mm512_fmadd_ps(value, value, square_sums[0]);
  }
  float sum = _mm512_reduce_add_ps(
      _mm512_add_ps(_mm512_add_ps(sums[0], sums[1]), _mm512_add_ps(sums[2], sums[3])));
  float sum_squares =
      _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(square_sums[0], square_sums[1]),
                                         _mm512_add_ps(square_sums[2], square_sums[3])));
  for (; index < count; ++index) {
    const float value = input[index];
    sum += value;
    sum_squares += value * value;
  }
  const float mean = sum / static_cast<float>(count);
  return {mean, sum_squares / static_cast<float>(count) - mean * mean};
}

void ApplyNormalizationAffineFloat32_AVX512(const float *input, const float *scale,
                                            const float *bias, float *output, std::size_t count,
                                            float center, float multiplier) {
  const __m512 center16 = _mm512_set1_ps(center);
  const __m512 multiplier16 = _mm512_set1_ps(multiplier);
  std::size_t index = 0;
  for (; index + 16 <= count; index += 16) {
    const __m512 value = _mm512_loadu_ps(input + index);
    const __m512 weight = _mm512_loadu_ps(scale + index);
    __m512 result =
        _mm512_mul_ps(_mm512_mul_ps(_mm512_sub_ps(value, center16), multiplier16), weight);
    if (bias != nullptr) {
      result = _mm512_add_ps(result, _mm512_loadu_ps(bias + index));
    }
    _mm512_storeu_ps(output + index, result);
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
