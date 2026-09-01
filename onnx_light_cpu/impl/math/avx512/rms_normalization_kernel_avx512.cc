// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/rms_normalization.h"

#include <immintrin.h>

#include <cmath>

namespace onnx_light_cpu {

void RmsNormalizationFloat32_AVX512(const float *input, const float *scale, float *output,
                                    std::size_t row_begin, std::size_t row_end, std::size_t width,
                                    float epsilon) {
  for (std::size_t row = row_begin; row < row_end; ++row) {
    const std::size_t offset = row * width;
    __m512 sum_squares = _mm512_setzero_ps();
    std::size_t column = 0;
    for (; column + 16 <= width; column += 16) {
      const __m512 value = _mm512_loadu_ps(input + offset + column);
      sum_squares = _mm512_fmadd_ps(value, value, sum_squares);
    }
    float sum = _mm512_reduce_add_ps(sum_squares);
    for (; column < width; ++column) {
      const float value = input[offset + column];
      sum += value * value;
    }

    const __m512 inverse_rms =
        _mm512_set1_ps(1.0F / std::sqrt(sum / static_cast<float>(width) + epsilon));
    column = 0;
    for (; column + 16 <= width; column += 16) {
      const __m512 value = _mm512_loadu_ps(input + offset + column);
      const __m512 weight = _mm512_loadu_ps(scale + column);
      _mm512_storeu_ps(output + offset + column,
                       _mm512_mul_ps(_mm512_mul_ps(value, inverse_rms), weight));
    }
    const float scalar_inverse_rms = _mm_cvtss_f32(_mm512_castps512_ps128(inverse_rms));
    for (; column < width; ++column) {
      output[offset + column] = input[offset + column] * scalar_inverse_rms * scale[column];
    }
  }
}

} // namespace onnx_light_cpu
