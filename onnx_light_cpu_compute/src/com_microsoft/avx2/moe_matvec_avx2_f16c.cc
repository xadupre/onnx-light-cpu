// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu_compute/math/half_conversion.h"

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

namespace onnx_light_cpu::detail {
namespace {

float Reduce(__m256 value) {
  const __m128 low = _mm256_castps256_ps128(value);
  const __m128 high = _mm256_extractf128_ps(value, 1);
  const __m128 sum = _mm_add_ps(low, high);
  const __m128 pair = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
  return _mm_cvtss_f32(_mm_add_ss(pair, _mm_shuffle_ps(pair, pair, 1)));
}

__m256 LoadFloat16(const std::uint16_t *source) {
  return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(source)));
}

} // namespace

void MoEMatVecFloat16Avx2F16c(const std::uint16_t *input, const std::uint16_t *weights,
                              float *output, std::size_t begin, std::size_t end,
                              std::size_t columns) {
  for (std::size_t row = begin; row < end; ++row) {
    const std::uint16_t *weight = weights + row * columns;
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    std::size_t column = 0;
    for (; column + 32 <= columns; column += 32) {
      sum0 = _mm256_add_ps(
          sum0, _mm256_mul_ps(LoadFloat16(input + column), LoadFloat16(weight + column)));
      sum1 = _mm256_add_ps(
          sum1, _mm256_mul_ps(LoadFloat16(input + column + 8), LoadFloat16(weight + column + 8)));
      sum2 = _mm256_add_ps(
          sum2, _mm256_mul_ps(LoadFloat16(input + column + 16), LoadFloat16(weight + column + 16)));
      sum3 = _mm256_add_ps(
          sum3, _mm256_mul_ps(LoadFloat16(input + column + 24), LoadFloat16(weight + column + 24)));
    }
    sum0 = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
    for (; column + 8 <= columns; column += 8) {
      sum0 = _mm256_add_ps(
          sum0, _mm256_mul_ps(LoadFloat16(input + column), LoadFloat16(weight + column)));
    }
    float value = Reduce(sum0);
    for (; column < columns; ++column) {
      value += Float16BitsToFloat(input[column]) * Float16BitsToFloat(weight[column]);
    }
    output[row] = value;
  }
}

void MoEMatVecFloat32Float16Avx2F16c(const float *input, const std::uint16_t *weights,
                                     float *output, std::size_t begin, std::size_t end,
                                     std::size_t columns) {
  for (std::size_t row = begin; row < end; ++row) {
    const std::uint16_t *weight = weights + row * columns;
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    std::size_t column = 0;
    for (; column + 32 <= columns; column += 32) {
      sum0 = _mm256_add_ps(
          sum0, _mm256_mul_ps(_mm256_loadu_ps(input + column), LoadFloat16(weight + column)));
      sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(_mm256_loadu_ps(input + column + 8),
                                               LoadFloat16(weight + column + 8)));
      sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(_mm256_loadu_ps(input + column + 16),
                                               LoadFloat16(weight + column + 16)));
      sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(_mm256_loadu_ps(input + column + 24),
                                               LoadFloat16(weight + column + 24)));
    }
    sum0 = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
    for (; column + 8 <= columns; column += 8) {
      sum0 = _mm256_add_ps(
          sum0, _mm256_mul_ps(_mm256_loadu_ps(input + column), LoadFloat16(weight + column)));
    }
    float value = Reduce(sum0);
    for (; column < columns; ++column) {
      value += input[column] * Float16BitsToFloat(weight[column]);
    }
    output[row] = value;
  }
}

} // namespace onnx_light_cpu::detail
