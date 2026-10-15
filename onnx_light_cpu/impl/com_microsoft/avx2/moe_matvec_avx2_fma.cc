// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

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

__m256 LoadBFloat16(const std::uint16_t *source) {
  const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i *>(source));
  const __m256i widened = _mm256_slli_epi32(_mm256_cvtepu16_epi32(packed), 16);
  return _mm256_castsi256_ps(widened);
}

} // namespace

void MoEMatVecFloat32Avx2Fma(const float *input, const float *weights, float *output,
                             std::size_t begin, std::size_t end, std::size_t columns) {
  for (std::size_t row = begin; row < end; ++row) {
    const float *weight = weights + row * columns;
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    std::size_t column = 0;
    for (; column + 32 <= columns; column += 32) {
      sum0 =
          _mm256_fmadd_ps(_mm256_loadu_ps(input + column), _mm256_loadu_ps(weight + column), sum0);
      sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(input + column + 8),
                             _mm256_loadu_ps(weight + column + 8), sum1);
      sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(input + column + 16),
                             _mm256_loadu_ps(weight + column + 16), sum2);
      sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(input + column + 24),
                             _mm256_loadu_ps(weight + column + 24), sum3);
    }
    sum0 = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
    for (; column + 8 <= columns; column += 8) {
      sum0 =
          _mm256_fmadd_ps(_mm256_loadu_ps(input + column), _mm256_loadu_ps(weight + column), sum0);
    }
    float value = Reduce(sum0);
    for (; column < columns; ++column) {
      value += input[column] * weight[column];
    }
    output[row] = value;
  }
}

void MoEMatVecBFloat16Avx2Fma(const std::uint16_t *input, const std::uint16_t *weights,
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
      sum0 = _mm256_fmadd_ps(LoadBFloat16(input + column), LoadBFloat16(weight + column), sum0);
      sum1 = _mm256_fmadd_ps(LoadBFloat16(input + column + 8), LoadBFloat16(weight + column + 8),
                             sum1);
      sum2 = _mm256_fmadd_ps(LoadBFloat16(input + column + 16), LoadBFloat16(weight + column + 16),
                             sum2);
      sum3 = _mm256_fmadd_ps(LoadBFloat16(input + column + 24), LoadBFloat16(weight + column + 24),
                             sum3);
    }
    sum0 = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
    for (; column + 8 <= columns; column += 8) {
      sum0 = _mm256_fmadd_ps(LoadBFloat16(input + column), LoadBFloat16(weight + column), sum0);
    }
    float value = Reduce(sum0);
    for (; column < columns; ++column) {
      const std::uint32_t left = static_cast<std::uint32_t>(input[column]) << 16;
      const std::uint32_t right = static_cast<std::uint32_t>(weight[column]) << 16;
      value += _mm_cvtss_f32(_mm_castsi128_ps(_mm_cvtsi32_si128(static_cast<int>(left)))) *
               _mm_cvtss_f32(_mm_castsi128_ps(_mm_cvtsi32_si128(static_cast<int>(right))));
    }
    output[row] = value;
  }
}

void MoEMatVecFloat32BFloat16Avx2Fma(const float *input, const std::uint16_t *weights,
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
      sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(input + column), LoadBFloat16(weight + column), sum0);
      sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(input + column + 8), LoadBFloat16(weight + column + 8),
                             sum1);
      sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(input + column + 16),
                             LoadBFloat16(weight + column + 16), sum2);
      sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(input + column + 24),
                             LoadBFloat16(weight + column + 24), sum3);
    }
    sum0 = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
    for (; column + 8 <= columns; column += 8) {
      sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(input + column), LoadBFloat16(weight + column), sum0);
    }
    float value = Reduce(sum0);
    for (; column < columns; ++column) {
      const std::uint32_t bits = static_cast<std::uint32_t>(weight[column]) << 16;
      value += input[column] *
               _mm_cvtss_f32(_mm_castsi128_ps(_mm_cvtsi32_si128(static_cast<int>(bits))));
    }
    output[row] = value;
  }
}

} // namespace onnx_light_cpu::detail
