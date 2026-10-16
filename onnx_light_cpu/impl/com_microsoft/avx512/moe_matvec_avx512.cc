// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

namespace onnx_light_cpu::detail {
namespace {

__m512 LoadBFloat16(const std::uint16_t *source) {
  const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(source));
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(packed), 16));
}

} // namespace

void MoEMatVecFloat32Avx512(const float *input, const float *weights, float *output,
                            std::size_t begin, std::size_t end, std::size_t columns) {
  for (std::size_t row = begin; row < end; ++row) {
    const float *weight = weights + row * columns;
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();
    std::size_t column = 0;
    for (; column + 64 <= columns; column += 64) {
      sum0 =
          _mm512_fmadd_ps(_mm512_loadu_ps(input + column), _mm512_loadu_ps(weight + column), sum0);
      sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(input + column + 16),
                             _mm512_loadu_ps(weight + column + 16), sum1);
      sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(input + column + 32),
                             _mm512_loadu_ps(weight + column + 32), sum2);
      sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(input + column + 48),
                             _mm512_loadu_ps(weight + column + 48), sum3);
    }
    sum0 = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
    for (; column + 16 <= columns; column += 16) {
      sum0 =
          _mm512_fmadd_ps(_mm512_loadu_ps(input + column), _mm512_loadu_ps(weight + column), sum0);
    }
    float value = _mm512_reduce_add_ps(sum0);
    for (; column < columns; ++column) {
      value += input[column] * weight[column];
    }
    output[row] = value;
  }
}

void MoEMatVecBFloat16Avx512(const std::uint16_t *input, const std::uint16_t *weights,
                             float *output, std::size_t begin, std::size_t end,
                             std::size_t columns) {
  for (std::size_t row = begin; row < end; ++row) {
    const std::uint16_t *weight = weights + row * columns;
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();
    std::size_t column = 0;
    for (; column + 64 <= columns; column += 64) {
      sum0 = _mm512_fmadd_ps(LoadBFloat16(input + column), LoadBFloat16(weight + column), sum0);
      sum1 = _mm512_fmadd_ps(LoadBFloat16(input + column + 16), LoadBFloat16(weight + column + 16),
                             sum1);
      sum2 = _mm512_fmadd_ps(LoadBFloat16(input + column + 32), LoadBFloat16(weight + column + 32),
                             sum2);
      sum3 = _mm512_fmadd_ps(LoadBFloat16(input + column + 48), LoadBFloat16(weight + column + 48),
                             sum3);
    }
    sum0 = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
    for (; column + 16 <= columns; column += 16) {
      sum0 = _mm512_fmadd_ps(LoadBFloat16(input + column), LoadBFloat16(weight + column), sum0);
    }
    float value = _mm512_reduce_add_ps(sum0);
    for (; column < columns; ++column) {
      const std::uint32_t left = static_cast<std::uint32_t>(input[column]) << 16;
      const std::uint32_t right = static_cast<std::uint32_t>(weight[column]) << 16;
      value += _mm_cvtss_f32(_mm_castsi128_ps(_mm_cvtsi32_si128(static_cast<int>(left)))) *
               _mm_cvtss_f32(_mm_castsi128_ps(_mm_cvtsi32_si128(static_cast<int>(right))));
    }
    output[row] = value;
  }
}

void MoEMatVecFloat32BFloat16Avx512(const float *input, const std::uint16_t *weights, float *output,
                                    std::size_t begin, std::size_t end, std::size_t columns) {
  for (std::size_t row = begin; row < end; ++row) {
    const std::uint16_t *weight = weights + row * columns;
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();
    std::size_t column = 0;
    for (; column + 64 <= columns; column += 64) {
      sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(input + column), LoadBFloat16(weight + column), sum0);
      sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(input + column + 16),
                             LoadBFloat16(weight + column + 16), sum1);
      sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(input + column + 32),
                             LoadBFloat16(weight + column + 32), sum2);
      sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(input + column + 48),
                             LoadBFloat16(weight + column + 48), sum3);
    }
    sum0 = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
    for (; column + 16 <= columns; column += 16) {
      sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(input + column), LoadBFloat16(weight + column), sum0);
    }
    float value = _mm512_reduce_add_ps(sum0);
    for (; column < columns; ++column) {
      const std::uint32_t bits = static_cast<std::uint32_t>(weight[column]) << 16;
      value += input[column] *
               _mm_cvtss_f32(_mm_castsi128_ps(_mm_cvtsi32_si128(static_cast<int>(bits))));
    }
    output[row] = value;
  }
}

} // namespace onnx_light_cpu::detail
