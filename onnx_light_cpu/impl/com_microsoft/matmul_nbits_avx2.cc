// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/com_microsoft/matmul_nbits_panel.h"

#include <immintrin.h>

namespace onnx_light_cpu::detail {

void NBitsPanelAvx2(const float *a, const float *b, float *sums, std::size_t rows,
                    std::size_t depth) {
  for (std::size_t row = 0; row < rows; ++row) {
    __m256 s0 = _mm256_loadu_ps(sums + row * kNBitsColumns);
    __m256 s1 = _mm256_loadu_ps(sums + row * kNBitsColumns + 8);
    __m256 s2 = _mm256_loadu_ps(sums + row * kNBitsColumns + 16);
    __m256 s3 = _mm256_loadu_ps(sums + row * kNBitsColumns + 24);
    for (std::size_t p = 0; p < depth; ++p) {
      const __m256 value = _mm256_set1_ps(a[row * kNBitsBlock + p]);
      const float *weights = b + p * kNBitsColumns;
      s0 = _mm256_add_ps(s0, _mm256_mul_ps(value, _mm256_loadu_ps(weights)));
      s1 = _mm256_add_ps(s1, _mm256_mul_ps(value, _mm256_loadu_ps(weights + 8)));
      s2 = _mm256_add_ps(s2, _mm256_mul_ps(value, _mm256_loadu_ps(weights + 16)));
      s3 = _mm256_add_ps(s3, _mm256_mul_ps(value, _mm256_loadu_ps(weights + 24)));
    }
    _mm256_storeu_ps(sums + row * kNBitsColumns, s0);
    _mm256_storeu_ps(sums + row * kNBitsColumns + 8, s1);
    _mm256_storeu_ps(sums + row * kNBitsColumns + 16, s2);
    _mm256_storeu_ps(sums + row * kNBitsColumns + 24, s3);
  }
}

} // namespace onnx_light_cpu::detail
