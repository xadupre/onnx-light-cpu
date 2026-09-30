// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/com_microsoft/matmul_nbits_panel.h"

#include <immintrin.h>

namespace onnx_light_cpu::detail {

void NBitsPanelAvx512(const float *a, const float *b, float *sums, std::size_t rows,
                      std::size_t depth) {
  constexpr std::size_t kRowBlock = 8;
  std::size_t row = 0;
  for (; row + kRowBlock <= rows; row += kRowBlock) {
    __m512 s0[kRowBlock];
    __m512 s1[kRowBlock];
    for (std::size_t r = 0; r < kRowBlock; ++r) {
      s0[r] = _mm512_loadu_ps(sums + (row + r) * kNBitsColumns);
      s1[r] = _mm512_loadu_ps(sums + (row + r) * kNBitsColumns + 16);
    }
    for (std::size_t p = 0; p < depth; ++p) {
      const float *weights = b + p * kNBitsColumns;
      const __m512 w0 = _mm512_loadu_ps(weights);
      const __m512 w1 = _mm512_loadu_ps(weights + 16);
      for (std::size_t r = 0; r < kRowBlock; ++r) {
        const __m512 value = _mm512_set1_ps(a[(row + r) * kNBitsBlock + p]);
        s0[r] = _mm512_add_ps(s0[r], _mm512_mul_ps(value, w0));
        s1[r] = _mm512_add_ps(s1[r], _mm512_mul_ps(value, w1));
      }
    }
    for (std::size_t r = 0; r < kRowBlock; ++r) {
      _mm512_storeu_ps(sums + (row + r) * kNBitsColumns, s0[r]);
      _mm512_storeu_ps(sums + (row + r) * kNBitsColumns + 16, s1[r]);
    }
  }
  for (; row < rows; ++row) {
    __m512 s0 = _mm512_loadu_ps(sums + row * kNBitsColumns);
    __m512 s1 = _mm512_loadu_ps(sums + row * kNBitsColumns + 16);
    for (std::size_t p = 0; p < depth; ++p) {
      const __m512 value = _mm512_set1_ps(a[row * kNBitsBlock + p]);
      const float *weights = b + p * kNBitsColumns;
      s0 = _mm512_add_ps(s0, _mm512_mul_ps(value, _mm512_loadu_ps(weights)));
      s1 = _mm512_add_ps(s1, _mm512_mul_ps(value, _mm512_loadu_ps(weights + 16)));
    }
    _mm512_storeu_ps(sums + row * kNBitsColumns, s0);
    _mm512_storeu_ps(sums + row * kNBitsColumns + 16, s1);
  }
}

} // namespace onnx_light_cpu::detail
