// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/com_microsoft/matmul_nbits_panel.h"

#include <immintrin.h>

namespace onnx_light_cpu::detail {

void NBitsPanelAvx512(const float *a, const float *b, float *sums, std::size_t rows,
                      std::size_t depth) {
  for (std::size_t row = 0; row < rows; ++row) {
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
