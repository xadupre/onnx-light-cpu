// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/logical/logical_kernels.h"

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace onnx_light_cpu {

void NotBool_AVX512(const uint8_t *input, uint8_t *output, std::size_t count) {
  const __m512i zero = _mm512_setzero_si512();
  std::size_t i = 0;
  for (; i + 64 <= count; i += 64) {
    const __m512i value = _mm512_loadu_si512(input + i);
    const __mmask64 is_zero = _mm512_cmpeq_epi8_mask(value, zero);
    const __m512i result = _mm512_maskz_set1_epi8(is_zero, 1);
    _mm512_storeu_si512(output + i, result);
  }
  for (; i < count; ++i) {
    output[i] = static_cast<uint8_t>(input[i] == 0 ? 1 : 0);
  }
}

} // namespace onnx_light_cpu
