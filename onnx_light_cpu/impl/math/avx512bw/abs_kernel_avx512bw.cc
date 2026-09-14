// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/math_kernels.h"

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace onnx_light_cpu {

void AbsInt8_AVX512(const int8_t *input, int8_t *output, std::size_t count) {
  std::size_t i = 0;
  const std::size_t stride = 64;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    __m512i v = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(input + i));
    v = _mm512_abs_epi8(v);
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(output + i), v);
  }
  for (; i < count; ++i) {
    const int v = static_cast<int>(input[i]);
    output[i] = static_cast<int8_t>(v < 0 ? -v : v);
  }
}

void AbsInt16_AVX512(const int16_t *input, int16_t *output, std::size_t count) {
  std::size_t i = 0;
  constexpr std::size_t stride = 32;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    const __m512i value = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(input + i));
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(output + i), _mm512_abs_epi16(value));
  }
  for (; i + 8 <= count; i += 8) {
    const __m128i value = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + i));
    const __m128i sign = _mm_srai_epi16(value, 15);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output + i),
                     _mm_sub_epi16(_mm_xor_si128(value, sign), sign));
  }
  for (; i < count; ++i) {
    const int value = input[i];
    output[i] = static_cast<int16_t>(value < 0 ? -value : value);
  }
}

} // namespace onnx_light_cpu
