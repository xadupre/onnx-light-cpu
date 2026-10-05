// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/abs_kernel_ssse3.h"

#include <tmmintrin.h>

namespace onnx_light_cpu {

void AbsInt8_SSSE3(const std::int8_t *input, std::int8_t *output, std::size_t count) {
  std::size_t i = 0;
  for (; i + 16 <= count; i += 16) {
    const __m128i value = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + i));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output + i), _mm_abs_epi8(value));
  }
  for (; i < count; ++i) {
    const int value = static_cast<int>(input[i]);
    output[i] = static_cast<std::int8_t>(value < 0 ? -value : value);
  }
}

void AbsInt16_SSSE3(const std::int16_t *input, std::int16_t *output, std::size_t count) {
  std::size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    const __m128i value = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + i));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output + i), _mm_abs_epi16(value));
  }
  for (; i < count; ++i) {
    const std::int32_t value = static_cast<std::int32_t>(input[i]);
    output[i] = static_cast<std::int16_t>(value < 0 ? -value : value);
  }
}

} // namespace onnx_light_cpu
