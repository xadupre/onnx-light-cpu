// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>

#include <immintrin.h>

namespace onnx_light_cpu {

void SplitCopy4x4_AVX2(const uint8_t *source, void *const *outputs, int64_t begin, int64_t end) {
  auto *output0 = static_cast<uint8_t *>(outputs[0]);
  auto *output1 = static_cast<uint8_t *>(outputs[1]);
  auto *output2 = static_cast<uint8_t *>(outputs[2]);
  auto *output3 = static_cast<uint8_t *>(outputs[3]);
  int64_t row = begin;
  for (; row + 4 <= end; row += 4) {
    __m128 values0 =
        _mm_castsi128_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(source + row * 16)));
    __m128 values1 = _mm_castsi128_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(source + (row + 1) * 16)));
    __m128 values2 = _mm_castsi128_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(source + (row + 2) * 16)));
    __m128 values3 = _mm_castsi128_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(source + (row + 3) * 16)));
    _MM_TRANSPOSE4_PS(values0, values1, values2, values3);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output0 + row * 4), _mm_castps_si128(values0));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output1 + row * 4), _mm_castps_si128(values1));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output2 + row * 4), _mm_castps_si128(values2));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output3 + row * 4), _mm_castps_si128(values3));
  }
  for (; row < end; ++row) {
    std::memcpy(output0 + row * 4, source + row * 16, 4);
    std::memcpy(output1 + row * 4, source + row * 16 + 4, 4);
    std::memcpy(output2 + row * 4, source + row * 16 + 8, 4);
    std::memcpy(output3 + row * 4, source + row * 16 + 12, 4);
  }
}

} // namespace onnx_light_cpu
