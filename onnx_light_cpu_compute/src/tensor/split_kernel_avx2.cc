// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>

#include <immintrin.h>

namespace onnx_light_cpu {

void SplitCopy4x2_AVX2(const uint8_t *source, void *const *outputs, int64_t begin, int64_t end) {
  auto *output0 = static_cast<uint8_t *>(outputs[0]);
  auto *output1 = static_cast<uint8_t *>(outputs[1]);
  auto *output2 = static_cast<uint8_t *>(outputs[2]);
  auto *output3 = static_cast<uint8_t *>(outputs[3]);
  const auto reorder4 = [](__m128i value) {
    value = _mm_shufflelo_epi16(value, _MM_SHUFFLE(3, 1, 2, 0));
    return _mm_shufflehi_epi16(value, _MM_SHUFFLE(3, 1, 2, 0));
  };
  int64_t row = begin;
  for (; row + 8 <= end; row += 8) {
    const __m128i rows01 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(source + row * 8));
    const __m128i rows23 =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(source + row * 8 + 16));
    const __m128i rows45 =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(source + row * 8 + 32));
    const __m128i rows67 =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(source + row * 8 + 48));
    const __m128i low03 = _mm_unpacklo_epi16(rows01, rows23);
    const __m128i high03 = _mm_unpackhi_epi16(rows01, rows23);
    const __m128i low47 = _mm_unpacklo_epi16(rows45, rows67);
    const __m128i high47 = _mm_unpackhi_epi16(rows45, rows67);
    const __m128i ab03 = reorder4(_mm_unpacklo_epi32(low03, high03));
    const __m128i cd03 = reorder4(_mm_unpackhi_epi32(low03, high03));
    const __m128i ab47 = reorder4(_mm_unpacklo_epi32(low47, high47));
    const __m128i cd47 = reorder4(_mm_unpackhi_epi32(low47, high47));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output0 + row * 2),
                     _mm_unpacklo_epi64(ab03, ab47));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output1 + row * 2),
                     _mm_unpackhi_epi64(ab03, ab47));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output2 + row * 2),
                     _mm_unpacklo_epi64(cd03, cd47));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output3 + row * 2),
                     _mm_unpackhi_epi64(cd03, cd47));
  }
  for (; row < end; ++row) {
    std::memcpy(output0 + row * 2, source + row * 8, 2);
    std::memcpy(output1 + row * 2, source + row * 8 + 2, 2);
    std::memcpy(output2 + row * 2, source + row * 8 + 4, 2);
    std::memcpy(output3 + row * 2, source + row * 8 + 6, 2);
  }
}

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
