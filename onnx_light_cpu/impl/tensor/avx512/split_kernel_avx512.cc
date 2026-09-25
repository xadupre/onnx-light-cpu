// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>

#include <immintrin.h>

namespace onnx_light_cpu {

void SplitCopy4x4_AVX512(const uint8_t *source, void *const *outputs, int64_t begin, int64_t end) {
  auto *output0 = static_cast<uint8_t *>(outputs[0]);
  auto *output1 = static_cast<uint8_t *>(outputs[1]);
  auto *output2 = static_cast<uint8_t *>(outputs[2]);
  auto *output3 = static_cast<uint8_t *>(outputs[3]);
  const __m512i columns01 =
      _mm512_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28, 1, 5, 9, 13, 17, 21, 25, 29);
  const __m512i columns23 =
      _mm512_setr_epi32(2, 6, 10, 14, 18, 22, 26, 30, 3, 7, 11, 15, 19, 23, 27, 31);
  // Keep the indices as the destructive operand so GCC does not reload each source vector.
  volatile const __mmask16 full_mask_storage = 0xffff;
  const __mmask16 full_mask = full_mask_storage;
  int64_t row = begin;
  for (; row + 16 <= end; row += 16) {
    const uint8_t *block = source + row * 16;
    const __m512i rows03 = _mm512_loadu_si512(block);
    const __m512i rows47 = _mm512_loadu_si512(block + 64);
    const __m512i rows811 = _mm512_loadu_si512(block + 128);
    const __m512i rows1215 = _mm512_loadu_si512(block + 192);
    const __m512i low01 = _mm512_mask2_permutex2var_epi32(rows03, columns01, full_mask, rows47);
    const __m512i high01 = _mm512_mask2_permutex2var_epi32(rows811, columns01, full_mask, rows1215);
    const __m512i low23 = _mm512_mask2_permutex2var_epi32(rows03, columns23, full_mask, rows47);
    const __m512i high23 = _mm512_mask2_permutex2var_epi32(rows811, columns23, full_mask, rows1215);
    _mm512_storeu_si512(output0 + row * 4, _mm512_shuffle_i64x2(low01, high01, 0x44));
    _mm512_storeu_si512(output1 + row * 4, _mm512_shuffle_i64x2(low01, high01, 0xee));
    _mm512_storeu_si512(output2 + row * 4, _mm512_shuffle_i64x2(low23, high23, 0x44));
    _mm512_storeu_si512(output3 + row * 4, _mm512_shuffle_i64x2(low23, high23, 0xee));
  }
  for (; row < end; ++row) {
    std::memcpy(output0 + row * 4, source + row * 16, 4);
    std::memcpy(output1 + row * 4, source + row * 16 + 4, 4);
    std::memcpy(output2 + row * 4, source + row * 16 + 8, 4);
    std::memcpy(output3 + row * 4, source + row * 16 + 12, 4);
  }
}

} // namespace onnx_light_cpu
