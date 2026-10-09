// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <immintrin.h>

#include <cstdint>
#include <cstring>

namespace onnx_light_cpu {

void MatMulNBitsInt8DotAvx512Vnni(const std::uint8_t *a, const std::uint8_t *b,
                                  std::int32_t *dots) {
  const __m512i offset = _mm512_set1_epi8(static_cast<char>(128));
  __m512i sum = _mm512_setzero_si512();
  for (int quad = 0; quad < 8; ++quad) {
    std::uint32_t value;
    std::memcpy(&value, a + quad * 4, sizeof(value));
    const __m512i activation = _mm512_set1_epi32(static_cast<int>(value));
    const __m256i source = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + quad * 32));
    const __m512i weight = _mm512_xor_si512(_mm512_zextsi256_si512(source), offset);
    sum = _mm512_dpbusd_epi32(sum, activation, weight);
  }
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(dots), _mm512_castsi512_si256(sum));
}

} // namespace onnx_light_cpu
