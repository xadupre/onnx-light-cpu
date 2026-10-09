// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <immintrin.h>

#include <cstdint>
#include <cstring>

namespace onnx_light_cpu {

void MatMulNBitsInt8DotAvxVnni(const std::uint8_t *a, const std::uint8_t *b, std::int32_t *dots) {
  const __m256i offset = _mm256_set1_epi8(static_cast<char>(128));
  __m256i sum = _mm256_setzero_si256();
  for (int quad = 0; quad < 8; ++quad) {
    std::uint32_t value;
    std::memcpy(&value, a + quad * 4, sizeof(value));
    const __m256i activation = _mm256_set1_epi32(static_cast<int>(value));
    const __m256i weight = _mm256_xor_si256(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + quad * 32)), offset);
    sum = _mm256_dpbusd_epi32(sum, activation, weight);
  }
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(dots), sum);
}

} // namespace onnx_light_cpu
