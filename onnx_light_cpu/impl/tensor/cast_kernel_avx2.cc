// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/tensor/cast_simd.h"

#include <cstring>
#include <immintrin.h>

namespace onnx_light_cpu::detail {

std::size_t CastFloat32ToBFloat16_AVX2(const std::uint8_t *src, std::uint8_t *dst,
                                       std::size_t count) {
  std::size_t i = 0;
  for (; count - i >= 8; i += 8) {
    __m256i bits;
    std::memcpy(&bits, src + i * 4, sizeof(bits));
    const __m256i high = _mm256_srli_epi32(bits, 16);
    const __m256i nan = _mm256_cmpgt_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff)),
                                           _mm256_set1_epi32(0x7f800000));
    const __m256i bias =
        _mm256_add_epi32(_mm256_set1_epi32(0x7fff), _mm256_and_si256(high, _mm256_set1_epi32(1)));
    const __m256i rounded = _mm256_srli_epi32(_mm256_add_epi32(bits, bias), 16);
    const __m256i result =
        _mm256_blendv_epi8(rounded, _mm256_or_si256(high, _mm256_set1_epi32(0x40)), nan);
    const __m128i packed =
        _mm_packus_epi32(_mm256_castsi256_si128(result), _mm256_extracti128_si256(result, 1));
    std::memcpy(dst + i * 2, &packed, sizeof(packed));
  }
  return i;
}

std::size_t CastBFloat16ToFloat32_AVX2(const std::uint8_t *src, std::uint8_t *dst,
                                       std::size_t count) {
  std::size_t i = 0;
  for (; count - i >= 8; i += 8) {
    __m128i half;
    std::memcpy(&half, src + i * 2, sizeof(half));
    const __m256i value = _mm256_slli_epi32(_mm256_cvtepu16_epi32(half), 16);
    std::memcpy(dst + i * 4, &value, sizeof(value));
  }
  return i;
}

} // namespace onnx_light_cpu::detail
