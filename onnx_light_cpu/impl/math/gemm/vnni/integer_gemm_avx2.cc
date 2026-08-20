// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Exact AVX2 UINT8 x INT8 dot-product fallback for CPUs without VNNI.

#include "onnx_light_cpu/impl/math/gemm/vnni/integer_gemm_vnni.h"

#include <bit>
#include <cstdint>

#include <immintrin.h>

namespace onnx_light_cpu {

namespace detail {

std::int32_t IntegerDotU8S8Avx2(const std::uint8_t *ua, const std::int8_t *sb, std::int64_t depth) {
  __m256i accumulator = _mm256_setzero_si256();
  const __m256i low_mask = _mm256_set1_epi8(0x7f);
  const __m256i ones = _mm256_set1_epi16(1);
  std::int64_t index = 0;
  for (; index + 32 <= depth; index += 32) {
    const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ua + index));
    const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(sb + index));
    const __m256i a_low = _mm256_and_si256(a, low_mask);
    const __m256i a_high = _mm256_andnot_si256(low_mask, a);
    const __m256i products_low = _mm256_maddubs_epi16(a_low, b);
    const __m256i products_high = _mm256_maddubs_epi16(a_high, b);
    accumulator = _mm256_add_epi32(accumulator, _mm256_madd_epi16(products_low, ones));
    accumulator = _mm256_add_epi32(accumulator, _mm256_madd_epi16(products_high, ones));
  }

  __m128i total128 =
      _mm_add_epi32(_mm256_castsi256_si128(accumulator), _mm256_extracti128_si256(accumulator, 1));
  total128 = _mm_add_epi32(total128, _mm_shuffle_epi32(total128, _MM_SHUFFLE(1, 0, 3, 2)));
  total128 = _mm_add_epi32(total128, _mm_shuffle_epi32(total128, _MM_SHUFFLE(2, 3, 0, 1)));
  std::uint32_t total = static_cast<std::uint32_t>(_mm_cvtsi128_si32(total128));
  for (; index < depth; ++index) {
    total += static_cast<std::uint32_t>(static_cast<std::int32_t>(ua[index]) *
                                        static_cast<std::int32_t>(sb[index]));
  }
  return std::bit_cast<std::int32_t>(total);
}

} // namespace detail

} // namespace onnx_light_cpu
