// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/gemm/vnni/integer_gemm_vnni.h"

#include <bit>
#include <cstdint>

#include <immintrin.h>

namespace onnx_light_cpu::detail {

std::int32_t IntegerDotU8S8AvxVnni(const std::uint8_t *ua, const std::int8_t *sb,
                                   std::int64_t depth) {
  __m256i accumulator = _mm256_setzero_si256();
  std::int64_t i = 0;
  for (; i + 32 <= depth; i += 32) {
    const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ua + i));
    const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(sb + i));
    accumulator = _mm256_dpbusd_epi32(accumulator, va, vb);
  }
  const __m128i lanes =
      _mm_add_epi32(_mm256_castsi256_si128(accumulator), _mm256_extracti128_si256(accumulator, 1));
  const __m128i pairs = _mm_add_epi32(lanes, _mm_shuffle_epi32(lanes, 0x4e));
  const __m128i sum = _mm_add_epi32(pairs, _mm_shuffle_epi32(pairs, 0xb1));
  std::uint32_t total = std::bit_cast<std::uint32_t>(_mm_cvtsi128_si32(sum));
  if (i + 16 <= depth) {
    const __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(ua + i));
    const __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i *>(sb + i));
    const __m128i tail = _mm_dpbusd_epi32(_mm_setzero_si128(), va, vb);
    const __m128i tail_pairs = _mm_add_epi32(tail, _mm_shuffle_epi32(tail, 0x4e));
    const __m128i tail_sum = _mm_add_epi32(tail_pairs, _mm_shuffle_epi32(tail_pairs, 0xb1));
    total += static_cast<std::uint32_t>(_mm_cvtsi128_si32(tail_sum));
    i += 16;
  }
  for (; i < depth; ++i) {
    total += static_cast<std::uint32_t>(static_cast<std::int32_t>(ua[i]) *
                                        static_cast<std::int32_t>(sb[i]));
  }
  return std::bit_cast<std::int32_t>(total);
}

} // namespace onnx_light_cpu::detail
