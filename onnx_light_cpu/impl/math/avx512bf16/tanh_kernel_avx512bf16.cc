// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/avx512/tanh_kernel_avx512.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace onnx_light_cpu {
namespace {

__m512 BFloat16ToFloat32(const std::uint16_t *values) {
  const __m256i halves = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(values));
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(halves), 16));
}

void StoreBFloat16(__m512 values, std::uint16_t *output) {
  const __m512i bits = _mm512_castps_si512(values);
  const __m512i magnitude = _mm512_and_si512(bits, _mm512_set1_epi32(0x7fffffff));
  const __mmask16 is_nan =
      _mm512_cmp_epu32_mask(magnitude, _mm512_set1_epi32(0x7f800000), _MM_CMPINT_GT);
  const __m512 quiet_values =
      _mm512_castsi512_ps(_mm512_mask_or_epi32(bits, is_nan, bits, _mm512_set1_epi32(0x00400000)));
  const __m256bh converted = _mm512_cvtneps_pbh(quiet_values);
  std::memcpy(output, &converted, sizeof(converted));
  unsigned subnormal =
      _mm512_cmp_epu32_mask(magnitude, _mm512_setzero_si512(), _MM_CMPINT_NE) &
      _mm512_cmp_epu32_mask(magnitude, _mm512_set1_epi32(0x00800000), _MM_CMPINT_LT);
  if (subnormal != 0) {
    alignas(64) std::uint32_t value_bits[16];
    _mm512_store_si512(value_bits, bits);
    while (subnormal != 0) {
      const unsigned lane = std::countr_zero(subnormal);
      const std::uint32_t lane_bits = value_bits[lane];
      output[lane] =
          static_cast<std::uint16_t>((lane_bits + 0x7fffu + ((lane_bits >> 16) & 1u)) >> 16);
      subnormal &= subnormal - 1;
    }
  }
}

} // namespace

void TanhBFloat16_AVX512BF16(const std::uint16_t *input, std::uint16_t *output, std::size_t count) {
  std::size_t index = 0;
  for (; index + 16 <= count; index += 16) {
    const __m512 values = BFloat16ToFloat32(input + index);
    StoreBFloat16(detail::TanhFloat32Vector_AVX512(values), output + index);
  }
  if (index < count) {
    alignas(32) std::uint16_t input_tail[16]{};
    alignas(32) std::uint16_t output_tail[16];
    const std::size_t tail_bytes = (count - index) * sizeof(std::uint16_t);
    std::memcpy(input_tail, input + index, tail_bytes);
    StoreBFloat16(detail::TanhFloat32Vector_AVX512(BFloat16ToFloat32(input_tail)), output_tail);
    std::memcpy(output + index, output_tail, tail_bytes);
  }
}

} // namespace onnx_light_cpu
