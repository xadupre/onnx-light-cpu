// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/avx512/exp_kernel_avx512.h"

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

__m512 SwiGLU(__m512 gate, __m512 value, __m512 alpha) {
  const __m512 exponent = detail::ExpFloat32Vector_AVX512(
      _mm512_mul_ps(_mm512_sub_ps(_mm512_setzero_ps(), alpha), gate));
  return _mm512_div_ps(_mm512_mul_ps(gate, value), _mm512_add_ps(_mm512_set1_ps(1.0f), exponent));
}

} // namespace

void SwiGLUBFloat16_AVX512BF16(const std::uint16_t *gate, const std::uint16_t *value,
                               std::uint16_t *output, std::size_t count, float alpha) {
  const __m512 alpha_vector = _mm512_set1_ps(alpha);
  std::size_t index = 0;
  for (; index + 16 <= count; index += 16) {
    const __m512 gate_vector = BFloat16ToFloat32(gate + index);
    const __m512 value_vector = BFloat16ToFloat32(value + index);
    StoreBFloat16(SwiGLU(gate_vector, value_vector, alpha_vector), output + index);
  }
  if (index < count) {
    alignas(32) std::uint16_t gate_tail[16]{};
    alignas(32) std::uint16_t value_tail[16]{};
    alignas(32) std::uint16_t output_tail[16];
    const std::size_t tail_bytes = (count - index) * sizeof(std::uint16_t);
    std::memcpy(gate_tail, gate + index, tail_bytes);
    std::memcpy(value_tail, value + index, tail_bytes);
    StoreBFloat16(SwiGLU(BFloat16ToFloat32(gate_tail), BFloat16ToFloat32(value_tail), alpha_vector),
                  output_tail);
    std::memcpy(output + index, output_tail, tail_bytes);
  }
}

} // namespace onnx_light_cpu
