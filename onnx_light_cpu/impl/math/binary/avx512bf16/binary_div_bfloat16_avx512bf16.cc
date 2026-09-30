// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/binary/binary_arithmetic_kernel.h"

#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <immintrin.h>

namespace onnx_light_cpu {
namespace {

__m512 LoadBFloat16(const std::uint16_t *source) {
  const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(source));
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(packed), 16));
}

void StoreBFloat16(__m512 value, std::uint16_t *destination) {
  const __m512i bits = _mm512_castps_si512(value);
  const __m512i upper = _mm512_srli_epi32(bits, 16);
  const __m512i rounding =
      _mm512_add_epi32(_mm512_set1_epi32(0x7fff), _mm512_and_si512(upper, _mm512_set1_epi32(1)));
  const __m512i rounded = _mm512_srli_epi32(_mm512_add_epi32(bits, rounding), 16);
  const __mmask16 is_nan =
      _mm512_cmp_epu32_mask(_mm512_and_si512(bits, _mm512_set1_epi32(0x7fffffff)),
                            _mm512_set1_epi32(0x7f800000), _MM_CMPINT_GT);
  const __m512i narrowed =
      _mm512_mask_mov_epi32(rounded, is_nan, _mm512_or_si512(upper, _mm512_set1_epi32(0x40)));
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(destination), _mm512_cvtusepi32_epi16(narrowed));
}

template <bool LeftScalar, bool RightScalar>
void BinaryDivBFloat16(const std::uint16_t *left, const std::uint16_t *right, std::uint16_t *out,
                       std::size_t count) {
  const float scalar_left = LeftScalar ? detail::Bfloat16BitsToFloat(*left) : 0.0f;
  const float scalar_right = RightScalar ? detail::Bfloat16BitsToFloat(*right) : 0.0f;
  const __m512 vector_left = _mm512_set1_ps(scalar_left);
  const __m512 vector_right = _mm512_set1_ps(scalar_right);
  std::size_t i = 0;
  for (; i + 16 <= count; i += 16) {
    const __m512 a = LeftScalar ? vector_left : LoadBFloat16(left + i);
    const __m512 b = RightScalar ? vector_right : LoadBFloat16(right + i);
    StoreBFloat16(_mm512_div_ps(a, b), out + i);
  }
  for (; i < count; ++i) {
    const float a = LeftScalar ? scalar_left : detail::Bfloat16BitsToFloat(left[i]);
    const float b = RightScalar ? scalar_right : detail::Bfloat16BitsToFloat(right[i]);
    out[i] = detail::FloatToBFloat16Bits(a / b);
  }
}

} // namespace

void BinaryDivBFloat16_AVX512BF16(const std::uint16_t *left, const std::uint16_t *right,
                                  std::uint16_t *out, std::size_t count) {
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
  if (count < 16) {
    BinaryDivBFloat16_AVX2_FMA(left, right, out, count);
    return;
  }
#endif
  BinaryDivBFloat16<false, false>(left, right, out, count);
}

void BinaryDivBFloat16Left_AVX512BF16(std::uint16_t left, const std::uint16_t *right,
                                      std::uint16_t *out, std::size_t count) {
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
  if (count < 16) {
    BinaryDivBFloat16Left_AVX2_FMA(left, right, out, count);
    return;
  }
#endif
  BinaryDivBFloat16<true, false>(&left, right, out, count);
}

void BinaryDivBFloat16Right_AVX512BF16(const std::uint16_t *left, std::uint16_t right,
                                       std::uint16_t *out, std::size_t count) {
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
  if (count < 16) {
    BinaryDivBFloat16Right_AVX2_FMA(left, right, out, count);
    return;
  }
#endif
  BinaryDivBFloat16<false, true>(left, &right, out, count);
}

} // namespace onnx_light_cpu
