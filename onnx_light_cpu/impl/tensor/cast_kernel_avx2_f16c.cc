// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/tensor/cast_simd.h"

#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <cstring>
#include <immintrin.h>

namespace onnx_light_cpu::detail {
namespace {

__m256i HalfNanLanes(__m256i half) {
  return _mm256_cmpgt_epi16(_mm256_and_si256(half, _mm256_set1_epi16(0x7fff)),
                            _mm256_set1_epi16(0x7c00));
}

__m128i HalfNanLanes(__m128i half) {
  return _mm_cmpgt_epi16(_mm_and_si128(half, _mm_set1_epi16(0x7fff)), _mm_set1_epi16(0x7c00));
}

void StoreCanonicalHalf(__m128i half, std::uint8_t *dst) {
  const __m128i nan = HalfNanLanes(half);
  if (_mm_movemask_epi8(nan) != 0) {
    const __m128i canonical =
        _mm_or_si128(_mm_and_si128(half, _mm_set1_epi16(-32768)), _mm_set1_epi16(0x7e00));
    half = _mm_or_si128(_mm_and_si128(nan, canonical), _mm_andnot_si128(nan, half));
  }
  std::memcpy(dst, &half, sizeof(half));
}

} // namespace

std::size_t CastFloat32ToFloat16_AVX2_F16C(const std::uint8_t *src, std::uint8_t *dst,
                                           std::size_t count) {
  const unsigned int mxcsr = _mm_getcsr();
  _mm_setcsr(mxcsr | _MM_MASK_MASK);
  std::size_t i = 0;
  for (; count - i >= 32; i += 32) {
    __m256 value0;
    __m256 value1;
    __m256 value2;
    __m256 value3;
    std::memcpy(&value0, src + (i + 0) * sizeof(float), sizeof(value0));
    std::memcpy(&value1, src + (i + 8) * sizeof(float), sizeof(value1));
    std::memcpy(&value2, src + (i + 16) * sizeof(float), sizeof(value2));
    std::memcpy(&value3, src + (i + 24) * sizeof(float), sizeof(value3));
    const __m128i half0 = _mm256_cvtps_ph(value0, _MM_FROUND_TO_NEAREST_INT);
    const __m128i half1 = _mm256_cvtps_ph(value1, _MM_FROUND_TO_NEAREST_INT);
    const __m128i half2 = _mm256_cvtps_ph(value2, _MM_FROUND_TO_NEAREST_INT);
    const __m128i half3 = _mm256_cvtps_ph(value3, _MM_FROUND_TO_NEAREST_INT);
    const __m256i halves01 = _mm256_set_m128i(half1, half0);
    const __m256i halves23 = _mm256_set_m128i(half3, half2);
    const __m256i any_nan = _mm256_or_si256(HalfNanLanes(halves01), HalfNanLanes(halves23));
    if (_mm256_movemask_epi8(any_nan) == 0) {
      std::memcpy(dst + (i + 0) * sizeof(std::uint16_t), &halves01, sizeof(halves01));
      std::memcpy(dst + (i + 16) * sizeof(std::uint16_t), &halves23, sizeof(halves23));
    } else {
      StoreCanonicalHalf(half0, dst + (i + 0) * sizeof(std::uint16_t));
      StoreCanonicalHalf(half1, dst + (i + 8) * sizeof(std::uint16_t));
      StoreCanonicalHalf(half2, dst + (i + 16) * sizeof(std::uint16_t));
      StoreCanonicalHalf(half3, dst + (i + 24) * sizeof(std::uint16_t));
    }
  }
  for (; count - i >= 8; i += 8) {
    __m256 value;
    std::memcpy(&value, src + i * sizeof(float), sizeof(value));
    const __m128i half = _mm256_cvtps_ph(value, _MM_FROUND_TO_NEAREST_INT);
    StoreCanonicalHalf(half, dst + i * sizeof(std::uint16_t));
  }
  _mm_setcsr(mxcsr);
  return i;
}

std::size_t CastFloat16ToFloat32_AVX2_F16C(const std::uint8_t *src, std::uint8_t *dst,
                                           std::size_t count) {
  std::size_t i = 0;
  for (; count - i >= 64; i += 64) {
    __m256i half0;
    __m256i half1;
    __m256i half2;
    __m256i half3;
    std::memcpy(&half0, src + (i + 0) * sizeof(std::uint16_t), sizeof(half0));
    std::memcpy(&half1, src + (i + 16) * sizeof(std::uint16_t), sizeof(half1));
    std::memcpy(&half2, src + (i + 32) * sizeof(std::uint16_t), sizeof(half2));
    std::memcpy(&half3, src + (i + 48) * sizeof(std::uint16_t), sizeof(half3));
    const __m256i any_nan =
        _mm256_or_si256(_mm256_or_si256(HalfNanLanes(half0), HalfNanLanes(half1)),
                        _mm256_or_si256(HalfNanLanes(half2), HalfNanLanes(half3)));
    if (_mm256_movemask_epi8(any_nan) != 0) {
      for (std::size_t j = 0; j < 64; ++j) {
        std::uint16_t scalar;
        std::memcpy(&scalar, src + (i + j) * sizeof(scalar), sizeof(scalar));
        const float converted = Float16BitsToFloat(scalar);
        std::memcpy(dst + (i + j) * sizeof(converted), &converted, sizeof(converted));
      }
      continue;
    }
    const __m256 value0 = _mm256_cvtph_ps(_mm256_castsi256_si128(half0));
    const __m256 value1 = _mm256_cvtph_ps(_mm256_extracti128_si256(half0, 1));
    const __m256 value2 = _mm256_cvtph_ps(_mm256_castsi256_si128(half1));
    const __m256 value3 = _mm256_cvtph_ps(_mm256_extracti128_si256(half1, 1));
    const __m256 value4 = _mm256_cvtph_ps(_mm256_castsi256_si128(half2));
    const __m256 value5 = _mm256_cvtph_ps(_mm256_extracti128_si256(half2, 1));
    const __m256 value6 = _mm256_cvtph_ps(_mm256_castsi256_si128(half3));
    const __m256 value7 = _mm256_cvtph_ps(_mm256_extracti128_si256(half3, 1));
    std::memcpy(dst + (i + 0) * sizeof(float), &value0, sizeof(value0));
    std::memcpy(dst + (i + 8) * sizeof(float), &value1, sizeof(value1));
    std::memcpy(dst + (i + 16) * sizeof(float), &value2, sizeof(value2));
    std::memcpy(dst + (i + 24) * sizeof(float), &value3, sizeof(value3));
    std::memcpy(dst + (i + 32) * sizeof(float), &value4, sizeof(value4));
    std::memcpy(dst + (i + 40) * sizeof(float), &value5, sizeof(value5));
    std::memcpy(dst + (i + 48) * sizeof(float), &value6, sizeof(value6));
    std::memcpy(dst + (i + 56) * sizeof(float), &value7, sizeof(value7));
  }
  for (; count - i >= 16; i += 16) {
    __m256i half;
    std::memcpy(&half, src + i * sizeof(std::uint16_t), sizeof(half));
    if (_mm256_movemask_epi8(HalfNanLanes(half)) != 0) {
      for (std::size_t j = 0; j < 16; ++j) {
        std::uint16_t scalar;
        std::memcpy(&scalar, src + (i + j) * sizeof(scalar), sizeof(scalar));
        const float converted = Float16BitsToFloat(scalar);
        std::memcpy(dst + (i + j) * sizeof(converted), &converted, sizeof(converted));
      }
      continue;
    }
    const __m256 value0 = _mm256_cvtph_ps(_mm256_castsi256_si128(half));
    const __m256 value1 = _mm256_cvtph_ps(_mm256_extracti128_si256(half, 1));
    std::memcpy(dst + (i + 0) * sizeof(float), &value0, sizeof(value0));
    std::memcpy(dst + (i + 8) * sizeof(float), &value1, sizeof(value1));
  }
  return i;
}

} // namespace onnx_light_cpu::detail
