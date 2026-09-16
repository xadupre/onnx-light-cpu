// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/tensor/cast_simd.h"

#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <cstring>
#include <immintrin.h>

namespace onnx_light_cpu::detail {

std::size_t CastFloat32ToFloat16_F16C(const std::uint8_t *src, std::uint8_t *dst,
                                      std::size_t count) {
  // VCVTPS2PH ignores _MM_FROUND_NO_EXC. Match the integer-based codec even
  // when the caller unmasks floating-point exceptions, and preserve its flags.
  const unsigned int mxcsr = _mm_getcsr();
  _mm_setcsr(mxcsr | _MM_MASK_MASK);
  std::size_t i = 0;
  for (; count - i >= 8; i += 8) {
    __m256 value;
    std::memcpy(&value, src + i * 4, sizeof(value));
    const __m128i low = _mm_castps_si128(_mm256_castps256_ps128(value));
    const __m128i high = _mm_castps_si128(_mm256_extractf128_ps(value, 1));
    const __m128i mask = _mm_set1_epi32(0x7fffffff);
    const __m128i inf = _mm_set1_epi32(0x7f800000);
    const __m128i nan = _mm_or_si128(_mm_cmpgt_epi32(_mm_and_si128(low, mask), inf),
                                     _mm_cmpgt_epi32(_mm_and_si128(high, mask), inf));
    // F16C preserves NaN payloads and quiets signaling NaNs. Cast instead
    // canonicalizes all float32 NaNs, without raising an invalid exception.
    if (_mm_movemask_epi8(nan) != 0) {
      for (std::size_t j = i; j < i + 8; ++j) {
        float scalar;
        std::memcpy(&scalar, src + j * 4, sizeof(scalar));
        const auto half = FloatToFloat16Bits(scalar);
        std::memcpy(dst + j * 2, &half, sizeof(half));
      }
    } else {
      const __m128i half = _mm256_cvtps_ph(value, _MM_FROUND_TO_NEAREST_INT);
      std::memcpy(dst + i * 2, &half, sizeof(half));
    }
  }
  _mm_setcsr(mxcsr);
  return i;
}

std::size_t CastFloat16ToFloat32_F16C(const std::uint8_t *src, std::uint8_t *dst,
                                      std::size_t count) {
  std::size_t i = 0;
  for (; count - i >= 8; i += 8) {
    __m128i half;
    std::memcpy(&half, src + i * 2, sizeof(half));
    const __m128i nan =
        _mm_cmpgt_epi16(_mm_and_si128(half, _mm_set1_epi16(0x7fff)), _mm_set1_epi16(0x7c00));
    // Preserve signaling NaN bits exactly, as the scalar decoder does.
    if (_mm_movemask_epi8(nan) != 0) {
      for (std::size_t j = i; j < i + 8; ++j) {
        std::uint16_t scalar;
        std::memcpy(&scalar, src + j * 2, sizeof(scalar));
        const float value = Float16BitsToFloat(scalar);
        std::memcpy(dst + j * 4, &value, sizeof(value));
      }
    } else {
      const __m256 value = _mm256_cvtph_ps(half);
      std::memcpy(dst + i * 4, &value, sizeof(value));
    }
  }
  return i;
}

} // namespace onnx_light_cpu::detail
