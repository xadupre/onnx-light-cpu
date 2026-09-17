// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/tensor/cast_simd.h"

#include <array>
#include <cstring>
#include <immintrin.h>

namespace onnx_light_cpu::detail {
namespace {

bool Float32LanesFitInt32(__m256 value) {
  const __m256 lower = _mm256_cmp_ps(value, _mm256_set1_ps(-0x1p31f), _CMP_GE_OQ);
  const __m256 upper = _mm256_cmp_ps(value, _mm256_set1_ps(0x1p31f), _CMP_LT_OQ);
  return _mm256_movemask_ps(_mm256_and_ps(lower, upper)) == 0xff;
}

bool Float32LanesFitInt64(__m256 value) {
  const __m256 lower = _mm256_cmp_ps(value, _mm256_set1_ps(-0x1p63f), _CMP_GE_OQ);
  const __m256 upper = _mm256_cmp_ps(value, _mm256_set1_ps(0x1p63f), _CMP_LT_OQ);
  return _mm256_movemask_ps(_mm256_and_ps(lower, upper)) == 0xff;
}

__m128i PackLowBytes(__m256i value) {
  const __m128i words =
      _mm_packus_epi32(_mm256_castsi256_si128(value), _mm256_extracti128_si256(value, 1));
  return _mm_packus_epi16(words, _mm_setzero_si128());
}

__m128i PackBoolBytes(__m256i first, __m256i second) {
  const __m256i interleaved = _mm256_packs_epi32(first, second);
  const __m256i ordered = _mm256_permute4x64_epi64(interleaved, _MM_SHUFFLE(3, 1, 2, 0));
  return _mm_packs_epi16(_mm256_castsi256_si128(ordered), _mm256_extracti128_si256(ordered, 1));
}

} // namespace

std::size_t CastFloat32ToInt32_AVX2(const std::uint8_t *src, std::uint8_t *dst, std::size_t count) {
  std::size_t i = 0;
  for (; count - i >= 8; i += 8) {
    __m256 value;
    std::memcpy(&value, src + i * 4, sizeof(value));
    if (!Float32LanesFitInt32(value)) {
      break;
    }
    const __m256i converted = _mm256_cvttps_epi32(value);
    std::memcpy(dst + i * 4, &converted, sizeof(converted));
  }
  return i;
}

std::size_t CastFloat32ToInt64_AVX2(const std::uint8_t *src, std::uint8_t *dst, std::size_t count) {
  std::size_t i = 0;
  for (; count - i >= 8; i += 8) {
    __m256 value;
    std::memcpy(&value, src + i * 4, sizeof(value));
    if (!Float32LanesFitInt64(value)) {
      break;
    }
    alignas(32) std::array<float, 8> lanes;
    std::memcpy(lanes.data(), &value, sizeof(value));
    for (std::size_t lane = 0; lane < lanes.size(); ++lane) {
      const std::int64_t converted = static_cast<std::int64_t>(lanes[lane]);
      std::memcpy(dst + (i + lane) * 8, &converted, sizeof(converted));
    }
  }
  return i;
}

std::size_t CastFloat32ToInt8_AVX2(const std::uint8_t *src, std::uint8_t *dst, std::size_t count) {
  std::size_t i = 0;
  for (; count - i >= 8; i += 8) {
    __m256 value;
    std::memcpy(&value, src + i * 4, sizeof(value));
    if (!Float32LanesFitInt32(value)) {
      break;
    }
    const __m256i converted = _mm256_and_si256(_mm256_cvttps_epi32(value), _mm256_set1_epi32(0xff));
    const __m128i packed = PackLowBytes(converted);
    std::memcpy(dst + i, &packed, 8);
  }
  return i;
}

std::size_t CastFloat32ToUint8_AVX2(const std::uint8_t *src, std::uint8_t *dst, std::size_t count) {
  return CastFloat32ToInt8_AVX2(src, dst, count);
}

std::size_t CastFloat32ToBool_AVX2(const std::uint8_t *src, std::uint8_t *dst, std::size_t count) {
  std::size_t i = 0;
  const __m256 zero = _mm256_setzero_ps();
  const __m256i one = _mm256_set1_epi32(1);
  for (; count - i >= 16; i += 16) {
    __m256 first;
    __m256 second;
    std::memcpy(&first, src + i * 4, sizeof(first));
    std::memcpy(&second, src + (i + 8) * 4, sizeof(second));
    const __m256i first_nonzero =
        _mm256_and_si256(_mm256_castps_si256(_mm256_cmp_ps(first, zero, _CMP_NEQ_UQ)), one);
    const __m256i second_nonzero =
        _mm256_and_si256(_mm256_castps_si256(_mm256_cmp_ps(second, zero, _CMP_NEQ_UQ)), one);
    const __m128i packed = PackBoolBytes(first_nonzero, second_nonzero);
    std::memcpy(dst + i, &packed, sizeof(packed));
  }
  for (; count - i >= 8; i += 8) {
    __m256 value;
    std::memcpy(&value, src + i * 4, sizeof(value));
    const __m256 nonzero = _mm256_cmp_ps(value, zero, _CMP_NEQ_UQ);
    const __m256i converted = _mm256_and_si256(_mm256_castps_si256(nonzero), one);
    const __m128i packed = PackLowBytes(converted);
    std::memcpy(dst + i, &packed, 8);
  }
  return i;
}

std::size_t CastBoolToFloat32_AVX2(const std::uint8_t *src, std::uint8_t *dst, std::size_t count) {
  std::size_t i = 0;
  const __m256i zero = _mm256_setzero_si256();
  const __m256i one = _mm256_set1_epi32(1);
  for (; count - i >= 8; i += 8) {
    __m128i bytes = _mm_setzero_si128();
    std::memcpy(&bytes, src + i, 8);
    const __m256i values = _mm256_cvtepu8_epi32(bytes);
    const __m256 converted =
        _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_cmpgt_epi32(values, zero), one));
    std::memcpy(dst + i * 4, &converted, sizeof(converted));
  }
  return i;
}

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
