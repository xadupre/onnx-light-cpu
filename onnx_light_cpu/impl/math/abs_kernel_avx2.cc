// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace onnx_light_cpu {

void AbsFloat16_AVX2(const uint16_t *input, uint16_t *output, std::size_t count) {
  constexpr std::uint16_t kFloat16AbsMask = 0x7FFF;
  const __m256i mask = _mm256_set1_epi16(static_cast<short>(kFloat16AbsMask));
  std::size_t i = 0;
  const std::size_t stride = 16;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + i));
    v = _mm256_and_si256(v, mask);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + i), v);
  }
  for (; i < count; ++i) {
    output[i] = static_cast<uint16_t>(input[i] & kFloat16AbsMask);
  }
}

void AbsInt8_AVX2(const int8_t *input, int8_t *output, std::size_t count) {
  std::size_t i = 0;
  const std::size_t stride = 32;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + i));
    v = _mm256_abs_epi8(v);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + i), v);
  }
  for (; i < count; ++i) {
    const int v = static_cast<int>(input[i]);
    output[i] = static_cast<int8_t>(v < 0 ? -v : v);
  }
}

void AbsInt16_AVX2(const int16_t *input, int16_t *output, std::size_t count) {
  std::size_t i = 0;
  constexpr std::size_t stride = 16;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    const __m256i value = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + i));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + i), _mm256_abs_epi16(value));
  }
  for (; i + 8 <= count; i += 8) {
    const __m128i value = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + i));
    const __m128i sign = _mm_srai_epi16(value, 15);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output + i),
                     _mm_sub_epi16(_mm_xor_si128(value, sign), sign));
  }
  for (; i < count; ++i) {
    const int value = input[i];
    output[i] = static_cast<int16_t>(value < 0 ? -value : value);
  }
}

void AbsInt32_AVX2(const int32_t *input, int32_t *output, std::size_t count) {
  std::size_t i = 0;
  const std::size_t stride = 8;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + i));
    v = _mm256_abs_epi32(v);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + i), v);
  }
  for (; i < count; ++i) {
    output[i] = input[i] < 0 ? -input[i] : input[i];
  }
}

void AbsInt64_AVX2(const int64_t *input, int64_t *output, std::size_t count) {
  // AVX2 has no _mm256_abs_epi64; emulate with arithmetic shift + xor + sub.
  std::size_t i = 0;
  const std::size_t stride = 4;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + i));
    __m256i sign = _mm256_srai_epi32(v, 31);
    sign = _mm256_shuffle_epi32(sign, 0xF5);
    __m256i abs_v = _mm256_sub_epi64(_mm256_xor_si256(v, sign), sign);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + i), abs_v);
  }
  for (; i < count; ++i) {
    output[i] = input[i] < 0 ? -input[i] : input[i];
  }
}

} // namespace onnx_light_cpu
