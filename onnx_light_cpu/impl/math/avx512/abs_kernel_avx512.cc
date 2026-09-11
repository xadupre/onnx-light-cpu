// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/math_kernels.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace onnx_light_cpu {

void AbsFloat32_AVX512(const float *input, float *output, std::size_t count) {
  const __m512i sign_mask = _mm512_set1_epi32(0x7FFFFFFF);
  std::size_t i = 0;
  constexpr std::size_t stride = 16;
  const std::size_t aligned_count = count - count % stride;
  for (; i < aligned_count; i += stride) {
    const __m512 value = _mm512_loadu_ps(input + i);
    _mm512_storeu_ps(output + i,
                     _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(value), sign_mask)));
  }
  for (; i < count; ++i) {
    output[i] = std::fabs(input[i]);
  }
}

void AbsFloat32_AVX512Streaming(const float *input, float *output, std::size_t count) {
  constexpr std::size_t stride = 16;
  std::size_t i = 0;
  while (i < count && reinterpret_cast<std::uintptr_t>(output + i) % 64 != 0) {
    output[i] = std::fabs(input[i]);
    ++i;
  }
  const __m512i sign_mask = _mm512_set1_epi32(0x7FFFFFFF);
  const std::size_t streamed_count = i + (count - i) / stride * stride;
  const bool streamed = i < streamed_count;
  for (; i < streamed_count; i += stride) {
    const __m512 value = _mm512_loadu_ps(input + i);
    _mm512_stream_ps(output + i,
                     _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(value), sign_mask)));
  }
  for (; i < count; ++i) {
    output[i] = std::fabs(input[i]);
  }
  if (streamed) {
    _mm_sfence();
  }
}

void AbsFloat64_AVX512(const double *input, double *output, std::size_t count) {
  const __m512i sign_mask = _mm512_set1_epi64(0x7FFFFFFFFFFFFFFF);
  std::size_t i = 0;
  const std::size_t stride = 8;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    const __m512d v = _mm512_loadu_pd(input + i);
    _mm512_storeu_pd(output + i,
                     _mm512_castsi512_pd(_mm512_and_si512(_mm512_castpd_si512(v), sign_mask)));
  }
  for (; i < count; ++i) {
    output[i] = std::fabs(input[i]);
  }
}

void AbsFloat16_AVX512(const uint16_t *input, uint16_t *output, std::size_t count) {
  constexpr std::uint16_t kFloat16AbsMask = 0x7FFF;
  const __m512i mask = _mm512_set1_epi32(0x7FFF7FFF);
  std::size_t i = 0;
  const std::size_t stride = 32;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    __m512i v = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(input + i));
    v = _mm512_and_si512(v, mask);
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(output + i), v);
  }
  for (; i < count; ++i) {
    output[i] = static_cast<uint16_t>(input[i] & kFloat16AbsMask);
  }
}

void AbsInt32_AVX512(const int32_t *input, int32_t *output, std::size_t count) {
  std::size_t i = 0;
  const std::size_t stride = 16;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    __m512i v = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(input + i));
    v = _mm512_abs_epi32(v);
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(output + i), v);
  }
  for (; i < count; ++i) {
    output[i] = input[i] < 0 ? -input[i] : input[i];
  }
}

void AbsInt64_AVX512(const int64_t *input, int64_t *output, std::size_t count) {
  std::size_t i = 0;
  const std::size_t stride = 8;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    __m512i v = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(input + i));
    v = _mm512_abs_epi64(v);
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(output + i), v);
  }
  for (; i < count; ++i) {
    output[i] = input[i] < 0 ? -input[i] : input[i];
  }
}

} // namespace onnx_light_cpu
