// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace onnx_light_cpu {

void AbsFloat32_AVX(const float *input, float *output, std::size_t count) {
  // Clear sign bit: AND with 0x7FFFFFFF using 256-bit registers.
  const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
  std::size_t i = 0;
  const std::size_t stride = 8;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    __m256 v = _mm256_loadu_ps(input + i);
    v = _mm256_and_ps(v, sign_mask);
    _mm256_storeu_ps(output + i, v);
  }
  for (; i < count; ++i) {
    output[i] = std::fabs(input[i]);
  }
}

void AbsFloat32_AVXStreaming(const float *input, float *output, std::size_t count) {
  std::size_t i = 0;
  while (i < count && reinterpret_cast<std::uintptr_t>(output + i) % 64 != 0) {
    output[i] = std::fabs(input[i]);
    ++i;
  }
  const __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
  const std::size_t end = i + (count - i) / 16 * 16;
  const bool streamed = i < end;
  for (; i < end; i += 16) {
    const __m256 v0 = _mm256_and_ps(_mm256_loadu_ps(input + i), mask);
    const __m256 v1 = _mm256_and_ps(_mm256_loadu_ps(input + i + 8), mask);
    _mm256_stream_ps(output + i, v0);
    _mm256_stream_ps(output + i + 8, v1);
  }
  for (; i < count; ++i) {
    output[i] = std::fabs(input[i]);
  }
  if (streamed) {
    _mm_sfence();
  }
}

void AbsFloat64_AVX(const double *input, double *output, std::size_t count) {
  const __m256d sign_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
  std::size_t i = 0;
  const std::size_t stride = 4;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    __m256d v = _mm256_loadu_pd(input + i);
    v = _mm256_and_pd(v, sign_mask);
    _mm256_storeu_pd(output + i, v);
  }
  for (; i < count; ++i) {
    output[i] = std::fabs(input[i]);
  }
}

} // namespace onnx_light_cpu
