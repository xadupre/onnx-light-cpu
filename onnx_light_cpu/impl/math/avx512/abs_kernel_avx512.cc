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

} // namespace onnx_light_cpu
