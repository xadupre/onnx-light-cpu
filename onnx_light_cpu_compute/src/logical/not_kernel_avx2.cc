// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace onnx_light_cpu {

void NotBool_AVX2(const uint8_t *input, uint8_t *output, std::size_t count) {
  const __m256i zero = _mm256_setzero_si256();
  const __m256i one = _mm256_set1_epi8(1);
  std::size_t i = 0;
  const std::size_t stride = 32;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + i));
    __m256i eq = _mm256_cmpeq_epi8(v, zero);
    v = _mm256_and_si256(eq, one);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + i), v);
  }
  for (; i < count; ++i) {
    output[i] = static_cast<uint8_t>(input[i] == 0 ? 1 : 0);
  }
}

} // namespace onnx_light_cpu
