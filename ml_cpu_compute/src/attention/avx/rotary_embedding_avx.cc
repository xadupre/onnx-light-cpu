// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "ml_cpu_compute/attention/rotary_embedding.h"

#include <immintrin.h>

namespace ml_cpu_compute {

std::size_t RotaryEmbeddingSplitHalf_AVX(const void *input, const void *cos, const void *sin,
                                         void *output, std::size_t half) {
  const auto *x = static_cast<const float *>(input);
  const auto *c = static_cast<const float *>(cos);
  const auto *s = static_cast<const float *>(sin);
  auto *y = static_cast<float *>(output);
  const std::size_t vector_end = half - half % 8;
  std::size_t i = 0;
  for (; i < vector_end; i += 8) {
    const auto a = _mm256_loadu_ps(x + i);
    const auto b = _mm256_loadu_ps(x + half + i);
    const auto cv = _mm256_loadu_ps(c + i);
    const auto sv = _mm256_loadu_ps(s + i);
    _mm256_storeu_ps(y + i, _mm256_sub_ps(_mm256_mul_ps(cv, a), _mm256_mul_ps(sv, b)));
    _mm256_storeu_ps(y + half + i, _mm256_add_ps(_mm256_mul_ps(sv, a), _mm256_mul_ps(cv, b)));
  }
  return i;
}

} // namespace ml_cpu_compute
