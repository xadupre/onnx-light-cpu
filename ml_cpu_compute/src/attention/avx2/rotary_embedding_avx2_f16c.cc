// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "ml_cpu_compute/attention/rotary_embedding.h"

#include <cstdint>
#include <immintrin.h>

namespace ml_cpu_compute {

std::size_t RotaryEmbeddingSplitHalf_AVX2_F16C(const void *input, const void *cos, const void *sin,
                                               void *output, std::size_t half) {
  const auto *x = static_cast<const std::uint16_t *>(input);
  const auto *c = static_cast<const std::uint16_t *>(cos);
  const auto *s = static_cast<const std::uint16_t *>(sin);
  auto *y = static_cast<std::uint16_t *>(output);
  std::size_t i = 0;
  for (; half - i >= 8; i += 8) {
    const auto a = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(x + i)));
    const auto b =
        _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(x + half + i)));
    const auto cv = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(c + i)));
    const auto sv = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(s + i)));
    const auto real = _mm256_sub_ps(_mm256_mul_ps(cv, a), _mm256_mul_ps(sv, b));
    const auto imag = _mm256_add_ps(_mm256_mul_ps(sv, a), _mm256_mul_ps(cv, b));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(y + i),
                     _mm256_cvtps_ph(real, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(y + half + i),
                     _mm256_cvtps_ph(imag, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }
  return i;
}

} // namespace ml_cpu_compute
