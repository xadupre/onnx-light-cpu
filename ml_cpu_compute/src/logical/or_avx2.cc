// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "ml_cpu_compute/logical/logical.h"

#include <cstring>
#include <immintrin.h>

namespace ml_cpu_compute {
namespace {

template <bool RightScalar>
void OrBool(const void *left, const void *right, void *output, std::size_t count) {
  if (count == 0) {
    return;
  }
  const auto *a = static_cast<const std::uint8_t *>(left);
  const auto *b = static_cast<const std::uint8_t *>(right);
  auto *out = static_cast<std::uint8_t *>(output);
  if constexpr (RightScalar) {
    if (*b != 0) {
      std::memset(out, 1, count);
      return;
    }
  }
  const __m256i zero = _mm256_setzero_si256();
  const __m256i one = _mm256_set1_epi8(1);
  std::size_t i = 0;
  for (; count - i >= 32; i += 32) {
    __m256i value = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
    if constexpr (!RightScalar) {
      value = _mm256_or_si256(value, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i)));
    }
    const __m256i result = _mm256_andnot_si256(_mm256_cmpeq_epi8(value, zero), one);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i), result);
  }
  for (; i < count; ++i) {
    if constexpr (RightScalar) {
      out[i] = static_cast<std::uint8_t>(a[i] != 0);
    } else {
      out[i] = static_cast<std::uint8_t>((a[i] | b[i]) != 0);
    }
  }
}

} // namespace

void BulkOrBool_AVX2(const void *left, const void *right, void *output, std::size_t count) {
  OrBool<false>(left, right, output, count);
}

void BulkOrBoolLeft_AVX2(const void *left, const void *right, void *output, std::size_t count) {
  OrBool<true>(right, left, output, count);
}

void BulkOrBoolRight_AVX2(const void *left, const void *right, void *output, std::size_t count) {
  OrBool<true>(left, right, output, count);
}

} // namespace ml_cpu_compute
