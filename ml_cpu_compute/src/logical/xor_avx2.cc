// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "ml_cpu_compute/logical/logical.h"

#include <immintrin.h>

namespace ml_cpu_compute {
namespace {

template <bool LeftScalar, bool RightScalar>
void XorBool(const void *left, const void *right, void *output, std::size_t count) {
  if (count == 0) {
    return;
  }
  const auto *a = static_cast<const std::uint8_t *>(left);
  const auto *b = static_cast<const std::uint8_t *>(right);
  auto *out = static_cast<std::uint8_t *>(output);
  const bool left_value = LeftScalar && *a != 0;
  const bool right_value = RightScalar && *b != 0;
  const __m256i zero = _mm256_setzero_si256();
  const __m256i one = _mm256_set1_epi8(1);
  const __m256i left_scalar = _mm256_set1_epi8(left_value ? 0 : -1);
  const __m256i right_scalar = _mm256_set1_epi8(right_value ? 0 : -1);
  std::size_t i = 0;
  for (; count - i >= 32; i += 32) {
    const __m256i left_zero =
        LeftScalar
            ? left_scalar
            : _mm256_cmpeq_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i)), zero);
    const __m256i right_zero =
        RightScalar
            ? right_scalar
            : _mm256_cmpeq_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i)), zero);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i),
                        _mm256_and_si256(_mm256_xor_si256(left_zero, right_zero), one));
  }
  for (; i < count; ++i) {
    const bool lhs = LeftScalar ? left_value : a[i] != 0;
    const bool rhs = RightScalar ? right_value : b[i] != 0;
    out[i] = static_cast<std::uint8_t>(lhs != rhs);
  }
}

} // namespace

void BulkXorBool_AVX2(const void *left, const void *right, void *output, std::size_t count) {
  XorBool<false, false>(left, right, output, count);
}

void BulkXorBoolLeft_AVX2(const void *left, const void *right, void *output, std::size_t count) {
  XorBool<true, false>(left, right, output, count);
}

void BulkXorBoolRight_AVX2(const void *left, const void *right, void *output, std::size_t count) {
  XorBool<false, true>(left, right, output, count);
}

} // namespace ml_cpu_compute
