// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/binary/binary_comparison_kernel.h"

#include <bit>
#include <cstring>
#include <immintrin.h>
#include <type_traits>

namespace onnx_light_cpu {
namespace {

template <int Predicate, typename T> __mmask8 CompareMask(__m512i a, __m512i b) {
  if constexpr (std::is_unsigned_v<T>) {
    return _mm512_cmp_epu64_mask(a, b, Predicate);
  } else {
    return _mm512_cmp_epi64_mask(a, b, Predicate);
  }
}

template <typename T>
void Compare(const T *left, const T *right, std::uint8_t *out, std::size_t count, BinaryOperator op,
             bool left_scalar, bool right_scalar) {
  if (count == 0) {
    return;
  }
  const auto a_scalar = _mm512_set1_epi64(left_scalar ? std::bit_cast<long long>(*left) : 0);
  const auto b_scalar = _mm512_set1_epi64(right_scalar ? std::bit_cast<long long>(*right) : 0);
  std::size_t i = 0;
  for (; count - i >= 8; i += 8) {
    const auto a = left_scalar ? a_scalar : _mm512_loadu_si512(left + i);
    const auto b = right_scalar ? b_scalar : _mm512_loadu_si512(right + i);
    __mmask8 mask;
    switch (op) {
    case BinaryOperator::kEqual:
      mask = CompareMask<_MM_CMPINT_EQ, T>(a, b);
      break;
    case BinaryOperator::kGreater:
      mask = CompareMask<_MM_CMPINT_GT, T>(a, b);
      break;
    case BinaryOperator::kGreaterOrEqual:
      mask = CompareMask<_MM_CMPINT_GE, T>(a, b);
      break;
    case BinaryOperator::kLess:
      mask = CompareMask<_MM_CMPINT_LT, T>(a, b);
      break;
    case BinaryOperator::kLessOrEqual:
      mask = CompareMask<_MM_CMPINT_LE, T>(a, b);
      break;
    default:
      mask = 0;
      break;
    }
    // Expand to canonical bool bytes without requiring AVX-512BW.
    const auto packed = _mm512_cvtepi64_epi8(_mm512_maskz_set1_epi64(mask, 1));
    std::memcpy(out + i, &packed, 8);
  }
  for (; i < count; ++i) {
    out[i] = detail::CompareInteger64(left[left_scalar ? 0 : i], right[right_scalar ? 0 : i], op);
  }
}

} // namespace

void BinaryCompareInt64_AVX512(const std::int64_t *left, const std::int64_t *right,
                               std::uint8_t *out, std::size_t count, BinaryOperator op,
                               bool left_scalar, bool right_scalar) {
  Compare(left, right, out, count, op, left_scalar, right_scalar);
}

void BinaryCompareUInt64_AVX512(const std::uint64_t *left, const std::uint64_t *right,
                                std::uint8_t *out, std::size_t count, BinaryOperator op,
                                bool left_scalar, bool right_scalar) {
  Compare(left, right, out, count, op, left_scalar, right_scalar);
}

} // namespace onnx_light_cpu
