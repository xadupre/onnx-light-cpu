// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "ml_cpu_compute/math/binary/binary_comparison.h"

#include <bit>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <type_traits>

namespace ml_cpu_compute {
namespace {

template <typename T>
void Compare(const T *left, const T *right, std::uint8_t *out, std::size_t count, BinaryOperator op,
             bool left_scalar, bool right_scalar) {
  if (count == 0) {
    return;
  }
  const auto a_scalar = _mm256_set1_epi64x(left_scalar ? std::bit_cast<long long>(*left) : 0);
  const auto b_scalar = _mm256_set1_epi64x(right_scalar ? std::bit_cast<long long>(*right) : 0);
  const auto bias = _mm256_set1_epi64x(std::numeric_limits<long long>::min());
  std::size_t i = 0;
  for (; count - i >= 4; i += 4) {
    auto a =
        left_scalar ? a_scalar : _mm256_loadu_si256(reinterpret_cast<const __m256i *>(left + i));
    auto b =
        right_scalar ? b_scalar : _mm256_loadu_si256(reinterpret_cast<const __m256i *>(right + i));
    if constexpr (std::is_unsigned_v<T>) {
      // Flipping the sign bit maps unsigned ordering onto signed ordering.
      a = _mm256_xor_si256(a, bias);
      b = _mm256_xor_si256(b, bias);
    }
    __m256i result;
    bool invert = false;
    switch (op) {
    case BinaryOperator::kEqual:
      result = _mm256_cmpeq_epi64(a, b);
      break;
    case BinaryOperator::kGreater:
    case BinaryOperator::kLessOrEqual:
      result = _mm256_cmpgt_epi64(a, b);
      invert = op == BinaryOperator::kLessOrEqual;
      break;
    case BinaryOperator::kLess:
    case BinaryOperator::kGreaterOrEqual:
      result = _mm256_cmpgt_epi64(b, a);
      invert = op == BinaryOperator::kGreaterOrEqual;
      break;
    default:
      result = _mm256_setzero_si256();
      break;
    }
    auto mask = static_cast<std::uint32_t>(_mm256_movemask_pd(_mm256_castsi256_pd(result)));
    mask ^= invert ? 15U : 0U;
    const std::uint32_t bytes = (mask * 0x00204081U) & 0x01010101U;
    std::memcpy(out + i, &bytes, sizeof(bytes));
  }
  for (; i < count; ++i) {
    out[i] = detail::CompareInteger64(left[left_scalar ? 0 : i], right[right_scalar ? 0 : i], op);
  }
}

template <bool OrEqual, bool LeftScalar = false, bool RightScalar = false>
void BulkFloatComparison(const void *left, const void *right, void *out, std::size_t count) {
  if (count == 0) {
    return;
  }
  const auto *a = static_cast<const float *>(left);
  const auto *b = static_cast<const float *>(right);
  auto *output = static_cast<std::uint8_t *>(out);
  const auto scalar_a = _mm256_set1_ps(LeftScalar ? *a : 0.0f);
  const auto scalar_b = _mm256_set1_ps(RightScalar ? *b : 0.0f);
  const auto ones = _mm256_set1_epi8(1);
  const auto order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
  const auto compare = [&](std::size_t i) {
    const auto av = LeftScalar ? scalar_a : _mm256_loadu_ps(a + i);
    const auto bv = RightScalar ? scalar_b : _mm256_loadu_ps(b + i);
    return _mm256_castps_si256(_mm256_cmp_ps(av, bv, OrEqual ? _CMP_GE_OQ : _CMP_GT_OQ));
  };
  std::size_t i = 0;
  for (; count - i >= 32; i += 32) {
    const auto ab = _mm256_packs_epi32(compare(i), compare(i + 8));
    const auto cd = _mm256_packs_epi32(compare(i + 16), compare(i + 24));
    // Packing is lane-local; restore element order before storing 0/1 bytes.
    const auto bytes = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(ab, cd), order);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + i), _mm256_and_si256(bytes, ones));
  }
  for (; count - i >= 8; i += 8) {
    const auto mask = compare(i);
    const auto words =
        _mm_packs_epi32(_mm256_castsi256_si128(mask), _mm256_extracti128_si256(mask, 1));
    const auto bytes = _mm_and_si128(_mm_packs_epi16(words, words), _mm_set1_epi8(1));
    _mm_storel_epi64(reinterpret_cast<__m128i *>(output + i), bytes);
  }
  for (; i < count; ++i) {
    const float av = a[LeftScalar ? 0 : i];
    const float bv = b[RightScalar ? 0 : i];
    output[i] = OrEqual ? av >= bv : av > bv;
  }
}

} // namespace

void BulkFloatGreater_AVX2(const void *left, const void *right, void *out, std::size_t count) {
  BulkFloatComparison<false>(left, right, out, count);
}

void BulkFloatLess_AVX2(const void *left, const void *right, void *out, std::size_t count) {
  BulkFloatComparison<false>(right, left, out, count);
}

void BulkFloatLessLeft_AVX2(const void *left, const void *right, void *out, std::size_t count) {
  BulkFloatComparison<false, false, true>(right, left, out, count);
}

void BulkFloatLessRight_AVX2(const void *left, const void *right, void *out, std::size_t count) {
  BulkFloatComparison<false, true, false>(right, left, out, count);
}

void BulkFloatLessOrEqual_AVX2(const void *left, const void *right, void *out, std::size_t count) {
  BulkFloatComparison<true>(right, left, out, count);
}

void BulkFloatLessOrEqualLeft_AVX2(const void *left, const void *right, void *out,
                                   std::size_t count) {
  BulkFloatComparison<true, false, true>(right, left, out, count);
}

void BulkFloatLessOrEqualRight_AVX2(const void *left, const void *right, void *out,
                                    std::size_t count) {
  BulkFloatComparison<true, true, false>(right, left, out, count);
}

void BulkFloatGreaterOrEqual_AVX2(const void *left, const void *right, void *out,
                                  std::size_t count) {
  BulkFloatComparison<true>(left, right, out, count);
}

void BulkFloatGreaterOrEqualLeft_AVX2(const void *left, const void *right, void *out,
                                      std::size_t count) {
  BulkFloatComparison<true, true>(left, right, out, count);
}

void BulkFloatGreaterOrEqualRight_AVX2(const void *left, const void *right, void *out,
                                       std::size_t count) {
  BulkFloatComparison<true, false, true>(left, right, out, count);
}

void BinaryCompareInt64_AVX2(const std::int64_t *left, const std::int64_t *right, std::uint8_t *out,
                             std::size_t count, BinaryOperator op, bool left_scalar,
                             bool right_scalar) {
  Compare(left, right, out, count, op, left_scalar, right_scalar);
}

void BinaryCompareUInt64_AVX2(const std::uint64_t *left, const std::uint64_t *right,
                              std::uint8_t *out, std::size_t count, BinaryOperator op,
                              bool left_scalar, bool right_scalar) {
  Compare(left, right, out, count, op, left_scalar, right_scalar);
}

} // namespace ml_cpu_compute
