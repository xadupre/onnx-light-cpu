// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/binary/binary_comparison_kernel.h"

#include <bit>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <type_traits>

namespace onnx_light_cpu {
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

} // namespace

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

} // namespace onnx_light_cpu
