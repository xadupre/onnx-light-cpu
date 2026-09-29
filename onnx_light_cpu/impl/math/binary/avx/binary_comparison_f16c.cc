// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/binary/binary_comparison_kernel.h"

#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <array>
#include <cstring>
#include <immintrin.h>

namespace onnx_light_cpu {
namespace {

constexpr std::array<std::uint64_t, 256> MakeExpandedMasks() {
  std::array<std::uint64_t, 256> masks{};
  for (std::size_t mask = 0; mask < masks.size(); ++mask) {
    for (std::size_t lane = 0; lane < 8; ++lane) {
      masks[mask] |= static_cast<std::uint64_t>((mask >> lane) & 1U) << (lane * 8U);
    }
  }
  return masks;
}

constexpr auto kExpandedMasks = MakeExpandedMasks();

template <int Predicate>
void CompareFloat16F16C(const std::uint16_t *left, const std::uint16_t *right, std::uint8_t *out,
                        std::size_t count, bool left_scalar, bool right_scalar) {
  const __m256 scalar_left =
      left_scalar ? _mm256_set1_ps(detail::Float16BitsToFloat(*left)) : _mm256_setzero_ps();
  const __m256 scalar_right =
      right_scalar ? _mm256_set1_ps(detail::Float16BitsToFloat(*right)) : _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    const __m256 a =
        left_scalar ? scalar_left
                    : _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(left + i)));
    const __m256 b =
        right_scalar
            ? scalar_right
            : _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(right + i)));
    const unsigned mask = static_cast<unsigned>(_mm256_movemask_ps(_mm256_cmp_ps(a, b, Predicate)));
    std::memcpy(out + i, &kExpandedMasks[mask], sizeof(std::uint64_t));
  }

  const float scalar_a = left_scalar ? detail::Float16BitsToFloat(*left) : 0.0f;
  const float scalar_b = right_scalar ? detail::Float16BitsToFloat(*right) : 0.0f;
  for (; i < count; ++i) {
    const float a = left_scalar ? scalar_a : detail::Float16BitsToFloat(left[i]);
    const float b = right_scalar ? scalar_b : detail::Float16BitsToFloat(right[i]);
    if constexpr (Predicate == _CMP_EQ_OQ) {
      out[i] = a == b ? 1U : 0U;
    } else if constexpr (Predicate == _CMP_GT_OQ) {
      out[i] = a > b ? 1U : 0U;
    } else if constexpr (Predicate == _CMP_GE_OQ) {
      out[i] = a >= b ? 1U : 0U;
    } else if constexpr (Predicate == _CMP_LT_OQ) {
      out[i] = a < b ? 1U : 0U;
    } else {
      out[i] = a <= b ? 1U : 0U;
    }
  }
}

} // namespace

void BinaryCompareFloat16_F16C(const std::uint16_t *left, const std::uint16_t *right,
                               std::uint8_t *out, std::size_t count, BinaryComparisonKind kind,
                               bool left_scalar, bool right_scalar) {
  switch (kind) {
  case BinaryComparisonKind::kEqual:
    CompareFloat16F16C<_CMP_EQ_OQ>(left, right, out, count, left_scalar, right_scalar);
    break;
  case BinaryComparisonKind::kGreater:
    CompareFloat16F16C<_CMP_GT_OQ>(left, right, out, count, left_scalar, right_scalar);
    break;
  case BinaryComparisonKind::kGreaterOrEqual:
    CompareFloat16F16C<_CMP_GE_OQ>(left, right, out, count, left_scalar, right_scalar);
    break;
  case BinaryComparisonKind::kLess:
    CompareFloat16F16C<_CMP_LT_OQ>(left, right, out, count, left_scalar, right_scalar);
    break;
  case BinaryComparisonKind::kLessOrEqual:
    CompareFloat16F16C<_CMP_LE_OQ>(left, right, out, count, left_scalar, right_scalar);
    break;
  }
}

} // namespace onnx_light_cpu
