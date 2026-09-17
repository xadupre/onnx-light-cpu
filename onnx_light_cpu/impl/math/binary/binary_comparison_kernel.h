// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include "onnx_light_cpu/impl/math/binary/binary_manifest.h"
#include "onnx_light_cpu/impl/simd_level.h"

namespace onnx_light_cpu {

enum class BinaryComparisonKind : std::uint8_t {
  kEqual,
  kGreater,
  kGreaterOrEqual,
  kLess,
  kLessOrEqual,
};

void BinaryCompareFloat16(const std::uint16_t *left, const std::uint16_t *right, std::uint8_t *out,
                          std::size_t count, BinaryComparisonKind kind, bool left_scalar,
                          bool right_scalar);

#ifdef ONNX_LIGHT_CPU_HAVE_F16C
void BinaryCompareFloat16_F16C(const std::uint16_t *left, const std::uint16_t *right,
                               std::uint8_t *out, std::size_t count, BinaryComparisonKind kind,
                               bool left_scalar, bool right_scalar);
#endif

/// The actual vector path used for a range of this length (kNone means scalar).
SimdLevel BinaryComparisonInt64SimdLevel(std::size_t count);
/// Cached preferred dispatch; use the count overload to account for short ranges.
const char *BinaryCompareInt64Implementation();
const char *BinaryCompareInt64Implementation(std::size_t count);

void BinaryCompareInt64(const std::int64_t *left, const std::int64_t *right, std::uint8_t *out,
                        std::size_t count, BinaryOperator op, bool left_scalar, bool right_scalar);
void BinaryCompareUInt64(const std::uint64_t *left, const std::uint64_t *right, std::uint8_t *out,
                         std::size_t count, BinaryOperator op, bool left_scalar, bool right_scalar);

#ifdef ONNX_LIGHT_CPU_HAVE_AVX2
void BinaryCompareInt64_AVX2(const std::int64_t *left, const std::int64_t *right, std::uint8_t *out,
                             std::size_t count, BinaryOperator op, bool left_scalar,
                             bool right_scalar);
void BinaryCompareUInt64_AVX2(const std::uint64_t *left, const std::uint64_t *right,
                              std::uint8_t *out, std::size_t count, BinaryOperator op,
                              bool left_scalar, bool right_scalar);
#endif

#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
void BinaryCompareInt64_AVX512(const std::int64_t *left, const std::int64_t *right,
                               std::uint8_t *out, std::size_t count, BinaryOperator op,
                               bool left_scalar, bool right_scalar);
void BinaryCompareUInt64_AVX512(const std::uint64_t *left, const std::uint64_t *right,
                                std::uint8_t *out, std::size_t count, BinaryOperator op,
                                bool left_scalar, bool right_scalar);
#endif

namespace detail {

template <typename T> bool CompareInteger64(T left, T right, BinaryOperator op) {
  switch (op) {
  case BinaryOperator::kEqual:
    return left == right;
  case BinaryOperator::kGreater:
    return left > right;
  case BinaryOperator::kGreaterOrEqual:
    return left >= right;
  case BinaryOperator::kLess:
    return left < right;
  case BinaryOperator::kLessOrEqual:
    return left <= right;
  default:
    return false;
  }
}

} // namespace detail

} // namespace onnx_light_cpu
