// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/binary/binary_comparison_kernel.h"

#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/impl/simd_level.h"

namespace onnx_light_cpu {
namespace {

bool Compare(float left, float right, BinaryComparisonKind kind) {
  switch (kind) {
  case BinaryComparisonKind::kEqual:
    return left == right;
  case BinaryComparisonKind::kGreater:
    return left > right;
  case BinaryComparisonKind::kGreaterOrEqual:
    return left >= right;
  case BinaryComparisonKind::kLess:
    return left < right;
  case BinaryComparisonKind::kLessOrEqual:
    return left <= right;
  }
  return false;
}

template <typename T>
void CompareInteger64Scalar(const T *left, const T *right, std::uint8_t *out, std::size_t count,
                            BinaryOperator op, bool left_scalar, bool right_scalar) {
  for (std::size_t i = 0; i < count; ++i) {
    out[i] = detail::CompareInteger64(left[left_scalar ? 0 : i], right[right_scalar ? 0 : i], op);
  }
}

} // namespace

void BinaryCompareFloat16(const std::uint16_t *left, const std::uint16_t *right, std::uint8_t *out,
                          std::size_t count, BinaryComparisonKind kind, bool left_scalar,
                          bool right_scalar) {
#ifdef ONNX_LIGHT_CPU_HAVE_F16C
  static const bool use_f16c = CpuSupportsF16C();
  if (use_f16c) {
    BinaryCompareFloat16_F16C(left, right, out, count, kind, left_scalar, right_scalar);
    return;
  }
#endif

  const float scalar_left = left_scalar ? detail::Float16BitsToFloat(*left) : 0.0f;
  const float scalar_right = right_scalar ? detail::Float16BitsToFloat(*right) : 0.0f;
  for (std::size_t i = 0; i < count; ++i) {
    const float a = left_scalar ? scalar_left : detail::Float16BitsToFloat(left[i]);
    const float b = right_scalar ? scalar_right : detail::Float16BitsToFloat(right[i]);
    out[i] = Compare(a, b, kind) ? 1U : 0U;
  }
}

SimdLevel BinaryComparisonInt64SimdLevel(std::size_t count) {
#if defined(ONNX_LIGHT_CPU_HAVE_AVX512) || defined(ONNX_LIGHT_CPU_HAVE_AVX2)
  static const SimdLevel level = DetectSimdLevel();
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
  if (count >= 8 && level >= SimdLevel::kAVX512) {
    return SimdLevel::kAVX512;
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2
  if (count >= 4 && level >= SimdLevel::kAVX2) {
    return SimdLevel::kAVX2;
  }
#endif
#else
  (void)count;
#endif
  return SimdLevel::kNone;
}

const char *BinaryCompareInt64Implementation(std::size_t count) {
  switch (BinaryComparisonInt64SimdLevel(count)) {
  case SimdLevel::kAVX512:
    return "avx512";
  case SimdLevel::kAVX2:
    return "avx2";
  default:
    return "scalar";
  }
}

const char *BinaryCompareInt64Implementation() {
  static const char *const implementation = BinaryCompareInt64Implementation(8);
  return implementation;
}

void BinaryCompareInt64(const std::int64_t *left, const std::int64_t *right, std::uint8_t *out,
                        std::size_t count, BinaryOperator op, bool left_scalar, bool right_scalar) {
  const auto level = BinaryComparisonInt64SimdLevel(count);
  (void)level;
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
  if (level == SimdLevel::kAVX512) {
    BinaryCompareInt64_AVX512(left, right, out, count, op, left_scalar, right_scalar);
    return;
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2
  if (level == SimdLevel::kAVX2) {
    BinaryCompareInt64_AVX2(left, right, out, count, op, left_scalar, right_scalar);
    return;
  }
#endif
  CompareInteger64Scalar(left, right, out, count, op, left_scalar, right_scalar);
}

void BinaryCompareUInt64(const std::uint64_t *left, const std::uint64_t *right, std::uint8_t *out,
                         std::size_t count, BinaryOperator op, bool left_scalar,
                         bool right_scalar) {
  const auto level = BinaryComparisonInt64SimdLevel(count);
  (void)level;
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
  if (level == SimdLevel::kAVX512) {
    BinaryCompareUInt64_AVX512(left, right, out, count, op, left_scalar, right_scalar);
    return;
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2
  if (level == SimdLevel::kAVX2) {
    BinaryCompareUInt64_AVX2(left, right, out, count, op, left_scalar, right_scalar);
    return;
  }
#endif
  CompareInteger64Scalar(left, right, out, count, op, left_scalar, right_scalar);
}

} // namespace onnx_light_cpu
