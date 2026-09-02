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

} // namespace onnx_light_cpu
