// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

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

} // namespace onnx_light_cpu
