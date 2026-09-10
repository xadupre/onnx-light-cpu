// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace onnx_light_cpu::detail {

enum class ShiftInputMode : std::uint8_t {
  kContiguous,
  kLeftScalar,
  kRightScalar,
};

template <typename T, bool Left, ShiftInputMode Mode>
void BinaryBitShift_Scalar(const T *left, const T *right, T *out, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    const T lhs = left[Mode == ShiftInputMode::kLeftScalar ? 0 : i];
    const T rhs = right[Mode == ShiftInputMode::kRightScalar ? 0 : i];
    out[i] = Left ? static_cast<T>(lhs << rhs) : static_cast<T>(lhs >> rhs);
  }
}

#ifdef ONNX_LIGHT_CPU_HAVE_AVX2
template <typename T, bool Left, ShiftInputMode Mode>
void BinaryBitShift_AVX2(const T *left, const T *right, T *out, std::size_t count);

template <typename T> bool BinaryBitShiftHasInvalidAmount_AVX2(const T *shifts, std::size_t count);
#endif

} // namespace onnx_light_cpu::detail
