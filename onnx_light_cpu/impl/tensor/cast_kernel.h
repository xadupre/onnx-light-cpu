// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu/impl/data_type.h"

#include <cstddef>

namespace onnx_light_cpu {

/// Fixed-width numeric Cast types, independent of onnx-light.
bool IsCastNumericType(DataType type) noexcept;

/// Returns the numeric storage width, or throws for an unsupported type.
std::size_t CastElementSize(DataType type);

/// Converts contiguous, possibly unaligned, non-overlapping buffers.
/// Identity copies preserve bits, including BOOL bytes and NaN payloads.
/// Integer conversions are direct C++20 modular conversions, never via double.
/// Floating-to-integer truncates toward zero. Narrow destinations wrap through
/// INT64 when that intermediate is representable; NaN becomes zero, and values
/// outside the intermediate range clamp to the destination bounds. INT64 and
/// UINT64 destinations clamp to their own bounds. BOOL is value != 0.
/// FLOAT16/BFLOAT16 use round-to-nearest-even via FLOAT (also from DOUBLE).
/// Validates types, byte arithmetic, null buffers and overlap before writes.
void CastConvert(const void *input, DataType from, void *output, DataType to, std::size_t count);

} // namespace onnx_light_cpu
