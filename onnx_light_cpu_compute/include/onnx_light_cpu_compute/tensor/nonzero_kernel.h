// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace onnx_light_cpu {

/// Counts nonzero bytes in a validated BOOL buffer of total elements.
/// total must be nonnegative; input may be null when total is zero.
int64_t CountNonZeroBool(const uint8_t *input, int64_t total);

/// Writes row-major coordinates into a disjoint [rank, count] INT64 buffer.
/// Dimensions and total must be validated, count must equal CountNonZeroBool,
/// and output byte size must fit int64_t and size_t. Input must remain unchanged
/// between counting and writing. For rank == 0 or count == 0, nothing is accessed.
void WriteNonZeroBoolIndices(const uint8_t *input, int64_t total, const int64_t *dimensions,
                             int64_t rank, int64_t count, int64_t *output);

} // namespace onnx_light_cpu
