// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace onnx_light_cpu {

/// One nonempty input's contiguous contribution to each output row.
struct ConcatInput {
  const uint8_t *data;
  std::size_t row_bytes;
  std::size_t output_offset;
};

/// Copies validated, non-overlapping buffers. Inputs are ordered by output_offset;
/// their row sizes sum to row_bytes, and all byte products fit int64_t and size_t.
void ConcatCopy(std::span<const ConcatInput> inputs, void *output, int64_t outer,
                std::size_t row_bytes);

} // namespace onnx_light_cpu
