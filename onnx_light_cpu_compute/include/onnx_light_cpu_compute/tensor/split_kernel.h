// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace onnx_light_cpu {

/// Copies an owned, contiguous output from equally strided input rows.
/// The caller validates sizes, products, offsets and disjoint input/output storage.
/// Empty work permits null pointers; all byte counts must fit INT64 and size_t.
void SplitCopy(const void *data, void *output, int64_t rows, std::size_t input_row_bytes,
               std::size_t output_row_bytes, std::size_t offset_bytes);

/// Copies every split in one row traversal; four 32-bit outputs use a SIMD transpose.
void SplitCopyOutputs(const void *data, std::span<void *const> outputs, int64_t rows,
                      std::size_t input_row_bytes, std::span<const std::size_t> output_row_bytes);

} // namespace onnx_light_cpu
