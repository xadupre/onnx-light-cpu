// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace onnx_light_cpu {

/// Copies an owned, contiguous output from equally strided input rows.
/// The caller validates sizes, products, offsets and disjoint input/output storage.
/// Empty work permits null pointers; all byte counts must fit INT64 and size_t.
void SplitCopy(const void *data, void *output, int64_t rows, std::size_t input_row_bytes,
               std::size_t output_row_bytes, std::size_t offset_bytes);

} // namespace onnx_light_cpu
