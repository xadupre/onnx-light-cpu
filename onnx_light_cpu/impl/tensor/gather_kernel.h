// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace onnx_light_cpu {

/// Copies contiguous slices using validated dimensions and in-range indices.
/// Input/output buffers must not overlap. All products must fit int64_t and size_t.
void GatherSlices(const void *data, const int32_t *indices, void *output, int64_t outer,
                  int64_t axis_size, int64_t index_count, std::size_t slice_bytes);
void GatherSlices(const void *data, const int64_t *indices, void *output, int64_t outer,
                  int64_t axis_size, int64_t index_count, std::size_t slice_bytes);

} // namespace onnx_light_cpu
