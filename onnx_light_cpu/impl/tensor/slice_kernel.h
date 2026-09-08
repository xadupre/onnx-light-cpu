// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace onnx_light_cpu {

/// Validated byte layout. Each independent chunk is contiguous in both tensors.
struct SliceCopyPlan {
  static constexpr std::size_t kMaxRank = 16;
  std::size_t rank = 0;
  std::array<int64_t, kMaxRank> dimensions{};
  std::array<int64_t, kMaxRank> strides{};
  std::array<int64_t, kMaxRank> rewinds{};
  int64_t source_offset = 0;
  int64_t chunks = 0;
  std::size_t chunk_bytes = 0;
  std::size_t element_bytes = 0;
};

/// Copies a checked layout with disjoint input/output buffers; never throws in workers.
void SliceCopy(const void *data, void *output, const SliceCopyPlan &plan);

} // namespace onnx_light_cpu
