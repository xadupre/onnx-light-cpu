// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/tensor/gather_kernel.h"

#include "onnx_light_cpu/impl/execution.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace onnx_light_cpu {
namespace {

template <std::size_t Width, typename Index>
void CopyRange(const uint8_t *data, const Index *indices, uint8_t *output, int64_t axis_size,
               int64_t index_count, std::size_t slice_bytes, int64_t begin, int64_t end) {
  int64_t outer = begin / index_count;
  int64_t qi = begin % index_count;
  while (begin < end) {
    const int64_t count = std::min(index_count - qi, end - begin);
    const uint8_t *source = data + static_cast<std::size_t>(outer * axis_size) * slice_bytes;
    uint8_t *destination = output + static_cast<std::size_t>(begin) * slice_bytes;
    for (int64_t i = 0; i < count; ++i) {
      int64_t index = indices[qi + i];
      if (index < 0) {
        index += axis_size;
      }
      // Constant-size memcpy gives scalar loads/stores without alignment or aliasing assumptions.
      std::memcpy(destination + static_cast<std::size_t>(i) * slice_bytes,
                  source + static_cast<std::size_t>(index) * slice_bytes,
                  Width == 0 ? slice_bytes : Width);
    }
    begin += count;
    ++outer;
    qi = 0;
  }
}

template <typename Index>
void Gather(const void *data, const Index *indices, void *output, int64_t outer, int64_t axis_size,
            int64_t index_count, std::size_t slice_bytes) {
  if (outer == 0 || index_count == 0 || slice_bytes == 0) {
    return;
  }
  const int64_t total = outer * index_count;
  const auto copy = [&](int64_t begin, int64_t end) {
    const auto *source = static_cast<const uint8_t *>(data);
    auto *destination = static_cast<uint8_t *>(output);
    switch (slice_bytes) {
    case 1:
      CopyRange<1>(source, indices, destination, axis_size, index_count, slice_bytes, begin, end);
      break;
    case 2:
      CopyRange<2>(source, indices, destination, axis_size, index_count, slice_bytes, begin, end);
      break;
    case 4:
      CopyRange<4>(source, indices, destination, axis_size, index_count, slice_bytes, begin, end);
      break;
    case 8:
      CopyRange<8>(source, indices, destination, axis_size, index_count, slice_bytes, begin, end);
      break;
    default:
      CopyRange<0>(source, indices, destination, axis_size, index_count, slice_bytes, begin, end);
    }
  };
  constexpr std::size_t kParallelBytes = 1024 * 1024;
  constexpr std::size_t kBlockBytes = 256 * 1024;
  const auto *executor = CurrentExecutionExecutor();
  if (static_cast<std::size_t>(total) * slice_bytes < kParallelBytes ||
      ExecutionInParallelRegion() || executor == nullptr || executor->run_blocks == nullptr ||
      ExecutionThreadCount() <= 1 ||
      total > std::numeric_limits<int64_t>::max() - ExecutionThreadCount()) {
    copy(0, total);
    return;
  }
  ExecutionSchedule schedule;
  schedule.min_parallel_size = static_cast<int64_t>(1 + (kParallelBytes - 1) / slice_bytes);
  schedule.min_block_size = static_cast<int64_t>(1 + (kBlockBytes - 1) / slice_bytes);
  schedule.max_participants = ExecutionThreadCount();
  ExecuteRanges(total, schedule, copy);
}

} // namespace

void GatherSlices(const void *data, const int32_t *indices, void *output, int64_t outer,
                  int64_t axis_size, int64_t index_count, std::size_t slice_bytes) {
  Gather(data, indices, output, outer, axis_size, index_count, slice_bytes);
}

void GatherSlices(const void *data, const int64_t *indices, void *output, int64_t outer,
                  int64_t axis_size, int64_t index_count, std::size_t slice_bytes) {
  Gather(data, indices, output, outer, axis_size, index_count, slice_bytes);
}

} // namespace onnx_light_cpu
