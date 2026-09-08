// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/tensor/slice_kernel.h"

#include "onnx_light_cpu/impl/execution.h"

#include <cstring>
#include <limits>

namespace onnx_light_cpu {
namespace {

template <typename Fn> void Schedule(int64_t count, std::size_t bytes, Fn copy) {
  constexpr std::size_t kParallelBytes = 1024 * 1024;
  constexpr std::size_t kBlockBytes = 256 * 1024;
  const auto *executor = CurrentExecutionExecutor();
  if (static_cast<std::size_t>(count) * bytes < kParallelBytes || ExecutionInParallelRegion() ||
      executor == nullptr || executor->run_blocks == nullptr || ExecutionThreadCount() <= 1 ||
      count > std::numeric_limits<int64_t>::max() - ExecutionThreadCount()) {
    copy(0, count);
    return;
  }
  ExecutionSchedule schedule;
  schedule.min_parallel_size = static_cast<int64_t>(1 + (kParallelBytes - 1) / bytes);
  schedule.min_block_size = static_cast<int64_t>(1 + (kBlockBytes - 1) / bytes);
  schedule.max_participants = ExecutionThreadCount();
  ExecuteRanges(count, schedule, copy);
}

template <std::size_t Width>
void CopyStrided(const uint8_t *data, uint8_t *output, const SliceCopyPlan &plan, int64_t begin,
                 int64_t end) {
  std::array<int64_t, SliceCopyPlan::kMaxRank> coordinates{};
  int64_t remaining = begin;
  int64_t offset = plan.source_offset;
  for (std::size_t i = plan.rank; i-- > 0;) {
    coordinates[i] = remaining % plan.dimensions[i];
    remaining /= plan.dimensions[i];
    offset += coordinates[i] * plan.strides[i];
  }
  for (int64_t chunk = begin; chunk < end; ++chunk) {
    std::memcpy(output + static_cast<std::size_t>(chunk) * plan.chunk_bytes, data + offset,
                Width == 0 ? plan.chunk_bytes : Width);
    if (chunk + 1 == end) {
      break;
    }
    for (std::size_t i = plan.rank; i-- > 0;) {
      if (++coordinates[i] < plan.dimensions[i]) {
        offset += plan.strides[i];
        break;
      }
      coordinates[i] = 0;
      // Rewind directly to a valid coordinate, never advancing one step past the allocation.
      offset -= plan.rewinds[i];
    }
  }
}

} // namespace

void SliceCopy(const void *data, void *output, const SliceCopyPlan &plan) {
  if (plan.chunks == 0) {
    return;
  }
  const auto *source = static_cast<const uint8_t *>(data);
  auto *destination = static_cast<uint8_t *>(output);
  if (plan.rank == 0) {
    const int64_t count = static_cast<int64_t>(plan.chunk_bytes / plan.element_bytes);
    Schedule(count, plan.element_bytes, [&](int64_t begin, int64_t end) {
      const std::size_t offset = static_cast<std::size_t>(begin) * plan.element_bytes;
      std::memcpy(destination + offset, source + plan.source_offset + offset,
                  static_cast<std::size_t>(end - begin) * plan.element_bytes);
    });
    return;
  }
  auto copy = [&](int64_t begin, int64_t end) {
    switch (plan.chunk_bytes) {
    case 1:
      CopyStrided<1>(source, destination, plan, begin, end);
      break;
    case 2:
      CopyStrided<2>(source, destination, plan, begin, end);
      break;
    case 4:
      CopyStrided<4>(source, destination, plan, begin, end);
      break;
    case 8:
      CopyStrided<8>(source, destination, plan, begin, end);
      break;
    default:
      CopyStrided<0>(source, destination, plan, begin, end);
    }
  };
  Schedule(plan.chunks, plan.chunk_bytes, copy);
}

} // namespace onnx_light_cpu
