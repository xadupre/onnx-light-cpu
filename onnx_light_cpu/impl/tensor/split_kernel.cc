// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/tensor/split_kernel.h"

#include "onnx_light_cpu/impl/execution.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace onnx_light_cpu {

namespace {

template <std::size_t Width>
void CopyFixedRows(const uint8_t *source, uint8_t *destination, std::size_t input_row_bytes,
                   int64_t begin, int64_t end) {
  // Scheduled byte boundaries are multiples of 64, hence aligned to every supported Width.
  const std::size_t first_row = static_cast<std::size_t>(begin) / Width;
  const std::size_t last_row = static_cast<std::size_t>(end) / Width;
  for (std::size_t row = first_row; row < last_row; ++row) {
    std::memcpy(destination + row * Width, source + row * input_row_bytes, Width);
  }
}

} // namespace

void SplitCopy(const void *data, void *output, int64_t rows, std::size_t input_row_bytes,
               std::size_t output_row_bytes, std::size_t offset_bytes) {
  if (rows == 0 || output_row_bytes == 0) {
    return;
  }
  const auto total = static_cast<int64_t>(static_cast<std::size_t>(rows) * output_row_bytes);
  const auto copy = [&](int64_t begin, int64_t end) {
    const auto *source = static_cast<const uint8_t *>(data) + offset_bytes;
    auto *destination = static_cast<uint8_t *>(output);
    if (input_row_bytes == output_row_bytes || rows == 1) {
      std::memcpy(destination + begin, source + begin, static_cast<std::size_t>(end - begin));
      return;
    }
    switch (output_row_bytes) {
    case 1:
      CopyFixedRows<1>(source, destination, input_row_bytes, begin, end);
      return;
    case 2:
      CopyFixedRows<2>(source, destination, input_row_bytes, begin, end);
      return;
    case 4:
      CopyFixedRows<4>(source, destination, input_row_bytes, begin, end);
      return;
    case 8:
      CopyFixedRows<8>(source, destination, input_row_bytes, begin, end);
      return;
    default:
      break;
    }
    std::size_t position = static_cast<std::size_t>(begin);
    std::size_t row = position / output_row_bytes;
    std::size_t column = position % output_row_bytes;
    while (position < static_cast<std::size_t>(end)) {
      const std::size_t count =
          std::min(output_row_bytes - column, static_cast<std::size_t>(end) - position);
      std::memcpy(destination + position, source + row * input_row_bytes + column, count);
      position += count;
      ++row;
      column = 0;
    }
  };
  constexpr int64_t kParallelBytes = 1024 * 1024;
  constexpr int64_t kBlockBytes = 256 * 1024;
  const auto *executor = CurrentExecutionExecutor();
  if (total < kParallelBytes || ExecutionInParallelRegion() || executor == nullptr ||
      executor->run_blocks == nullptr || ExecutionThreadCount() <= 1 ||
      total > std::numeric_limits<int64_t>::max() / 2) {
    copy(0, total);
    return;
  }
  ExecutionSchedule schedule;
  schedule.min_parallel_size = kParallelBytes;
  schedule.min_block_size = kBlockBytes;
  schedule.max_participants = ExecutionThreadCount();
  ExecuteRanges(total, schedule, int64_t{64}, copy);
}

void SplitCopyOutputs(const void *data, std::span<void *const> outputs, int64_t rows,
                      std::size_t input_row_bytes, std::span<const std::size_t> output_row_bytes) {
  if (rows == 0 || input_row_bytes == 0) {
    return;
  }
  const auto copy = [&](int64_t begin, int64_t end) {
    const auto *source = static_cast<const uint8_t *>(data);
    for (int64_t row = begin; row < end; ++row) {
      std::size_t offset = 0;
      for (std::size_t output = 0; output < outputs.size(); ++output) {
        const std::size_t width = output_row_bytes[output];
        if (width != 0) {
          std::memcpy(static_cast<uint8_t *>(outputs[output]) +
                          static_cast<std::size_t>(row) * width,
                      source + static_cast<std::size_t>(row) * input_row_bytes + offset, width);
        }
        offset += width;
      }
    }
  };
  constexpr int64_t kParallelBytes = 1024 * 1024;
  constexpr int64_t kBlockBytes = 256 * 1024;
  if (input_row_bytes > static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
    copy(0, rows);
    return;
  }
  const int64_t parallel_rows =
      std::max<int64_t>(1, kParallelBytes / static_cast<int64_t>(input_row_bytes));
  const int64_t block_rows =
      std::max<int64_t>(1, kBlockBytes / static_cast<int64_t>(input_row_bytes));
  const auto *executor = CurrentExecutionExecutor();
  if (rows < parallel_rows || ExecutionInParallelRegion() || executor == nullptr ||
      executor->run_blocks == nullptr || ExecutionThreadCount() <= 1) {
    copy(0, rows);
    return;
  }
  ExecuteRanges(rows, ExecutionSchedule{parallel_rows, block_rows, ExecutionThreadCount()}, copy);
}

} // namespace onnx_light_cpu
