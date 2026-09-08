// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/tensor/concat_kernel.h"

#include "onnx_light_cpu/impl/execution.h"

#include <algorithm>
#include <cstring>

namespace onnx_light_cpu {
namespace {

void CopyBytes(uint8_t *output, const uint8_t *input, std::size_t count) {
  switch (count) {
  case 1:
    std::memcpy(output, input, 1);
    break;
  case 2:
    std::memcpy(output, input, 2);
    break;
  case 4:
    std::memcpy(output, input, 4);
    break;
  case 8:
    std::memcpy(output, input, 8);
    break;
  default:
    std::memcpy(output, input, count);
  }
}

void CopyRange(std::span<const ConcatInput> inputs, uint8_t *output, std::size_t row_bytes,
               std::size_t begin, std::size_t end) {
  if (inputs.size() == 1) {
    std::memcpy(output + begin, inputs[0].data + begin, end - begin);
    return;
  }
  std::size_t row = begin / row_bytes;
  std::size_t offset = begin % row_bytes;
  auto input = std::upper_bound(inputs.begin(), inputs.end(), offset,
                                [](std::size_t value, const ConcatInput &piece) {
                                  return value < piece.output_offset + piece.row_bytes;
                                });
  while (begin < end) {
    const std::size_t piece_offset = offset - input->output_offset;
    const std::size_t count = std::min(input->row_bytes - piece_offset, end - begin);
    CopyBytes(output + begin, input->data + row * input->row_bytes + piece_offset, count);
    begin += count;
    offset += count;
    if (offset == row_bytes) {
      ++row;
      offset = 0;
      input = inputs.begin();
    } else {
      ++input;
    }
  }
}

template <std::size_t Width>
void CopyNarrowRange(std::span<const ConcatInput> inputs, uint8_t *output, std::size_t row_bytes,
                     std::size_t begin, std::size_t end) {
  if (begin % row_bytes != 0) {
    const std::size_t head = std::min(row_bytes - begin % row_bytes, end - begin);
    CopyRange(inputs, output, row_bytes, begin, begin + head);
    begin += head;
  }
  const std::size_t rows = (end - begin) / row_bytes;
  const std::size_t first_row = begin / row_bytes;
  for (std::size_t row = 0; row < rows; ++row) {
    uint8_t *destination = output + begin + row * row_bytes;
    for (const ConcatInput &input : inputs) {
      std::memcpy(destination + input.output_offset, input.data + (first_row + row) * Width, Width);
    }
  }
  begin += rows * row_bytes;
  if (begin != end) {
    CopyRange(inputs, output, row_bytes, begin, end);
  }
}

} // namespace

void ConcatCopy(std::span<const ConcatInput> inputs, void *output, int64_t outer,
                std::size_t row_bytes) {
  if (outer == 0 || row_bytes == 0) {
    return;
  }
  const std::size_t bytes = static_cast<std::size_t>(outer) * row_bytes;
  auto *destination = static_cast<uint8_t *>(output);
  auto copy = &CopyRange;
  if (inputs.size() > 1 && inputs.front().row_bytes <= 8 &&
      std::all_of(inputs.begin(), inputs.end(), [&](const ConcatInput &input) {
        return input.row_bytes == inputs.front().row_bytes;
      })) {
    switch (inputs.front().row_bytes) {
    case 1:
      copy = &CopyNarrowRange<1>;
      break;
    case 2:
      copy = &CopyNarrowRange<2>;
      break;
    case 4:
      copy = &CopyNarrowRange<4>;
      break;
    case 8:
      copy = &CopyNarrowRange<8>;
      break;
    }
  }
  const auto *executor = CurrentExecutionExecutor();
  constexpr std::size_t kTileBytes = 64 * 1024;
  if (bytes < 1024 * 1024 || ExecutionInParallelRegion() || executor == nullptr ||
      executor->run_blocks == nullptr || ExecutionThreadCount() <= 1) {
    copy(inputs, destination, row_bytes, 0, bytes);
    return;
  }
  // Byte tiles split large axis-0 spans too, with one dispatch for the whole concatenation.
  // Scheduling tiles rather than individual bytes also bounds executor rounding arithmetic.
  const int64_t tiles = static_cast<int64_t>(bytes / kTileBytes + (bytes % kTileBytes != 0));
  ExecutionSchedule schedule;
  schedule.min_parallel_size = 16;
  schedule.min_block_size = 4;
  schedule.max_participants = ExecutionThreadCount();
  ExecuteRanges(tiles, schedule, [&](int64_t begin, int64_t end) {
    const std::size_t first = static_cast<std::size_t>(begin) * kTileBytes;
    const std::size_t last = end == tiles ? bytes : static_cast<std::size_t>(end) * kTileBytes;
    copy(inputs, destination, row_bytes, first, last);
  });
}

} // namespace onnx_light_cpu
