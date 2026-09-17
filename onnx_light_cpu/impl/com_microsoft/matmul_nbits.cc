// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/com_microsoft/matmul_nbits.h"

#include "onnx_light_cpu/impl/checked_arithmetic.h"
#include "onnx_light_cpu/impl/execution.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace onnx_light_cpu {

void MatMulNBitsFloat32(const float *a, const std::uint8_t *b, const float *scales,
                        const float *bias, float *y, std::size_t rows, std::size_t k, std::size_t n,
                        std::size_t block_size, const MatMulNBitsExecutionTuning &tuning) {
  if (block_size != 32) {
    throw std::invalid_argument("onnx_light_cpu::MatMulNBits: block_size must be 32.");
  }
  if (rows == 0 || n == 0) {
    return;
  }
  const std::size_t k_blocks =
      CheckedAdd(k, block_size - 1, "MatMulNBits", "K blocks") / block_size;
  const std::size_t blob_size = block_size / 2;
  const std::size_t packed_column_stride =
      CheckedMultiply(k_blocks, blob_size, "MatMulNBits", "packed column stride");
  CheckedMultiply(rows, k, "MatMulNBits", "activation element count");
  CheckedMultiply(n, packed_column_stride, "MatMulNBits", "packed weight element count");
  CheckedMultiply(n, k_blocks, "MatMulNBits", "scale element count");
  const std::size_t output_size = CheckedMultiply(rows, n, "MatMulNBits", "output element count");
  if (output_size > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument(
        "onnx_light_cpu::MatMulNBits: output element count exceeds int64_t.");
  }
  const std::int64_t output_count = static_cast<std::int64_t>(output_size);
  const ExecutionSchedule schedule{
      static_cast<std::int64_t>(std::max<std::size_t>(tuning.parallel_threshold_outputs, 1)),
      static_cast<std::int64_t>(std::max<std::size_t>(tuning.target_block_outputs, 1)),
      tuning.max_participants,
  };
  ExecuteRanges(output_count, schedule, [&](std::int64_t begin, std::int64_t end) {
    for (std::int64_t output_index = begin; output_index < end; ++output_index) {
      const std::size_t index = static_cast<std::size_t>(output_index);
      const std::size_t row = index / n;
      const std::size_t column = index - row * n;
      const float *a_row = a + row * k;
      const std::uint8_t *b_column = b + column * packed_column_stride;
      const float *column_scales = scales + column * k_blocks;
      float sum = bias == nullptr ? 0.0f : bias[column];
      for (std::size_t block = 0; block < k_blocks; ++block) {
        const std::size_t block_begin = block * block_size;
        const std::size_t block_end = std::min(block_begin + block_size, k);
        const std::uint8_t *packed = b_column + block * blob_size;
        const float scale = column_scales[block];
        for (std::size_t offset = 0; block_begin + offset < block_end; ++offset) {
          const std::uint8_t byte = packed[offset / 2];
          const std::uint8_t quantized =
              (offset & 1) == 0 ? byte & 0x0f : static_cast<std::uint8_t>(byte >> 4);
          sum += a_row[block_begin + offset] * (static_cast<float>(quantized) - 8.0f) * scale;
        }
      }
      y[index] = sum;
    }
  });
}

} // namespace onnx_light_cpu
