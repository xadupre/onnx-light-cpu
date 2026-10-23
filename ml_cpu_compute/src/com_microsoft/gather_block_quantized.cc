// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "ml_cpu_compute/com_microsoft/gather_block_quantized.h"

#include "ml_cpu_compute/checked_arithmetic.h"
#include "ml_cpu_compute/execution.h"
#include "ml_cpu_compute/math/half_conversion.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace ml_cpu_compute {
namespace {

constexpr const char *kRoutine = "GatherBlockQuantized";

std::size_t Multiply(std::size_t a, std::size_t b) {
  const auto result = CheckedMultiply(a, b, kRoutine, "buffer size");
  CheckedStride(result, kRoutine, "buffer size");
  return result;
}

void ValidatePointer(const void *pointer, std::size_t bytes) {
  if (bytes != 0 && pointer == nullptr) {
    throw std::invalid_argument("GatherBlockQuantized: missing buffer.");
  }
}

bool Overlaps(const void *a, std::size_t a_bytes, const void *b, std::size_t b_bytes) {
  if (a_bytes == 0 || b_bytes == 0) {
    return false;
  }
  const auto x = reinterpret_cast<std::uintptr_t>(a);
  const auto y = reinterpret_cast<std::uintptr_t>(b);
  return x <= y ? y - x < a_bytes : x - y < b_bytes;
}

template <typename Scale, typename Index>
void Compute(const uint8_t *data, const Index *indices, const Scale *scales,
             const uint8_t *zero_points, Scale *output, const GatherBlockQuantizedParameters &p) {
  if ((p.bits != 2 && p.bits != 4 && p.bits != 8) || p.block_size < 16 ||
      (p.block_size & (p.block_size - 1)) != 0) {
    throw std::invalid_argument("GatherBlockQuantized: invalid bits or block_size.");
  }
  const auto rows = CheckedDimension(p.rows, kRoutine, "rows");
  const auto middle = CheckedDimension(p.rows_per_gather, kRoutine, "rows_per_gather");
  const auto packed = CheckedDimension(p.packed_columns, kRoutine, "packed_columns");
  const auto count = CheckedDimension(p.index_count, kRoutine, "index_count");
  const auto block_size = CheckedDimension(p.block_size, kRoutine, "block_size");
  const auto components = static_cast<std::size_t>(8 / p.bits);
  const auto columns = Multiply(packed, components);
  const auto blocks = columns / block_size + (columns % block_size != 0);
  const auto zp_columns = blocks / components + (blocks % components != 0);
  const auto data_rows = Multiply(rows, middle);
  const auto output_rows = Multiply(count, middle);
  const auto data_bytes = Multiply(data_rows, packed);
  const auto scale_bytes = Multiply(Multiply(data_rows, blocks), sizeof(Scale));
  const auto zp_bytes = Multiply(data_rows, zp_columns);
  const auto index_bytes = Multiply(count, sizeof(Index));
  const auto output_bytes = Multiply(Multiply(output_rows, columns), sizeof(Scale));
  const auto work = Multiply(output_rows, blocks);
  ValidatePointer(data, data_bytes);
  ValidatePointer(indices, index_bytes);
  ValidatePointer(scales, scale_bytes);
  ValidatePointer(output, output_bytes);
  if (Overlaps(output, output_bytes, data, data_bytes) ||
      Overlaps(output, output_bytes, indices, index_bytes) ||
      Overlaps(output, output_bytes, scales, scale_bytes) ||
      (zero_points != nullptr && Overlaps(output, output_bytes, zero_points, zp_bytes))) {
    throw std::invalid_argument("GatherBlockQuantized: output overlaps an input.");
  }
  if (work == 0) {
    return;
  }
  const unsigned bits = static_cast<unsigned>(p.bits);
  const unsigned mask = (1u << bits) - 1;
  ExecuteRanges(
      static_cast<int64_t>(work), static_cast<double>(std::min(block_size, columns)) * 4.0,
      [&](int64_t begin, int64_t end) noexcept {
        for (auto unit = static_cast<std::size_t>(begin); unit < static_cast<std::size_t>(end);
             ++unit) {
          const auto output_row = unit / blocks;
          const auto block = unit % blocks;
          const auto first = block * block_size;
          const auto length = std::min(block_size, columns - first);
          Scale *destination = output + output_row * columns + first;
          int64_t index = indices[output_row / middle];
          if (index < -p.rows || index >= p.rows) {
            std::fill_n(destination, length, Scale{0});
            continue;
          }
          if (index < 0) {
            index += p.rows;
          }
          const auto source_row = static_cast<std::size_t>(index) * middle + output_row % middle;
          const uint8_t *source = data + source_row * packed;
          const auto scale_value = scales[source_row * blocks + block];
          float scale;
          if constexpr (std::is_same_v<Scale, float>) {
            scale = scale_value;
          } else {
            scale = detail::Float16BitsToFloat(scale_value);
          }
          const int zp = zero_points == nullptr
                             ? 1 << (bits - 1)
                             : (zero_points[source_row * zp_columns + block / components] >>
                                ((block % components) * bits)) &
                                   mask;
          for (std::size_t j = 0; j < length; ++j) {
            const auto column = first + j;
            const int value =
                (source[column / components] >> ((column % components) * bits)) & mask;
            const float result = static_cast<float>(value - zp) * scale;
            if constexpr (std::is_same_v<Scale, float>) {
              destination[j] = result;
            } else {
              destination[j] = detail::FloatToFloat16Bits(result);
            }
          }
        }
      });
}

} // namespace

#define ML_CPU_DEFINE_GATHER_BLOCK_QUANTIZED(Scale, Index)                                         \
  void GatherBlockQuantized(const uint8_t *data, const Index *indices, const Scale *scales,        \
                            const uint8_t *zero_points, Scale *output,                             \
                            const GatherBlockQuantizedParameters &parameters) {                    \
    Compute(data, indices, scales, zero_points, output, parameters);                               \
  }

ML_CPU_DEFINE_GATHER_BLOCK_QUANTIZED(float, int32_t)
ML_CPU_DEFINE_GATHER_BLOCK_QUANTIZED(float, int64_t)
ML_CPU_DEFINE_GATHER_BLOCK_QUANTIZED(uint16_t, int32_t)
ML_CPU_DEFINE_GATHER_BLOCK_QUANTIZED(uint16_t, int64_t)

#undef ML_CPU_DEFINE_GATHER_BLOCK_QUANTIZED

} // namespace ml_cpu_compute
