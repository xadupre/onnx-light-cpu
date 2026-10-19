// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "ml_cpu_compute/com_microsoft/qmoe.h"

#include "ml_cpu_compute/math/half_conversion.h"

#include <immintrin.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace ml_cpu_compute {
namespace {

float LoadScale(const QMoEWeight &weight, std::size_t index) {
  if (weight.scale_type == MoEDataType::kFloat32) {
    return static_cast<const float *>(weight.scales)[index];
  }
  const auto bits = static_cast<const std::uint16_t *>(weight.scales)[index];
  return weight.scale_type == MoEDataType::kFloat16 ? detail::Float16BitsToFloat(bits)
                                                    : detail::Bfloat16BitsToFloat(bits);
}

} // namespace

void DequantizeQMoEAvx2(QMoEWeight weight, float *output, std::size_t experts, std::size_t inputs,
                        std::size_t outputs, const QMoEAttributes &attributes, std::size_t begin,
                        std::size_t end) {
  const std::size_t factor = 8 / attributes.expert_weight_bits;
  const std::size_t rows = weight.output_major ? outputs : inputs;
  const std::size_t columns = weight.output_major ? inputs : outputs;
  const std::size_t packed = columns / factor + (columns % factor != 0);
  const std::size_t blocks =
      attributes.block_size == 0
          ? 1
          : columns / attributes.block_size + (columns % attributes.block_size != 0);
  for (std::size_t row = begin; row < end; ++row) {
    const auto expert = row / rows;
    const auto stored_row = row % rows;
    const auto *source = weight.data + row * packed;
    auto *destination = weight.output_major ? output + row * inputs
                                            : output + expert * outputs * inputs + stored_row;
    for (std::size_t block = 0; block < blocks; ++block) {
      const auto column_begin = attributes.block_size == 0 ? 0 : block * attributes.block_size;
      const auto column_end = attributes.block_size == 0
                                  ? columns
                                  : std::min(columns, column_begin + attributes.block_size);
      const auto scale_index = attributes.block_size == 0
                                   ? expert * rows + stored_row
                                   : (expert * rows + stored_row) * blocks + block;
      const auto scale = LoadScale(weight, scale_index);
      std::size_t column = column_begin;
      if (attributes.expert_weight_bits == 4) {
        for (; column + 8 <= column_end; column += 8) {
          std::int32_t packed_values;
          std::memcpy(&packed_values, source + column / 2, sizeof(packed_values));
          const auto bytes = _mm_cvtsi32_si128(packed_values);
          const auto mask = _mm_set1_epi8(0x0f);
          const auto low = _mm_and_si128(bytes, mask);
          const auto high = _mm_and_si128(_mm_srli_epi32(bytes, 4), mask);
          const auto interleaved = _mm_unpacklo_epi8(low, high);
          auto quantized = _mm256_cvtepu8_epi32(interleaved);
          quantized = _mm256_sub_epi32(quantized, _mm256_set1_epi32(8));
          const auto values = _mm256_mul_ps(_mm256_cvtepi32_ps(quantized), _mm256_set1_ps(scale));
          if (weight.output_major) {
            _mm256_storeu_ps(destination + column, values);
          } else {
            float lanes[8];
            _mm256_storeu_ps(lanes, values);
            for (std::size_t lane = 0; lane < 8; ++lane) {
              destination[(column + lane) * inputs] = lanes[lane];
            }
          }
        }
      } else {
        for (; column + 8 <= column_end; column += 8) {
          std::int64_t packed_values;
          std::memcpy(&packed_values, source + column, sizeof(packed_values));
          const auto bytes = _mm_cvtsi64_si128(packed_values);
          auto quantized =
              weight.unsigned_int8 ? _mm256_cvtepu8_epi32(bytes) : _mm256_cvtepi8_epi32(bytes);
          if (weight.unsigned_int8) {
            quantized = _mm256_sub_epi32(quantized, _mm256_set1_epi32(128));
          }
          const auto values = _mm256_mul_ps(_mm256_cvtepi32_ps(quantized), _mm256_set1_ps(scale));
          if (weight.output_major) {
            _mm256_storeu_ps(destination + column, values);
          } else {
            float lanes[8];
            _mm256_storeu_ps(lanes, values);
            for (std::size_t lane = 0; lane < 8; ++lane) {
              destination[(column + lane) * inputs] = lanes[lane];
            }
          }
        }
      }
      for (; column < column_end; ++column) {
        const auto byte = source[column / factor];
        const int quantized = attributes.expert_weight_bits == 4
                                  ? static_cast<int>((byte >> (4 * (column % 2))) & 15) - 8
                              : weight.unsigned_int8 ? static_cast<int>(byte) - 128
                              : byte < 128           ? static_cast<int>(byte)
                                                     : static_cast<int>(byte) - 256;
        const auto output_index = weight.output_major ? column : column * inputs;
        destination[output_index] = static_cast<float>(quantized) * scale;
      }
    }
  }
}

} // namespace ml_cpu_compute
