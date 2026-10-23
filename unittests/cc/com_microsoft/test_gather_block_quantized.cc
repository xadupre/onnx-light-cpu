// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "ml_cpu_compute/com_microsoft/gather_block_quantized.h"
#include "ml_cpu_compute/execution.h"
#include "ml_cpu_compute/math/half_conversion.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

using ml_cpu_compute::GatherBlockQuantized;
using ml_cpu_compute::GatherBlockQuantizedParameters;

template <typename Scale> Scale Encode(float value) {
  if constexpr (std::is_same_v<Scale, float>) {
    return value;
  } else {
    return ml_cpu_compute::detail::FloatToFloat16Bits(value);
  }
}

template <typename Scale, typename Index>
void Check(int64_t bits, int64_t packed_columns, int64_t block_size, bool with_zp) {
  const int64_t rows = 3;
  const int64_t middle = 6;
  const int64_t components = 8 / bits;
  const int64_t columns = packed_columns * components;
  const int64_t blocks = (columns + block_size - 1) / block_size;
  const int64_t zp_columns = (blocks + components - 1) / components;
  const unsigned mask = (1u << bits) - 1;
  const std::vector<Index> indices = {2,
                                      -3,
                                      0,
                                      -1,
                                      1,
                                      2,
                                      -4,
                                      3,
                                      std::numeric_limits<Index>::min(),
                                      std::numeric_limits<Index>::max()};
  std::vector<uint8_t> data(rows * middle * packed_columns, 0);
  std::vector<Scale> scales(rows * middle * blocks);
  // Padding is deliberately nonzero to catch zero-point packing across rows.
  std::vector<uint8_t> zp(rows * middle * zp_columns, 0xff);
  for (int64_t r = 0; r < rows * middle; ++r) {
    for (int64_t c = 0; c < columns; ++c) {
      const auto value = static_cast<unsigned>((r * 13 + c * 7 + 1) & mask);
      data[r * packed_columns + c / components] |=
          static_cast<uint8_t>(value << ((c % components) * bits));
    }
    for (int64_t b = 0; b < blocks; ++b) {
      scales[r * blocks + b] = Encode<Scale>(static_cast<float>(r + b + 1) / 16.0f);
      auto &byte = zp[r * zp_columns + b / components];
      const auto shift = (b % components) * bits;
      byte = static_cast<uint8_t>((byte & ~(mask << shift)) |
                                  (static_cast<unsigned>((r * 3 + b) & mask) << shift));
    }
  }
  std::vector<Scale> output(indices.size() * middle * columns + 2, Encode<Scale>(42));
  const GatherBlockQuantizedParameters p{
      rows, middle, packed_columns, static_cast<int64_t>(indices.size()), bits, block_size};
  GatherBlockQuantized(data.data(), indices.data(), scales.data(), with_zp ? zp.data() : nullptr,
                       output.data() + 1, p);
  EXPECT_EQ(output.front(), Encode<Scale>(42));
  EXPECT_EQ(output.back(), Encode<Scale>(42));
  for (std::size_t i = 0; i < indices.size(); ++i) {
    int64_t index = indices[i];
    const bool invalid = index < -rows || index >= rows;
    if (index < 0 && !invalid) {
      index += rows;
    }
    for (int64_t m = 0; m < middle; ++m) {
      for (int64_t c = 0; c < columns; ++c) {
        float expected = 0;
        if (!invalid) {
          const auto r = index * middle + m;
          const auto block = c / block_size;
          const int value = static_cast<int>((r * 13 + c * 7 + 1) & mask);
          const int zero = with_zp ? static_cast<int>((r * 3 + block) & mask) : 1 << (bits - 1);
          expected = static_cast<float>(value - zero) * static_cast<float>(r + block + 1) / 16.0f;
        }
        EXPECT_EQ(output[1 + (i * middle + m) * columns + c], Encode<Scale>(expected));
      }
    }
  }
}

TEST(GatherBlockQuantized, AllBitsTypesTailsAndZeroPoints) {
  for (int64_t bits : {2, 4, 8}) {
    for (int64_t columns : {1, 8, 9, 17, 33, 65}) {
      for (int64_t block : {16, 32, 128}) {
        for (bool zp : {false, true}) {
          Check<float, int32_t>(bits, columns, block, zp);
          Check<float, int64_t>(bits, columns, block, zp);
          Check<uint16_t, int32_t>(bits, columns, block, zp);
          Check<uint16_t, int64_t>(bits, columns, block, zp);
        }
      }
    }
  }
}

TEST(GatherBlockQuantized, EmptyDimensionsAndNoSourceRows) {
  for (const auto &p :
       {GatherBlockQuantizedParameters{3, 2, 7, 0}, GatherBlockQuantizedParameters{3, 0, 7, 1},
        GatherBlockQuantizedParameters{3, 2, 0, 1}}) {
    std::vector<uint8_t> data(42);
    std::vector<float> scales(6);
    const int32_t index = 0;
    EXPECT_NO_THROW(GatherBlockQuantized(data.data(), &index, scales.data(), nullptr,
                                         static_cast<float *>(nullptr), p));
  }
  const int64_t indices[] = {0, -1};
  std::vector<float> output(64, 42);
  GatherBlockQuantized(nullptr, indices, static_cast<const float *>(nullptr), nullptr,
                       output.data(), {0, 1, 16, 2});
  EXPECT_EQ(output, std::vector<float>(64, 0));
}

TEST(GatherBlockQuantized, SpecialScalesAndInvalidRows) {
  const uint8_t data[] = {0x88, 0x99, 0x77};
  const int64_t indices[] = {0, 1, 2, 3};
  const float scales[] = {std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN(), -0.0f};
  float output[8];
  GatherBlockQuantized(data, indices, scales, nullptr, output, {3, 1, 1, 4});
  EXPECT_TRUE(std::isnan(output[0]));
  EXPECT_TRUE(std::isnan(output[2]));
  EXPECT_EQ(output[4], 0);
  EXPECT_FALSE(std::signbit(output[4]));
  EXPECT_EQ(output[6], 0);
  EXPECT_FALSE(std::signbit(output[6]));
}

TEST(GatherBlockQuantized, LargeBlockSizeAndHalfRounding) {
  const uint8_t data[] = {0x97};
  const int64_t index = 0;
  const uint16_t scale = ml_cpu_compute::detail::FloatToFloat16Bits(0.3333f);
  uint16_t output[2];
  GatherBlockQuantized(data, &index, &scale, nullptr, output, {1, 1, 1, 1, 4, int64_t{1} << 62});
  const float value = ml_cpu_compute::detail::Float16BitsToFloat(scale);
  EXPECT_EQ(output[0], ml_cpu_compute::detail::FloatToFloat16Bits(-value));
  EXPECT_EQ(output[1], ml_cpu_compute::detail::FloatToFloat16Bits(value));
}

TEST(GatherBlockQuantized, RejectsInvalidParametersOverflowNullAndOverlap) {
  uint8_t data[32]{};
  int32_t index = 0;
  float scales[2]{1, 1};
  float output[64]{};
  for (auto p :
       {GatherBlockQuantizedParameters{-1, 1, 16, 1}, GatherBlockQuantizedParameters{1, -1, 16, 1},
        GatherBlockQuantizedParameters{1, 1, -1, 1}, GatherBlockQuantizedParameters{1, 1, 16, -1},
        GatherBlockQuantizedParameters{1, 1, 16, 1, 3},
        GatherBlockQuantizedParameters{1, 1, 16, 1, 4, 8},
        GatherBlockQuantizedParameters{1, 1, 16, 1, 4, 24},
        GatherBlockQuantizedParameters{1, 1, std::numeric_limits<int64_t>::max(), 1},
        GatherBlockQuantizedParameters{1, 2, 16, std::numeric_limits<int64_t>::max()}}) {
    EXPECT_THROW(GatherBlockQuantized(data, &index, scales, nullptr, output, p),
                 std::invalid_argument);
  }
  const GatherBlockQuantizedParameters p{1, 1, 16, 1};
  EXPECT_THROW(GatherBlockQuantized(nullptr, &index, scales, nullptr, output, p),
               std::invalid_argument);
  EXPECT_THROW(
      GatherBlockQuantized(data, static_cast<const int32_t *>(nullptr), scales, nullptr, output, p),
      std::invalid_argument);
  EXPECT_THROW(
      GatherBlockQuantized(data, &index, static_cast<const float *>(nullptr), nullptr, output, p),
      std::invalid_argument);
  EXPECT_THROW(GatherBlockQuantized(data, &index, output, nullptr, output, p),
               std::invalid_argument);
  EXPECT_THROW(
      GatherBlockQuantized(reinterpret_cast<uint8_t *>(output), &index, scales, nullptr, output, p),
      std::invalid_argument);
  EXPECT_THROW(
      GatherBlockQuantized(data, &index, scales, reinterpret_cast<uint8_t *>(output), output, p),
      std::invalid_argument);
  EXPECT_THROW(
      GatherBlockQuantized(data, &index, scales, nullptr, static_cast<float *>(nullptr), p),
      std::invalid_argument);
  EXPECT_THROW(GatherBlockQuantized(data, reinterpret_cast<const int32_t *>(output), scales,
                                    nullptr, output, p),
               std::invalid_argument);
}

struct ExecutorState {
  int calls = 0;
  int64_t blocks = 0;
};

void RunBlocks(void *context, int64_t blocks, void *task_context,
               ml_cpu_compute::ExecutionBlockFn task) {
  auto &state = *static_cast<ExecutorState *>(context);
  ++state.calls;
  state.blocks = blocks;
  for (int64_t i = blocks; i-- > 0;) {
    task(task_context, i);
  }
}

TEST(GatherBlockQuantized, RuntimeExecutorAndNestedSerialExecution) {
  ExecutorState state;
  const ml_cpu_compute::ExecutionExecutorView executor{&state, 4, RunBlocks};
  const ml_cpu_compute::ExecutionExecutorScope scope(&executor);
  std::vector<uint8_t> data(8192, 0x99);
  const int32_t indices[] = {0, 0, 0, 0, -1, 0, 0, 0};
  std::vector<float> scales(1024, 2);
  std::vector<float> output(8 * 16384);
  const GatherBlockQuantizedParameters p{1, 1, 8192, 8, 4, 16};
  GatherBlockQuantized(data.data(), indices, scales.data(), nullptr, output.data(), p);
  EXPECT_EQ(state.calls, 1);
  EXPECT_GT(state.blocks, 1);
  EXPECT_EQ(output, std::vector<float>(output.size(), 2));
  ml_cpu_compute::ExecuteRanges(1, [&](int64_t, int64_t) {
    GatherBlockQuantized(data.data(), indices, scales.data(), nullptr, output.data(), p);
  });
  EXPECT_EQ(state.calls, 1);
  const uint8_t small_data = 0x99;
  const float small_scale = 2;
  float small_output[2];
  GatherBlockQuantized(&small_data, indices, &small_scale, nullptr, small_output, {1, 1, 1, 1});
  EXPECT_EQ(state.calls, 1);
  std::vector<float> huge_block_output(16);
  GatherBlockQuantized(&small_data, indices, &small_scale, nullptr, huge_block_output.data(),
                       {1, 1, 1, 8, 4, int64_t{1} << 62});
  EXPECT_EQ(state.calls, 1);
  EXPECT_EQ(huge_block_output, std::vector<float>(huge_block_output.size(), 2));
}

} // namespace
