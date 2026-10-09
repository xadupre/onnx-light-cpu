// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/com_microsoft/matmul_nbits.h"
#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

using namespace onnx_light_cpu;

template <typename Storage, typename Encode, typename Decode>
void CheckPanels(DataType type, std::size_t bits, Encode encode, Decode decode) {
  const std::size_t values_per_byte = 8 / bits;
  const std::size_t blob_size = 32 * bits / 8;
  const std::uint8_t mask = static_cast<std::uint8_t>((1U << bits) - 1U);
  const int zero_point = 1 << (bits - 1);
  for (std::size_t m : {0u, 1u, 2u, 7u, 8u, 9u, 17u}) {
    for (std::size_t n = 1; n <= 65; ++n) {
      for (std::size_t k : {0u, 1u, 15u, 31u, 32u, 33u, 65u}) {
        SCOPED_TRACE(::testing::Message() << m << ',' << k << ',' << n);
        const auto blocks = (k + 31) / 32;
        std::vector<Storage> a(m * k), scales(n * blocks), bias(n), output(m * n + 1);
        std::vector<std::uint8_t> packed(n * blocks * blob_size);
        for (std::size_t i = 0; i < a.size(); ++i) {
          a[i] = encode(static_cast<float>(static_cast<int>(i % 17) - 8) / 16);
        }
        for (std::size_t i = 0; i < scales.size(); ++i) {
          scales[i] = encode(static_cast<float>(i % 7 + 1) / 64);
        }
        for (std::size_t i = 0; i < bias.size(); ++i) {
          bias[i] = encode(static_cast<float>(i % 3) / 8);
        }
        for (std::size_t i = 0; i < packed.size(); ++i) {
          packed[i] = static_cast<std::uint8_t>(i * 37 + i / 11);
        }
        const auto original = packed;
        for (bool with_bias : {false, true}) {
          output.back() = encode(-42);
          for (int repeat = 0; repeat < 2; ++repeat) {
            MatMulNBits(a.data(), packed.data(), scales.data(), with_bias ? bias.data() : nullptr,
                        output.data(), type, m, k, n, bits, 32);
            for (std::size_t r = 0; r < m; ++r) {
              for (std::size_t c = 0; c < n; ++c) {
                float expected = with_bias ? decode(bias[c]) : 0;
                for (std::size_t p = 0; p < k; ++p) {
                  const auto index = c * blocks + p / 32;
                  const std::size_t offset = p % 32;
                  const int q = (packed[index * blob_size + offset / values_per_byte] >>
                                 ((offset % values_per_byte) * bits)) &
                                mask;
                  expected += decode(a[r * k + p]) * (q - zero_point) * decode(scales[index]);
                }
                EXPECT_EQ(output[r * n + c], encode(expected));
              }
            }
            EXPECT_EQ(output.back(), encode(-42));
            EXPECT_EQ(packed, original);
          }
        }
      }
    }
  }
}

TEST(MatMulNBits, PanelTailsAndRepeatedPackedConstantsAllTypes) {
  for (std::size_t bits : {2u, 4u, 8u}) {
    CheckPanels<float>(DataType::FLOAT, bits, [](float x) { return x; }, [](float x) { return x; });
    CheckPanels<std::uint16_t>(DataType::FLOAT16, bits, detail::FloatToFloat16Bits,
                               detail::Float16BitsToFloat);
    CheckPanels<std::uint16_t>(DataType::BFLOAT16, bits, detail::FloatToBFloat16Bits,
                               detail::Bfloat16BitsToFloat);
  }
}

TEST(MatMulNBits, WidePanelRowBoundariesMatchReference) {
  constexpr std::size_t k = 33;
  constexpr std::size_t n = 33;
  constexpr std::size_t blocks = 2;
  for (std::size_t bits : {2u, 4u, 8u}) {
    const std::size_t values_per_byte = 8 / bits;
    const std::size_t blob_size = 32 * bits / 8;
    const std::uint8_t mask = static_cast<std::uint8_t>((1U << bits) - 1U);
    const int zero_point = 1 << (bits - 1);
    for (std::size_t rows : {127u, 128u, 129u}) {
      SCOPED_TRACE(::testing::Message() << bits << ',' << rows);
      std::vector<float> a(rows * k), scales(n * blocks), output(rows * n);
      std::vector<std::uint8_t> packed(n * blocks * blob_size);
      for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<float>(static_cast<int>(i % 19) - 9) / 16.0f;
      }
      for (std::size_t i = 0; i < scales.size(); ++i) {
        scales[i] = static_cast<float>(i % 7 + 1) / 64.0f;
      }
      for (std::size_t i = 0; i < packed.size(); ++i) {
        packed[i] = static_cast<std::uint8_t>(i * 37 + i / 11);
      }

      MatMulNBits(a.data(), packed.data(), scales.data(), nullptr, output.data(), DataType::FLOAT,
                  rows, k, n, bits, 32);

      for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < n; ++column) {
          float expected = 0.0f;
          for (std::size_t p = 0; p < k; ++p) {
            const std::size_t block = p / 32;
            const std::size_t offset = p % 32;
            const std::size_t packed_index =
                (column * blocks + block) * blob_size + offset / values_per_byte;
            const int q = (packed[packed_index] >> ((offset % values_per_byte) * bits)) & mask;
            expected += a[row * k + p] * static_cast<float>(q - zero_point) *
                        scales[column * blocks + block];
          }
          EXPECT_FLOAT_EQ(output[row * n + column], expected);
        }
      }
    }
  }
}

struct InlineExecutor {
  int dispatches = 0;
  std::int64_t blocks = 0;
  static void Run(void *context, std::int64_t count, void *task_context, ExecutionBlockFn task) {
    auto &self = *static_cast<InlineExecutor *>(context);
    ++self.dispatches;
    self.blocks = count;
    for (std::int64_t block = count; block-- > 0;) {
      task(task_context, block);
    }
  }
};

TEST(MatMulNBits, PreparedInt8QwenShapesAndTails) {
  for (const auto [k, n] : {std::pair{512u, 32000u}, std::pair{512u, 1376u}, std::pair{1376u, 512u},
                            std::pair{512u, 1024u}, std::pair{512u, 512u}, std::pair{96u, 13u}}) {
    SCOPED_TRACE(::testing::Message() << k << ',' << n);
    const std::size_t rows = n == 13 ? 3 : 128;
    const std::size_t blocks = k / 32;
    const std::size_t slots = (n / 8 + (n % 8 != 0)) * blocks * 8;
    std::vector<float> a(rows * k), scales(n * blocks), bias(n), packed_scales(slots), y(rows * n);
    std::vector<std::uint8_t> b(n * k), packed(slots * 32);
    std::vector<std::int32_t> sums(slots);
    for (std::size_t i = 0; i < a.size(); ++i) {
      a[i] = static_cast<float>(static_cast<int>(i * 17 % 127) - 63) / 63.0f;
    }
    for (std::size_t i = 0; i < b.size(); ++i) {
      b[i] = static_cast<std::uint8_t>((i * 37 + i / 29) % 256);
    }
    for (std::size_t i = 0; i < scales.size(); ++i) {
      scales[i] = static_cast<float>(i % 7 + 1) / 1024.0f;
    }
    for (std::size_t i = 0; i < n; ++i) {
      bias[i] = static_cast<float>(i % 5) / 16;
    }
    PackMatMulNBitsInt8(b.data(), scales.data(), packed.data(), sums.data(), packed_scales.data(),
                        k, n);
    for (std::int64_t threads : {1, 2, 4, 8}) {
      for (bool with_bias : {false, true}) {
        MatMulNBitsInt8Float32(a.data(), packed.data(), sums.data(), packed_scales.data(),
                               with_bias ? bias.data() : nullptr, y.data(), rows, k, n, threads);
        for (std::size_t row :
             {0u, static_cast<unsigned>(rows / 2), static_cast<unsigned>(rows - 1)}) {
          for (std::size_t col : {0u, static_cast<unsigned>(n / 2), static_cast<unsigned>(n - 1)}) {
            float expected = with_bias ? bias[col] : 0.0f;
            for (std::size_t block = 0; block < blocks; ++block) {
              float maximum = 0;
              for (std::size_t p = 0; p < 32; ++p) {
                maximum = std::max(maximum, std::abs(a[row * k + block * 32 + p]));
              }
              std::int32_t dot = 0;
              for (std::size_t p = 0; p < 32; ++p) {
                const float value = a[row * k + block * 32 + p];
                const int q = static_cast<int>(std::nearbyint(value * (63.0f / maximum)));
                dot += q * (static_cast<int>(b[(col * blocks + block) * 32 + p]) - 128);
              }
              expected +=
                  static_cast<float>(dot) * (maximum / 63.0f) * scales[col * blocks + block];
            }
            EXPECT_NEAR(y[row * n + col], expected, 0.0002f);
            float unquantized = with_bias ? bias[col] : 0.0f;
            for (std::size_t p = 0; p < k; ++p) {
              unquantized += a[row * k + p] *
                             static_cast<float>(static_cast<int>(b[col * k + p]) - 128) *
                             scales[col * blocks + p / 32];
            }
            EXPECT_NEAR(y[row * n + col], unquantized, 0.3f + 0.05f * std::abs(unquantized));
          }
        }
      }
    }
  }
}

TEST(MatMulNBits, PreparedInt8UsesRuntimeExecutor) {
  std::vector<float> a(9 * 32, 0.5f), scales(8, 0.25f), y(9 * 8);
  std::vector<std::uint8_t> b(8 * 32, 129), packed(8 * 32);
  std::vector<std::int32_t> sums(8);
  PackMatMulNBitsInt8(b.data(), scales.data(), packed.data(), sums.data(), scales.data(), 32, 8);
  InlineExecutor executor;
  ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run};
  ExecutionExecutorScope scope(&view);
  MatMulNBitsInt8Float32(a.data(), packed.data(), sums.data(), scales.data(), nullptr, y.data(), 1,
                         32, 8, 4);
  EXPECT_EQ(executor.dispatches, 0);
  MatMulNBitsInt8Float32(a.data(), packed.data(), sums.data(), scales.data(), nullptr, y.data(), 9,
                         32, 8, 4);
  EXPECT_EQ(executor.dispatches, 1);
  EXPECT_GT(executor.blocks, 1);
}

TEST(MatMulNBits, PreparedInt8HandlesNonFiniteAndTinyActivations) {
  constexpr std::size_t k = 32;
  constexpr std::size_t n = 9;
  std::vector<std::uint8_t> b(n * k, 129), packed(2 * 256);
  std::vector<std::int32_t> sums(16);
  std::vector<float> scales(n, 1e30f), packed_scales(16), a(3 * k), y(3 * n);
  std::fill_n(a.data(), k, 1e-40f);
  std::fill_n(a.data() + k, k, 0.5f);
  a[k] = std::numeric_limits<float>::quiet_NaN();
  std::fill_n(a.data() + 2 * k, k, 0.5f);
  a[2 * k] = std::numeric_limits<float>::infinity();
  PackMatMulNBitsInt8(b.data(), scales.data(), packed.data(), sums.data(), packed_scales.data(), k,
                      n);
  MatMulNBitsInt8Float32(a.data(), packed.data(), sums.data(), packed_scales.data(), nullptr,
                         y.data(), 3, k, n, 2);
  for (std::size_t column = 0; column < n; ++column) {
    EXPECT_NEAR(y[column], 32e-10f, 1e-10f);
    EXPECT_TRUE(std::isnan(y[n + column]));
    EXPECT_TRUE(std::isinf(y[2 * n + column]));
  }
}

TEST(MatMulNBits, PanelSchedulingAndNestedSuppression) {
  const std::vector<float> a(9 * 33, 1);
  const std::vector<float> scales(513 * 2, 0.5f);
  const std::vector<std::uint8_t> b(513 * 32, 0x99);
  std::vector<float> y(9 * 513);
  InlineExecutor executor;
  ExecutionExecutorView view{&executor, 8, &InlineExecutor::Run};
  ExecutionExecutorScope scope(&view);
  MatMulNBitsFloat32(a.data(), b.data(), scales.data(), nullptr, y.data(), 1, 33, 2, 32);
  EXPECT_EQ(executor.dispatches, 0);
  MatMulNBitsExecutionTuning tuning;
  tuning.parallel_threshold_outputs = 513;
  MatMulNBitsFloat32(a.data(), b.data(), scales.data(), nullptr, y.data(), 8, 33, 64, 32, tuning);
  EXPECT_EQ(executor.dispatches, 0);
  tuning.parallel_threshold_outputs = 512;
  MatMulNBitsFloat32(a.data(), b.data(), scales.data(), nullptr, y.data(), 8, 33, 64, 32, tuning);
  EXPECT_EQ(executor.dispatches, 1);
  EXPECT_EQ(executor.blocks, 2);
  executor = {};
  tuning.parallel_threshold_outputs = 577;
  MatMulNBitsFloat32(a.data(), b.data(), scales.data(), nullptr, y.data(), 9, 33, 64, 32, tuning);
  EXPECT_EQ(executor.dispatches, 0);
  MatMulNBitsFloat32(a.data(), b.data(), scales.data(), nullptr, y.data(), 9, 33, 513, 32);
  EXPECT_EQ(executor.dispatches, 1);
  EXPECT_GT(executor.blocks, 1);
  for (float value : y) {
    EXPECT_EQ(value, 16.5f);
  }
  {
    detail::ExecutionRegionScope nested;
    MatMulNBitsFloat32(a.data(), b.data(), scales.data(), nullptr, y.data(), 9, 33, 513, 32);
  }
  EXPECT_EQ(executor.dispatches, 1);
}

TEST(MatMulNBits, PanelSpecialValuesAndEmptyOutputs) {
  std::vector<float> a(33, 1), scales(66, 1), y(33);
  const std::vector<std::uint8_t> b(33 * 32, 0x99);
  for (float value :
       {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN()}) {
    a[32] = value;
    MatMulNBitsFloat32(a.data(), b.data(), scales.data(), nullptr, y.data(), 1, 33, 33, 32);
    for (float actual : y) {
      if (std::isnan(value)) {
        EXPECT_TRUE(std::isnan(actual));
      } else {
        EXPECT_EQ(actual, value);
      }
    }
  }
  MatMulNBitsFloat32(nullptr, nullptr, nullptr, nullptr, nullptr, 1, 33, 0, 32);
  EXPECT_THROW(MatMulNBitsFloat32(nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 64),
               std::invalid_argument);
}

TEST(MatMulNBits, RejectsOverflowBeforeDispatchOrBufferAccess) {
  const auto maximum = std::numeric_limits<std::size_t>::max();
  EXPECT_THROW(MatMulNBitsFloat32(nullptr, nullptr, nullptr, nullptr, nullptr, 1, maximum, 1, 32),
               std::invalid_argument);
  EXPECT_THROW(MatMulNBitsFloat32(nullptr, nullptr, nullptr, nullptr, nullptr, maximum, 33, 1, 32),
               std::invalid_argument);
  EXPECT_THROW(MatMulNBitsFloat32(nullptr, nullptr, nullptr, nullptr, nullptr, 1, 33, maximum, 32),
               std::invalid_argument);
  EXPECT_THROW(MatMulNBitsFloat32(nullptr, nullptr, nullptr, nullptr, nullptr, maximum, 0, 2, 32),
               std::invalid_argument);
}

} // namespace
