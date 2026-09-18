// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/com_microsoft/matmul_nbits.h"
#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

namespace {

using namespace onnx_light_cpu;

template <typename Storage, typename Encode, typename Decode>
void CheckPanels(DataType type, Encode encode, Decode decode) {
  for (std::size_t m : {0u, 1u, 2u, 7u, 8u, 9u, 17u}) {
    for (std::size_t n = 1; n <= 65; ++n) {
      for (std::size_t k : {0u, 1u, 15u, 31u, 32u, 33u, 65u}) {
        SCOPED_TRACE(::testing::Message() << m << ',' << k << ',' << n);
        const auto blocks = (k + 31) / 32;
        std::vector<Storage> a(m * k), scales(n * blocks), bias(n), output(m * n + 1);
        std::vector<std::uint8_t> packed(n * blocks * 16);
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
                        output.data(), type, m, k, n, 4, 32);
            for (std::size_t r = 0; r < m; ++r) {
              for (std::size_t c = 0; c < n; ++c) {
                float expected = with_bias ? decode(bias[c]) : 0;
                for (std::size_t p = 0; p < k; ++p) {
                  const auto index = c * blocks + p / 32;
                  const int q = (packed[index * 16 + (p % 32) / 2] >> ((p % 2) * 4)) & 15;
                  expected += decode(a[r * k + p]) * (q - 8) * decode(scales[index]);
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
  CheckPanels<float>(DataType::FLOAT, [](float x) { return x; }, [](float x) { return x; });
  CheckPanels<std::uint16_t>(DataType::FLOAT16, detail::FloatToFloat16Bits,
                             detail::Float16BitsToFloat);
  CheckPanels<std::uint16_t>(DataType::BFLOAT16, detail::FloatToBFloat16Bits,
                             detail::Bfloat16BitsToFloat);
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

} // namespace
