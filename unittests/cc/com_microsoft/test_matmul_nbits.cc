// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/com_microsoft/matmul_nbits.h"
#include "onnx_light_cpu/impl/com_microsoft/matmul_nbits_int8_tile.h"
#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/impl/simd_level.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <thread>
#include <vector>

namespace onnx_light_cpu {
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2
void MatMulNBitsInt8TileAvx2(const MatMulNBitsInt8Tile &tile);
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVXVNNI
void MatMulNBitsInt8TileAvxVnni(const MatMulNBitsInt8Tile &tile);
#endif
#if defined(ONNX_LIGHT_CPU_HAVE_AVX512VNNI) && defined(ONNX_LIGHT_CPU_HAVE_AVX512BW)
void MatMulNBitsInt8TileAvx512Vnni(const MatMulNBitsInt8Tile &tile);
#endif
} // namespace onnx_light_cpu

namespace {

using namespace onnx_light_cpu;

template <typename Storage, typename Encode, typename Decode>
void CheckPanels(DataType type, std::size_t bits, Encode encode, Decode decode,
                 std::size_t block_size = 32, int zero_mode = 0) {
  const std::size_t values_per_byte = 8 / bits;
  const std::size_t blob_size = block_size * bits / 8;
  const std::uint8_t mask = static_cast<std::uint8_t>((1U << bits) - 1U);
  const int zero_point = 1 << (bits - 1);
  const std::vector<std::size_t> depths = block_size == 32
                                              ? std::vector<std::size_t>{0, 1, 15, 31, 32, 33, 65}
                                              : std::vector<std::size_t>{0,
                                                                         1,
                                                                         block_size - 1,
                                                                         block_size,
                                                                         block_size + 1,
                                                                         2 * block_size + 3,
                                                                         5 * block_size - 1};
  for (std::size_t m : {0u, 1u, 2u, 7u, 8u, 9u, 17u}) {
    if (block_size != 32 && m != 0 && m != 1 && m != 9) {
      continue;
    }
    for (std::size_t n = 1; n <= 65; ++n) {
      if (block_size != 32 && n != 1 && n != 33) {
        continue;
      }
      for (std::size_t k : depths) {
        SCOPED_TRACE(::testing::Message() << m << ',' << k << ',' << n);
        const auto blocks = (k + block_size - 1) / block_size;
        const auto zero_stride = (blocks + values_per_byte - 1) / values_per_byte;
        std::vector<std::uint8_t> packed_zeros(n * zero_stride, 0);
        std::vector<Storage> float_zeros(n * blocks);
        for (std::size_t c = 0; c < n; ++c) {
          for (std::size_t block = 0; block < blocks; ++block) {
            const auto zero = (c + block + 1) & mask;
            packed_zeros[c * zero_stride + block / values_per_byte] |=
                static_cast<std::uint8_t>(zero << ((block % values_per_byte) * bits));
            float_zeros[c * blocks + block] = encode(static_cast<float>(zero) + 0.5f);
          }
        }
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
          for (int repeat = 0; repeat < (block_size == 32 ? 2 : 1); ++repeat) {
            MatMulNBits(a.data(), packed.data(), scales.data(), with_bias ? bias.data() : nullptr,
                        output.data(), type, m, k, n, bits, block_size,
                        kDefaultMatMulNBitsExecutionTuning,
                        zero_mode == 1   ? static_cast<const void *>(packed_zeros.data())
                        : zero_mode == 2 ? static_cast<const void *>(float_zeros.data())
                                         : nullptr,
                        zero_mode != 2);
            for (std::size_t r = 0; r < m; ++r) {
              for (std::size_t c = 0; c < n; ++c) {
                float expected = with_bias ? decode(bias[c]) : 0;
                for (std::size_t p = 0; p < k; ++p) {
                  const auto index = c * blocks + p / block_size;
                  const std::size_t offset = p % block_size;
                  const int q = (packed[index * blob_size + offset / values_per_byte] >>
                                 ((offset % values_per_byte) * bits)) &
                                mask;
                  const float zero = zero_mode == 0   ? zero_point
                                     : zero_mode == 1 ? ((c + p / block_size + 1) & mask)
                                                      : decode(float_zeros[index]);
                  expected += decode(a[r * k + p]) * ((q - zero) * decode(scales[index]));
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

TEST(MatMulNBits, Non32GroupsAndExplicitZeroPointsAllTypes) {
  for (std::size_t block_size : {16u, 64u, 128u, 256u}) {
    for (std::size_t bits : {2u, 4u, 8u}) {
      for (int zero_mode : {0, 1, 2}) {
        SCOPED_TRACE(::testing::Message() << bits << ',' << block_size << ',' << zero_mode);
        CheckPanels<float>(
            DataType::FLOAT, bits, [](float x) { return x; }, [](float x) { return x; }, block_size,
            zero_mode);
        CheckPanels<std::uint16_t>(DataType::FLOAT16, bits, detail::FloatToFloat16Bits,
                                   detail::Float16BitsToFloat, block_size, zero_mode);
        CheckPanels<std::uint16_t>(DataType::BFLOAT16, bits, detail::FloatToBFloat16Bits,
                                   detail::Bfloat16BitsToFloat, block_size, zero_mode);
      }
    }
  }
}

TEST(MatMulNBits, EncodedInt3DirectAffineMatchesReferenceAtByteAndBlockBoundaries) {
  constexpr std::size_t k = 35;
  constexpr std::size_t n = 2;
  constexpr std::size_t blocks = 2;
  std::vector<std::uint8_t> packed(n * blocks * 12, 0), zero_points{4, 1, 7, 0};
  std::vector<float> a(k), scales{0.5f, 0.25f, 1.0f, 2.0f}, output(n);
  for (std::size_t p = 0; p < k; ++p) {
    a[p] = static_cast<float>(static_cast<int>(p % 7) - 3) / 4;
  }
  for (std::size_t column = 0; column < n; ++column) {
    for (std::size_t p = 0; p < k; ++p) {
      const unsigned code = static_cast<unsigned>((column * 3 + p * 5) % 8);
      const std::size_t bit = (column * blocks + p / 32) * 96 + (p % 32) * 3;
      packed[bit / 8] |= static_cast<std::uint8_t>(code << (bit % 8));
      if (bit % 8 > 5) {
        packed[bit / 8 + 1] |= static_cast<std::uint8_t>(code >> (8 - bit % 8));
      }
    }
  }
  for (bool asymmetric : {false, true}) {
    const std::uint8_t *zeros = asymmetric ? zero_points.data() : nullptr;
    MatMulNBitsEncodedInt3(a.data(), packed.data(), scales.data(), zeros, nullptr, output.data(),
                           DataType::FLOAT, 1, k, n);
    for (std::size_t column = 0; column < n; ++column) {
      float expected = 0;
      for (std::size_t p = 0; p < k; ++p) {
        const std::size_t block = column * blocks + p / 32;
        const int code = static_cast<int>((column * 3 + p * 5) % 8);
        expected += a[p] * (code - (zeros ? zeros[block] : 4)) * scales[block];
      }
      EXPECT_NEAR(output[column], expected, 1e-5f);
    }
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

struct ConcurrentExecutor {
  int dispatches = 0;
  std::int64_t blocks = 0;
  static void Run(void *context, std::int64_t count, void *task_context, ExecutionBlockFn task) {
    auto &self = *static_cast<ConcurrentExecutor *>(context);
    ++self.dispatches;
    self.blocks = count;
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(count));
    for (std::int64_t block = 0; block < count; ++block) {
      workers.emplace_back([=] { task(task_context, block); });
    }
    for (auto &worker : workers) {
      worker.join();
    }
  }
};

TEST(MatMulNBits, PreparedInt8QwenShapesAndTails) {
  for (const auto [k, n] : {std::pair{512u, 32000u}, std::pair{512u, 1376u}, std::pair{1376u, 512u},
                            std::pair{512u, 1024u}, std::pair{512u, 512u}, std::pair{96u, 13u}}) {
    SCOPED_TRACE(::testing::Message() << k << ',' << n);
    const std::size_t rows = n == 13 ? 9 : 128;
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
        ConcurrentExecutor executor;
        ExecutionExecutorView view{&executor, threads, &ConcurrentExecutor::Run};
        {
          ExecutionExecutorScope scope(&view);
          MatMulNBitsInt8Float32(a.data(), packed.data(), sums.data(), packed_scales.data(),
                                 with_bias ? bias.data() : nullptr, y.data(), rows, k, n,
                                 {1, 16, threads});
        }
        EXPECT_EQ(executor.dispatches > 0, threads > 1);
        if (threads > 1) {
          EXPECT_GT(executor.blocks, 1);
          EXPECT_LE(executor.blocks, threads);
        }
        for (std::size_t row :
             {0u, static_cast<unsigned>(rows / 2), static_cast<unsigned>(rows - 1)}) {
          for (std::size_t col = 0; col < n; ++col) {
            float expected = with_bias ? bias[col] : 0.0f;
            for (std::size_t block = 0; block < blocks; ++block) {
              float maximum = 0;
              for (std::size_t p = 0; p < 32; ++p) {
                maximum = std::max(maximum, std::abs(a[row * k + block * 32 + p]));
              }
              std::int32_t dot = 0;
              for (std::size_t p = 0; p < 32; ++p) {
                const float value = a[row * k + block * 32 + p];
                const int q = static_cast<int>(std::nearbyint(value * (127.0f / maximum)));
                dot += q * (static_cast<int>(b[(col * blocks + block) * 32 + p]) - 128);
              }
              expected +=
                  static_cast<float>(dot) * (maximum / 127.0f) * scales[col * blocks + block];
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

TEST(MatMulNBits, PreparedInt8DecodeReusesPackedWeights) {
  for (const auto [k, n] : {std::pair{512u, 32000u}, std::pair{512u, 1376u}, std::pair{1376u, 512u},
                            std::pair{512u, 1024u}, std::pair{512u, 512u}, std::pair{96u, 13u}}) {
    SCOPED_TRACE(::testing::Message() << k << ',' << n);
    const std::size_t blocks = k / 32;
    const std::size_t slots = ((n + 7) / 8) * blocks * 8;
    std::vector<std::uint8_t> b(n * k), weights(slots * 32);
    std::vector<std::int32_t> sums(slots);
    std::vector<float> scales(n * blocks), packed_scales(slots), bias(n);
    for (std::size_t i = 0; i < b.size(); ++i) {
      b[i] = static_cast<std::uint8_t>((i * 37 + i / 29) % 256);
    }
    for (std::size_t i = 0; i < scales.size(); ++i) {
      scales[i] = static_cast<float>(i % 7 + 1) / 1024.0f;
    }
    for (std::size_t i = 0; i < n; ++i) {
      bias[i] = static_cast<float>(i % 5) / 16;
    }
    PackMatMulNBitsInt8(b.data(), scales.data(), weights.data(), sums.data(), packed_scales.data(),
                        k, n);
    const auto original_weights = weights;
    for (std::size_t rows : {1u, 2u, 4u}) {
      std::vector<float> a(rows * k), y(rows * n, -42.0f);
      for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<float>(static_cast<int>(i * 17 % 127) - 63) / 63.0f;
      }
      const auto scratch_size = MatMulNBitsInt8WorkspaceElements(rows, k, n);
      EXPECT_EQ(scratch_size * sizeof(double), rows * (k + blocks * sizeof(double)));
      std::vector<double> scratch(scratch_size + 1, -42.0);
      ConcurrentExecutor executor;
      ExecutionExecutorView view{&executor, 4, &ConcurrentExecutor::Run};
      for (int repeat = 0; repeat < 2; ++repeat) {
        ExecutionExecutorScope scope(&view);
        MatMulNBitsInt8Float32(a.data(), weights.data(), sums.data(), packed_scales.data(),
                               bias.data(), y.data(), rows, k, n, {1, 1, 4},
                               {scratch.data(), scratch_size});
        for (std::size_t row = 0; row < rows; ++row) {
          for (std::size_t col : {0u, 7u, 8u, 12u, 127u, 511u, 1023u, 1375u, 31999u}) {
            if (col >= n) {
              continue;
            }
            float expected = bias[col];
            for (std::size_t block = 0; block < blocks; ++block) {
              float maximum = 0.0f;
              for (std::size_t p = 0; p < 32; ++p) {
                maximum = std::max(maximum, std::abs(a[row * k + block * 32 + p]));
              }
              std::int32_t dot = 0;
              for (std::size_t p = 0; p < 32; ++p) {
                const auto value = a[row * k + block * 32 + p];
                const int q = maximum == 0.0f
                                  ? 0
                                  : static_cast<int>(std::nearbyint(value * (127.0f / maximum)));
                dot += q * (static_cast<int>(b[(col * blocks + block) * 32 + p]) - 128);
              }
              expected +=
                  static_cast<float>(dot) * (maximum / 127.0f) * scales[col * blocks + block];
            }
            EXPECT_NEAR(y[row * n + col], expected, 0.0002f);
          }
        }
      }
      EXPECT_EQ(scratch.back(), -42.0);
      EXPECT_EQ(weights, original_weights);
      if (n == 13) {
        EXPECT_EQ(executor.dispatches, 0);
      } else {
        EXPECT_GT(executor.dispatches, 0);
        EXPECT_LE(executor.blocks, 4);
        const auto dispatches = executor.dispatches;
        {
          ExecutionExecutorScope scope(&view);
          MatMulNBitsInt8Float32(a.data(), weights.data(), sums.data(), packed_scales.data(),
                                 bias.data(), y.data(), rows, k, n, {rows * n + 1, 1, 4},
                                 {scratch.data(), scratch_size});
        }
        EXPECT_EQ(executor.dispatches, dispatches);
      }
    }
  }
}

TEST(MatMulNBits, PreparedInt8DecodeHonorsActualOutputThresholdWithColumnTails) {
  constexpr std::size_t k = 512;
  for (std::size_t rows : {1u, 2u, 4u}) {
    for (std::size_t n : {511u, 512u, 513u}) {
      SCOPED_TRACE(::testing::Message() << rows << ',' << n);
      const std::size_t blocks = k / 32;
      const std::size_t slots = (n / 8 + (n % 8 != 0)) * blocks * 8;
      std::vector<float> a(rows * k, 0.5f), scales(n * blocks, 0.25f), packed_scales(slots),
          y(rows * n + 1, -42.0f);
      std::vector<std::uint8_t> b(n * k, 129), weights(slots * 32);
      std::vector<std::int32_t> sums(slots);
      PackMatMulNBitsInt8(b.data(), scales.data(), weights.data(), sums.data(),
                          packed_scales.data(), k, n);
      InlineExecutor executor;
      ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run};
      ExecutionExecutorScope scope(&view);
      const auto run = [&](std::size_t threshold) {
        MatMulNBitsInt8Float32(a.data(), weights.data(), sums.data(), packed_scales.data(), nullptr,
                               y.data(), rows, k, n, {threshold, 1, 4});
        for (std::size_t index = 0; index < rows * n; ++index) {
          EXPECT_FLOAT_EQ(y[index], 64.0f);
        }
        EXPECT_FLOAT_EQ(y.back(), -42.0f);
      };
      run(rows * n + 1);
      EXPECT_EQ(executor.dispatches, 0);
      run(rows * n);
      EXPECT_EQ(executor.dispatches, 1);
      EXPECT_GT(executor.blocks, 1);
      EXPECT_LE(executor.blocks, 4);
      {
        detail::ExecutionRegionScope nested;
        run(rows * n);
      }
      EXPECT_EQ(executor.dispatches, 1);
    }
  }
}

TEST(MatMulNBits, PreparedInt8ZeroReductionReturnsBiasWithoutDispatch) {
  InlineExecutor executor;
  ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run};
  ExecutionExecutorScope scope(&view);
  for (std::size_t rows : {1u, 2u, 4u, 5u}) {
    for (std::size_t n : {1u, 13u, 512u}) {
      SCOPED_TRACE(::testing::Message() << rows << ',' << n);
      std::vector<float> bias(n), y(rows * n + 1, -42.0f);
      for (std::size_t column = 0; column < n; ++column) {
        bias[column] = static_cast<float>(column) - 7.0f;
      }
      EXPECT_EQ(MatMulNBitsInt8WorkspaceElements(rows, 0, n), 0U);
      for (bool with_bias : {false, true}) {
        MatMulNBitsInt8Float32(nullptr, nullptr, nullptr, nullptr,
                               with_bias ? bias.data() : nullptr, y.data(), rows, 0, n);
        for (std::size_t index = 0; index < rows * n; ++index) {
          EXPECT_FLOAT_EQ(y[index], with_bias ? bias[index % n] : 0.0f);
        }
        EXPECT_FLOAT_EQ(y.back(), -42.0f);
      }
    }
  }
  EXPECT_EQ(executor.dispatches, 0);
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
                         32, 8, {1, 16, 4});
  EXPECT_EQ(executor.dispatches, 0);
  MatMulNBitsInt8Float32(a.data(), packed.data(), sums.data(), scales.data(), nullptr, y.data(), 9,
                         32, 8, {1, 16, 4});
  EXPECT_EQ(executor.dispatches, 1);
  EXPECT_GT(executor.blocks, 1);
  for (float value : y) {
    EXPECT_FLOAT_EQ(value, 4.0f);
  }
  std::fill(y.begin(), y.end(), -1.0f);
  {
    detail::ExecutionRegionScope nested;
    MatMulNBitsInt8Float32(a.data(), packed.data(), sums.data(), scales.data(), nullptr, y.data(),
                           9, 32, 8, {1, 16, 4});
  }
  EXPECT_EQ(executor.dispatches, 1);
  for (float value : y) {
    EXPECT_FLOAT_EQ(value, 4.0f);
  }
  MatMulNBitsInt8Float32(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 32, 8,
                         {1, 16, 4});
  EXPECT_EQ(executor.dispatches, 1);
}

TEST(MatMulNBits, PreparedInt8PreservesExtremeScaleProducts) {
  constexpr std::size_t rows = 9;
  constexpr std::size_t k = 32;
  constexpr std::size_t n = 9;
  struct TestCase {
    float activation;
    std::uint8_t weight;
    float scale;
  };
  for (const TestCase test : {TestCase{1e30f, 128, 1e12f}, TestCase{-1e30f, 128, 1e12f},
                              TestCase{1e-20f, 129, 1e-24f}, TestCase{-1e-20f, 129, 1e-24f}}) {
    SCOPED_TRACE(::testing::Message() << test.activation << ',' << test.scale);
    std::vector<float> a(rows * k, test.activation), scales(n, test.scale), packed_scales(16),
        y(rows * n, -1.0f);
    std::vector<std::uint8_t> b(n * k, test.weight), packed(2 * 256);
    std::vector<std::int32_t> sums(16);
    PackMatMulNBitsInt8(b.data(), scales.data(), packed.data(), sums.data(), packed_scales.data(),
                        k, n);
    const float expected =
        static_cast<float>(static_cast<double>(test.activation) *
                           (static_cast<int>(test.weight) - 128) * test.scale * k);
    if (test.weight != 128) {
      ASSERT_NE(expected, 0.0f);
    }
    for (std::int64_t threads : {1, 4}) {
      ConcurrentExecutor executor;
      ExecutionExecutorView view{&executor, threads, &ConcurrentExecutor::Run};
      ExecutionExecutorScope scope(&view);
      MatMulNBitsInt8Float32(a.data(), packed.data(), sums.data(), packed_scales.data(), nullptr,
                             y.data(), rows, k, n, {1, 16, threads});
      for (float value : y) {
        EXPECT_EQ(value, expected);
      }
    }
  }
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
                         y.data(), 3, k, n, {1, 16, 2});
  for (std::size_t column = 0; column < n; ++column) {
    EXPECT_NEAR(y[column], 32e-10f, 1e-10f);
    EXPECT_TRUE(std::isnan(y[n + column]));
    EXPECT_TRUE(std::isinf(y[2 * n + column]));
  }
}

TEST(MatMulNBits, PreparedInt8PreservesNonFiniteScalesAndBiasedOverflow) {
  constexpr std::size_t rows = 9;
  constexpr std::size_t k = 32;
  constexpr std::size_t n = 9;
  for (float scale :
       {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(), 12.0f}) {
    for (bool zero_weight : {false, true}) {
      const bool finite = std::isfinite(scale);
      std::vector<float> a(rows * k, finite ? 1e36f : 0.5f), scales(n, scale),
          bias(n, finite ? -1e38f : 0.0f), packed_scales(16), y(rows * n), expected(rows * n);
      std::vector<std::uint8_t> b(n * k, 129), packed(2 * 256);
      std::vector<std::int32_t> sums(16);
      if (zero_weight) {
        for (std::size_t column = 0; column < n; ++column) {
          b[column * k] = 128;
        }
      }
      MatMulNBits(a.data(), b.data(), scales.data(), bias.data(), expected.data(), DataType::FLOAT,
                  rows, k, n, 8, 32);
      PackMatMulNBitsInt8(b.data(), scales.data(), packed.data(), sums.data(), packed_scales.data(),
                          k, n);
      for (std::int64_t threads : {1, 4}) {
        ConcurrentExecutor executor;
        ExecutionExecutorView view{&executor, threads, &ConcurrentExecutor::Run};
        ExecutionExecutorScope scope(&view);
        MatMulNBitsInt8Float32(a.data(), packed.data(), sums.data(), packed_scales.data(),
                               bias.data(), y.data(), rows, k, n, {1, 16, threads});
        for (std::size_t i = 0; i < y.size(); ++i) {
          if (std::isnan(expected[i])) {
            EXPECT_TRUE(std::isnan(y[i]));
          } else if (std::isinf(expected[i])) {
            EXPECT_EQ(y[i], expected[i]);
          } else {
            ASSERT_TRUE(std::isfinite(y[i]));
            EXPECT_NEAR(y[i] / expected[i], 1.0f, 1e-5f);
          }
        }
      }
    }
  }
}

TEST(MatMulNBits, PreparedInt8FullRangeAcrossEveryAvailableIsa) {
  using TileFn = void (*)(const MatMulNBitsInt8Tile &);
  std::vector<TileFn> kernels;
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2
  if (DetectSimdLevel() >= SimdLevel::kAVX2) {
    kernels.push_back(MatMulNBitsInt8TileAvx2);
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVXVNNI
  if (CpuSupportsAvxVnni()) {
    kernels.push_back(MatMulNBitsInt8TileAvxVnni);
  }
#endif
#if defined(ONNX_LIGHT_CPU_HAVE_AVX512VNNI) && defined(ONNX_LIGHT_CPU_HAVE_AVX512BW)
  if (CpuSupportsAvx512Vnni() && CpuSupportsAvx512BW()) {
    kernels.push_back(MatMulNBitsInt8TileAvx512Vnni);
  }
#endif
  if (kernels.empty()) {
    GTEST_SKIP() << "No compiled INT8 SIMD implementation is available on this CPU.";
  }
  constexpr std::size_t n = 13;
  for (std::size_t k : {32u, 96u}) {
    const std::size_t blocks = k / 32;
    std::vector<std::uint8_t> activations(4 * k), b(n * k), packed(2 * blocks * 256);
    std::vector<std::int32_t> sums(2 * blocks * 8);
    std::vector<float> scales(n * blocks, 0.125f), packed_scales(sums.size()), bias(n, 0.5f),
        y(4 * n + 1, -42.0f);
    std::vector<double> activation_scales(4 * blocks, 0.25);
    for (std::size_t i = 0; i < activations.size(); ++i) {
      activations[i] = i % 4 < 2 ? 255 : 1;
    }
    for (std::size_t i = 0; i < b.size(); ++i) {
      b[i] = i % 4 < 2 ? 255 : 0;
    }
    PackMatMulNBitsInt8(b.data(), scales.data(), packed.data(), sums.data(), packed_scales.data(),
                        k, n);
    for (const auto kernel : kernels) {
      for (std::size_t rows = 1; rows <= 4; ++rows) {
        for (std::size_t group = 0; group < 2; ++group) {
          const std::size_t index = group * blocks * 8;
          kernel({activations.data(), activation_scales.data(), packed.data() + index * 32,
                  sums.data() + index, packed_scales.data() + index, bias.data() + group * 8,
                  y.data() + group * 8, rows, k, n, std::min<std::size_t>(8, n - group * 8)});
        }
        for (std::size_t row = 0; row < rows; ++row) {
          for (std::size_t column = 0; column < n; ++column) {
            float expected = bias[column];
            for (std::size_t p = 0; p < k; ++p) {
              expected += (static_cast<int>(activations[row * k + p]) - 128) * 0.25f *
                          (static_cast<int>(b[column * k + p]) - 128) * 0.125f;
            }
            EXPECT_EQ(y[row * n + column], expected);
          }
        }
        EXPECT_EQ(y.back(), -42.0f);
      }
    }
  }
}

TEST(MatMulNBits, PreparedInt8WorkspaceIsCheckedBeforeDispatch) {
  constexpr std::size_t rows = 9, k = 32, n = 9;
  std::vector<float> a(rows * k, 0.5f), scales(n, 0.25f), packed_scales(16), y(rows * n);
  std::vector<std::uint8_t> b(n * k, 129), packed(512);
  std::vector<std::int32_t> sums(16);
  PackMatMulNBitsInt8(b.data(), scales.data(), packed.data(), sums.data(), packed_scales.data(), k,
                      n);
  InlineExecutor executor;
  ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run};
  ExecutionExecutorScope scope(&view);
  const MatMulNBitsExecutionTuning tuning{1, 16, 4};
  const auto count = MatMulNBitsInt8WorkspaceElements(rows, k, n, tuning);
  EXPECT_EQ(count * sizeof(double), 4 * 5 * k);
  std::vector<double> workspace(count + 1, -42.0);
  MatMulNBitsInt8Float32(a.data(), packed.data(), sums.data(), packed_scales.data(), nullptr,
                         y.data(), rows, k, n, tuning, {workspace.data(), count});
  EXPECT_EQ(workspace.back(), -42.0);
  for (float value : y) {
    EXPECT_EQ(value, 4.0f);
  }
  EXPECT_EQ(executor.dispatches, 1);
  EXPECT_THROW(MatMulNBitsInt8Float32(a.data(), packed.data(), sums.data(), packed_scales.data(),
                                      nullptr, y.data(), rows, k, n, tuning,
                                      {workspace.data(), count - 1}),
               std::invalid_argument);
  EXPECT_EQ(executor.dispatches, 1);
  EXPECT_THROW(MatMulNBitsInt8WorkspaceElements(rows, k + 1, n, tuning), std::invalid_argument);
  if constexpr (sizeof(std::size_t) >= sizeof(std::int64_t)) {
    EXPECT_THROW(
        MatMulNBitsInt8WorkspaceElements(std::numeric_limits<std::size_t>::max(), k, n, tuning),
        std::invalid_argument);
  }
  EXPECT_EQ(MatMulNBitsInt8WorkspaceElements(0, k, n, tuning), 0U);
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
  EXPECT_NO_THROW(MatMulNBitsFloat32(nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 64));
  EXPECT_THROW(MatMulNBitsFloat32(nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 24),
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
  InlineExecutor executor;
  ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run};
  ExecutionExecutorScope scope(&view);
  for (DataType type : {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16}) {
    SCOPED_TRACE(static_cast<int>(type));
    const std::size_t width = type == DataType::FLOAT ? sizeof(float) : sizeof(std::uint16_t);
    const auto check = [&](std::size_t rows, std::size_t k, std::size_t n) {
      EXPECT_THROW(MatMulNBitsEncodedInt3(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                          type, rows, k, n, {1, 1, 4}),
                   std::invalid_argument);
    };
    check(maximum / 33 + 1, 33, 1);
    check(1, 33, maximum / 24 + 1);
    check(1, maximum / width + 1, 1);
    check(maximum / width + 1, 0, 1);
    check(maximum, 0, 2);
    check(maximum, 0, 1);
    EXPECT_NO_THROW(MatMulNBitsEncodedInt3(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                           type, 0, maximum, maximum));
    EXPECT_NO_THROW(MatMulNBitsEncodedInt3(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                           type, maximum, maximum, 0));
  }
  EXPECT_EQ(executor.dispatches, 0);
}

} // namespace
