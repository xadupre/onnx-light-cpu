// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/tensor/cast_kernel.h"

#include "onnx_light_cpu/impl/execution.h"

#include <gtest/gtest.h>

#include <array>
#include <cfenv>
#include <cstring>
#include <vector>

namespace {

using namespace onnx_light_cpu;

constexpr std::array kLevels{SimdLevel::kNone, SimdLevel::kSSE2, SimdLevel::kAVX, SimdLevel::kAVX2,
                             SimdLevel::kAVX512};

void Compare(const void *input, DataType from, DataType to, std::size_t count,
             std::size_t source_offset = 0, std::size_t output_offset = 0) {
  const auto source_bytes = count * CastElementSize(from);
  const auto output_bytes = count * CastElementSize(to);
  std::vector<std::uint8_t> source(source_offset + source_bytes);
  if (source_bytes != 0) {
    std::memcpy(source.data() + source_offset, input, source_bytes);
  }
  std::vector<std::uint8_t> expected(output_offset + output_bytes + 16, 0xa5);
  const auto *data = source.empty() ? nullptr : source.data() + source_offset;
  CastConvert(data, from, expected.data() + output_offset, to, count, SimdLevel::kNone);
  for (auto level : kLevels) {
    SCOPED_TRACE(static_cast<int>(level));
    std::vector<std::uint8_t> actual(expected.size(), 0xa5);
    CastConvert(data, from, actual.data() + output_offset, to, count, level);
    EXPECT_EQ(actual, expected);
  }
}

TEST(CastKernel, AllHalfAndBfloat16BitPatterns) {
  std::vector<std::uint16_t> values(65536);
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<std::uint16_t>(i);
  }
  for (auto from : {DataType::FLOAT16, DataType::BFLOAT16}) {
    Compare(values.data(), from, DataType::FLOAT, values.size(), 1, 3);
  }
}

TEST(CastKernel, FloatRoundingBoundariesAndSpecialValues) {
  std::vector<std::uint32_t> values;
  // Every bfloat16 boundary, and adjacent float32 values, includes both signs,
  // subnormals, infinities and signaling/quiet NaNs with different payloads.
  for (std::uint32_t high = 0; high < 65536; ++high) {
    for (std::uint32_t low : {0u, 0x0fffu, 0x1000u, 0x1001u, 0x7fffu, 0x8000u, 0x8001u, 0xffffu}) {
      values.push_back((high << 16) | low);
    }
  }
  std::uint32_t state = 42;
  for (int i = 0; i < 65536; ++i) {
    state = state * 1664525u + 1013904223u;
    values.push_back(state);
  }
  for (auto to : {DataType::FLOAT16, DataType::BFLOAT16}) {
    Compare(values.data(), DataType::FLOAT, to, values.size(), 3, 1);
  }
}

TEST(CastKernel, EmptyTailsAndUnalignedBuffers) {
  std::array<std::uint32_t, 65> values{};
  const std::array<std::uint32_t, 9> special{0u,          0x80000000u, 0x7f800000u,
                                             0xff800000u, 0x7f800001u, 0xffc12345u,
                                             0x33800000u, 0x3f801000u, 0x477ff000u};
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = special[i % special.size()];
  }
  for (std::size_t count = 0; count <= values.size(); ++count) {
    SCOPED_TRACE(count);
    for (std::size_t offset = 0; offset < 8; ++offset) {
      for (auto half : {DataType::FLOAT16, DataType::BFLOAT16}) {
        Compare(values.data(), DataType::FLOAT, half, count, offset, 7 - offset);
        Compare(values.data(), half, DataType::FLOAT, count, offset, 7 - offset);
      }
    }
  }
}

TEST(CastKernel, ConversionPathAndForcedFallback) {
  EXPECT_STREQ(CastConversionPath(DataType::FLOAT, DataType::FLOAT16, 0), "Cast.empty");
  EXPECT_STREQ(CastConversionPath(DataType::FLOAT, DataType::FLOAT, 32), "Cast.copy");
  EXPECT_STREQ(CastConversionPath(DataType::DOUBLE, DataType::INT64, 32), "Cast.scalar");
  for (auto half : {DataType::FLOAT16, DataType::BFLOAT16}) {
    EXPECT_STREQ(CastConversionPath(DataType::FLOAT, half, 7), "Cast.scalar");
    for (auto level : {SimdLevel::kNone, SimdLevel::kSSE2}) {
      EXPECT_STREQ(CastConversionPath(DataType::FLOAT, half, 32, level), "Cast.scalar");
      EXPECT_STREQ(CastConversionPath(half, DataType::FLOAT, 32, level), "Cast.scalar");
    }
  }
  EXPECT_STREQ(CastConversionPath(DataType::FLOAT, DataType::BFLOAT16, 32, SimdLevel::kAVX),
               "Cast.scalar");
  EXPECT_STREQ(CastConversionPath(DataType::BFLOAT16, DataType::FLOAT, 32, SimdLevel::kAVX),
               "Cast.scalar");
#ifdef ONNX_LIGHT_CPU_HAVE_AVX_F16C
  if (DetectSimdLevel() >= SimdLevel::kAVX && CpuSupportsF16C()) {
    EXPECT_STREQ(CastConversionPath(DataType::FLOAT, DataType::FLOAT16, 32, SimdLevel::kAVX),
                 "Cast.float32_to_float16.f16c");
    EXPECT_STREQ(CastConversionPath(DataType::FLOAT16, DataType::FLOAT, 32, SimdLevel::kAVX),
                 "Cast.float16_to_float32.f16c");
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2
  if (DetectSimdLevel() >= SimdLevel::kAVX2) {
    EXPECT_STREQ(CastConversionPath(DataType::FLOAT, DataType::BFLOAT16, 32),
                 "Cast.float32_to_bfloat16.avx2");
    EXPECT_STREQ(CastConversionPath(DataType::BFLOAT16, DataType::FLOAT, 32),
                 "Cast.bfloat16_to_float32.avx2");
  }
#endif
}

TEST(CastKernel, RoundingModeDoesNotChangeHalfConversion) {
  const int original = std::fegetround();
  const std::array<float, 17> values{0.0f,        -0.0f,        0x1p-25f,   -0x1p-25f,   0x1.8p-24f,
                                     -0x1.8p-24f, 0x1.ffcp-15f, 0x1.002p0f, -0x1.002p0f, 0x1.006p0f,
                                     0x1.01p0f,   -0x1.01p0f,   0x1.03p0f,  65504.0f,    65520.0f,
                                     -65520.0f,   1.0f};
  for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    EXPECT_EQ(std::fesetround(mode), 0);
    for (auto half : {DataType::FLOAT16, DataType::BFLOAT16}) {
      Compare(values.data(), DataType::FLOAT, half, values.size());
    }
  }
  EXPECT_EQ(std::fesetround(original), 0);
}

TEST(CastKernel, ParallelRangesAndNestedBypass) {
  struct Executor {
    int calls = 0;
    int64_t blocks = 0;
    static void Run(void *context, int64_t count, void *task_context, ExecutionBlockFn task) {
      auto &self = *static_cast<Executor *>(context);
      ++self.calls;
      self.blocks = count;
      for (int64_t block = 0; block < count; ++block) {
        task(task_context, block);
      }
    }
  } executor;
  ExecutionExecutorView view{&executor, 4, &Executor::Run};
  ExecutionExecutorScope scope(&view);
  std::vector<float> values(262145, 1.5f);
  for (auto half : {DataType::FLOAT16, DataType::BFLOAT16}) {
    for (const auto &[from, to] :
         {std::pair{DataType::FLOAT, half}, std::pair{half, DataType::FLOAT}}) {
      const int before = executor.calls;
      Compare(values.data(), from, to, 17);
      EXPECT_EQ(executor.calls, before);
      Compare(values.data(), from, to, values.size(), 1, 3);
      EXPECT_GT(executor.calls, before);
      EXPECT_GT(executor.blocks, 1);
      EXPECT_LE(executor.blocks, view.effective_threads);
      const int after = executor.calls;
      {
        detail::ExecutionRegionScope region;
        Compare(values.data(), from, to, values.size());
      }
      EXPECT_EQ(executor.calls, after);
    }
  }
}

} // namespace
