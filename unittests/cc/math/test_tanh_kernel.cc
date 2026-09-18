// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/impl/math/math_kernels.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

using FloatKernel = void (*)(const float *, float *, std::size_t);
using HalfKernel = void (*)(const std::uint16_t *, std::uint16_t *, std::size_t);

std::vector<FloatKernel> FloatKernels() {
  std::vector<FloatKernel> kernels{onnx_light_cpu::TanhFloat32, onnx_light_cpu::TanhFloat32_Scalar};
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
  if (onnx_light_cpu::DetectSimdLevel() >= onnx_light_cpu::SimdLevel::kAVX2 &&
      onnx_light_cpu::CpuSupportsFma()) {
    kernels.push_back(onnx_light_cpu::TanhFloat32_AVX2_FMA);
  }
#endif
  return kernels;
}

void ExpectTanh(float input, float actual) {
  SCOPED_TRACE(::testing::Message() << "input bits=" << std::bit_cast<std::uint32_t>(input));
  const float expected = static_cast<float>(std::tanh(static_cast<double>(input)));
  if (std::isnan(input)) {
    EXPECT_TRUE(std::isnan(actual));
    EXPECT_EQ(std::signbit(actual), std::signbit(input));
    EXPECT_NE(std::bit_cast<std::uint32_t>(actual) & 0x00400000u, 0u);
  } else if (std::fabs(input) <= std::numeric_limits<float>::min() || std::isinf(input)) {
    EXPECT_EQ(std::bit_cast<std::uint32_t>(actual), std::bit_cast<std::uint32_t>(expected));
  } else {
    EXPECT_NEAR(actual, expected, std::fabs(expected) * 5e-7f);
    EXPECT_LE(std::fabs(actual), 1.0f);
    EXPECT_EQ(std::signbit(actual), std::signbit(input));
  }
}

std::vector<float> Corpus() {
  const float inf = std::numeric_limits<float>::infinity();
  const float tiny = std::numeric_limits<float>::denorm_min();
  const float normal = std::numeric_limits<float>::min();
  return {-inf,
          -std::numeric_limits<float>::max(),
          -100.0f,
          -10.0f,
          -9.0f,
          -1.0f,
          -0.25f,
          -0x1p-12f,
          -normal,
          -tiny,
          -0.0f,
          0.0f,
          tiny,
          std::nextafter(normal, 0.0f),
          normal,
          0x1p-12f,
          std::nextafter(0.25f, 0.0f),
          0.25f,
          std::nextafter(0.25f, 1.0f),
          1.0f,
          9.0f,
          10.0f,
          100.0f,
          std::numeric_limits<float>::max(),
          inf,
          std::bit_cast<float>(0x7fc12345u),
          std::bit_cast<float>(0xffc12345u),
          std::bit_cast<float>(0x7f812345u),
          std::bit_cast<float>(0xff812345u)};
}

TEST(TanhKernel, EmptyNullBuffers) {
  for (const auto kernel : FloatKernels()) {
    kernel(nullptr, nullptr, 0);
  }
  onnx_light_cpu::TanhFloat16(nullptr, nullptr, 0);
  onnx_light_cpu::TanhBFloat16(nullptr, nullptr, 0);
}

TEST(TanhKernel, Float32SizesTailsUnalignedInPlaceAndSpecialValues) {
  const auto corpus = Corpus();
  std::vector<std::size_t> sizes{127, 128, 129, 1023, 1024, 1025, 65535};
  for (std::size_t count = 0; count <= 65; ++count) {
    sizes.push_back(count);
  }
  for (const auto kernel : FloatKernels()) {
    for (const auto count : sizes) {
      for (const std::size_t offset : {1, 3, 8}) {
        for (const bool in_place : {false, true}) {
          SCOPED_TRACE(::testing::Message()
                       << "count=" << count << " offset=" << offset << " inplace=" << in_place);
          std::vector<float> input(count + offset + 1, 42.0f);
          std::vector<float> output(input.size(), 42.0f);
          for (std::size_t i = 0; i < count; ++i) {
            input[offset + i] = corpus[i % corpus.size()];
          }
          float *result = (in_place ? input.data() : output.data()) + offset;
          kernel(input.data() + offset, result, count);
          for (std::size_t i = 0; i < count; ++i) {
            ExpectTanh(corpus[i % corpus.size()], result[i]);
          }
          for (const auto *buffer : {&input, &output}) {
            for (std::size_t i = 0; i < offset; ++i) {
              EXPECT_EQ((*buffer)[i], 42.0f);
            }
            EXPECT_EQ(buffer->back(), 42.0f);
          }
        }
      }
    }
  }
}

TEST(TanhKernel, Float32DenseAndRandomBitPatterns) {
  std::vector<float> input(131072), output(input.size());
  std::uint32_t state = 123;
  for (std::size_t i = 0; i < input.size() / 2; ++i) {
    input[2 * i] = -10.0f + 20.0f * static_cast<float>(i) / 65535.0f;
    state = state * 1664525u + 1013904223u;
    input[2 * i + 1] = std::bit_cast<float>(state);
  }
  for (const auto kernel : FloatKernels()) {
    kernel(input.data(), output.data(), input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
      ExpectTanh(input[i], output[i]);
    }
  }
}

float HalfToFloat(std::uint16_t bits, bool bfloat16) {
  return bfloat16 ? onnx_light_cpu::detail::Bfloat16BitsToFloat(bits)
                  : onnx_light_cpu::detail::Float16BitsToFloat(bits);
}

std::uint16_t FloatToHalf(float value, bool bfloat16) {
  return bfloat16 ? onnx_light_cpu::detail::FloatToBFloat16Bits(value)
                  : onnx_light_cpu::detail::FloatToFloat16Bits(value);
}

void ExpectHalf(std::uint16_t input, std::uint16_t actual, bool bfloat16) {
  const float value = HalfToFloat(input, bfloat16);
  const float expected = static_cast<float>(std::tanh(static_cast<double>(value)));
  if (std::isnan(value)) {
    EXPECT_TRUE(std::isnan(HalfToFloat(actual, bfloat16)));
    EXPECT_EQ(actual & 0x8000u, input & 0x8000u);
  } else {
    const auto bits = FloatToHalf(expected, bfloat16);
    // Float32 approximation may cross a half-precision rounding midpoint.
    EXPECT_LE(std::abs(static_cast<int>(actual) - static_cast<int>(bits)), 1);
    if (expected == 0.0f || std::fabs(value) < 0x1p-12f || std::isinf(value)) {
      EXPECT_EQ(actual, bits);
    }
  }
}

TEST(TanhKernel, ExhaustiveFloat16AndBFloat16) {
  for (const bool bfloat16 : {false, true}) {
    const HalfKernel kernel = bfloat16 ? onnx_light_cpu::TanhBFloat16 : onnx_light_cpu::TanhFloat16;
    for (const bool in_place : {false, true}) {
      std::vector<std::uint16_t> input(65538, 42), output(input.size(), 42);
      for (std::uint32_t bits = 0; bits < 65536; ++bits) {
        input[bits + 1] = static_cast<std::uint16_t>(bits);
      }
      auto *result = (in_place ? input.data() : output.data()) + 1;
      kernel(input.data() + 1, result, 65536);
      for (std::uint32_t bits = 0; bits < 65536; ++bits) {
        SCOPED_TRACE(::testing::Message() << "bits=" << bits << " bfloat16=" << bfloat16);
        ExpectHalf(static_cast<std::uint16_t>(bits), result[bits], bfloat16);
      }
      EXPECT_EQ(input.front(), 42);
      EXPECT_EQ(input.back(), 42);
      EXPECT_EQ(output.front(), 42);
      EXPECT_EQ(output.back(), 42);
    }
  }
}

TEST(TanhKernel, HalfSizesTailsAndConversionBlockBoundaries) {
  std::vector<std::size_t> sizes{1023, 1024, 1025, 2047, 2048, 2049};
  for (std::size_t count = 0; count <= 65; ++count) {
    sizes.push_back(count);
  }
  const auto corpus = Corpus();
  for (const bool bfloat16 : {false, true}) {
    const HalfKernel kernel = bfloat16 ? onnx_light_cpu::TanhBFloat16 : onnx_light_cpu::TanhFloat16;
    for (const auto count : sizes) {
      for (const bool in_place : {false, true}) {
        std::vector<std::uint16_t> input(count + 2, 42), output(input.size(), 42);
        for (std::size_t i = 0; i < count; ++i) {
          input[i + 1] = FloatToHalf(corpus[i % corpus.size()], bfloat16);
        }
        auto *result = (in_place ? input.data() : output.data()) + 1;
        kernel(input.data() + 1, result, count);
        for (std::size_t i = 0; i < count; ++i) {
          ExpectHalf(FloatToHalf(corpus[i % corpus.size()], bfloat16), result[i], bfloat16);
        }
        EXPECT_EQ(input.front(), 42);
        EXPECT_EQ(input.back(), 42);
        EXPECT_EQ(output.front(), 42);
        EXPECT_EQ(output.back(), 42);
      }
    }
  }
}

struct InlineExecutor {
  std::int64_t dispatches = 0;
  std::int64_t blocks = 0;

  static void Run(void *context, std::int64_t count, void *task_context,
                  onnx_light_cpu::ExecutionBlockFn task) {
    auto &self = *static_cast<InlineExecutor *>(context);
    ++self.dispatches;
    self.blocks = count;
    for (std::int64_t i = count; i > 0; --i) {
      task(task_context, i - 1);
    }
  }
};

TEST(TanhKernel, ExecutorSmallLargeAndNestedRanges) {
  InlineExecutor executor;
  const onnx_light_cpu::ExecutionExecutorView view{&executor, 4, InlineExecutor::Run};
  const onnx_light_cpu::ExecutionExecutorScope scope(&view);
  constexpr std::size_t count = 1048579;
  std::vector<float> input(count), output(count, 42.0f);
  for (std::size_t i = 0; i < count; ++i) {
    input[i] = static_cast<float>(i % 101) / 13.0f - 4.0f;
  }
  onnx_light_cpu::TanhFloat32(input.data(), output.data(), 65);
  EXPECT_EQ(executor.dispatches, 0);
  onnx_light_cpu::TanhFloat32(input.data(), output.data(), count);
  EXPECT_EQ(executor.dispatches, 1);
  EXPECT_GT(executor.blocks, 1);
  for (std::size_t i = 0; i < count; ++i) {
    ExpectTanh(input[i], output[i]);
  }
  for (const bool bfloat16 : {false, true}) {
    const HalfKernel kernel = bfloat16 ? onnx_light_cpu::TanhBFloat16 : onnx_light_cpu::TanhFloat16;
    std::vector<std::uint16_t> half_input(count), half_output(count, 42);
    for (std::size_t i = 0; i < count; ++i) {
      half_input[i] = FloatToHalf(input[i], bfloat16);
    }
    const auto before = executor.dispatches;
    kernel(half_input.data(), half_output.data(), 65);
    EXPECT_EQ(executor.dispatches, before);
    kernel(half_input.data(), half_output.data(), count);
    EXPECT_EQ(executor.dispatches, before + 1);
    EXPECT_GT(executor.blocks, 1);
    for (std::size_t i = 0; i < count; ++i) {
      ExpectHalf(half_input[i], half_output[i], bfloat16);
    }
    const auto nested_before = executor.dispatches;
    onnx_light_cpu::ExecuteRanges(2, onnx_light_cpu::ExecutionSchedule{1, 1, 2}, 1,
                                  [&](std::int64_t, std::int64_t) {
                                    EXPECT_TRUE(onnx_light_cpu::ExecutionInParallelRegion());
                                    onnx_light_cpu::TanhFloat32(input.data(), output.data(), count);
                                    kernel(half_input.data(), half_output.data(), count);
                                  });
    EXPECT_EQ(executor.dispatches, nested_before + 1);
  }
}

} // namespace
