// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/math_kernels.h"
#include "onnx_light_cpu/kernels/math/sigmoid_softmax_kernel.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace {

namespace rt = ONNX_LIGHT_NAMESPACE::core::runtime;

rt::KernelContext Context() { return rt::KernelContext(rt::OpsetId(std::string(), 13)); }

std::vector<float> Values(std::size_t count) {
  std::vector<float> values(count);
  for (std::size_t index = 0; index < count; ++index) {
    values[index] = static_cast<float>(static_cast<int>(index * 41 % 10001) - 5000) * 0.002f;
  }
  return values;
}

struct InlineExecutor {
  std::int64_t dispatches = 0;
  std::int64_t blocks = 0;
  bool nested = false;

  static void Run(void *context, std::int64_t blocks, void *task_context,
                  onnx_light_cpu::ExecutionBlockFn task) {
    auto &self = *static_cast<InlineExecutor *>(context);
    self.nested |= onnx_light_cpu::ExecutionInParallelRegion();
    ++self.dispatches;
    self.blocks = blocks;
    for (std::int64_t block = blocks; block > 0; --block) {
      task(task_context, block - 1);
    }
  }
};

TEST(OnnxLightSigmoidSoftmaxKernel, SigmoidMatchesReferenceIncludingTailsAndAliasing) {
  const onnx_light_cpu::SigmoidKernel kernel(Context());
  for (std::int64_t count : {0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1023, 1024, 1025}) {
    const auto values = Values(static_cast<std::size_t>(count));
    auto input = rt::Tensor::FromFloat("", {count}, values);
    kernel(input, input);
    for (std::int64_t index = 0; index < count; ++index) {
      const double expected = 1.0 / (1.0 + std::exp(-static_cast<double>(values[index])));
      EXPECT_NEAR(input.AsFloat()[index], expected, 2e-6) << count << "," << index;
    }
  }
}

TEST(OnnxLightSigmoidSoftmaxKernel, SoftmaxMatchesReferenceIncludingTailsAndAliasing) {
  const onnx_light_cpu::SoftmaxKernel kernel(Context());
  for (std::int64_t columns : {1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1023, 1025}) {
    constexpr std::int64_t rows = 3;
    const auto values = Values(static_cast<std::size_t>(rows * columns));
    auto input = rt::Tensor::FromFloat("", {rows, columns}, values);
    kernel(input, -1, input);
    for (std::int64_t row = 0; row < rows; ++row) {
      const auto begin = values.begin() + row * columns;
      const float maximum = *std::max_element(begin, begin + columns);
      double sum = 0;
      for (std::int64_t column = 0; column < columns; ++column) {
        sum += std::exp(static_cast<double>(begin[column]) - maximum);
      }
      for (std::int64_t column = 0; column < columns; ++column) {
        const double expected = std::exp(static_cast<double>(begin[column]) - maximum) / sum;
        EXPECT_NEAR(input.AsFloat()[row * columns + column], expected, 2e-6)
            << columns << "," << row << "," << column;
      }
    }
  }
}

TEST(OnnxLightSigmoidSoftmaxKernel, SpecialValues) {
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  auto input = rt::Tensor::FromFloat("", {9}, {-inf, inf, nan, -0.0f, 0, -100, 100, 1, -1});
  const auto sigmoid = onnx_light_cpu::SigmoidKernel(Context())(input);
  EXPECT_EQ(sigmoid.AsFloat()[0], 0.0f);
  EXPECT_EQ(sigmoid.AsFloat()[1], 1.0f);
  EXPECT_TRUE(std::isnan(sigmoid.AsFloat()[2]));
  EXPECT_NEAR(sigmoid.AsFloat()[3], 0.5f, 1e-7f);
  EXPECT_NEAR(sigmoid.AsFloat()[4], 0.5f, 1e-7f);

  input = rt::Tensor::FromFloat("", {3, 3}, {0, -inf, 0, -inf, -inf, -inf, 0, nan, 1});
  const auto softmax = onnx_light_cpu::SoftmaxKernel(Context())(input, -1);
  EXPECT_EQ(softmax.AsFloat()[0], 0.5f);
  EXPECT_EQ(softmax.AsFloat()[1], 0.0f);
  EXPECT_EQ(softmax.AsFloat()[2], 0.5f);
  for (std::size_t index = 3; index < 9; ++index) {
    EXPECT_TRUE(std::isnan(softmax.AsFloat()[index])) << index;
  }
}

TEST(OnnxLightSigmoidSoftmaxKernel, SigmoidExtremeValuesInEveryTail) {
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float maximum = std::numeric_limits<float>::max();
  const std::vector<float> extremes = {-inf,    -maximum, -104, -100,    -90, -87.34f,
                                       -87.33f, -20,      -1,   -0.0f,   0,   1,
                                       20,      87.33f,   100,  maximum, inf, nan};
  const onnx_light_cpu::SigmoidKernel kernel(Context());
  for (std::int64_t count = 1; count <= 65; ++count) {
    std::vector<float> values(static_cast<std::size_t>(count));
    for (std::size_t index = 0; index < values.size(); ++index) {
      values[index] = extremes[(index + count) % extremes.size()];
    }
    auto input = rt::Tensor::FromFloat("", {count}, values);
    kernel(input, input);
    for (std::size_t index = 0; index < values.size(); ++index) {
      SCOPED_TRACE(std::to_string(count) + "," + std::to_string(index));
      if (std::isnan(values[index])) {
        EXPECT_TRUE(std::isnan(input.AsFloat()[index]));
        continue;
      }
      const double exponent = std::exp(-std::abs(static_cast<double>(values[index])));
      const double expected = values[index] < 0 ? exponent / (1 + exponent) : 1 / (1 + exponent);
      EXPECT_NEAR(input.AsFloat()[index], expected,
                  std::max(expected * 3e-7, 2.0 * std::numeric_limits<float>::denorm_min()));
      if (std::isinf(values[index])) {
        EXPECT_EQ(input.AsFloat()[index], values[index] < 0 ? 0.0f : 1.0f);
      }
    }
  }
}

TEST(OnnxLightSigmoidSoftmaxKernel, SoftmaxExtremeRowsIncludingTails) {
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const onnx_light_cpu::SoftmaxKernel kernel(Context());
  for (std::int64_t columns : {7, 8, 9, 15, 16, 17, 31, 32, 33, 1023, 1024, 1025}) {
    constexpr std::int64_t rows = 5;
    auto values = Values(static_cast<std::size_t>(rows * columns));
    for (std::int64_t column = 0; column < columns; ++column) {
      values[column] = -86.0f - static_cast<float>(column % 20);
      values[3 * columns + column] = -inf;
      values[4 * columns + column] = -std::numeric_limits<float>::max();
    }
    values[columns - 1] = 0;
    values[2 * columns - 1] = nan;
    values[3 * columns - 1] = inf;
    values[5 * columns - 1] = std::numeric_limits<float>::max();
    auto input = rt::Tensor::FromFloat("", {rows, columns}, values);
    kernel(input, -1, input);
    for (std::int64_t row = 0; row < rows; ++row) {
      const auto begin = values.begin() + row * columns;
      const double maximum = *std::max_element(begin, begin + columns);
      double sum = 0;
      for (std::int64_t column = 0; column < columns; ++column) {
        sum += std::exp(static_cast<double>(begin[column]) - maximum);
      }
      for (std::int64_t column = 0; column < columns; ++column) {
        SCOPED_TRACE(std::to_string(columns) + "," + std::to_string(row) + "," +
                     std::to_string(column));
        const double expected = std::exp(static_cast<double>(begin[column]) - maximum) / sum;
        const float actual = input.AsFloat()[row * columns + column];
        if (std::isnan(expected)) {
          EXPECT_TRUE(std::isnan(actual));
        } else {
          EXPECT_NEAR(actual, expected,
                      std::max(expected * 3e-7, 2.0 * std::numeric_limits<float>::denorm_min()));
        }
      }
    }
  }
}

TEST(OnnxLightSigmoidSoftmaxKernel, UsesRuntimeExecutorForLargeInputs) {
  InlineExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run};
  onnx_light_cpu::ExecutionExecutorScope scope(&view);
  const auto small = rt::Tensor::FromFloat("", {1, 17}, Values(17));
  const onnx_light_cpu::SigmoidKernel sigmoid(Context());
  const onnx_light_cpu::SoftmaxKernel softmax(Context());
  sigmoid(small);
  softmax(small, -1);
  EXPECT_EQ(executor.dispatches, 0);
  const auto large = rt::Tensor::FromFloat("", {256, 1025}, Values(256 * 1025));
  view.effective_threads = 1;
  const auto expected_sigmoid = sigmoid(large);
  const auto expected_softmax = softmax(large, -1);
  EXPECT_EQ(executor.dispatches, 0);
  view.effective_threads = 4;
  const auto actual_sigmoid = sigmoid(large);
  EXPECT_TRUE(std::equal(expected_sigmoid.AsFloat(),
                         expected_sigmoid.AsFloat() + large.element_count(),
                         actual_sigmoid.AsFloat()));
  EXPECT_EQ(executor.dispatches, 1);
  EXPECT_GT(executor.blocks, 1);
  EXPECT_LE(executor.blocks, view.effective_threads);
  EXPECT_FALSE(executor.nested);
  executor = {};
  const auto actual_softmax = softmax(large, -1);
  EXPECT_TRUE(std::equal(expected_softmax.AsFloat(),
                         expected_softmax.AsFloat() + large.element_count(),
                         actual_softmax.AsFloat()));
  EXPECT_EQ(executor.dispatches, 1);
  EXPECT_GT(executor.blocks, 1);
  EXPECT_LE(executor.blocks, view.effective_threads);
  EXPECT_FALSE(executor.nested);

  executor = {};
  {
    onnx_light_cpu::detail::ExecutionRegionScope region;
    sigmoid(large);
    softmax(large, -1);
  }
  EXPECT_EQ(executor.dispatches, 0);
}

#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
TEST(OnnxLightSigmoidSoftmaxKernel, SmallAvx2SoftmaxAvoidsParallelDispatch) {
  if (onnx_light_cpu::DetectSimdLevel() != onnx_light_cpu::SimdLevel::kAVX2 ||
      !onnx_light_cpu::CpuSupportsFma()) {
    GTEST_SKIP() << "AVX2-specific scheduling is unavailable";
  }
  InlineExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 32, &InlineExecutor::Run};
  onnx_light_cpu::ExecutionExecutorScope scope(&view);
  const onnx_light_cpu::SoftmaxKernel softmax(Context());
  for (std::int64_t rows : {1, 32, 64, 128, 255}) {
    const auto input = rt::Tensor::FromFloat("", {rows, 1024}, Values(rows * 1024));
    softmax(input, -1);
    EXPECT_EQ(executor.dispatches, 0) << rows;
  }
}

TEST(OnnxLightSigmoidSoftmaxKernel, RegisteredFloat32UsesFusedAvx2Implementation) {
  if (onnx_light_cpu::DetectSimdLevel() < onnx_light_cpu::SimdLevel::kAVX2 ||
      !onnx_light_cpu::CpuSupportsFma()) {
    GTEST_SKIP() << "AVX2/FMA is unavailable";
  }
  const onnx_light_cpu::SigmoidKernel sigmoid(Context());
  const onnx_light_cpu::SoftmaxKernel softmax(Context());
  // Exact equality checks dispatch, not just the scalar fallback's accuracy.
  for (std::int64_t columns : {7, 8, 9, 15, 16, 17, 1023, 1024, 1025}) {
    constexpr std::int64_t rows = 3;
    const auto values = Values(static_cast<std::size_t>(rows * columns));
    const auto input = rt::Tensor::FromFloat("", {rows, columns}, values);
    std::vector<float> expected(values.size());
    onnx_light_cpu::SigmoidFloat32_AVX2_FMA(values.data(), expected.data(), expected.size());
    const auto sigmoid_output = sigmoid(input);
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), sigmoid_output.AsFloat()))
        << "Sigmoid columns=" << columns;
    onnx_light_cpu::SoftmaxFloat32_AVX2_FMA(values.data(), expected.data(), rows, columns);
    const auto softmax_output = softmax(input, -1);
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), softmax_output.AsFloat()))
        << "Softmax columns=" << columns;
  }
}
#endif

} // namespace
