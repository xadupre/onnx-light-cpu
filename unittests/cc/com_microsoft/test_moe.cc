// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/com_microsoft/moe.h"
#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace onnx_light_cpu {
namespace {

struct InlineExecutor {
  std::int64_t dispatches = 0;
  std::int64_t blocks = 0;

  static void Run(void *context, std::int64_t blocks, void *task_context, ExecutionBlockFn task) {
    auto &self = *static_cast<InlineExecutor *>(context);
    ++self.dispatches;
    self.blocks = blocks;
    for (std::int64_t block = 0; block < blocks; ++block) {
      task(task_context, block);
    }
  }

  static ExecutionParallelPlan Plan(void *, std::int64_t, const ExecutionWorkCost &, std::int64_t,
                                    std::int64_t) {
    return {1, 4};
  }
};

TEST(MoE, SelectsTopExpertsAndNormalizes) {
  const float input[] = {1.0f, 2.0f};
  const float router[] = {0.0f, 1.0f, 2.0f};
  const float fc1[] = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
  const float fc2[] = {1.0f, 2.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  float output[2] = {};
  MoEAttributes attributes;
  attributes.activation = "identity";
  attributes.k = 2;
  attributes.normalize = true;
  ComputeMoE(input, router, fc1, nullptr, fc2, nullptr, output, 1, 2, 3, 1, false, attributes);
  const float top = std::exp(2.0f) / (std::exp(1.0f) + std::exp(2.0f));
  EXPECT_NEAR(output[0], 1.0f + 2.0f * top, 1e-5f);
  EXPECT_NEAR(output[1], 2.0f + 2.0f * top, 1e-5f);
  attributes.normalize = false;
  ComputeMoE(input, router, fc1, nullptr, fc2, nullptr, output, 1, 2, 3, 1, false, attributes);
  EXPECT_LT(output[0], 1.0f + 2.0f * top);
}

TEST(MoE, InterleavedSwiGLUAndFloat16Output) {
  const float input[] = {2.0f};
  const float router[] = {0.0f};
  const float fc1[] = {1.0f, 2.0f};
  const float fc2[] = {3.0f};
  float output[] = {0.0f};
  MoEAttributes attributes;
  attributes.activation = "swiglu";
  attributes.limit = 1.0f;
  attributes.alpha = 2.0f;
  attributes.beta = 0.5f;
  ComputeMoE(input, router, fc1, nullptr, fc2, nullptr, output, 1, 1, 1, 1, false, attributes);
  EXPECT_NEAR(output[0], 3.0f * (1.0f / (1.0f + std::exp(-2.0f))) * 1.5f, 1e-5f);

  const std::uint16_t half_input[] = {detail::FloatToFloat16Bits(2.0f)};
  const std::uint16_t half_router[] = {0};
  const std::uint16_t half_fc1[] = {detail::FloatToFloat16Bits(1.0f),
                                    detail::FloatToFloat16Bits(2.0f)};
  const std::uint16_t half_fc2[] = {detail::FloatToFloat16Bits(3.0f)};
  std::uint16_t half_output[] = {0};
  ComputeMoE(half_input, half_router, half_fc1, nullptr, half_fc2, nullptr, half_output, 1, 1, 1, 1,
             true, attributes);
  EXPECT_NEAR(detail::Float16BitsToFloat(half_output[0]), output[0], 0.01f);
}

TEST(MoE, EmptyTokensDoNotReadRouter) {
  MoEAttributes attributes;
  float output = 42.0f;
  ComputeMoE(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &output, 0, 2, 1, 1, false,
             attributes);
  EXPECT_FLOAT_EQ(output, 42.0f);
}

TEST(MoE, UsesRuntimeExecutorAcrossIndependentTokens) {
  constexpr std::size_t tokens = 8;
  const std::vector<float> input(tokens, 2.0f);
  const std::vector<float> router(tokens, 0.0f);
  const float fc1[] = {3.0f};
  const float fc2[] = {4.0f};
  std::vector<float> output(tokens);
  MoEAttributes attributes;
  attributes.activation = "identity";

  InlineExecutor executor;
  ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run, &InlineExecutor::Plan};
  {
    ExecutionExecutorScope scope(&view);
    ComputeMoE(input.data(), router.data(), fc1, nullptr, fc2, nullptr, output.data(), tokens, 1, 1,
               1, false, attributes);
  }

  EXPECT_EQ(executor.dispatches, 1);
  EXPECT_EQ(executor.blocks, 4);
  for (float value : output) {
    EXPECT_FLOAT_EQ(value, 24.0f);
  }
}

TEST(MoE, SingleTokenAndNestedExecutionRemainSerial) {
  const std::vector<float> input(8, 2.0f);
  const std::vector<float> router(8, 0.0f);
  const float fc1[] = {3.0f};
  const float fc2[] = {4.0f};
  float output = 0.0f;
  MoEAttributes attributes;
  attributes.activation = "identity";

  InlineExecutor executor;
  ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run, &InlineExecutor::Plan};
  {
    ExecutionExecutorScope scope(&view);
    ComputeMoE(input.data(), router.data(), fc1, nullptr, fc2, nullptr, &output, 1, 1, 1, 1, false,
               attributes);
    ExecuteRanges(2, ExecutionSchedule{1, 1, 4, 0}, [&](std::int64_t, std::int64_t) {
      std::vector<float> nested_output(8);
      ComputeMoE(input.data(), router.data(), fc1, nullptr, fc2, nullptr, nested_output.data(), 8,
                 1, 1, 1, false, attributes);
    });
  }

  EXPECT_FLOAT_EQ(output, 24.0f);
  EXPECT_EQ(executor.dispatches, 1);
}

} // namespace
} // namespace onnx_light_cpu
