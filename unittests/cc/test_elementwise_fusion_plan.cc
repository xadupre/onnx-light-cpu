// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/binary/elementwise_fusion_plan.h"

#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

namespace BinaryDataType = onnx_light_cpu::BinaryDataType;
using onnx_light_cpu::ElementwiseFusionGuards;
using onnx_light_cpu::ElementwiseFusionPlan;
using onnx_light_cpu::ExecutionBlockFn;

struct InlineExecutor {
  std::int64_t dispatches = 0;

  static void Run(void *context, std::int64_t blocks, void *task_context, ExecutionBlockFn task) {
    ++static_cast<InlineExecutor *>(context)->dispatches;
    for (std::int64_t block = 0; block < blocks; ++block) {
      task(task_context, block);
    }
  }
};

ElementwiseFusionGuards Guards(std::span<const std::size_t> consumers) {
  return ElementwiseFusionGuards{consumers, false, true, true};
}

TEST(ElementwiseFusionPlan, SwiGLUFloat32PreservesOperationOrder) {
  const std::array<std::int64_t, 3> shape{1, 1, 3072};
  const std::array<std::size_t, 2> consumers{1, 1};
  auto plan = ElementwiseFusionPlan::TryCreateSwiGLUGate(BinaryDataType::FLOAT, shape, shape, shape,
                                                         shape, shape, Guards(consumers));
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->workspace_bytes(), 0u);

  std::vector<float> gate(plan->element_count());
  std::vector<float> up(plan->element_count());
  std::vector<float> output(plan->element_count());
  for (std::size_t i = 0; i < gate.size(); ++i) {
    gate[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 4.0f;
    up[i] = static_cast<float>(static_cast<int>(i % 11) - 5) / 3.0f;
  }
  plan->Execute(gate.data(), up.data(), nullptr, output.data());
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float sigmoid = 1.0f / (1.0f + std::exp(-gate[i]));
    const float inner = gate[i] * sigmoid;
    EXPECT_NEAR(output[i], inner * up[i], 3e-6f) << i;
  }
}

TEST(ElementwiseFusionPlan, SwiGLUBFloat16RoundsEveryGraphIntermediate) {
  const std::array<std::int64_t, 3> shape{1, 1, 3072};
  const std::array<std::size_t, 2> consumers{1, 1};
  auto plan = ElementwiseFusionPlan::TryCreateSwiGLUGate(BinaryDataType::BFLOAT16, shape, shape,
                                                         shape, shape, shape, Guards(consumers));
  ASSERT_TRUE(plan.has_value());

  std::vector<float> source(plan->element_count());
  std::vector<std::uint16_t> gate(source.size());
  std::vector<std::uint16_t> up(source.size());
  std::vector<std::uint16_t> output(source.size());
  for (std::size_t i = 0; i < source.size(); ++i) {
    source[i] = static_cast<float>(static_cast<int>(i % 19) - 9) / 7.0f;
  }
  onnx_light_cpu::detail::ConvertFloat32ToBFloat16(source.data(), gate.data(), source.size());
  std::reverse(source.begin(), source.end());
  onnx_light_cpu::detail::ConvertFloat32ToBFloat16(source.data(), up.data(), source.size());
  plan->Execute(gate.data(), up.data(), nullptr, output.data());

  for (std::size_t i = 0; i < output.size(); ++i) {
    const float g = onnx_light_cpu::detail::Bfloat16BitsToFloat(gate[i]);
    const float u = onnx_light_cpu::detail::Bfloat16BitsToFloat(up[i]);
    const auto sigmoid_bits =
        onnx_light_cpu::detail::FloatToBFloat16Bits(1.0f / (1.0f + std::exp(-g)));
    const float sigmoid = onnx_light_cpu::detail::Bfloat16BitsToFloat(sigmoid_bits);
    const auto inner_bits = onnx_light_cpu::detail::FloatToBFloat16Bits(g * sigmoid);
    const float inner = onnx_light_cpu::detail::Bfloat16BitsToFloat(inner_bits);
    EXPECT_EQ(output[i], onnx_light_cpu::detail::FloatToBFloat16Bits(inner * u)) << i;
  }
}

TEST(ElementwiseFusionPlan, ScaledMaskedScoresBroadcastsMaskInOneTraversal) {
  const std::array<std::int64_t, 4> scores_shape{1, 16, 32, 1024};
  const std::array<std::int64_t, 4> mask_shape{1, 1, 32, 1024};
  const std::array<std::size_t, 1> consumers{1};
  auto plan = ElementwiseFusionPlan::TryCreateScaledMaskedScores(
      BinaryDataType::FLOAT, scores_shape, std::span<const std::int64_t>{}, mask_shape,
      scores_shape, scores_shape, Guards(consumers));
  ASSERT_TRUE(plan.has_value());

  std::vector<float> scores(plan->element_count());
  std::vector<float> mask(32 * 1024);
  std::vector<float> output(plan->element_count());
  const float scale = 0.125f;
  for (std::size_t i = 0; i < scores.size(); ++i) {
    scores[i] = static_cast<float>(i % 23) - 11.0f;
  }
  for (std::size_t i = 0; i < mask.size(); ++i) {
    mask[i] = static_cast<float>(i) / 32.0f;
  }
  InlineExecutor executor;
  const onnx_light_cpu::ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run};
  {
    onnx_light_cpu::ExecutionExecutorScope scope(&view);
    plan->Execute(scores.data(), &scale, mask.data(), output.data());
  }
  EXPECT_EQ(executor.dispatches, 1);
  for (std::size_t i = 0; i < output.size(); ++i) {
    EXPECT_EQ(output[i], scores[i] * scale + mask[i % mask.size()]);
  }
}

TEST(ElementwiseFusionPlan, RejectsUnsupportedGraphAndSafetyGuards) {
  const std::array<std::int64_t, 3> shape{1, 1, 3072};
  const std::array<std::int64_t, 3> wrong_shape{1, 2, 3072};
  const std::array<std::size_t, 2> exclusive{1, 1};
  const std::array<std::size_t, 2> shared{1, 2};
  EXPECT_FALSE(ElementwiseFusionPlan::TryCreateSwiGLUGate(BinaryDataType::FLOAT, shape, shape,
                                                          shape, shape, shape, Guards(shared)));
  EXPECT_FALSE(ElementwiseFusionPlan::TryCreateSwiGLUGate(
      BinaryDataType::FLOAT, shape, shape, shape, wrong_shape, shape, Guards(exclusive)));

  auto alias = Guards(exclusive);
  alias.has_unsafe_alias = true;
  EXPECT_FALSE(ElementwiseFusionPlan::TryCreateSwiGLUGate(BinaryDataType::FLOAT, shape, shape,
                                                          shape, shape, shape, alias));
  auto dynamic = Guards(exclusive);
  dynamic.has_static_shapes = false;
  EXPECT_FALSE(ElementwiseFusionPlan::TryCreateSwiGLUGate(BinaryDataType::FLOAT, shape, shape,
                                                          shape, shape, shape, dynamic));
  auto numerical = Guards(exclusive);
  numerical.numerical_contract_matches = false;
  EXPECT_FALSE(ElementwiseFusionPlan::TryCreateSwiGLUGate(BinaryDataType::FLOAT, shape, shape,
                                                          shape, shape, shape, numerical));
  EXPECT_FALSE(ElementwiseFusionPlan::TryCreateSwiGLUGate(BinaryDataType::INT32, shape, shape,
                                                          shape, shape, shape, Guards(exclusive)));
}

TEST(ElementwiseFusionPlan, RejectsOutOfScopeScaledMaskVariantsAndRuntimeAliasing) {
  const std::array<std::int64_t, 4> scores_shape{1, 16, 1, 128};
  const std::array<std::int64_t, 4> mask_shape{1, 1, 1, 128};
  const std::array<std::int64_t, 1> vector_scale{1};
  const std::array<std::size_t, 1> consumers{1};
  EXPECT_FALSE(ElementwiseFusionPlan::TryCreateScaledMaskedScores(
      BinaryDataType::FLOAT, scores_shape, vector_scale, mask_shape, scores_shape, scores_shape,
      Guards(consumers)));
  EXPECT_FALSE(ElementwiseFusionPlan::TryCreateScaledMaskedScores(
      BinaryDataType::BFLOAT16, scores_shape, std::span<const std::int64_t>{}, mask_shape,
      scores_shape, scores_shape, Guards(consumers)));

  auto plan = ElementwiseFusionPlan::TryCreateScaledMaskedScores(
      BinaryDataType::FLOAT, scores_shape, std::span<const std::int64_t>{}, mask_shape,
      scores_shape, scores_shape, Guards(consumers));
  ASSERT_TRUE(plan.has_value());
  std::vector<float> scores(plan->element_count());
  std::array<float, 128> mask{};
  const float scale = 1.0f;
  EXPECT_THROW(plan->Execute(scores.data(), &scale, mask.data(), scores.data()),
               std::invalid_argument);
}

} // namespace
