// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu_compute/com_microsoft/moe.h"
#include "onnx_light_cpu_compute/com_microsoft/moe_matvec.h"
#include "onnx_light_cpu_compute/execution.h"
#include "onnx_light_cpu_compute/math/half_conversion.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace onnx_light_cpu {
namespace {

struct ScratchStorage {
  std::vector<MoERoute> routes;
  std::vector<float> values;
  std::size_t token_blocks = 0;

  ScratchStorage(std::size_t tokens, std::size_t hidden, std::size_t experts,
                 std::size_t intermediate, MoEDataType data_type, const MoEAttributes &attributes) {
    const MoEScratchSize size =
        ComputeMoEScratchSize(tokens, hidden, experts, intermediate, data_type, attributes);
    routes.resize(size.route_count);
    values.resize(size.value_count);
    token_blocks = size.token_blocks;
  }

  MoEScratch Get() {
    return {routes.data(), routes.size(), values.data(), values.size(), token_blocks};
  }
};

void RunMoE(const void *input, const void *router, const void *fc1, const void *bias1,
            const void *fc2, const void *bias2, void *output, std::size_t tokens,
            std::size_t hidden, std::size_t experts, std::size_t intermediate,
            MoEDataType data_type, const MoEAttributes &attributes) {
  ScratchStorage scratch(tokens, hidden, experts, intermediate, data_type, attributes);
  ComputeMoE(input, router, fc1, bias1, fc2, bias2, output, tokens, hidden, experts, intermediate,
             data_type, attributes, scratch.Get());
}

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
  RunMoE(input, router, fc1, nullptr, fc2, nullptr, output, 1, 2, 3, 1, MoEDataType::kFloat32,
         attributes);
  const float top = std::exp(2.0f) / (std::exp(1.0f) + std::exp(2.0f));
  EXPECT_NEAR(output[0], 1.0f + 2.0f * top, 1e-5f);
  EXPECT_NEAR(output[1], 2.0f + 2.0f * top, 1e-5f);
  attributes.normalize = false;
  RunMoE(input, router, fc1, nullptr, fc2, nullptr, output, 1, 2, 3, 1, MoEDataType::kFloat32,
         attributes);
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
  RunMoE(input, router, fc1, nullptr, fc2, nullptr, output, 1, 1, 1, 1, MoEDataType::kFloat32,
         attributes);
  EXPECT_NEAR(output[0], 3.0f * (1.0f / (1.0f + std::exp(-2.0f))) * 1.5f, 1e-5f);

  const std::uint16_t half_input[] = {detail::FloatToFloat16Bits(2.0f)};
  const std::uint16_t half_router[] = {0};
  const std::uint16_t half_fc1[] = {detail::FloatToFloat16Bits(1.0f),
                                    detail::FloatToFloat16Bits(2.0f)};
  const std::uint16_t half_fc2[] = {detail::FloatToFloat16Bits(3.0f)};
  std::uint16_t half_output[] = {0};
  RunMoE(half_input, half_router, half_fc1, nullptr, half_fc2, nullptr, half_output, 1, 1, 1, 1,
         MoEDataType::kFloat16, attributes);
  EXPECT_NEAR(detail::Float16BitsToFloat(half_output[0]), output[0], 0.01f);
}

TEST(MoE, EmptyTokensDoNotReadRouter) {
  MoEAttributes attributes;
  float output = 42.0f;
  RunMoE(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &output, 0, 2, 1, 1,
         MoEDataType::kFloat32, attributes);
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
    const MoEScratchSize scratch_size =
        ComputeMoEScratchSize(tokens, 1, 1, 1, MoEDataType::kFloat32, attributes);
    EXPECT_EQ(scratch_size.token_blocks, 4U);
    EXPECT_EQ(scratch_size.route_count, 4U);
    RunMoE(input.data(), router.data(), fc1, nullptr, fc2, nullptr, output.data(), tokens, 1, 1, 1,
           MoEDataType::kFloat32, attributes);
  }

  EXPECT_EQ(executor.dispatches, 1);
  EXPECT_EQ(executor.blocks, 4);
  for (float value : output) {
    EXPECT_FLOAT_EQ(value, 24.0f);
  }
}

TEST(MoE, ReusesSingleBlockScratchAcrossLongPrompts) {
  MoEAttributes attributes;
  attributes.activation = "swiglu";
  attributes.k = 8;
  const MoEScratchSize scratch_size =
      ComputeMoEScratchSize(32768, 2880, 32, 2880, MoEDataType::kBFloat16, attributes);

  EXPECT_EQ(scratch_size.token_blocks, 1U);
  EXPECT_EQ(scratch_size.route_count, 32U);
  EXPECT_EQ(scratch_size.value_count, 14432U);
  EXPECT_EQ(scratch_size.value_count * sizeof(float), 57728U);
}

TEST(MoE, RejectsMissingScratchBeforeParallelDispatch) {
  InlineExecutor executor;
  ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run, &InlineExecutor::Plan};
  ExecutionExecutorScope scope(&view);
  MoEAttributes attributes;
  attributes.activation = "identity";

  EXPECT_THROW(ComputeMoE(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 8, 1, 1, 1,
                          MoEDataType::kFloat32, attributes, {}),
               std::invalid_argument);
  EXPECT_EQ(executor.dispatches, 0);

  attributes.k = 2;
  EXPECT_THROW(ComputeMoE(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1, 1024, 2,
                          512, MoEDataType::kFloat32, attributes, {}),
               std::invalid_argument);
  EXPECT_EQ(executor.dispatches, 0);
}

TEST(MoE, ParallelExpertsClearUnderflowedRouteResults) {
  constexpr std::size_t hidden = 1024;
  constexpr std::size_t intermediate = 512;
  constexpr std::size_t experts = 2;
  const std::vector<float> float_input(hidden, 0.0f);
  const std::array<float, experts> float_router{0.0f, -200.0f};
  const std::vector<float> float_fc1(experts * intermediate * hidden, 0.0f);
  const std::vector<float> float_fc2(experts * hidden * intermediate, 0.0f);
  MoEAttributes attributes;
  attributes.activation = "identity";
  attributes.k = experts;

  InlineExecutor executor;
  ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run, &InlineExecutor::Plan};
  ExecutionExecutorScope scope(&view);

  std::vector<float> float_output(hidden);
  ScratchStorage float_scratch(1, hidden, experts, intermediate, MoEDataType::kFloat32, attributes);
  std::fill(float_scratch.values.begin(), float_scratch.values.end(), 7.0f);
  ComputeMoE(float_input.data(), float_router.data(), float_fc1.data(), nullptr, float_fc2.data(),
             nullptr, float_output.data(), 1, hidden, experts, intermediate, MoEDataType::kFloat32,
             attributes, float_scratch.Get());
  EXPECT_TRUE(std::all_of(float_output.begin(), float_output.end(),
                          [](float value) { return value == 0.0f; }));

  const std::vector<std::uint16_t> low_input(hidden, 0);
  const std::vector<std::uint16_t> low_fc1(experts * intermediate * hidden, 0);
  const std::vector<std::uint16_t> low_fc2(experts * hidden * intermediate, 0);
  for (const MoEDataType data_type : {MoEDataType::kFloat16, MoEDataType::kBFloat16}) {
    const std::array<std::uint16_t, experts> low_router{
        0, data_type == MoEDataType::kFloat16 ? detail::FloatToFloat16Bits(-200.0f)
                                              : detail::FloatToBFloat16Bits(-200.0f)};
    std::vector<std::uint16_t> low_output(hidden);
    ScratchStorage low_scratch(1, hidden, experts, intermediate, data_type, attributes);
    std::fill(low_scratch.values.begin(), low_scratch.values.end(), 7.0f);
    ComputeMoE(low_input.data(), low_router.data(), low_fc1.data(), nullptr, low_fc2.data(),
               nullptr, low_output.data(), 1, hidden, experts, intermediate, data_type, attributes,
               low_scratch.Get());
    EXPECT_TRUE(std::all_of(low_output.begin(), low_output.end(),
                            [](std::uint16_t value) { return value == 0; }));
  }
  EXPECT_EQ(executor.dispatches, 3);
}

TEST(MoE, ParallelExpertsMatchNestedSerialForNonzeroValuesAndBiases) {
  constexpr std::size_t hidden = 1024;
  constexpr std::size_t intermediate = 512;
  constexpr std::size_t experts = 2;
  std::vector<float> input(hidden, 0.0f);
  std::array<float, experts> router{0.25f, -0.5f};
  std::vector<float> fc1(experts * intermediate * hidden, 0.0f);
  std::vector<float> bias1(experts * intermediate, 0.0f);
  std::vector<float> fc2(experts * hidden * intermediate, 0.0f);
  std::vector<float> bias2(experts * hidden, 0.0f);
  input[0] = 1.0f;
  fc1[0] = 1.0f;
  fc1[intermediate * hidden] = 2.0f;
  bias1[0] = 0.5f;
  bias1[intermediate] = -0.25f;
  fc2[0] = 3.0f;
  fc2[hidden * intermediate] = 5.0f;
  bias2[0] = 0.25f;
  bias2[hidden] = -0.5f;
  bias2[1] = 0.125f;
  bias2[hidden + 1] = -0.25f;

  InlineExecutor executor;
  ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run, &InlineExecutor::Plan};
  ExecutionExecutorScope scope(&view);
  for (const bool normalize : {false, true}) {
    MoEAttributes attributes;
    attributes.activation = "identity";
    attributes.k = experts;
    attributes.normalize = normalize;
    for (const MoEDataType data_type :
         {MoEDataType::kFloat32, MoEDataType::kFloat16, MoEDataType::kBFloat16}) {
      ScratchStorage parallel_scratch(1, hidden, experts, intermediate, data_type, attributes);
      ScratchStorage serial_scratch(1, hidden, experts, intermediate, data_type, attributes);
      const std::int64_t dispatches_before = executor.dispatches;
      if (data_type == MoEDataType::kFloat32) {
        std::vector<float> parallel_output(hidden), serial_output(hidden);
        ComputeMoE(input.data(), router.data(), fc1.data(), bias1.data(), fc2.data(), bias2.data(),
                   parallel_output.data(), 1, hidden, experts, intermediate, data_type, attributes,
                   parallel_scratch.Get());
        EXPECT_EQ(executor.dispatches, dispatches_before + 1);
        ExecuteRanges(1, ExecutionSchedule{1, 1, 4, 0}, [&](std::int64_t, std::int64_t) {
          ComputeMoE(input.data(), router.data(), fc1.data(), bias1.data(), fc2.data(),
                     bias2.data(), serial_output.data(), 1, hidden, experts, intermediate,
                     data_type, attributes, serial_scratch.Get());
        });
        for (std::size_t index = 0; index < hidden; ++index) {
          EXPECT_NEAR(parallel_output[index], serial_output[index], 1e-6f);
        }
        EXPECT_NE(parallel_output[0], 0.0f);
        EXPECT_NE(parallel_output[1], 0.0f);
      } else {
        const auto encode = [data_type](float value) {
          return data_type == MoEDataType::kFloat16 ? detail::FloatToFloat16Bits(value)
                                                    : detail::FloatToBFloat16Bits(value);
        };
        const auto decode = [data_type](std::uint16_t value) {
          return data_type == MoEDataType::kFloat16 ? detail::Float16BitsToFloat(value)
                                                    : detail::Bfloat16BitsToFloat(value);
        };
        std::vector<std::uint16_t> low_input(input.size()), low_router(router.size()),
            low_fc1(fc1.size()), low_bias1(bias1.size()), low_fc2(fc2.size()),
            low_bias2(bias2.size());
        std::transform(input.begin(), input.end(), low_input.begin(), encode);
        std::transform(router.begin(), router.end(), low_router.begin(), encode);
        std::transform(fc1.begin(), fc1.end(), low_fc1.begin(), encode);
        std::transform(bias1.begin(), bias1.end(), low_bias1.begin(), encode);
        std::transform(fc2.begin(), fc2.end(), low_fc2.begin(), encode);
        std::transform(bias2.begin(), bias2.end(), low_bias2.begin(), encode);
        std::vector<std::uint16_t> parallel_output(hidden), serial_output(hidden);
        ComputeMoE(low_input.data(), low_router.data(), low_fc1.data(), low_bias1.data(),
                   low_fc2.data(), low_bias2.data(), parallel_output.data(), 1, hidden, experts,
                   intermediate, data_type, attributes, parallel_scratch.Get());
        EXPECT_EQ(executor.dispatches, dispatches_before + 1);
        ExecuteRanges(1, ExecutionSchedule{1, 1, 4, 0}, [&](std::int64_t, std::int64_t) {
          ComputeMoE(low_input.data(), low_router.data(), low_fc1.data(), low_bias1.data(),
                     low_fc2.data(), low_bias2.data(), serial_output.data(), 1, hidden, experts,
                     intermediate, data_type, attributes, serial_scratch.Get());
        });
        for (std::size_t index = 0; index < hidden; ++index) {
          EXPECT_NEAR(decode(parallel_output[index]), decode(serial_output[index]), 1e-6f);
        }
        EXPECT_NE(decode(parallel_output[0]), 0.0f);
        EXPECT_NE(decode(parallel_output[1]), 0.0f);
      }
      EXPECT_EQ(executor.dispatches, dispatches_before + 1);
    }
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
    RunMoE(input.data(), router.data(), fc1, nullptr, fc2, nullptr, &output, 1, 1, 1, 1,
           MoEDataType::kFloat32, attributes);
    ScratchStorage nested_scratch(8, 1, 1, 1, MoEDataType::kFloat32, attributes);
    std::array<float, 8> nested_output{};
    ExecuteRanges(2, ExecutionSchedule{1, 1, 4, 0}, [&](std::int64_t, std::int64_t) {
      ComputeMoE(input.data(), router.data(), fc1, nullptr, fc2, nullptr, nested_output.data(), 8,
                 1, 1, 1, MoEDataType::kFloat32, attributes, nested_scratch.Get());
    });
  }

  EXPECT_FLOAT_EQ(output, 24.0f);
  EXPECT_EQ(executor.dispatches, 1);
}

TEST(MoE, BFloat16UsesFloatAccumulation) {
  const std::uint16_t input[] = {detail::FloatToBFloat16Bits(2.0f)};
  const std::uint16_t router[] = {detail::FloatToBFloat16Bits(0.0f)};
  const std::uint16_t fc1[] = {detail::FloatToBFloat16Bits(3.0f)};
  const std::uint16_t fc2[] = {detail::FloatToBFloat16Bits(4.0f)};
  std::uint16_t output = 0;
  MoEAttributes attributes;
  attributes.activation = "identity";

  RunMoE(input, router, fc1, nullptr, fc2, nullptr, &output, 1, 1, 1, 1, MoEDataType::kBFloat16,
         attributes);

  EXPECT_FLOAT_EQ(detail::Bfloat16BitsToFloat(output), 24.0f);
}

TEST(MoE, MatVecSimdWidthsAndTailsMatchReference) {
  for (const std::size_t columns : {7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65}) {
    constexpr std::size_t rows = 3;
    std::vector<float> input(columns);
    std::vector<float> weights(rows * columns);
    for (std::size_t column = 0; column < columns; ++column) {
      input[column] = static_cast<float>(static_cast<int>(column % 7) - 3) / 8.0f;
    }
    for (std::size_t index = 0; index < weights.size(); ++index) {
      weights[index] = static_cast<float>(static_cast<int>(index % 11) - 5) / 16.0f;
    }
    std::vector<std::uint16_t> float16_input(columns), bfloat16_input(columns);
    std::vector<std::uint16_t> float16_weights(weights.size()), bfloat16_weights(weights.size());
    detail::ConvertFloat32ToFloat16(input.data(), float16_input.data(), columns);
    detail::ConvertFloat32ToBFloat16(input.data(), bfloat16_input.data(), columns);
    detail::ConvertFloat32ToFloat16(weights.data(), float16_weights.data(), weights.size());
    detail::ConvertFloat32ToBFloat16(weights.data(), bfloat16_weights.data(), weights.size());

    std::vector<float> result(rows), float16_result(rows), bfloat16_result(rows);
    std::vector<float> mixed_float16_result(rows), mixed_bfloat16_result(rows);
    detail::MoEMatVecFloat32(input.data(), weights.data(), result.data(), rows, columns);
    detail::MoEMatVecFloat16(float16_input.data(), float16_weights.data(), float16_result.data(),
                             rows, columns);
    detail::MoEMatVecBFloat16(bfloat16_input.data(), bfloat16_weights.data(),
                              bfloat16_result.data(), rows, columns);
    detail::MoEMatVecFloat32Float16(input.data(), float16_weights.data(),
                                    mixed_float16_result.data(), rows, columns);
    detail::MoEMatVecFloat32BFloat16(input.data(), bfloat16_weights.data(),
                                     mixed_bfloat16_result.data(), rows, columns);

    for (std::size_t row = 0; row < rows; ++row) {
      float expected = 0.0f;
      float float16_expected = 0.0f;
      float bfloat16_expected = 0.0f;
      float mixed_float16_expected = 0.0f;
      float mixed_bfloat16_expected = 0.0f;
      for (std::size_t column = 0; column < columns; ++column) {
        const std::size_t index = row * columns + column;
        expected += input[column] * weights[index];
        const float float16_value = detail::Float16BitsToFloat(float16_weights[index]);
        const float bfloat16_value = detail::Bfloat16BitsToFloat(bfloat16_weights[index]);
        float16_expected += detail::Float16BitsToFloat(float16_input[column]) * float16_value;
        bfloat16_expected += detail::Bfloat16BitsToFloat(bfloat16_input[column]) * bfloat16_value;
        mixed_float16_expected += input[column] * float16_value;
        mixed_bfloat16_expected += input[column] * bfloat16_value;
      }
      EXPECT_NEAR(result[row], expected, 1e-5f);
      EXPECT_NEAR(float16_result[row], float16_expected, 1e-5f);
      EXPECT_NEAR(bfloat16_result[row], bfloat16_expected, 1e-5f);
      EXPECT_NEAR(mixed_float16_result[row], mixed_float16_expected, 1e-5f);
      EXPECT_NEAR(mixed_bfloat16_result[row], mixed_bfloat16_expected, 1e-5f);
    }
  }
}

TEST(MoE, MatVecRowParallelModesMatchNestedSerial) {
  constexpr std::size_t rows = 512;
  constexpr std::size_t columns = 1024;
  std::vector<float> input(columns);
  std::vector<float> weights(rows * columns);
  for (std::size_t column = 0; column < columns; ++column) {
    input[column] = static_cast<float>(static_cast<int>(column % 7) - 3) / 8.0f;
  }
  for (std::size_t index = 0; index < weights.size(); ++index) {
    weights[index] = static_cast<float>(static_cast<int>(index % 11) - 5) / 16.0f;
  }
  std::vector<std::uint16_t> float16_input(columns), bfloat16_input(columns);
  std::vector<std::uint16_t> float16_weights(weights.size()), bfloat16_weights(weights.size());
  detail::ConvertFloat32ToFloat16(input.data(), float16_input.data(), columns);
  detail::ConvertFloat32ToBFloat16(input.data(), bfloat16_input.data(), columns);
  detail::ConvertFloat32ToFloat16(weights.data(), float16_weights.data(), weights.size());
  detail::ConvertFloat32ToBFloat16(weights.data(), bfloat16_weights.data(), weights.size());

  InlineExecutor executor;
  ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run, &InlineExecutor::Plan};
  ExecutionExecutorScope scope(&view);
  const auto check_mode = [&](const auto &run) {
    std::vector<float> parallel(rows, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> serial(rows, std::numeric_limits<float>::quiet_NaN());
    const std::int64_t dispatches_before = executor.dispatches;
    run(parallel.data());
    EXPECT_EQ(executor.dispatches, dispatches_before + 1);
    EXPECT_GT(executor.blocks, 1);
    ExecuteRanges(1, ExecutionSchedule{1, 1, 4, 0},
                  [&](std::int64_t, std::int64_t) { run(serial.data()); });
    EXPECT_EQ(executor.dispatches, dispatches_before + 1);
    for (std::size_t row = 0; row < rows; ++row) {
      EXPECT_TRUE(std::isfinite(parallel[row]));
      EXPECT_FLOAT_EQ(parallel[row], serial[row]);
    }
  };

  check_mode([&](float *output) {
    detail::MoEMatVecFloat32(input.data(), weights.data(), output, rows, columns);
  });
  check_mode([&](float *output) {
    detail::MoEMatVecFloat16(float16_input.data(), float16_weights.data(), output, rows, columns);
  });
  check_mode([&](float *output) {
    detail::MoEMatVecBFloat16(bfloat16_input.data(), bfloat16_weights.data(), output, rows,
                              columns);
  });
  check_mode([&](float *output) {
    detail::MoEMatVecFloat32Float16(input.data(), float16_weights.data(), output, rows, columns);
  });
  check_mode([&](float *output) {
    detail::MoEMatVecFloat32BFloat16(input.data(), bfloat16_weights.data(), output, rows, columns);
  });
}

} // namespace
} // namespace onnx_light_cpu
