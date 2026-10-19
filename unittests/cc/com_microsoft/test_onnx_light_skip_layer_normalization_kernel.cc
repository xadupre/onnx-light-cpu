// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/com_microsoft/skip_layer_normalization_kernel.h"
#include "onnx_light_cpu_compute/execution.h"
#include "onnx_light_cpu_compute/math/half_conversion.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
namespace rt = ONNX_LIGHT_NAMESPACE::core::runtime;
using rt::DataType;
using rt::Shape;
using rt::Tensor;

struct InlineExecutor {
  std::int64_t dispatches = 0;
  std::int64_t blocks = 0;

  static void Run(void *context, std::int64_t num_blocks, void *task_context,
                  onnx_light_cpu::ExecutionBlockFn task) {
    auto &self = *static_cast<InlineExecutor *>(context);
    ++self.dispatches;
    self.blocks = num_blocks;
    for (std::int64_t block = 0; block < num_blocks; ++block) {
      task(task_context, block);
    }
  }
};

rt::KernelContext MakeCtx() { return rt::KernelContext(rt::OpsetId("com.microsoft", 1)); }

Tensor MakeTensor(DataType type, const Shape &shape, const std::vector<float> &values) {
  if (type == DataType::FLOAT16) {
    return rt::MakeFloat16Tensor("", shape, values);
  }
  if (type == DataType::BFLOAT16) {
    return rt::MakeBfloat16Tensor("", shape, values);
  }
  return Tensor::FromFloat("", shape, values);
}

float Value(const Tensor &tensor, std::size_t index) {
  if (tensor.data_type == DataType::FLOAT) {
    return tensor.AsFloat()[index];
  }
  const auto value = reinterpret_cast<const std::uint16_t *>(tensor.bytes())[index];
  return tensor.data_type == DataType::FLOAT16 ? onnx_light_cpu::detail::Float16BitsToFloat(value)
                                               : onnx_light_cpu::detail::Bfloat16BitsToFloat(value);
}

TEST(SkipLayerNormalizationKernel, ComputesAffineStatisticsAndResidual) {
  const onnx_light_cpu::SkipLayerNormalizationKernel kernel(MakeCtx());
  for (const auto type : {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16}) {
    const auto input = MakeTensor(type, {1, 2}, {1, -3});
    const auto skip = MakeTensor(type, {1, 2}, {0.5F, 0.5F});
    const auto gamma = MakeTensor(type, {2}, {2, -3});
    const auto beta = MakeTensor(type, {2}, {1, -1});
    const auto bias = MakeTensor(type, {2}, {0.5F, 0.5F});
    const auto result = kernel(input, skip, gamma, &beta, &bias, 0, true, true, true);
    EXPECT_EQ(Value(result.output, 0), 3);
    EXPECT_EQ(Value(result.output, 1), 2);
    EXPECT_EQ(Value(result.input_skip_bias_sum, 0), 2);
    EXPECT_EQ(Value(result.input_skip_bias_sum, 1), -2);
    EXPECT_FLOAT_EQ(result.mean.AsFloat()[0], 0);
    EXPECT_FLOAT_EQ(result.inv_std_var.AsFloat()[0], 0.5F);
  }
}

TEST(SkipLayerNormalizationKernel, SupportsBroadcastSkipAndEmptyRows) {
  const onnx_light_cpu::SkipLayerNormalizationKernel kernel(MakeCtx());
  const auto input = Tensor::FromFloat("", {2, 2, 2}, {1, 3, 2, 4, 5, 7, 6, 8});
  const auto skip = Tensor::FromFloat("", {2, 2}, {1, 1, 2, 2});
  const auto gamma = Tensor::FromFloat("", {2}, {1, 1});
  const auto result = kernel(input, skip, gamma, nullptr, nullptr, 0, true, true, true);
  for (std::size_t row = 0; row < 4; ++row) {
    EXPECT_FLOAT_EQ(result.output.AsFloat()[row * 2], -1);
    EXPECT_FLOAT_EQ(result.output.AsFloat()[row * 2 + 1], 1);
  }
  const auto empty = Tensor::FromFloat("", {0, 2, 2}, {});
  const auto empty_result = kernel(empty, empty, gamma, nullptr, nullptr, 1e-5F, true);
  EXPECT_EQ(empty_result.output.size_bytes(), 0U);
  EXPECT_EQ(empty_result.input_skip_bias_sum.size_bytes(), 0U);
}

TEST(SkipLayerNormalizationKernel, HandlesFloat32VectorWidthsAndTails) {
  const onnx_light_cpu::SkipLayerNormalizationKernel kernel(MakeCtx());
  for (const std::size_t width : {32U, 33U}) {
    std::vector<float> residual(width, 0.0F);
    for (std::size_t index = 0; index < 32; ++index) {
      residual[index] = index % 2 == 0 ? -1.0F : 1.0F;
    }
    const auto input = Tensor::FromFloat("", {1, static_cast<std::int64_t>(width)},
                                         std::vector<float>(width, 0.0F));
    const auto skip = Tensor::FromFloat("", {1, static_cast<std::int64_t>(width)}, residual);
    const auto gamma =
        Tensor::FromFloat("", {static_cast<std::int64_t>(width)}, std::vector<float>(width, 1.0F));
    const auto result = kernel(input, skip, gamma, nullptr, nullptr, 0.0F, true, true, true);
    const float inverse = std::sqrt(static_cast<float>(width) / 32.0F);
    EXPECT_NEAR(result.mean.AsFloat()[0], 0.0F, 1.0e-6F);
    EXPECT_NEAR(result.inv_std_var.AsFloat()[0], inverse, 1.0e-6F);
    for (std::size_t index = 0; index < width; ++index) {
      EXPECT_FLOAT_EQ(result.input_skip_bias_sum.AsFloat()[index], residual[index]);
      EXPECT_NEAR(result.output.AsFloat()[index], residual[index] * inverse, 1.0e-6F);
    }
  }
}

TEST(SkipLayerNormalizationKernel, RunsOptionalBiasWithoutBeta) {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("SkipLayerNormalization");
  node.set_domain("com.microsoft");
  for (const char *name : {"input", "skip", "gamma", "", "bias"}) {
    node.add_input(name);
  }
  for (const char *name : {"output", "mean", "inverse", "sum"}) {
    node.add_output(name);
  }
  rt::RuntimeContext context(MakeCtx());
  context.Set("input", Tensor::FromFloat("", {1, 2}, {1, -3}));
  context.Set("skip", Tensor::FromFloat("", {1, 2}, {0.5F, 0.5F}));
  context.Set("gamma", Tensor::FromFloat("", {2}, {2, -3}));
  context.Set("bias", Tensor::FromFloat("", {2}, {0.5F, 0.5F}));
  onnx_light_cpu::SkipLayerNormalizationKernel(node, MakeCtx()).Run(context);
  EXPECT_FLOAT_EQ(context.Get("output").AsFloat()[0], 2);
  EXPECT_FLOAT_EQ(context.Get("output").AsFloat()[1], 3);
  EXPECT_FLOAT_EQ(context.Get("mean").AsFloat()[0], 0);
  EXPECT_FLOAT_EQ(context.Get("inverse").AsFloat()[0], 0.5F);
}

TEST(SkipLayerNormalizationKernel, RejectsInvalidContracts) {
  const onnx_light_cpu::SkipLayerNormalizationKernel kernel(MakeCtx());
  const auto input = Tensor::FromFloat("", {2, 3}, std::vector<float>(6, 1));
  const auto gamma = Tensor::FromFloat("", {3}, {1, 1, 1});
  const auto short_vector = Tensor::FromFloat("", {2}, {1, 1});
  EXPECT_THROW((void)kernel(input, input, short_vector), std::invalid_argument);
  EXPECT_THROW((void)kernel(input, input, gamma, &short_vector), std::invalid_argument);
  EXPECT_THROW((void)kernel(input, input, gamma, nullptr, &short_vector), std::invalid_argument);
  const auto wrong_skip = Tensor::FromFloat("", {3, 2}, std::vector<float>(6, 1));
  EXPECT_THROW((void)kernel(input, wrong_skip, gamma), std::invalid_argument);
  for (const float epsilon :
       {-1.F, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
    EXPECT_THROW((void)kernel(input, input, gamma, nullptr, nullptr, epsilon),
                 std::invalid_argument);
  }
}

TEST(SkipLayerNormalizationKernel, UsesRuntimeExecutorForLargeInputsOnly) {
  const onnx_light_cpu::SkipLayerNormalizationKernel kernel(MakeCtx());
  InlineExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run};
  onnx_light_cpu::ExecutionExecutorScope scope(&view);
  const auto gamma = Tensor::FromFloat("", {4096}, std::vector<float>(4096, 1));

  const auto small = Tensor::FromFloat("", {1, 4096}, std::vector<float>(4096, 1));
  (void)kernel(small, small, gamma);
  EXPECT_EQ(executor.dispatches, 0);

  const auto large = Tensor::FromFloat("", {64, 4096}, std::vector<float>(64 * 4096, 1));
  (void)kernel(large, large, gamma);
  EXPECT_EQ(executor.dispatches, 1);
  EXPECT_GT(executor.blocks, 1);
  EXPECT_LE(executor.blocks, 4);
}

} // namespace
