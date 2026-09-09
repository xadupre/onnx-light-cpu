// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/math/normalization_helpers.h"
#include "onnx_light_cpu/kernels/math/normalization_kernel.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/normalization_kernel.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace {

namespace norm = onnx_light_cpu::normalization;
namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

struct InlineExecutor {
  std::int64_t dispatches = 0;
  std::int64_t blocks = 0;

  static void Run(void *context, std::int64_t num_blocks, void *task_context,
                  onnx_light_cpu::ExecutionBlockFn task) {
    auto &self = *static_cast<InlineExecutor *>(context);
    ++self.dispatches;
    self.blocks = num_blocks;
    for (std::int64_t block = num_blocks; block > 0; --block) {
      task(task_context, block - 1);
    }
  }
};

rt_ns::KernelContext MakeContext(std::int64_t version) {
  return rt_ns::KernelContext(rt_ns::OpsetId(std::string(), version));
}

void AddIntAttribute(ONNX_LIGHT_NAMESPACE::NodeProto &node, const char *name, std::int64_t value) {
  auto *attribute = node.add_attribute();
  attribute->set_name(name);
  attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::INT);
  attribute->set_i(value);
}

void AddFloatAttribute(ONNX_LIGHT_NAMESPACE::NodeProto &node, const char *name, float value) {
  auto *attribute = node.add_attribute();
  attribute->set_name(name);
  attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::FLOAT);
  attribute->set_f(value);
}

rt_ns::Tensor MakeTensor(rt_ns::DataType type, const rt_ns::Shape &shape,
                         const std::vector<float> &values) {
  switch (type) {
  case rt_ns::DataType::FLOAT:
    return rt_ns::Tensor::FromFloat("", shape, values);
  case rt_ns::DataType::DOUBLE:
    return rt_ns::Tensor::FromDouble("", shape, std::vector<double>(values.begin(), values.end()));
  case rt_ns::DataType::FLOAT16:
    return rt_ns::MakeFloat16Tensor("", shape, values);
  case rt_ns::DataType::BFLOAT16:
    return rt_ns::MakeBfloat16Tensor("", shape, values);
  default:
    throw std::invalid_argument("unsupported test data type");
  }
}

double Value(const rt_ns::Tensor &tensor, std::size_t index) {
  return norm::TensorReader(tensor, "test", "tensor").LoadDouble(index);
}

double Tolerance(rt_ns::DataType type) {
  return type == rt_ns::DataType::FLOAT16 || type == rt_ns::DataType::BFLOAT16 ? 2.0e-2 : 1.0e-5;
}

float RoundFloat(rt_ns::DataType type, float value) {
  if (type == rt_ns::DataType::FLOAT16) {
    return onnx_light_cpu::detail::Float16BitsToFloat(
        onnx_light_cpu::detail::FloatToFloat16Bits(value));
  }
  if (type == rt_ns::DataType::BFLOAT16) {
    return onnx_light_cpu::detail::Bfloat16BitsToFloat(
        onnx_light_cpu::detail::FloatToBFloat16Bits(value));
  }
  return value;
}

TEST(OnnxLightNormalizationKernel, BatchNormalizationSupportsMixedParameterTypes) {
  const onnx_light_cpu::BatchNormalizationKernel kernel(MakeContext(15));
  const rt_ns::Tensor x = rt_ns::MakeFloat16Tensor("", {1, 2, 2}, {1.0F, 3.0F, 2.0F, 6.0F});
  const rt_ns::Tensor scale = rt_ns::Tensor::FromFloat("", {2}, {2.0F, 0.5F});
  const rt_ns::Tensor bias = rt_ns::Tensor::FromFloat("", {2}, {1.0F, -1.0F});
  const rt_ns::Tensor mean = rt_ns::MakeBfloat16Tensor("", {2}, {2.0F, 4.0F});
  const rt_ns::Tensor variance = rt_ns::MakeBfloat16Tensor("", {2}, {1.0F, 4.0F});
  const rt_ns::Tensor y = kernel(x, scale, bias, mean, variance, 0.0F);

  EXPECT_EQ(y.data_type, static_cast<std::int32_t>(rt_ns::DataType::FLOAT16));
  EXPECT_NEAR(Value(y, 0), -1.0, 2.0e-3);
  EXPECT_NEAR(Value(y, 1), 3.0, 2.0e-3);
  EXPECT_NEAR(Value(y, 2), -1.5, 2.0e-3);
  EXPECT_NEAR(Value(y, 3), -0.5, 2.0e-3);
}

TEST(OnnxLightNormalizationKernel, BatchNormalizationTrainingModeUpdatesRunningStatistics) {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("BatchNormalization");
  for (const char *name : {"X", "scale", "B", "mean", "variance"}) {
    node.add_input(name);
  }
  node.add_output("Y");
  node.add_output("running_mean");
  node.add_output("running_var");
  AddIntAttribute(node, "training_mode", 1);
  AddFloatAttribute(node, "epsilon", 0.0F);
  AddFloatAttribute(node, "momentum", 0.75F);
  rt_ns::RuntimeContext rt(MakeContext(15));
  rt.Set("X",
         rt_ns::MakeFloat16Tensor("", {2, 2, 2}, {1.0F, 3.0F, 2.0F, 6.0F, 5.0F, 7.0F, 4.0F, 8.0F}));
  rt.Set("scale", rt_ns::Tensor::FromFloat("", {2}, {2.0F, 0.5F}));
  rt.Set("B", rt_ns::Tensor::FromFloat("", {2}, {1.0F, -1.0F}));
  rt.Set("mean", rt_ns::MakeBfloat16Tensor("", {2}, {10.0F, 20.0F}));
  rt.Set("variance", rt_ns::MakeBfloat16Tensor("", {2}, {9.0F, 13.0F}));
  onnx_light_cpu::BatchNormalizationKernel kernel(MakeContext(15));
  kernel.set_node(node);
  ASSERT_NO_THROW(kernel.Run(rt));

  const rt_ns::Tensor &y = rt.Get("Y");
  const double sqrt_five = std::sqrt(5.0);
  EXPECT_EQ(y.data_type, static_cast<std::int32_t>(rt_ns::DataType::FLOAT16));
  EXPECT_NEAR(Value(y, 0), 1.0 - 6.0 / sqrt_five, 3.0e-3);
  EXPECT_NEAR(Value(y, 1), 1.0 - 2.0 / sqrt_five, 3.0e-3);
  EXPECT_NEAR(Value(y, 2), -1.0 - 1.5 / sqrt_five, 3.0e-3);
  EXPECT_NEAR(Value(y, 7), -1.0 + 1.5 / sqrt_five, 3.0e-3);
  EXPECT_EQ(rt.Get("running_mean").data_type, static_cast<std::int32_t>(rt_ns::DataType::BFLOAT16));
  EXPECT_EQ(rt.Get("running_var").data_type, static_cast<std::int32_t>(rt_ns::DataType::BFLOAT16));
  EXPECT_NEAR(Value(rt.Get("running_mean"), 0), 8.5, 5.0e-2);
  EXPECT_NEAR(Value(rt.Get("running_mean"), 1), 16.25, 5.0e-2);
  EXPECT_NEAR(Value(rt.Get("running_var"), 0), 8.0, 5.0e-2);
  EXPECT_NEAR(Value(rt.Get("running_var"), 1), 11.0, 5.0e-2);
}

TEST(OnnxLightNormalizationKernel, BatchTrainingAlignedAndTailSlicesMatchBuiltin) {
  const onnx_light_cpu::BatchNormalizationKernel kernel(MakeContext(15));
  const ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::BatchNormalization reference(MakeContext(15));
  for (const std::int64_t spatial : {1, 3, 4, 7, 8, 63, 64, 65, 68}) {
    const rt_ns::Shape shape{4, 3, spatial};
    std::vector<float> values(static_cast<std::size_t>(shape.product()));
    for (std::size_t i = 0; i < values.size(); ++i) {
      const std::size_t channel = (i / static_cast<std::size_t>(spatial)) % 3;
      values[i] = channel == 0 ? 8.0F
                               : static_cast<float>(static_cast<int>(i % 17) - 8) * 0.25F +
                                     static_cast<float>(channel) * 8.0F;
    }
    const auto x = rt_ns::Tensor::FromFloat("", shape, values);
    const auto [expected_y, expected_mean, expected_variance] = reference.TrainingForward(
        x, MakeTensor(rt_ns::DataType::FLOAT, {3}, {0.5F, -1.25F, 2.0F}),
        MakeTensor(rt_ns::DataType::FLOAT, {3}, {1.0F, 0.5F, -0.25F}),
        MakeTensor(rt_ns::DataType::FLOAT, {3}, {2.0F, 3.0F, 4.0F}),
        MakeTensor(rt_ns::DataType::FLOAT, {3}, {4.0F, 9.0F, 16.0F}), 1.0e-5F, 0.75F);
    for (const auto parameter_type : {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE}) {
      SCOPED_TRACE(::testing::Message() << "spatial=" << spatial
                                        << " parameter_type=" << static_cast<int>(parameter_type));
      const auto scale = MakeTensor(parameter_type, {3}, {0.5F, -1.25F, 2.0F});
      const auto bias = MakeTensor(parameter_type, {3}, {1.0F, 0.5F, -0.25F});
      const auto mean = MakeTensor(parameter_type, {3}, {2.0F, 3.0F, 4.0F});
      const auto variance = MakeTensor(parameter_type, {3}, {4.0F, 9.0F, 16.0F});
      const auto actual = kernel.Compute(x, scale, bias, mean, variance, true, 1.0e-5F, 0.75F);
      ASSERT_TRUE(actual.running_mean);
      ASSERT_TRUE(actual.running_variance);
      EXPECT_EQ(actual.running_mean->data_type, static_cast<std::int32_t>(parameter_type));
      EXPECT_EQ(actual.running_variance->data_type, static_cast<std::int32_t>(parameter_type));
      for (std::size_t i = 0; i < values.size(); ++i) {
        EXPECT_NEAR(Value(actual.y, i), Value(expected_y, i), 1.0e-5) << i;
      }
      for (std::size_t channel = 0; channel < 3; ++channel) {
        EXPECT_NEAR(Value(*actual.running_mean, channel), Value(expected_mean, channel), 1.0e-5);
        EXPECT_NEAR(Value(*actual.running_variance, channel), Value(expected_variance, channel),
                    1.0e-5);
      }
    }
  }
}

TEST(OnnxLightNormalizationKernel, BatchNormalizationTrainingAllowsOptionalStatistics) {
  for (const std::vector<std::string> &outputs :
       {std::vector<std::string>{"Y"}, std::vector<std::string>{"Y", "", ""},
        std::vector<std::string>{"Y", "running_mean", ""},
        std::vector<std::string>{"Y", "", "running_var"}}) {
    ONNX_LIGHT_NAMESPACE::NodeProto node;
    node.set_op_type("BatchNormalization");
    for (const char *name : {"X", "scale", "B", "mean", "variance"}) {
      node.add_input(name);
    }
    for (const std::string &output : outputs) {
      node.add_output(output);
    }
    AddIntAttribute(node, "training_mode", 1);

    rt_ns::RuntimeContext rt(MakeContext(15));
    rt.Set("X", rt_ns::Tensor::FromFloat("", {1, 1, 2}, {1.0F, 3.0F}));
    rt.Set("scale", rt_ns::Tensor::FromFloat("", {1}, {1.0F}));
    rt.Set("B", rt_ns::Tensor::FromFloat("", {1}, {0.0F}));
    rt.Set("mean", rt_ns::Tensor::FromFloat("", {1}, {4.0F}));
    rt.Set("variance", rt_ns::Tensor::FromFloat("", {1}, {9.0F}));
    onnx_light_cpu::BatchNormalizationKernel kernel(MakeContext(15));
    kernel.set_node(node);
    ASSERT_NO_THROW(kernel.Run(rt));
    EXPECT_TRUE(rt.Has("Y"));
    EXPECT_EQ(rt.Has("running_mean"), outputs.size() == 3 && !outputs[1].empty());
    EXPECT_EQ(rt.Has("running_var"), outputs.size() == 3 && !outputs[2].empty());
  }
}

TEST(OnnxLightNormalizationKernel, BatchNormalizationInferenceAllowsEmptyOutputPlaceholders) {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("BatchNormalization");
  for (const char *name : {"X", "scale", "B", "mean", "variance"}) {
    node.add_input(name);
  }
  node.add_output("Y");
  node.add_output("");
  node.add_output("");

  rt_ns::RuntimeContext rt(MakeContext(15));
  rt.Set("X", rt_ns::Tensor::FromFloat("", {1, 1, 2}, {1.0F, 3.0F}));
  rt.Set("scale", rt_ns::Tensor::FromFloat("", {1}, {1.0F}));
  rt.Set("B", rt_ns::Tensor::FromFloat("", {1}, {0.0F}));
  rt.Set("mean", rt_ns::Tensor::FromFloat("", {1}, {2.0F}));
  rt.Set("variance", rt_ns::Tensor::FromFloat("", {1}, {1.0F}));
  onnx_light_cpu::BatchNormalizationKernel kernel(MakeContext(15));
  kernel.set_node(node);
  ASSERT_NO_THROW(kernel.Run(rt));
  EXPECT_TRUE(rt.Has("Y"));
}

TEST(OnnxLightNormalizationKernel, BatchNormalizationRejectsInvalidOutputCounts) {
  auto make_runtime = []() {
    rt_ns::RuntimeContext rt(MakeContext(15));
    rt.Set("X", rt_ns::Tensor::FromFloat("", {1, 1, 2}, {1.0F, 2.0F}));
    rt.Set("scale", rt_ns::Tensor::FromFloat("", {1}, {1.0F}));
    rt.Set("B", rt_ns::Tensor::FromFloat("", {1}, {0.0F}));
    rt.Set("mean", rt_ns::Tensor::FromFloat("", {1}, {0.0F}));
    rt.Set("variance", rt_ns::Tensor::FromFloat("", {1}, {1.0F}));
    return rt;
  };
  auto make_node = [](bool training) {
    ONNX_LIGHT_NAMESPACE::NodeProto node;
    node.set_op_type("BatchNormalization");
    for (const char *name : {"X", "scale", "B", "mean", "variance"}) {
      node.add_input(name);
    }
    node.add_output("Y");
    if (training) {
      AddIntAttribute(node, "training_mode", 1);
    }
    return node;
  };

  ONNX_LIGHT_NAMESPACE::NodeProto inference = make_node(false);
  inference.add_output("running_mean");
  inference.add_output("running_var");
  rt_ns::RuntimeContext inference_rt = make_runtime();
  onnx_light_cpu::BatchNormalizationKernel inference_kernel(MakeContext(15));
  inference_kernel.set_node(inference);
  EXPECT_THROW(inference_kernel.Run(inference_rt), std::invalid_argument);

  ONNX_LIGHT_NAMESPACE::NodeProto training = make_node(true);
  training.add_output("");
  rt_ns::RuntimeContext training_rt = make_runtime();
  onnx_light_cpu::BatchNormalizationKernel training_kernel(MakeContext(15));
  training_kernel.set_node(training);
  EXPECT_THROW(training_kernel.Run(training_rt), std::invalid_argument);
}

TEST(OnnxLightNormalizationKernel, GroupNormalizationUsesNchwGroupsAndAffine) {
  const onnx_light_cpu::GroupNormalizationKernel kernel(MakeContext(21));
  const rt_ns::Tensor x = rt_ns::Tensor::FromFloat(
      "", {1, 4, 2}, {-3.0F, -1.0F, 1.0F, 3.0F, 10.0F, 12.0F, 14.0F, 16.0F});
  const rt_ns::Tensor scale = rt_ns::Tensor::FromFloat("", {4}, {1.0F, 2.0F, 1.0F, 0.5F});
  const rt_ns::Tensor bias = rt_ns::Tensor::FromFloat("", {4}, {0.0F, 1.0F, -1.0F, 2.0F});
  const rt_ns::Tensor y = kernel(x, scale, bias, 2, 0.0F);

  const float inv = 1.0F / std::sqrt(5.0F);
  EXPECT_NEAR(Value(y, 0), -3.0F * inv, 1.0e-6);
  EXPECT_NEAR(Value(y, 3), 3.0F * inv * 2.0F + 1.0F, 1.0e-6);
  EXPECT_NEAR(Value(y, 4), -3.0F * inv - 1.0F, 1.0e-6);
  EXPECT_NEAR(Value(y, 7), 3.0F * inv * 0.5F + 2.0F, 1.0e-6);
  EXPECT_THROW(kernel(x, scale, bias, 3), std::invalid_argument);
  EXPECT_THROW(kernel(x, scale, bias, 2, 1.0e-5F, 16), std::invalid_argument);
}

TEST(OnnxLightNormalizationKernel, GroupNormalizationOpset18UsesGroupAffineParameters) {
  const onnx_light_cpu::GroupNormalizationKernel kernel(MakeContext(18));
  const rt_ns::Tensor x = rt_ns::Tensor::FromFloat(
      "", {1, 4, 2}, {-3.0F, -1.0F, 1.0F, 3.0F, 10.0F, 12.0F, 14.0F, 16.0F});
  const rt_ns::Tensor scale = rt_ns::Tensor::FromFloat("", {2}, {2.0F, 0.5F});
  const rt_ns::Tensor bias = rt_ns::Tensor::FromFloat("", {2}, {1.0F, -2.0F});
  const rt_ns::Tensor y = kernel(x, scale, bias, 2, 0.0F);

  const float inv = 1.0F / std::sqrt(5.0F);
  EXPECT_NEAR(Value(y, 0), -3.0F * inv * 2.0F + 1.0F, 1.0e-6);
  EXPECT_NEAR(Value(y, 3), 3.0F * inv * 2.0F + 1.0F, 1.0e-6);
  EXPECT_NEAR(Value(y, 4), -3.0F * inv * 0.5F - 2.0F, 1.0e-6);
  EXPECT_NEAR(Value(y, 7), 3.0F * inv * 0.5F - 2.0F, 1.0e-6);
}

TEST(OnnxLightNormalizationKernel, GroupNormalizationOpset18UsesDoubleMoments) {
  const onnx_light_cpu::GroupNormalizationKernel kernel(MakeContext(18));
  const rt_ns::Tensor x =
      rt_ns::Tensor::FromDouble("", {1, 1, 4}, {10000000.1, 10000000.2, 10000000.3, 10000000.4});
  const rt_ns::Tensor scale = rt_ns::Tensor::FromDouble("", {1}, {1.0});
  const rt_ns::Tensor bias = rt_ns::Tensor::FromDouble("", {1}, {0.0});
  const rt_ns::Tensor y = kernel(x, scale, bias, 1, 1.0F, 16);

  EXPECT_FLOAT_EQ(static_cast<float>(Value(x, 0)), static_cast<float>(Value(x, 3)));
  const std::array<double, 4> reference = {-0.14884168, -0.04961389, 0.04961389, 0.14884168};
  for (std::size_t i = 0; i < reference.size(); ++i) {
    EXPECT_NEAR(Value(y, i), reference[i], 3.0e-4);
  }
}

TEST(OnnxLightNormalizationKernel, GroupNormalizationUsesFloatStashForDoubleInput) {
  const onnx_light_cpu::GroupNormalizationKernel kernel(MakeContext(21));
  const rt_ns::Tensor x = rt_ns::Tensor::FromDouble("", {1, 1, 2}, {100000001.0, 100000002.0});
  const rt_ns::Tensor scale = rt_ns::Tensor::FromDouble("", {1}, {1.0});
  const rt_ns::Tensor bias = rt_ns::Tensor::FromDouble("", {1}, {0.0});
  const rt_ns::Tensor y = kernel(x, scale, bias, 1);
  EXPECT_DOUBLE_EQ(Value(y, 0), 0.0);
  EXPECT_DOUBLE_EQ(Value(y, 1), 0.0);
}

TEST(OnnxLightNormalizationKernel, GroupNormalizationRoundsNormalizedValueBeforeAffine) {
  for (rt_ns::DataType type : {rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16}) {
    const onnx_light_cpu::GroupNormalizationKernel kernel(MakeContext(21));
    const rt_ns::Tensor x = MakeTensor(type, {1, 1, 4}, {-3.0F, -2.0F, -1.0F, -0.5F});
    const rt_ns::Tensor scale = MakeTensor(type, {1}, {0.05F});
    const rt_ns::Tensor bias = MakeTensor(type, {1}, {-0.25F});
    const rt_ns::Tensor y = kernel(x, scale, bias, 1);

    const float mean = -1.625F;
    const float variance = 0.921875F;
    const float inverse_std_dev = 1.0F / std::sqrt(variance + 1.0e-5F);
    const float stored_scale = RoundFloat(type, 0.05F);
    const float stored_bias = RoundFloat(type, -0.25F);
    for (std::size_t i = 0; i < 4; ++i) {
      const float input = static_cast<float>(Value(x, i));
      const float normalized = (input - mean) * inverse_std_dev;
      const float scaled = RoundFloat(type, RoundFloat(type, normalized) * stored_scale);
      const float expected = RoundFloat(type, scaled + stored_bias);
      EXPECT_FLOAT_EQ(static_cast<float>(Value(y, i)), expected);
    }
    const std::size_t discriminating_index = 2;
    const float discriminating_input = static_cast<float>(Value(x, discriminating_index));
    const float normalized = RoundFloat(type, (discriminating_input - mean) * inverse_std_dev);
    const float unrounded = RoundFloat(type, normalized * stored_scale + stored_bias);
    EXPECT_NE(static_cast<float>(Value(y, discriminating_index)), unrounded);
    const std::uint16_t reference_bits =
        type == rt_ns::DataType::FLOAT16 ? std::uint16_t{0xb2f6} : std::uint16_t{0xbe5e};
    EXPECT_EQ(reinterpret_cast<const std::uint16_t *>(y.bytes())[discriminating_index],
              reference_bits);
  }
}

TEST(OnnxLightNormalizationKernel, InstanceNormalizationSupportsEveryFloatType) {
  for (rt_ns::DataType type : {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE,
                               rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16}) {
    const onnx_light_cpu::InstanceNormalizationKernel kernel(MakeContext(22));
    const rt_ns::Tensor x = MakeTensor(type, {1, 2, 3}, {1.0F, 2.0F, 3.0F, 2.0F, 4.0F, 6.0F});
    const rt_ns::Tensor scale = MakeTensor(type, {2}, {1.0F, 0.5F});
    const rt_ns::Tensor bias = MakeTensor(type, {2}, {0.0F, 1.0F});
    const rt_ns::Tensor y = kernel(x, scale, bias, 0.0F);
    const double inv = std::sqrt(1.5);
    EXPECT_NEAR(Value(y, 0), -inv, Tolerance(type));
    EXPECT_NEAR(Value(y, 1), 0.0, Tolerance(type));
    EXPECT_NEAR(Value(y, 2), inv, Tolerance(type));
    EXPECT_NEAR(Value(y, 3), 1.0 - 0.5 * inv, Tolerance(type));
    EXPECT_NEAR(Value(y, 5), 1.0 + 0.5 * inv, Tolerance(type));
  }
}

TEST(OnnxLightNormalizationKernel, ContiguousMomentsRemainStableForLargeOffsets) {
  std::vector<float> values(10000);
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = i % 4 == 0 ? 99.0F : (i % 4 == 2 ? 101.0F : 100.0F);
  }
  const norm::Moments<float> moments =
      norm::ComputeContiguousMoments<rt_ns::DataType::FLOAT>(values.data(), values.size());
  const norm::Moments<float> stashed =
      norm::ComputeContiguousFloatMoments<rt_ns::DataType::FLOAT>(values.data(), values.size());
  const onnx_light_cpu::Float32NormalizationMoments shared =
      onnx_light_cpu::ComputeNormalizationMomentsFloat32(values.data(), values.size());

  EXPECT_FLOAT_EQ(moments.mean, 100.0F);
  EXPECT_FLOAT_EQ(moments.variance, 0.5F);
  EXPECT_FLOAT_EQ(stashed.mean, 100.0F);
  EXPECT_FLOAT_EQ(stashed.variance, 0.5F);
  EXPECT_FLOAT_EQ(shared.mean, 100.0F);
  EXPECT_FLOAT_EQ(shared.variance, 0.5F);
}

TEST(OnnxLightNormalizationKernel, InstanceAndMvnFloat32ContiguousMomentsMatchCenteredReference) {
  const onnx_light_cpu::InstanceNormalizationKernel instance(MakeContext(22));
  const onnx_light_cpu::MeanVarianceNormalizationKernel mvn(MakeContext(13));
  std::vector<rt_ns::Shape> shapes = {{4, 32, 16, 16}, {8, 16, 128}, {4, 16, 32, 8}, {1, 256, 128}};
  for (std::int64_t width : {1, 7, 8, 9, 15, 16, 17, 31, 63, 64, 65, 127, 129, 255, 257}) {
    shapes.push_back({2, 3, width});
  }
  for (std::int64_t width = 32; width < 48; ++width) {
    shapes.push_back({2, 3, width});
  }
  for (const rt_ns::Shape &shape : shapes) {
    const std::size_t channels = static_cast<std::size_t>(shape[1]);
    const std::size_t rows = static_cast<std::size_t>(shape[0]) * channels;
    std::size_t width = 1;
    std::vector<std::int64_t> axes;
    for (std::size_t axis = 2; axis < shape.size(); ++axis) {
      width *= static_cast<std::size_t>(shape[axis]);
      axes.push_back(static_cast<std::int64_t>(axis));
    }
    SCOPED_TRACE(width);
    std::vector<float> values(rows * width);
    std::vector<float> scales(channels);
    std::vector<float> biases(channels);
    for (std::size_t channel = 0; channel < channels; ++channel) {
      scales[channel] = channel % 2 == 0 ? 0.5F : -1.25F;
      biases[channel] = static_cast<float>(channel % 3) * 0.25F;
    }
    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t i = 0; i < width; ++i) {
        const float variation = static_cast<float>(static_cast<int>((i * 7 + row) % 29) - 14);
        values[row * width + i] =
            row % 3 == 0 ? 8.0F : variation * 0.125F + (row % 3 == 1 ? 0.0F : 256.0F);
      }
    }
    const rt_ns::Tensor x = rt_ns::Tensor::FromFloat("", shape, values);
    const rt_ns::Tensor scale = rt_ns::Tensor::FromFloat("", {shape[1]}, scales);
    const rt_ns::Tensor bias = rt_ns::Tensor::FromFloat("", {shape[1]}, biases);
    const rt_ns::Tensor instance_y = instance(x, scale, bias, 1.0e-5F);
    const rt_ns::Tensor mvn_y = mvn(x, axes);
    for (std::size_t row = 0; row < rows; ++row) {
      double mean = 0.0;
      for (std::size_t i = 0; i < width; ++i) {
        mean += values[row * width + i];
      }
      mean /= static_cast<double>(width);
      double variance = 0.0;
      for (std::size_t i = 0; i < width; ++i) {
        const double delta = static_cast<double>(values[row * width + i]) - mean;
        variance += delta * delta;
      }
      variance /= static_cast<double>(width);
      for (std::size_t i = 0; i < width; ++i) {
        const std::size_t index = row * width + i;
        const double centered = static_cast<double>(values[index]) - mean;
        const double instance_expected =
            centered / std::sqrt(variance + 1.0e-5F) * scales[row % channels] +
            biases[row % channels];
        EXPECT_NEAR(Value(instance_y, index), instance_expected, 1.1e-3);
        EXPECT_NEAR(Value(mvn_y, index), centered / (std::sqrt(variance) + 1.0e-9F), 5.0e-5);
      }
    }
  }
}

TEST(OnnxLightNormalizationKernel, InstanceAndMvnFloat32ContiguousSpecialValues) {
  const onnx_light_cpu::InstanceNormalizationKernel instance(MakeContext(22));
  const onnx_light_cpu::MeanVarianceNormalizationKernel mvn(MakeContext(13));
  const rt_ns::Tensor scale = rt_ns::Tensor::FromFloat("", {1}, {1.0F});
  const rt_ns::Tensor bias = rt_ns::Tensor::FromFloat("", {1}, {0.0F});
  for (float special :
       {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()}) {
    std::vector<float> values(65, 1.0F);
    values.back() = special;
    const rt_ns::Tensor x = rt_ns::Tensor::FromFloat("", {1, 1, 65}, values);
    const rt_ns::Tensor instance_y = instance(x, scale, bias);
    const rt_ns::Tensor mvn_y = mvn(x, {2});
    for (std::size_t i = 0; i < values.size(); ++i) {
      EXPECT_TRUE(std::isnan(Value(instance_y, i)));
      EXPECT_TRUE(std::isnan(Value(mvn_y, i)));
    }
  }
  const rt_ns::Tensor constant =
      rt_ns::Tensor::FromFloat("", {1, 1, 64}, std::vector<float>(64, std::ldexp(1.0F, 70)));
  const rt_ns::Tensor instance_y = instance(constant, scale, bias);
  const rt_ns::Tensor mvn_y = mvn(constant, {2});
  for (std::size_t i = 0; i < 64; ++i) {
    EXPECT_EQ(Value(instance_y, i), 0.0);
    EXPECT_EQ(Value(mvn_y, i), 0.0);
  }
}

TEST(OnnxLightNormalizationKernel, MvnConstantSimdTailPreservesOriginalMean) {
  const onnx_light_cpu::MeanVarianceNormalizationKernel mvn(MakeContext(13));
  for (float value : {256.1F, -256.1F}) {
    const auto x = rt_ns::Tensor::FromFloat("", {1, 1, 34}, std::vector<float>(34, value));
    const auto y = mvn(x, {2});
    for (std::size_t i = 0; i < 34; ++i) {
      EXPECT_EQ(Value(y, i), 0.0);
    }
  }
}

TEST(OnnxLightNormalizationKernel, LayerNormalizationBroadcastsAndReturnsFloatStats) {
  const onnx_light_cpu::LayerNormalizationKernel kernel(MakeContext(17));
  const rt_ns::Tensor x = rt_ns::MakeBfloat16Tensor("", {2, 2, 2}, {1, 2, 3, 4, 2, 4, 6, 8});
  const rt_ns::Tensor scale = rt_ns::MakeBfloat16Tensor("", {2}, {1.0F, 2.0F});
  const rt_ns::Tensor bias = rt_ns::MakeBfloat16Tensor("", {1}, {0.5F});
  const onnx_light_cpu::LayerNormalizationResult result =
      kernel(x, scale, &bias, 1, 0.0F, 1, true, true);

  ASSERT_TRUE(result.mean);
  ASSERT_TRUE(result.inv_std_dev);
  EXPECT_EQ(result.mean->shape, (rt_ns::Shape{2, 1, 1}));
  EXPECT_EQ(result.mean->data_type, static_cast<std::int32_t>(rt_ns::DataType::FLOAT));
  EXPECT_NEAR(Value(*result.mean, 0), 2.5, 1.0e-6);
  EXPECT_NEAR(Value(*result.inv_std_dev, 0), 1.0 / std::sqrt(1.25), 1.0e-6);
  EXPECT_NEAR(Value(result.y, 0), (1.0 - 2.5) / std::sqrt(1.25) + 0.5, 2.0e-2);
  EXPECT_NEAR(Value(result.y, 1), 2.0 * (2.0 - 2.5) / std::sqrt(1.25) + 0.5, 2.0e-2);
  EXPECT_THROW(kernel(x, scale, &bias, 1, 1.0e-5F, 16), std::invalid_argument);
}

TEST(OnnxLightNormalizationKernel, LayerNormalizationRunHandlesOmittedBiasAndOutputs) {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("LayerNormalization");
  node.add_input("X");
  node.add_input("Scale");
  node.add_input("");
  node.add_output("Y");
  node.add_output("");
  node.add_output("InvStdDev");
  rt_ns::RuntimeContext rt(MakeContext(17));
  rt.Set("X", rt_ns::Tensor::FromFloat("", {2, 2}, {1.0F, 3.0F, 2.0F, 6.0F}));
  rt.Set("Scale", rt_ns::Tensor::FromFloat("", {2}, {1.0F, 1.0F}));
  onnx_light_cpu::LayerNormalizationKernel kernel(MakeContext(17));
  kernel.set_node(node);
  EXPECT_NO_THROW(kernel.Run(rt));
  EXPECT_EQ(rt.Get("Y").shape, (rt_ns::Shape{2, 2}));
  EXPECT_EQ(rt.Get("InvStdDev").shape, (rt_ns::Shape{2, 1}));
}

TEST(OnnxLightNormalizationKernel, LayerNormalizationAcceptsFullInputAffineShape) {
  const onnx_light_cpu::LayerNormalizationKernel kernel(MakeContext(17));
  const rt_ns::Tensor x = rt_ns::Tensor::FromFloat("", {2, 2}, {1.0F, 3.0F, 2.0F, 6.0F});
  const rt_ns::Tensor scale = rt_ns::Tensor::FromFloat("", {2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
  const onnx_light_cpu::LayerNormalizationResult result = kernel(x, scale);
  EXPECT_NEAR(Value(result.y, 0), -1.0, 2.0e-5);
  EXPECT_NEAR(Value(result.y, 1), 2.0, 2.0e-5);
  EXPECT_NEAR(Value(result.y, 2), -3.0, 2.0e-5);
  EXPECT_NEAR(Value(result.y, 3), 4.0, 2.0e-5);
}

TEST(OnnxLightNormalizationKernel, LayerNormalizationUsesFloatStashForDoubleInput) {
  const onnx_light_cpu::LayerNormalizationKernel kernel(MakeContext(17));
  const rt_ns::Tensor x = rt_ns::Tensor::FromDouble("", {1, 2}, {100000001.0, 100000002.0});
  const rt_ns::Tensor scale = rt_ns::Tensor::FromDouble("", {2}, {1.0, 1.0});
  const onnx_light_cpu::LayerNormalizationResult result = kernel(x, scale);
  EXPECT_DOUBLE_EQ(Value(result.y, 0), 0.0);
  EXPECT_DOUBLE_EQ(Value(result.y, 1), 0.0);
}

TEST(OnnxLightNormalizationKernel, LayerNormalizationRoundsNormalizedValueBeforeAffine) {
  for (rt_ns::DataType type : {rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16}) {
    const onnx_light_cpu::LayerNormalizationKernel kernel(MakeContext(17));
    const rt_ns::Tensor x = MakeTensor(type, {1, 4}, {-3.0F, -2.0F, -1.0F, -0.5F});
    const rt_ns::Tensor scale = MakeTensor(type, {4}, {0.05F, 0.05F, 0.05F, 0.05F});
    const rt_ns::Tensor bias = MakeTensor(type, {4}, {-0.25F, -0.25F, -0.25F, -0.25F});
    const onnx_light_cpu::LayerNormalizationResult result = kernel(x, scale, &bias);

    const float mean = -1.625F;
    const float variance = 0.921875F;
    const float inverse_std_dev = 1.0F / std::sqrt(variance + 1.0e-5F);
    const float stored_scale = RoundFloat(type, 0.05F);
    const float stored_bias = RoundFloat(type, -0.25F);
    for (std::size_t i = 0; i < 4; ++i) {
      const float input = static_cast<float>(Value(x, i));
      const float normalized = (input - mean) * inverse_std_dev;
      const float scaled = RoundFloat(type, RoundFloat(type, normalized) * stored_scale);
      const float expected = RoundFloat(type, scaled + stored_bias);
      EXPECT_FLOAT_EQ(static_cast<float>(Value(result.y, i)), expected);
    }
    const std::size_t discriminating_index = 2;
    const float discriminating_input = static_cast<float>(Value(x, discriminating_index));
    const float normalized = RoundFloat(type, (discriminating_input - mean) * inverse_std_dev);
    const float unrounded = RoundFloat(type, normalized * stored_scale + stored_bias);
    EXPECT_NE(static_cast<float>(Value(result.y, discriminating_index)), unrounded);
    const std::uint16_t reference_bits =
        type == rt_ns::DataType::FLOAT16 ? std::uint16_t{0xb2f6} : std::uint16_t{0xbe5e};
    EXPECT_EQ(reinterpret_cast<const std::uint16_t *>(result.y.bytes())[discriminating_index],
              reference_bits);
  }
}

TEST(OnnxLightNormalizationKernel, LayerNormalizationUsesExecutorForLargeRowSets) {
  constexpr std::size_t rows = 128;
  constexpr std::size_t width = 1024;
  std::vector<float> values(rows * width);
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<float>(i % width) * 0.01F;
  }
  const rt_ns::Tensor x = rt_ns::Tensor::FromFloat(
      "", {static_cast<std::int64_t>(rows), static_cast<std::int64_t>(width)}, values);
  const rt_ns::Tensor scale = rt_ns::Tensor::FromFloat("", {static_cast<std::int64_t>(width)},
                                                       std::vector<float>(width, 1.0F));
  InlineExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run};
  {
    onnx_light_cpu::ExecutionExecutorScope scope(&view);
    const onnx_light_cpu::LayerNormalizationResult result =
        onnx_light_cpu::LayerNormalizationKernel(MakeContext(17))(x, scale);
    EXPECT_TRUE(std::isfinite(Value(result.y, 0)));
  }
  EXPECT_EQ(executor.dispatches, 1);
  EXPECT_GT(executor.blocks, 1);
}

TEST(OnnxLightNormalizationKernel, LayerNormalizationFloat32SimdHandlesVectorTails) {
  constexpr std::size_t rows = 3;
  constexpr std::size_t width = 515;
  std::vector<float> values(rows * width);
  std::vector<float> scales(width);
  std::vector<float> biases(width);
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<float>(static_cast<int>(i % 31) - 15) * 0.125F;
  }
  for (std::size_t i = 0; i < width; ++i) {
    scales[i] = 0.5F + static_cast<float>(i % 7) * 0.125F;
    biases[i] = static_cast<float>(static_cast<int>(i % 5) - 2) * 0.25F;
  }
  const rt_ns::Tensor x = rt_ns::Tensor::FromFloat(
      "", {static_cast<std::int64_t>(rows), static_cast<std::int64_t>(width)}, values);
  const rt_ns::Tensor scale =
      rt_ns::Tensor::FromFloat("", {static_cast<std::int64_t>(width)}, scales);
  const rt_ns::Tensor bias =
      rt_ns::Tensor::FromFloat("", {static_cast<std::int64_t>(width)}, biases);
  const onnx_light_cpu::LayerNormalizationResult result =
      onnx_light_cpu::LayerNormalizationKernel(MakeContext(17))(x, scale, &bias);

  for (std::size_t row = 0; row < rows; ++row) {
    double mean = 0.0;
    double variance = 0.0;
    for (std::size_t i = 0; i < width; ++i) {
      mean += values[row * width + i];
    }
    mean /= static_cast<double>(width);
    for (std::size_t i = 0; i < width; ++i) {
      const double delta = static_cast<double>(values[row * width + i]) - mean;
      variance += delta * delta;
    }
    variance /= static_cast<double>(width);
    const double inverse_std_dev = 1.0 / std::sqrt(variance + 1.0e-5);
    for (std::size_t i : {std::size_t{0}, width - 2, width - 1}) {
      const double expected =
          (static_cast<double>(values[row * width + i]) - mean) * inverse_std_dev * scales[i] +
          biases[i];
      EXPECT_NEAR(Value(result.y, row * width + i), expected, 2.0e-5);
    }
  }
}

TEST(OnnxLightNormalizationKernel, LpNormalizationHandlesStridedAxesAndZeroNorm) {
  const onnx_light_cpu::LpNormalizationKernel kernel(MakeContext(22));
  const rt_ns::Tensor x = rt_ns::Tensor::FromDouble("", {2, 2}, {3.0, 0.0, 4.0, 0.0});
  const rt_ns::Tensor y = kernel(x, 0, 2);
  EXPECT_NEAR(Value(y, 0), 0.6, 1.0e-12);
  EXPECT_NEAR(Value(y, 2), 0.8, 1.0e-12);
  EXPECT_DOUBLE_EQ(Value(y, 1), 0.0);
  EXPECT_DOUBLE_EQ(Value(y, 3), 0.0);
  EXPECT_THROW(kernel(x, 2, 2), std::invalid_argument);
  EXPECT_THROW(kernel(x, 0, 3), std::invalid_argument);
}

TEST(OnnxLightNormalizationKernel, LpNormalizationFloat32ContiguousPreservesDoubleReduction) {
  const onnx_light_cpu::LpNormalizationKernel kernel(MakeContext(22));
  for (std::int64_t width : {1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 127, 128, 129}) {
    SCOPED_TRACE(width);
    const std::size_t count = static_cast<std::size_t>(width);
    std::vector<float> values(4 * count);
    for (std::size_t i = 0; i < count; ++i) {
      const float value = static_cast<float>(static_cast<int>(i % 13) - 6);
      values[count + i] = value;
      values[2 * count + i] = value * 1.0e20F;
      values[3 * count + i] = value * 1.0e-30F;
    }
    const rt_ns::Tensor x = rt_ns::Tensor::FromFloat("", {2, 2, width}, values);
    for (std::int64_t p : {1, 2}) {
      const rt_ns::Tensor y = kernel(x, -1, p);
      for (std::size_t row = 0; row < 4; ++row) {
        double sum = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
          const double value = values[row * count + i];
          sum += p == 1 ? std::abs(value) : value * value;
        }
        const double norm_value = p == 1 ? sum : std::sqrt(sum);
        for (std::size_t i = 0; i < count; ++i) {
          const double expected = norm_value == 0.0 ? 0.0 : values[row * count + i] / norm_value;
          EXPECT_NEAR(Value(y, row * count + i), expected, 1.0e-7);
        }
      }
    }
  }
}

TEST(OnnxLightNormalizationKernel, MeanVarianceNormalizationUsesRequestedAxes) {
  const onnx_light_cpu::MeanVarianceNormalizationKernel kernel(MakeContext(13));
  const rt_ns::Tensor x = rt_ns::MakeFloat16Tensor("", {2, 2}, {1.0F, 3.0F, 5.0F, 7.0F});
  const rt_ns::Tensor y = kernel(x, {1});
  EXPECT_NEAR(Value(y, 0), -1.0, 2.0e-3);
  EXPECT_NEAR(Value(y, 1), 1.0, 2.0e-3);
  EXPECT_NEAR(Value(y, 2), -1.0, 2.0e-3);
  EXPECT_NEAR(Value(y, 3), 1.0, 2.0e-3);
  EXPECT_THROW(kernel(x, {2}), std::invalid_argument);
}

TEST(OnnxLightNormalizationKernel, MeanVarianceNormalizationHandlesOverflowingRawMoments) {
  const onnx_light_cpu::MeanVarianceNormalizationKernel kernel(MakeContext(13));
  const rt_ns::Tensor x = rt_ns::Tensor::FromFloat("", {2, 1}, {2.0e20F, 2.0e20F});
  const rt_ns::Tensor y = kernel(x, {0});

  EXPECT_FLOAT_EQ(Value(y, 0), 0.0F);
  EXPECT_FLOAT_EQ(Value(y, 1), 0.0F);
}

TEST(OnnxLightNormalizationKernel, MeanVarianceNormalizationKeepsGenericMaskFallback) {
  const onnx_light_cpu::MeanVarianceNormalizationKernel kernel(MakeContext(13));
  const rt_ns::Tensor x =
      rt_ns::Tensor::FromFloat("", {2, 2, 2}, {1.0F, 10.0F, 3.0F, 14.0F, 2.0F, 8.0F, 6.0F, 16.0F});
  const rt_ns::Tensor y = kernel(x, {1});
  const std::vector<double> expected = {-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0};
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(Value(y, i), expected[i], 1.0e-6);
  }
}

TEST(OnnxLightNormalizationKernel, EveryKernelAcceptsAllFourFloatingPointTypes) {
  for (rt_ns::DataType type : {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE,
                               rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16}) {
    const rt_ns::Tensor x = MakeTensor(type, {1, 2, 2}, {1.0F, 3.0F, 2.0F, 6.0F});
    const rt_ns::Tensor affine_scale = MakeTensor(type, {2}, {1.0F, 0.5F});
    const rt_ns::Tensor bias = MakeTensor(type, {2}, {0.0F, 1.0F});
    const rt_ns::Tensor mean = MakeTensor(type, {2}, {2.0F, 4.0F});
    const rt_ns::Tensor variance = MakeTensor(type, {2}, {1.0F, 4.0F});

    EXPECT_EQ(onnx_light_cpu::BatchNormalizationKernel(MakeContext(15))(x, affine_scale, bias, mean,
                                                                        variance)
                  .data_type,
              static_cast<std::int32_t>(type));
    EXPECT_EQ(onnx_light_cpu::GroupNormalizationKernel(MakeContext(21))(x, affine_scale, bias, 2)
                  .data_type,
              static_cast<std::int32_t>(type));
    EXPECT_EQ(onnx_light_cpu::InstanceNormalizationKernel(MakeContext(22))(x, affine_scale, bias)
                  .data_type,
              static_cast<std::int32_t>(type));
    EXPECT_EQ(
        onnx_light_cpu::LayerNormalizationKernel(MakeContext(17))(x, affine_scale).y.data_type,
        static_cast<std::int32_t>(type));
    EXPECT_EQ(onnx_light_cpu::LpNormalizationKernel(MakeContext(22))(x).data_type,
              static_cast<std::int32_t>(type));
    EXPECT_EQ(onnx_light_cpu::MeanVarianceNormalizationKernel(MakeContext(13))(x, {2}).data_type,
              static_cast<std::int32_t>(type));
  }
}

} // namespace
