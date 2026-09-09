// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/kernels/com_microsoft/naive_skip_simplified_layer_normalization_kernel.h"
#include "onnx_light_cpu/kernels/com_microsoft/skip_simplified_layer_normalization_kernel.h"
#include "onnx_light_cpu/kernels/register_kernels.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
namespace rt = ONNX_LIGHT_NAMESPACE::core::runtime;
using ONNX_LIGHT_NAMESPACE::NodeProto;
using rt::DataType;
using rt::Shape;
using rt::Tensor;

rt::KernelContext MakeCtx() { return rt::KernelContext(rt::OpsetId("com.microsoft", 1)); }

NodeProto MakeNode(bool bias = false, bool residual = true) {
  NodeProto node;
  node.set_op_type("SkipSimplifiedLayerNormalization");
  node.set_domain("com.microsoft");
  node.add_input("input");
  node.add_input("skip");
  node.add_input("gamma");
  node.add_input(bias ? "bias" : "");
  node.add_output("output");
  if (residual) {
    node.add_output("");
    node.add_output("");
    node.add_output("sum");
  }
  return node;
}

Tensor MakeTensor(DataType type, const Shape &shape, const std::vector<float> &values) {
  if (type == DataType::BFLOAT16) {
    return rt::MakeBfloat16Tensor("", shape, values);
  }
  return type == DataType::FLOAT ? Tensor::FromFloat("", shape, values)
                                 : rt::MakeFloat16Tensor("", shape, values);
}

float Value(const Tensor &tensor, std::size_t index) {
  if (tensor.data_type == DataType::FLOAT) {
    return tensor.AsFloat()[index];
  }
  if (tensor.data_type == DataType::BFLOAT16) {
    return onnx_light_cpu::detail::Bfloat16BitsToFloat(
        reinterpret_cast<const std::uint16_t *>(tensor.bytes())[index]);
  }
  return onnx_light_cpu::detail::Float16BitsToFloat(
      reinterpret_cast<const std::uint16_t *>(tensor.bytes())[index]);
}

std::vector<std::uint8_t> Bytes(const Tensor &tensor) {
  return {tensor.bytes(), tensor.bytes() + tensor.size_bytes()};
}

template <typename Kernel>
class OnnxLightSkipSimplifiedLayerNormalizationKernel : public ::testing::Test {};
using Implementations =
    ::testing::Types<onnx_light_cpu::SkipSimplifiedLayerNormalizationKernel,
                     onnx_light_cpu::NaiveSkipSimplifiedLayerNormalizationKernel>;
TYPED_TEST_SUITE(OnnxLightSkipSimplifiedLayerNormalizationKernel, Implementations);

TYPED_TEST(OnnxLightSkipSimplifiedLayerNormalizationKernel, AnalyticalValuesAndInputImmutability) {
  for (auto type : {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16}) {
    const Tensor input = MakeTensor(type, {2, 2}, {1, -3, 3, -5});
    const Tensor skip = MakeTensor(type, {2, 2}, {0.5F, 0.5F, 0.5F, 0.5F});
    const Tensor gamma = MakeTensor(type, {2}, {2, -3});
    const Tensor bias = MakeTensor(type, {2}, {0.5F, 0.5F});
    const auto before_input = Bytes(input), before_skip = Bytes(skip);
    const auto before_gamma = Bytes(gamma), before_bias = Bytes(bias);
    const auto result = TypeParam(MakeCtx())(input, skip, gamma, &bias, 0, true);
    EXPECT_EQ(result.output.shape, input.shape);
    EXPECT_EQ(result.output.data_type, type);
    EXPECT_EQ(result.input_skip_bias_sum.shape, input.shape);
    EXPECT_EQ(result.input_skip_bias_sum.data_type, type);
    for (std::size_t i = 0; i < 4; ++i) {
      EXPECT_EQ(Value(result.output, i), i % 2 ? 3 : 2);
      EXPECT_EQ(Value(result.input_skip_bias_sum, i), (i < 2 ? 2 : 4) * (i % 2 ? -1 : 1));
    }
    EXPECT_EQ(Bytes(input), before_input);
    EXPECT_EQ(Bytes(skip), before_skip);
    EXPECT_EQ(Bytes(gamma), before_gamma);
    EXPECT_EQ(Bytes(bias), before_bias);
  }
}

TYPED_TEST(OnnxLightSkipSimplifiedLayerNormalizationKernel, EpsilonAndOmittedResidual) {
  const auto input = Tensor::FromFloat("", {1, 2}, {2, -2});
  const auto skip = Tensor::FromFloat("", {1, 2}, {0, 0});
  const auto gamma = Tensor::FromFloat("", {2}, {1, 2});
  const auto result = TypeParam(MakeCtx())(input, skip, gamma, nullptr, 12);
  EXPECT_FLOAT_EQ(result.output.AsFloat()[0], 0.5F);
  EXPECT_FLOAT_EQ(result.output.AsFloat()[1], -1.F);
  EXPECT_EQ(result.input_skip_bias_sum.size_bytes(), 0u);
  const auto zero = TypeParam(MakeCtx())(skip, skip, gamma);
  EXPECT_EQ(zero.output.AsFloat()[0], 0);
  EXPECT_EQ(zero.output.AsFloat()[1], 0);
}

TYPED_TEST(OnnxLightSkipSimplifiedLayerNormalizationKernel, HalfResidualRoundsAfterOptionalBias) {
  const auto input = rt::MakeFloat16Tensor("", {1, 2}, {1, 2});
  const auto skip = rt::MakeFloat16Tensor("", {1, 2}, {0.00048828125F, 0.0009765625F});
  const auto gamma = rt::MakeFloat16Tensor("", {2}, {1, 1});
  const auto bias = rt::MakeFloat16Tensor("", {2}, {0.00048828125F, 0.0009765625F});
  const TypeParam kernel(MakeCtx());
  const auto without_bias = kernel(input, skip, gamma, nullptr, 0, true);
  EXPECT_EQ(Value(without_bias.input_skip_bias_sum, 0), 1);
  EXPECT_EQ(Value(without_bias.input_skip_bias_sum, 1), 2);
  const auto with_bias = kernel(input, skip, gamma, &bias, 0, true);
  EXPECT_EQ(Value(with_bias.input_skip_bias_sum, 0), 1.0009765625F);
  EXPECT_EQ(Value(with_bias.input_skip_bias_sum, 1), 2.001953125F);
  EXPECT_EQ(Bytes(without_bias.output), Bytes(with_bias.output));
}

TYPED_TEST(OnnxLightSkipSimplifiedLayerNormalizationKernel,
           HalfNormalizationKeepsFloatResidualPrecision) {
  const auto input = rt::MakeFloat16Tensor("", {1, 2}, {1, 2});
  const auto skip = rt::MakeFloat16Tensor("", {1, 2}, {0.00048828125F, 0});
  const auto gamma = rt::MakeFloat16Tensor("", {2}, {1, 1});
  const auto result = TypeParam(MakeCtx())(input, skip, gamma, nullptr, 0, true);
  EXPECT_EQ(Value(result.input_skip_bias_sum, 0), 1.F);
  EXPECT_EQ(Value(result.input_skip_bias_sum, 1), 2.F);
  EXPECT_EQ(Value(result.output, 0), 0.6328125F);
  EXPECT_EQ(Value(result.output, 1), 1.2646484375F);
}

TYPED_TEST(OnnxLightSkipSimplifiedLayerNormalizationKernel,
           BfloatNormalizationKeepsFloatResidualPrecision) {
  const auto input = rt::MakeBfloat16Tensor("", {1, 2}, {1.F, 3.F});
  const auto skip = rt::MakeBfloat16Tensor("", {1, 2}, {0.00390625F, 0.F});
  const auto gamma = rt::MakeBfloat16Tensor("", {2}, {1.F, 1.F});
  const auto result = TypeParam(MakeCtx())(input, skip, gamma, nullptr, 0, true);
  EXPECT_EQ(Value(result.input_skip_bias_sum, 0), 1.F);
  EXPECT_EQ(Value(result.input_skip_bias_sum, 1), 3.F);
  EXPECT_EQ(Value(result.output, 0), 0.44921875F);
  EXPECT_EQ(Value(result.output, 1), 1.34375F);
}

TYPED_TEST(OnnxLightSkipSimplifiedLayerNormalizationKernel,
           HalfResidualOverflowDoesNotOverflowNormalizedOutput) {
  const auto input = rt::MakeFloat16Tensor("", {1, 2}, {65504.F, 65504.F});
  const auto gamma = rt::MakeFloat16Tensor("", {2}, {1.F, -2.F});
  for (bool residual : {false, true}) {
    const auto result = TypeParam(MakeCtx())(input, input, gamma, nullptr, 0, residual, true, true);
    EXPECT_EQ(Value(result.output, 0), 1.F);
    EXPECT_EQ(Value(result.output, 1), -2.F);
    EXPECT_EQ(result.mean.AsFloat()[0], 0.F);
    EXPECT_NEAR(result.inv_std_var.AsFloat()[0], 1.F / 131008.F, 1e-11F);
    if (residual) {
      EXPECT_EQ(Value(result.input_skip_bias_sum, 0), std::numeric_limits<float>::infinity());
      EXPECT_EQ(Value(result.input_skip_bias_sum, 1), std::numeric_limits<float>::infinity());
    }
  }
}

TYPED_TEST(OnnxLightSkipSimplifiedLayerNormalizationKernel, DefaultEpsilonIsOneTrillionth) {
  const auto input = Tensor::FromFloat("", {1, 1}, {1e-6F});
  const auto skip = Tensor::FromFloat("", {1, 1}, {0.F});
  const auto gamma = Tensor::FromFloat("", {1}, {1.F});
  const auto result = TypeParam(MakeCtx())(input, skip, gamma);
  EXPECT_NEAR(result.output.AsFloat()[0], 0.7071067811865476, 1e-7);
  auto node = MakeNode(false, false);
  rt::RuntimeContext context(MakeCtx());
  context.Set("input", input);
  context.Set("skip", skip);
  context.Set("gamma", gamma);
  TypeParam(node, MakeCtx()).Run(context);
  EXPECT_NEAR(context.Get("output").AsFloat()[0], 0.7071067811865476, 1e-7);
}

TYPED_TEST(OnnxLightSkipSimplifiedLayerNormalizationKernel, BfloatResidualUsesRoundToNearestEven) {
  const auto input = rt::MakeBfloat16Tensor("", {1, 2}, {1.F, 1.F});
  const auto skip = rt::MakeBfloat16Tensor("", {1, 2}, {0.00390625F, 0.01171875F});
  const auto gamma = rt::MakeBfloat16Tensor("", {2}, {0.F, 0.F});
  const auto result = TypeParam(MakeCtx())(input, skip, gamma, nullptr, 0, true);
  EXPECT_EQ(Value(result.input_skip_bias_sum, 0), 1.F);
  EXPECT_EQ(Value(result.input_skip_bias_sum, 1), 1.015625F);
  EXPECT_EQ(Value(result.output, 0), 0.F);
  EXPECT_EQ(Value(result.output, 1), 0.F);
}

TYPED_TEST(OnnxLightSkipSimplifiedLayerNormalizationKernel, RowsTailsAndNaiveParity) {
  for (auto type : {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16}) {
    for (std::int64_t width : {1, 3, 7, 16, 17, 31, 33, 129}) {
      for (std::int64_t rows : {1, 3, 65}) {
        std::vector<float> values(rows * width), skips(rows * width), scales(width), biases(width);
        for (std::size_t i = 0; i < values.size(); ++i) {
          values[i] = (static_cast<int>(i % 29) - 14) * 0.125F;
          skips[i] = (static_cast<int>(i % 13) - 6) * 0.0625F;
        }
        for (std::size_t i = 0; i < scales.size(); ++i) {
          scales[i] = (static_cast<int>(i % 7) - 3) * 0.25F;
          biases[i] = (static_cast<int>(i % 5) - 2) * 0.125F;
        }
        const auto input = MakeTensor(type, {rows, width}, values);
        const auto skip = MakeTensor(type, {rows, width}, skips);
        const auto gamma = MakeTensor(type, {width}, scales);
        const auto bias = MakeTensor(type, {width}, biases);
        for (bool with_bias : {false, true}) {
          const auto *bias_pointer = with_bias ? &bias : nullptr;
          const auto actual = TypeParam(MakeCtx())(input, skip, gamma, bias_pointer, 1e-5F, true);
          const auto expected = onnx_light_cpu::NaiveSkipSimplifiedLayerNormalizationKernel(
              MakeCtx())(input, skip, gamma, bias_pointer, 1e-5F, true);
          for (std::int64_t row = 0; row < rows; ++row) {
            double squared_sum = 0;
            for (std::int64_t col = 0; col < width; ++col) {
              const auto i = row * width + col;
              const double sum = values[i] + skips[i] + (with_bias ? biases[col] : 0);
              squared_sum += sum * sum;
              EXPECT_EQ(Value(actual.input_skip_bias_sum, i), sum);
            }
            for (std::int64_t col = 0; col < width; ++col) {
              const auto i = row * width + col;
              const double sum = values[i] + skips[i] + (with_bias ? biases[col] : 0);
              const double analytical = sum / std::sqrt(squared_sum / width + 1e-5) * scales[col];
              const double tolerance = type == DataType::FLOAT     ? 2e-6
                                       : type == DataType::FLOAT16 ? 2e-3
                                                                   : 2e-2;
              EXPECT_NEAR(Value(actual.output, i), analytical, tolerance);
              EXPECT_NEAR(Value(actual.output, i), Value(expected.output, i), tolerance);
            }
          }
        }
      }
    }
  }
}

TYPED_TEST(OnnxLightSkipSimplifiedLayerNormalizationKernel, BroadcastSkipAndEmptyRows) {
  for (auto type : {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16}) {
    const auto gamma = MakeTensor(type, {2}, {1, 1});
    const auto input = MakeTensor(type, {2, 2, 2}, {1, -1, 2, -2, 3, -3, 4, -4});
    for (const Shape &shape : {Shape{2, 2}, Shape{1, 2, 2}}) {
      const auto skip = MakeTensor(type, shape, {1, -1, 2, -2});
      const auto result = TypeParam(MakeCtx())(input, skip, gamma, nullptr, 0, true);
      for (std::size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(Value(result.output, i), i % 2 ? -1 : 1);
        EXPECT_EQ(Value(result.input_skip_bias_sum, i), Value(input, i) + Value(skip, i % 4));
      }
    }
    for (const Shape &shape : {Shape{0, 2}, Shape{0, 3, 2}, Shape{2, 0, 2}}) {
      const auto empty = MakeTensor(type, shape, {});
      const auto result = TypeParam(MakeCtx())(empty, empty, gamma, nullptr, 0, true);
      EXPECT_EQ(result.output.shape, shape);
      EXPECT_EQ(result.input_skip_bias_sum.shape, shape);
      EXPECT_EQ(result.output.size_bytes(), 0u);
      EXPECT_EQ(result.input_skip_bias_sum.size_bytes(), 0u);
    }
  }
}

TYPED_TEST(OnnxLightSkipSimplifiedLayerNormalizationKernel,
           RuntimeOptionalBiasAndResidualSlotThree) {
  for (bool bias : {false, true}) {
    for (bool residual : {false, true}) {
      auto node = MakeNode(bias, residual);
      ONNX_LIGHT_NAMESPACE::AddAttribute(node, "epsilon", 0.F);
      rt::RuntimeContext context(MakeCtx());
      context.Set("input", Tensor::FromFloat("", {1, 2}, {1, -3}));
      context.Set("skip", Tensor::FromFloat("", {1, 2}, {0.5F, 0.5F}));
      context.Set("gamma", Tensor::FromFloat("", {2}, {1, 1}));
      if (bias) {
        context.Set("bias", Tensor::FromFloat("", {2}, {0.5F, 0.5F}));
      }
      TypeParam kernel(node, MakeCtx());
      kernel.Run(context);
      const auto expected =
          TypeParam(MakeCtx())(context.Get("input"), context.Get("skip"), context.Get("gamma"),
                               bias ? &context.Get("bias") : nullptr, 0, residual);
      EXPECT_EQ(Bytes(context.Get("output")), Bytes(expected.output));
      if (residual) {
        EXPECT_EQ(Bytes(context.Get("sum")), Bytes(expected.input_skip_bias_sum));
      }
      if (!bias) {
        node.mutable_input()->resize(node.input_size() - 1);
        TypeParam(node, MakeCtx()).Run(context);
        EXPECT_EQ(Bytes(context.Get("output")), Bytes(expected.output));
      }
    }
  }
}

TYPED_TEST(OnnxLightSkipSimplifiedLayerNormalizationKernel,
           RejectsInvalidShapesTypesBuffersAndEpsilon) {
  const TypeParam kernel(MakeCtx());
  const auto input = Tensor::FromFloat("", {2, 3}, std::vector<float>(6, 1));
  const auto gamma = Tensor::FromFloat("", {3}, {1, 1, 1});
  for (const Shape &shape : {Shape{}, Shape{6}, Shape{1, 1, 2, 3}, Shape{2, 0}}) {
    auto bad = input;
    bad.shape = shape;
    EXPECT_THROW((void)kernel(bad, bad, gamma), std::invalid_argument);
  }
  for (const Shape &shape : {Shape{3, 2}, Shape{1, 3}, Shape{3}, Shape{1, 2, 3}}) {
    auto bad = input;
    bad.shape = shape;
    EXPECT_THROW((void)kernel(input, bad, gamma), std::invalid_argument);
  }
  for (const Shape &shape : {Shape{}, Shape{1, 3}, Shape{2}, Shape{4}}) {
    auto bad = gamma;
    bad.shape = shape;
    EXPECT_THROW((void)kernel(input, input, bad), std::invalid_argument);
    EXPECT_THROW((void)kernel(input, input, gamma, &bad), std::invalid_argument);
  }
  for (float epsilon :
       {-1.F, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
    EXPECT_THROW((void)kernel(input, input, gamma, nullptr, epsilon), std::invalid_argument);
  }
  const auto half = rt::MakeFloat16Tensor("", {2, 3}, std::vector<float>(6, 1));
  const auto half_gamma = rt::MakeFloat16Tensor("", {3}, {1, 1, 1});
  EXPECT_THROW((void)kernel(input, half, gamma), std::invalid_argument);
  EXPECT_THROW((void)kernel(input, input, half_gamma), std::invalid_argument);
  EXPECT_THROW((void)kernel(input, input, gamma, &half_gamma), std::invalid_argument);
  for (auto type : {DataType::DOUBLE, DataType::INT64, DataType::INT32}) {
    auto bad = input;
    bad.data_type = type;
    EXPECT_THROW((void)kernel(bad, bad, gamma), std::invalid_argument);
  }
  for (const Shape &shape :
       {Shape{2, 4}, Shape{2, -3}, Shape{std::numeric_limits<std::int64_t>::max(), 3}}) {
    auto bad = input;
    bad.shape = shape;
    EXPECT_THROW((void)kernel(bad, bad, gamma), std::invalid_argument);
  }
  auto oversized = Tensor::FromFloat("", {7}, std::vector<float>(7, 1));
  oversized.shape = {2, 3};
  EXPECT_THROW((void)kernel(oversized, input, gamma), std::invalid_argument);
  const auto zero_width = Tensor::FromFloat("", {2, 0}, {});
  const auto empty_gamma = Tensor::FromFloat("", {0}, {});
  EXPECT_THROW((void)kernel(zero_width, zero_width, empty_gamma), std::invalid_argument);
  const auto scalar = Tensor::FromFloat("", {}, {1.F});
  const auto scalar_gamma = Tensor::FromFloat("", {1}, {1.F});
  EXPECT_THROW((void)kernel(scalar, scalar, scalar_gamma), std::invalid_argument);
  const auto rank3 = Tensor::FromFloat("", {2, 3, 3}, std::vector<float>(18, 1.F));
  for (const Shape &shape : {Shape{1, 3}, Shape{1, 1, 3}, Shape{3}}) {
    const auto unsupported_broadcast = Tensor::FromFloat("", shape, {1.F, 1.F, 1.F});
    EXPECT_THROW((void)kernel(rank3, unsupported_broadcast, gamma), std::invalid_argument);
  }
}

TYPED_TEST(OnnxLightSkipSimplifiedLayerNormalizationKernel, RejectsInvalidNodeInputsAndOutputs) {
  rt::RuntimeContext context(MakeCtx());
  context.Set("input", Tensor::FromFloat("", {1, 2}, {1, 1}));
  context.Set("skip", Tensor::FromFloat("", {1, 2}, {1, 1}));
  context.Set("gamma", Tensor::FromFloat("", {2}, {1, 1}));
  std::vector<NodeProto> invalid;
  for (int slot = 0; slot < 3; ++slot) {
    auto node = MakeNode();
    node.ref_input()[slot] = "";
    invalid.push_back(node);
  }
  auto node = MakeNode();
  node.clear_input();
  invalid.push_back(node);
  node = MakeNode();
  node.add_input("extra");
  invalid.push_back(node);
  node = MakeNode();
  node.clear_output();
  invalid.push_back(node);
  node = MakeNode();
  node.ref_output()[0] = "";
  invalid.push_back(node);
  node = MakeNode();
  node.add_output("extra");
  invalid.push_back(node);
  node = MakeNode();
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "epsilon", -1.F);
  invalid.push_back(node);
  for (const auto &bad : invalid) {
    EXPECT_THROW(
        {
          TypeParam kernel(bad, MakeCtx());
          kernel.Run(context);
        },
        std::invalid_argument);
  }
}

TEST(OnnxLightSkipSimplifiedLayerNormalizationRegistration, BothMicrosoftImplementationSwitches) {
  for (auto implementation : {onnx_light_cpu::MicrosoftKernelImplementation::NAIVE,
                              onnx_light_cpu::MicrosoftKernelImplementation::OPTIMIZED}) {
    onnx_light_cpu::RegisterMicrosoftKernels(implementation);
    const auto &table = rt::KernelDispatchTable();
    EXPECT_NE(table.find("com.microsoft:SkipSimplifiedLayerNormalization"), table.end());
  }
}

TYPED_TEST(OnnxLightSkipSimplifiedLayerNormalizationKernel,
           RuntimeStatisticsPreserveOptionalSlots) {
  for (auto type : {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16}) {
    for (bool mean : {false, true}) {
      for (bool inverse : {false, true}) {
        for (bool residual : {false, true}) {
          auto node = MakeNode();
          node.ref_output()[1] = mean ? "mean" : "";
          node.ref_output()[2] = inverse ? "inverse" : "";
          if (!residual) {
            node.ref_output()[3] = "";
            while (node.output_size() > 1 && node.output(node.output_size() - 1).empty()) {
              node.mutable_output()->resize(node.output_size() - 1);
            }
          }
          ONNX_LIGHT_NAMESPACE::AddAttribute(node, "epsilon", 0.F);
          rt::RuntimeContext context(MakeCtx());
          context.Set("input", MakeTensor(type, {1, 2, 2}, {1, -1, 2, -2}));
          context.Set("skip", MakeTensor(type, {1, 2, 2}, {1, -1, 2, -2}));
          context.Set("gamma", MakeTensor(type, {2}, {1, 1}));
          TypeParam(node, MakeCtx()).Run(context);
          if (mean) {
            const auto &statistics = context.Get("mean");
            EXPECT_EQ(statistics.data_type, DataType::FLOAT);
            EXPECT_EQ(statistics.shape, (Shape{1, 2, 1}));
            EXPECT_EQ(statistics.AsFloat()[0], 0.F);
            EXPECT_EQ(statistics.AsFloat()[1], 0.F);
          }
          if (inverse) {
            const auto &statistics = context.Get("inverse");
            EXPECT_EQ(statistics.data_type, DataType::FLOAT);
            EXPECT_EQ(statistics.shape, (Shape{1, 2, 1}));
            EXPECT_FLOAT_EQ(statistics.AsFloat()[0], 0.5F);
            EXPECT_FLOAT_EQ(statistics.AsFloat()[1], 0.25F);
          }
          if (residual) {
            EXPECT_EQ(Value(context.Get("sum"), 0), 2.F);
          }
          EXPECT_EQ(Value(context.Get("output"), 0), 1.F);
        }
      }
    }
  }
}

struct InlineExecutor {
  std::int64_t blocks = 0;
  static void Run(void *context, std::int64_t count, void *task_context,
                  onnx_light_cpu::ExecutionBlockFn task) {
    auto &self = *static_cast<InlineExecutor *>(context);
    self.blocks += count;
    for (std::int64_t block = count; block > 0; --block) {
      task(task_context, block - 1);
    }
  }
};

TEST(OnnxLightSkipSimplifiedLayerNormalizationExecution, ParallelRowsMatchNaive) {
  constexpr std::int64_t rows = 257, width = 257;
  std::vector<float> values(rows * width);
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = (static_cast<int>(i % 31) - 15) / 13.F;
  }
  const auto input = Tensor::FromFloat("", {rows, width}, values);
  const auto gamma = Tensor::FromFloat("", {width}, std::vector<float>(width, 1.F));
  const auto expected = onnx_light_cpu::NaiveSkipSimplifiedLayerNormalizationKernel(MakeCtx())(
      input, input, gamma, nullptr, 1e-5F, true, true, true);
  InlineExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run};
  onnx_light_cpu::ExecutionExecutorScope scope(&view);
  const auto actual = onnx_light_cpu::SkipSimplifiedLayerNormalizationKernel(MakeCtx())(
      input, input, gamma, nullptr, 1e-5F, true, true, true);
  EXPECT_GT(executor.blocks, 1);
  EXPECT_EQ(Bytes(actual.input_skip_bias_sum), Bytes(expected.input_skip_bias_sum));
  for (std::size_t i = 0; i < values.size(); ++i) {
    EXPECT_NEAR(actual.output.AsFloat()[i], expected.output.AsFloat()[i], 2e-6F);
  }
  for (std::int64_t row = 0; row < rows; ++row) {
    EXPECT_EQ(actual.mean.AsFloat()[row], 0.F);
    EXPECT_NEAR(actual.inv_std_var.AsFloat()[row], expected.inv_std_var.AsFloat()[row], 2e-6F);
  }
}
} // namespace
