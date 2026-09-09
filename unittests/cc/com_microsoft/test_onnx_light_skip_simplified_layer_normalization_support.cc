// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"
#include "onnx_light_cpu/shapes/com_microsoft/shape_inference.h"

#include "onnx_core/builder/graph_builder.h"
#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {

namespace builder_ns = ONNX_LIGHT_NAMESPACE::core::builder;
namespace schema_ns = ONNX_LIGHT_NAMESPACE::core::schema;
namespace shapes_ns = ONNX_LIGHT_NAMESPACE::core::shapes;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;
using ONNX_LIGHT_NAMESPACE::NodeProto;
using sym_ns::SymDim;
using sym_ns::SymShape;
using sym_ns::SymTensor;
using sym_ns::TensorType;

constexpr const char *kOp = "SkipSimplifiedLayerNormalization";

NodeProto MakeNode(bool bias = false, bool sum = false, bool mean = false, bool inverse = false) {
  NodeProto node;
  node.set_domain(onnx_light_cpu::kMicrosoftDomain);
  node.set_op_type(kOp);
  for (const char *input : {"input", "skip", "gamma"}) {
    node.add_input(input);
  }
  if (bias) {
    node.add_input("bias");
  }
  node.add_output("output");
  if (mean || inverse || sum) {
    node.add_output(mean ? "mean" : "");
  }
  if (inverse || sum) {
    node.add_output(inverse ? "inv_std_var" : "");
  }
  if (sum) {
    node.add_output("input_skip_bias_sum");
  }
  return node;
}

shapes_ns::ShapesContext MakeContext(const SymShape &input, const SymShape &skip,
                                     const SymShape &gamma, TensorType type = TensorType::kFloat) {
  shapes_ns::ShapesContext ctx;
  ctx.Set("input", SymTensor(nullptr, type, input));
  ctx.Set("skip", SymTensor(nullptr, type, skip));
  ctx.Set("gamma", SymTensor(nullptr, type, gamma));
  return ctx;
}

TEST(SkipSimplifiedLayerNormalizationSupport, SchemaAndReadOnlyInventory) {
  const auto schemas = onnx_light_cpu::GetMicrosoftOpSchemasWithHistory(kOp);
  ASSERT_EQ(schemas.size(), 1U);
  const auto &schema = schemas[0];
  EXPECT_EQ(schema.domain(), onnx_light_cpu::kMicrosoftDomain);
  EXPECT_EQ(schema.since_version(), 1);
  EXPECT_FALSE(schema.has_function_implementation());
  EXPECT_EQ(schema.min_output(), 1);
  EXPECT_EQ(schema.max_output(), 4);
  ASSERT_EQ(schema.inputs().size(), 4U);
  EXPECT_EQ(schema.inputs()[3].name, "bias");
  ASSERT_EQ(schema.outputs().size(), 4U);
  EXPECT_EQ(schema.outputs()[3].name, "input_skip_bias_sum");
  ASSERT_EQ(schema.attributes().size(), 1U);
  EXPECT_EQ(schema.attributes()[0].name, "epsilon");
  EXPECT_FLOAT_EQ(static_cast<float>(std::get<double>(schema.attributes()[0].default_value)),
                  1.0e-12F);
  EXPECT_NO_THROW(schema.Verify(MakeNode()));
  EXPECT_NO_THROW(schema.Verify(MakeNode(true, true)));
  EXPECT_TRUE(onnx_light_cpu::GetMicrosoftOpSchemasWithHistory(kOp, false)[0].doc().empty());

  const auto records = onnx_light_cpu::CollectOperatorSupport();
  const auto record = std::find_if(records.begin(), records.end(),
                                   [](const auto &entry) { return entry.op_type == kOp; });
  ASSERT_NE(record, records.end());
  EXPECT_EQ(record->domain, onnx_light_cpu::kMicrosoftDomain);
  EXPECT_EQ(record->shape_inference_function,
            "onnx_light_cpu::ComputeShapeSkipSimplifiedLayerNormalization");
  EXPECT_EQ(record->peak_memory_function,
            "onnx_light_cpu::ComputePeakMemorySkipSimplifiedLayerNormalization");
  EXPECT_FALSE(record->has_gradient);
  EXPECT_TRUE(record->fusion_patterns.empty());
}

TEST(SkipSimplifiedLayerNormalizationSupport, RegistersGlobalShapeAndZeroScratch) {
  onnx_light_cpu::RegisterMicrosoftShapeAndMemoryFunctions();
  onnx_light_cpu::RegisterMicrosoftShapeAndMemoryFunctions();
  auto ctx = MakeContext({SymDim(2), SymDim(3)}, {SymDim(2), SymDim(3)}, {SymDim(3)});
  ctx.ComputeShapeNode(MakeNode(false, true));
  EXPECT_EQ(ctx.Get("output").Shape(), ctx.Get("input").Shape());
  EXPECT_EQ(ctx.Get("input_skip_bias_sum").Shape(), ctx.Get("input").Shape());
  EXPECT_FALSE(ctx.Has(""));
  EXPECT_EQ(shapes_ns::ComputePeakMemory(onnx_light_cpu::kMicrosoftDomain, kOp,
                                         sym_ns::Device::kCPU, {ctx.Get("input").Shape()}),
            0);
  EXPECT_EQ(onnx_light_cpu::ComputePeakMemorySkipSimplifiedLayerNormalization(
                sym_ns::Device::kCPU, {{SymDim("B"), SymDim("S"), SymDim("H")}}),
            0);
}

TEST(SkipSimplifiedLayerNormalizationSupport, SupportedTypesBroadcastsAndOptionalSlots) {
  const auto schema = onnx_light_cpu::GetMicrosoftOpSchemasWithHistory(kOp)[0];
  for (auto type : {TensorType::kFloat, TensorType::kFloat16, TensorType::kBfloat16}) {
    for (const auto &skip : std::vector<SymShape>{{SymDim(2), SymDim(3), SymDim(4)},
                                                  {SymDim(1), SymDim(3), SymDim(4)},
                                                  {SymDim(3), SymDim(4)}}) {
      for (bool bias : {false, true}) {
        auto ctx = MakeContext({SymDim(2), SymDim(3), SymDim(4)}, skip, {SymDim(4)}, type);
        ctx.Set("bias", SymTensor(nullptr, type, {SymDim(4)}));
        const auto node = MakeNode(bias, true);
        std::vector<std::optional<schema_ns::SchemaInputValue>> inputs = {
            ctx.Get("input"), ctx.Get("skip"), ctx.Get("gamma")};
        if (bias) {
          inputs.push_back(ctx.Get("bias"));
        }
        EXPECT_NO_THROW(schema.Verify(node, &inputs));
        onnx_light_cpu::ComputeShapeSkipSimplifiedLayerNormalization(ctx, node);
        EXPECT_EQ(ctx.Get("output").Shape(), ctx.Get("input").Shape());
        EXPECT_EQ(ctx.Get("output").Dtype(), type);
        EXPECT_EQ(ctx.Get("input_skip_bias_sum").Shape(), ctx.Get("input").Shape());
        EXPECT_EQ(ctx.Get("input_skip_bias_sum").Dtype(), type);
      }
    }
  }
  for (int outputs = 1; outputs <= 4; ++outputs) {
    auto ctx = MakeContext({SymDim(2), SymDim(4)}, {SymDim(2), SymDim(4)}, {SymDim(4)});
    auto node = MakeNode();
    node.add_input("");
    while (node.output_size() < outputs) {
      node.add_output("");
    }
    EXPECT_NO_THROW(onnx_light_cpu::ComputeShapeSkipSimplifiedLayerNormalization(ctx, node));
    EXPECT_FALSE(ctx.Has(""));
    EXPECT_FALSE(ctx.Has("input_skip_bias_sum"));
  }
}

TEST(SkipSimplifiedLayerNormalizationSupport, IndependentOptionalStatistics) {
  for (auto type : {TensorType::kFloat, TensorType::kFloat16, TensorType::kBfloat16}) {
    for (int flags = 0; flags < 8; ++flags) {
      const bool mean = flags & 1;
      const bool inverse = flags & 2;
      const bool sum = flags & 4;
      auto ctx = MakeContext({SymDim("B"), SymDim("S"), SymDim("H")}, {SymDim("S"), SymDim("H")},
                             {SymDim("H")}, type);
      onnx_light_cpu::ComputeShapeSkipSimplifiedLayerNormalization(
          ctx, MakeNode(false, sum, mean, inverse));
      EXPECT_EQ(ctx.Has("mean"), mean);
      EXPECT_EQ(ctx.Has("inv_std_var"), inverse);
      EXPECT_EQ(ctx.Has("input_skip_bias_sum"), sum);
      for (const char *name : {"mean", "inv_std_var"}) {
        if (ctx.Has(name)) {
          EXPECT_EQ(ctx.Get(name).Shape(), SymShape({SymDim("B"), SymDim("S"), SymDim(1)}));
          EXPECT_EQ(ctx.Get(name).Dtype(), TensorType::kFloat);
        }
      }
      EXPECT_FALSE(ctx.Has(""));
    }
  }
}

TEST(SkipSimplifiedLayerNormalizationSupport, SymbolicEqualityAndBatchBroadcastConstraints) {
  auto ctx = MakeContext({SymDim("B"), SymDim("S"), SymDim("H")},
                         {SymDim("BS"), SymDim("SS"), SymDim("HS")}, {SymDim("G")});
  ctx.Set("bias", SymTensor(nullptr, TensorType::kFloat, {SymDim("K")}));
  onnx_light_cpu::ComputeShapeSkipSimplifiedLayerNormalization(ctx, MakeNode(true, true));
  EXPECT_EQ(ctx.Get("output").Shape(), ctx.Get("input").Shape());
  EXPECT_EQ(ctx.Get("input_skip_bias_sum").Shape(), ctx.Get("input").Shape());
  EXPECT_EQ(ctx.Constraints().size(), 5U);
  EXPECT_TRUE(ctx.HasLessEqualConstraint("1", "H"));
  EXPECT_TRUE(ctx.HasLessEqualConstraint("H", std::to_string(std::numeric_limits<int>::max())));
  EXPECT_TRUE(std::any_of(ctx.Constraints().begin(), ctx.Constraints().end(), [](const auto &pair) {
    return pair.first == "(BS-1)*(BS-(B))" || pair.second == "(BS-1)*(BS-(B))";
  }));

  auto no_batch = MakeContext({SymDim("B"), SymDim("S"), SymDim("H")}, {SymDim("S"), SymDim("H")},
                              {SymDim("H")});
  onnx_light_cpu::ComputeShapeSkipSimplifiedLayerNormalization(no_batch, MakeNode());
  EXPECT_TRUE(no_batch.Constraints().empty());
  auto fixed_batch = MakeContext({SymDim("B"), SymDim(3), SymDim(4)},
                                 {SymDim(2), SymDim(3), SymDim(4)}, {SymDim(4)});
  onnx_light_cpu::ComputeShapeSkipSimplifiedLayerNormalization(fixed_batch, MakeNode());
  EXPECT_EQ(fixed_batch.Get("output").Shape()[0], SymDim("B"));
  EXPECT_EQ(fixed_batch.Constraints().size(), 1U);
}

TEST(SkipSimplifiedLayerNormalizationSupport, EmptyOuterDimensionsAndMissingDescriptors) {
  for (const auto &input : std::vector<SymShape>{{SymDim(int64_t{0}), SymDim(4)},
                                                 {SymDim(int64_t{0}), SymDim(3), SymDim(4)},
                                                 {SymDim(2), SymDim(int64_t{0}), SymDim(4)}}) {
    auto ctx = MakeContext(input, input, {SymDim(4)});
    onnx_light_cpu::ComputeShapeSkipSimplifiedLayerNormalization(ctx, MakeNode(false, true));
    EXPECT_EQ(ctx.Get("output").Shape(), input);
    EXPECT_EQ(ctx.Get("input_skip_bias_sum").Shape(), input);
  }
  shapes_ns::ShapesContext unknown;
  EXPECT_THROW(onnx_light_cpu::ComputeShapeSkipSimplifiedLayerNormalization(unknown, MakeNode()),
               std::invalid_argument);
  EXPECT_FALSE(unknown.Has("output"));
  for (int slot = 0; slot < 4; ++slot) {
    auto ctx = MakeContext({SymDim(2), SymDim(4)}, {SymDim(2), SymDim(4)}, {SymDim(4)});
    ctx.Set("bias", SymTensor(nullptr, TensorType::kFloat, {SymDim(4)}));
    auto node = MakeNode(true, true, true, true);
    node.ref_input()[slot] = "missing";
    EXPECT_THROW(onnx_light_cpu::ComputeShapeSkipSimplifiedLayerNormalization(ctx, node),
                 std::invalid_argument);
    for (const char *output : {"output", "mean", "inv_std_var", "input_skip_bias_sum"}) {
      EXPECT_FALSE(ctx.Has(output));
    }
  }
}

TEST(SkipSimplifiedLayerNormalizationSupport, RejectsInvalidShapes) {
  for (const auto &shape : std::vector<SymShape>{{},
                                                 {SymDim(4)},
                                                 {SymDim(1), SymDim(2), SymDim(3), SymDim(4)},
                                                 {SymDim(2), SymDim(int64_t{0})},
                                                 {SymDim(-1), SymDim(4)},
                                                 {SymDim(1), SymDim(int64_t{2147483648})}}) {
    auto ctx = MakeContext(shape, shape, {SymDim(4)});
    EXPECT_THROW(onnx_light_cpu::ComputeShapeSkipSimplifiedLayerNormalization(ctx, MakeNode()),
                 std::invalid_argument);
  }
  for (const auto &skip : std::vector<SymShape>{{SymDim(4)},
                                                {SymDim(1), SymDim(4)},
                                                {SymDim(3), SymDim(3), SymDim(4)},
                                                {SymDim(2), SymDim(1), SymDim(4)},
                                                {SymDim(2), SymDim(3), SymDim(1)}}) {
    auto ctx = MakeContext({SymDim(2), SymDim(3), SymDim(4)}, skip, {SymDim(4)});
    EXPECT_THROW(onnx_light_cpu::ComputeShapeSkipSimplifiedLayerNormalization(ctx, MakeNode()),
                 std::invalid_argument);
  }
  auto rank2 = MakeContext({SymDim(2), SymDim(4)}, {SymDim(1), SymDim(4)}, {SymDim(4)});
  EXPECT_THROW(onnx_light_cpu::ComputeShapeSkipSimplifiedLayerNormalization(rank2, MakeNode()),
               std::invalid_argument);
  for (const char *input : {"gamma", "bias"}) {
    for (const auto &shape : std::vector<SymShape>{{}, {SymDim(1)}, {SymDim(1), SymDim(4)}}) {
      auto ctx = MakeContext({SymDim(2), SymDim(4)}, {SymDim(2), SymDim(4)}, {SymDim(4)});
      ctx.Set("bias", SymTensor(nullptr, TensorType::kFloat, {SymDim(4)}));
      ctx.Set(input, SymTensor(nullptr, TensorType::kFloat, shape));
      EXPECT_THROW(
          onnx_light_cpu::ComputeShapeSkipSimplifiedLayerNormalization(ctx, MakeNode(true)),
          std::invalid_argument);
    }
  }
}

TEST(SkipSimplifiedLayerNormalizationSupport, RejectsTypesEpsilonAndMalformedSlots) {
  const auto schema = onnx_light_cpu::GetMicrosoftOpSchemasWithHistory(kOp)[0];
  for (auto type : {TensorType::kInt32, TensorType::kDouble, TensorType::kInt64}) {
    auto ctx = MakeContext({SymDim(2), SymDim(4)}, {SymDim(2), SymDim(4)}, {SymDim(4)}, type);
    std::vector<std::optional<schema_ns::SchemaInputValue>> inputs = {
        ctx.Get("input"), ctx.Get("skip"), ctx.Get("gamma")};
    EXPECT_THROW(schema.Verify(MakeNode(), &inputs), schema_ns::SchemaError);
    EXPECT_THROW(onnx_light_cpu::ComputeShapeSkipSimplifiedLayerNormalization(ctx, MakeNode()),
                 std::invalid_argument);
  }
  for (const char *input : {"skip", "gamma", "bias"}) {
    auto ctx = MakeContext({SymDim(2), SymDim(4)}, {SymDim(2), SymDim(4)}, {SymDim(4)});
    ctx.Set("bias", SymTensor(nullptr, TensorType::kFloat, {SymDim(4)}));
    ctx.Set(input, SymTensor(nullptr, TensorType::kFloat16, ctx.Get(input).Shape()));
    EXPECT_THROW(onnx_light_cpu::ComputeShapeSkipSimplifiedLayerNormalization(ctx, MakeNode(true)),
                 std::invalid_argument);
  }
  auto ctx = MakeContext({SymDim(2), SymDim(4)}, {SymDim(2), SymDim(4)}, {SymDim(4)});
  for (float epsilon :
       {-1.0F, std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN()}) {
    auto node = MakeNode();
    ONNX_LIGHT_NAMESPACE::AddAttribute(node, "epsilon", epsilon);
    EXPECT_THROW(onnx_light_cpu::ComputeShapeSkipSimplifiedLayerNormalization(ctx, node),
                 std::invalid_argument);
  }
  auto zero_epsilon = MakeNode();
  ONNX_LIGHT_NAMESPACE::AddAttribute(zero_epsilon, "epsilon", 0.0F);
  EXPECT_NO_THROW(onnx_light_cpu::ComputeShapeSkipSimplifiedLayerNormalization(ctx, zero_epsilon));
  for (int slot = 0; slot < 3; ++slot) {
    auto node = MakeNode(false, true);
    if (slot == 0) {
      node.ref_output()[slot] = "";
      EXPECT_THROW(onnx_light_cpu::ComputeShapeSkipSimplifiedLayerNormalization(ctx, node),
                   std::invalid_argument);
    }
    node = MakeNode();
    node.ref_input()[slot] = "";
    EXPECT_THROW(onnx_light_cpu::ComputeShapeSkipSimplifiedLayerNormalization(ctx, node),
                 std::invalid_argument);
  }
  NodeProto missing_gamma;
  missing_gamma.set_op_type(kOp);
  missing_gamma.set_domain(onnx_light_cpu::kMicrosoftDomain);
  missing_gamma.add_input("input");
  missing_gamma.add_input("skip");
  missing_gamma.add_output("output");
  EXPECT_THROW(onnx_light_cpu::ComputeShapeSkipSimplifiedLayerNormalization(ctx, missing_gamma),
               std::invalid_argument);
  auto extra_input = MakeNode(true);
  extra_input.add_input("extra");
  EXPECT_THROW(onnx_light_cpu::ComputeShapeSkipSimplifiedLayerNormalization(ctx, extra_input),
               std::invalid_argument);
  auto extra_output = MakeNode(false, true);
  extra_output.add_output("extra");
  EXPECT_THROW(onnx_light_cpu::ComputeShapeSkipSimplifiedLayerNormalization(ctx, extra_output),
               std::invalid_argument);
}

TEST(SkipSimplifiedLayerNormalizationSupport, NativeGraphBuilderSchemaCallbackAndModelReload) {
  onnx_light_cpu::RegisterMicrosoftShapeAndMemoryFunctions();
  const auto lookup = [](const std::string &op_type) {
    return onnx_light_cpu::GetMicrosoftOpSchemasWithHistory(op_type, false);
  };
  for (auto type : {TensorType::kFloat, TensorType::kFloat16, TensorType::kBfloat16}) {
    builder_ns::GraphBuilder builder("skip-normalization", lookup);
    const SymShape shape{SymDim("B"), SymDim("S"), SymDim(4)};
    builder.MakeInput("input", type, shape);
    builder.MakeInput("skip", type, {SymDim("S"), SymDim(4)});
    builder.MakeInput("gamma", type, {SymDim(4)});
    builder.MakeInput("bias", type, {SymDim(4)});
    const auto outputs = builder.MakeNode(kOp, {"input", "skip", "gamma", "bias"}, {"output"},
                                          onnx_light_cpu::kMicrosoftDomain);
    ASSERT_EQ(outputs.size(), 1U);
    EXPECT_EQ(builder.OpsetVersion(onnx_light_cpu::kMicrosoftDomain), 1);
    EXPECT_EQ(builder.GetShape("output").Shape(), shape);
    EXPECT_EQ(builder.GetShape("output").Dtype(), type);
    builder.MakeOutput("output");
    auto model = builder.ToModel();
    builder_ns::GraphBuilder loaded(model, lookup);
    EXPECT_EQ(loaded.GetShape("output").Shape(), shape);
    EXPECT_EQ(loaded.GetShape("output").Dtype(), type);
  }
}

TEST(SkipSimplifiedLayerNormalizationSupport, NativeGraphBuilderStatisticsAndResidual) {
  onnx_light_cpu::RegisterMicrosoftShapeAndMemoryFunctions();
  const auto lookup = [](const std::string &op_type) {
    return onnx_light_cpu::GetMicrosoftOpSchemasWithHistory(op_type, false);
  };
  for (const auto &outputs :
       std::vector<std::vector<std::string>>{{"output", "mean", "inv_std_var", "sum"},
                                             {"output", "", "inv_std_var"},
                                             {"output", "", "", "sum"}}) {
    builder_ns::GraphBuilder builder("skip-statistics", lookup);
    const SymShape shape{SymDim("B"), SymDim(3), SymDim(4)};
    builder.MakeInput("input", TensorType::kBfloat16, shape);
    builder.MakeInput("skip", TensorType::kBfloat16, shape);
    builder.MakeInput("gamma", TensorType::kBfloat16, {SymDim(4)});
    const auto actual = builder.MakeNode(kOp, {"input", "skip", "gamma"}, outputs,
                                         onnx_light_cpu::kMicrosoftDomain);
    ASSERT_EQ(actual.size(), outputs.size());
    EXPECT_EQ(builder.GetShape("output").Shape(), shape);
    for (size_t slot = 1; slot < actual.size(); ++slot) {
      if (!actual[slot].empty()) {
        EXPECT_EQ(builder.GetShape(actual[slot]).Dtype(),
                  slot == 3 ? TensorType::kBfloat16 : TensorType::kFloat);
        EXPECT_EQ(builder.GetShape(actual[slot]).Shape(),
                  slot == 3 ? shape : SymShape({SymDim("B"), SymDim(3), SymDim(1)}));
      }
      if (!outputs[slot].empty()) {
        builder.MakeOutput(outputs[slot]);
      }
    }
    builder.MakeOutput("output");
    const auto model = builder.ToModel();
    builder_ns::GraphBuilder loaded(model, lookup);
    for (size_t slot = 0; slot < outputs.size(); ++slot) {
      if (!outputs[slot].empty()) {
        EXPECT_EQ(loaded.GetShape(outputs[slot]).Shape(), builder.GetShape(outputs[slot]).Shape());
      }
    }
  }
}

} // namespace
