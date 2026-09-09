// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/schemas/ai_onnx/op_schema.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"
#include "onnx_light_cpu/shapes/ai_onnx/shape_inference.h"
#include "onnx_light_cpu/shapes/com_microsoft/shape_inference.h"

#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {

namespace shapes_ns = ONNX_LIGHT_NAMESPACE::core::shapes;
namespace schema_ns = ONNX_LIGHT_NAMESPACE::core::schema;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;
using ONNX_LIGHT_NAMESPACE::NodeProto;
using sym_ns::SymDim;
using sym_ns::SymShape;
using sym_ns::SymTensor;
using sym_ns::TensorType;

NodeProto MakeNode(int64_t axis = -1, int64_t stash_type = 1, bool statistics = true) {
  NodeProto node;
  node.set_op_type("SimplifiedLayerNormalization");
  node.add_input("X");
  node.add_input("Scale");
  node.add_output("Y");
  if (statistics) {
    node.add_output("inv_std_var");
  }
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "axis", axis);
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "stash_type", stash_type);
  return node;
}

shapes_ns::ShapesContext MakeContext(const SymShape &x, const SymShape &scale,
                                     TensorType x_type = TensorType::kFloat,
                                     TensorType scale_type = TensorType::kFloat) {
  shapes_ns::ShapesContext ctx;
  ctx.Set("X", SymTensor(nullptr, x_type, x));
  ctx.Set("Scale", SymTensor(nullptr, scale_type, scale));
  return ctx;
}

TEST(SimplifiedLayerNormalizationSupport, ExperimentalSchemaContract) {
  const auto schemas = onnx_light_cpu::GetExperimentalOpSchemasWithHistory();
  ASSERT_EQ(schemas.size(), 1U);
  const auto &schema = schemas[0];
  EXPECT_EQ(schema.name(), "SimplifiedLayerNormalization");
  EXPECT_EQ(schema.domain(), "ai.onnx");
  EXPECT_EQ(schema.since_version(), 1);
  EXPECT_NE(schema.doc().find("Experimental"), std::string::npos);
  EXPECT_FALSE(schema.has_function_implementation());
  EXPECT_EQ(schema.min_output(), 1);
  EXPECT_EQ(schema.max_output(), 2);
  EXPECT_EQ(schema.inputs()[0].type, "T");
  EXPECT_EQ(schema.inputs()[1].type, "V");
  EXPECT_EQ(schema.outputs()[0].type, "V");
  EXPECT_EQ(schema.outputs()[1].type, "U");
  EXPECT_EQ(schema.outputs()[1].name, "inv_std_var");
  EXPECT_EQ(schema.attributes().size(), 3U);
  EXPECT_EQ(std::get<int64_t>(schema.attributes()[0].default_value), -1);
  EXPECT_FLOAT_EQ(static_cast<float>(std::get<double>(schema.attributes()[1].default_value)),
                  1.0e-5f);
  EXPECT_EQ(std::get<int64_t>(schema.attributes()[2].default_value), 1);
  EXPECT_TRUE(onnx_light_cpu::GetExperimentalOpSchemasWithHistory("Missing").empty());
  EXPECT_TRUE(
      onnx_light_cpu::GetExperimentalOpSchemasWithHistory(schema.name(), false)[0].doc().empty());
  EXPECT_TRUE(onnx_light_cpu::GetMicrosoftOpSchemasWithHistory(schema.name()).empty());
  EXPECT_NO_THROW(schema.Verify(MakeNode(-1, 1, false)));
  auto empty_statistics = MakeNode();
  empty_statistics.ref_output()[1] = "";
  EXPECT_NO_THROW(schema.Verify(empty_statistics));
}

TEST(SimplifiedLayerNormalizationSupport, AllIndependentInputTypesAndStatisticsTypes) {
  const auto schema = onnx_light_cpu::GetExperimentalOpSchemasWithHistory()[0];
  const std::vector<TensorType> types = {TensorType::kFloat, TensorType::kFloat16,
                                         TensorType::kDouble, TensorType::kBfloat16};
  for (auto x_type : types) {
    for (auto scale_type : types) {
      for (int64_t stash_type : {1, 11}) {
        auto ctx = MakeContext({SymDim(2), SymDim(3), SymDim(4)}, {SymDim(2), SymDim(1), SymDim(4)},
                               x_type, scale_type);
        const auto node = MakeNode(1, stash_type);
        std::vector<std::optional<schema_ns::SchemaInputValue>> inputs = {ctx.Get("X"),
                                                                          ctx.Get("Scale")};
        EXPECT_NO_THROW(schema.Verify(node, &inputs));
        onnx_light_cpu::ComputeShapeSimplifiedLayerNormalization(ctx, node);
        EXPECT_EQ(ctx.Get("Y").Dtype(), scale_type);
        EXPECT_EQ(ctx.Get("Y").Shape(), ctx.Get("X").Shape());
        EXPECT_EQ(ctx.Get("inv_std_var").Shape(), SymShape({SymDim(2), SymDim(1), SymDim(1)}));
        EXPECT_EQ(ctx.Get("inv_std_var").Dtype(),
                  stash_type == 1 ? TensorType::kFloat : TensorType::kDouble);
      }
    }
  }
  auto invalid = MakeContext({SymDim(2)}, {}, TensorType::kInt32);
  std::vector<std::optional<schema_ns::SchemaInputValue>> inputs = {invalid.Get("X"),
                                                                    invalid.Get("Scale")};
  EXPECT_THROW(schema.Verify(MakeNode(), &inputs), schema_ns::SchemaError);
}

TEST(SimplifiedLayerNormalizationSupport, SymbolicShapesAndUnidirectionalConstraints) {
  auto ctx = MakeContext({SymDim("B"), SymDim("S"), SymDim("H")}, {SymDim("K")});
  onnx_light_cpu::ComputeShapeSimplifiedLayerNormalization(ctx, MakeNode(-2));
  EXPECT_EQ(ctx.Get("Y").Shape(), ctx.Get("X").Shape());
  EXPECT_EQ(ctx.Get("inv_std_var").Shape(), SymShape({SymDim("B"), SymDim(1), SymDim(1)}));
  EXPECT_EQ(ctx.Constraints().size(), 1U);
  const auto &constraint = *ctx.Constraints().begin();
  EXPECT_TRUE(constraint.first == "(K-1)*(K-(H))" || constraint.second == "(K-1)*(K-(H))");

  auto concrete_scale = MakeContext({SymDim("B"), SymDim("H")}, {SymDim(7)});
  onnx_light_cpu::ComputeShapeSimplifiedLayerNormalization(concrete_scale, MakeNode());
  EXPECT_EQ(concrete_scale.Get("Y").Shape()[1], SymDim("H"));
  EXPECT_EQ(concrete_scale.Constraints().size(), 1U);
}

TEST(SimplifiedLayerNormalizationSupport, OptionalStatisticsAndEmptyOuterRows) {
  for (bool empty_name : {false, true}) {
    auto ctx = MakeContext({SymDim(int64_t{0}), SymDim(3), SymDim(4)}, {});
    auto node = MakeNode(-2, 11, empty_name);
    if (empty_name) {
      node.ref_output()[1] = "";
    }
    EXPECT_NO_THROW(onnx_light_cpu::ComputeShapeSimplifiedLayerNormalization(ctx, node));
    EXPECT_EQ(ctx.Get("Y").Shape(), ctx.Get("X").Shape());
    EXPECT_FALSE(ctx.Has(""));
    EXPECT_FALSE(ctx.Has("inv_std_var"));
  }
  auto ctx = MakeContext({SymDim(3), SymDim(4)}, {SymDim(1)});
  onnx_light_cpu::ComputeShapeSimplifiedLayerNormalization(ctx, MakeNode(0));
  EXPECT_EQ(ctx.Get("inv_std_var").Shape(), SymShape({SymDim(1), SymDim(1)}));
}

TEST(SimplifiedLayerNormalizationSupport, RejectsInvalidShapesTypesAndAttributes) {
  for (const auto &shapes :
       std::vector<std::pair<SymShape, SymShape>>{{{}, {}},
                                                  {{SymDim(2), SymDim(int64_t{0})}, {}},
                                                  {{SymDim(2), SymDim(3)}, {SymDim(4)}},
                                                  {{SymDim(2), SymDim(1)}, {SymDim(3)}},
                                                  {{SymDim(3)}, {SymDim(1), SymDim(3)}}}) {
    auto ctx = MakeContext(shapes.first, shapes.second);
    EXPECT_THROW(onnx_light_cpu::ComputeShapeSimplifiedLayerNormalization(ctx, MakeNode()),
                 std::invalid_argument);
  }
  for (int64_t axis : {-3, 2}) {
    auto ctx = MakeContext({SymDim(2), SymDim(3)}, {});
    EXPECT_THROW(onnx_light_cpu::ComputeShapeSimplifiedLayerNormalization(ctx, MakeNode(axis)),
                 std::invalid_argument);
  }
  for (int64_t stash_type : {0, 10, 16}) {
    auto ctx = MakeContext({SymDim(2)}, {});
    EXPECT_THROW(
        onnx_light_cpu::ComputeShapeSimplifiedLayerNormalization(ctx, MakeNode(-1, stash_type)),
        std::invalid_argument);
  }
  for (bool integer_x : {false, true}) {
    auto ctx = MakeContext({SymDim(2)}, {}, integer_x ? TensorType::kInt32 : TensorType::kFloat,
                           integer_x ? TensorType::kFloat : TensorType::kInt64);
    EXPECT_THROW(onnx_light_cpu::ComputeShapeSimplifiedLayerNormalization(ctx, MakeNode()),
                 std::invalid_argument);
  }
  auto ctx = MakeContext({SymDim(2)}, {});
  auto node = MakeNode();
  node.ref_output()[0] = "";
  EXPECT_THROW(onnx_light_cpu::ComputeShapeSimplifiedLayerNormalization(ctx, node),
               std::invalid_argument);
}

TEST(SimplifiedLayerNormalizationSupport, PermitsIeeeEpsilonAndUnknownInputs) {
  for (float epsilon :
       {-1.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
    auto ctx = MakeContext({SymDim(2)}, {});
    auto node = MakeNode();
    ONNX_LIGHT_NAMESPACE::AddAttribute(node, "epsilon", epsilon);
    EXPECT_NO_THROW(onnx_light_cpu::ComputeShapeSimplifiedLayerNormalization(ctx, node));
  }
  shapes_ns::ShapesContext unknown;
  EXPECT_NO_THROW(onnx_light_cpu::ComputeShapeSimplifiedLayerNormalization(unknown, MakeNode()));
  EXPECT_FALSE(unknown.Has("Y"));
  EXPECT_FALSE(unknown.Has("inv_std_var"));
}

TEST(SimplifiedLayerNormalizationSupport, RegistersAliasesAndZeroScratchWithoutTrainingSupport) {
  onnx_light_cpu::RegisterExperimentalShapeAndMemoryFunctions();
  onnx_light_cpu::RegisterExperimentalShapeAndMemoryFunctions();
  for (const char *domain : {"", "ai.onnx"}) {
    auto ctx = MakeContext({SymDim(2), SymDim(3)}, {});
    auto node = MakeNode();
    node.set_domain(domain);
    ctx.ComputeShapeNode(node);
    EXPECT_EQ(ctx.Get("Y").Shape(), ctx.Get("X").Shape());
    EXPECT_EQ(shapes_ns::ComputePeakMemory(domain, node.op_type(), sym_ns::Device::kCPU,
                                           {ctx.Get("X").Shape(), ctx.Get("Scale").Shape()}),
              0);
  }
  const auto records = onnx_light_cpu::CollectOperatorSupport();
  const auto record = std::find_if(records.begin(), records.end(), [](const auto &entry) {
    return entry.op_type == "SimplifiedLayerNormalization";
  });
  ASSERT_NE(record, records.end());
  EXPECT_EQ(record->domain, "ai.onnx");
  EXPECT_EQ(record->shape_inference_function,
            "onnx_light_cpu::ComputeShapeSimplifiedLayerNormalization");
  EXPECT_EQ(record->peak_memory_function,
            "onnx_light_cpu::ComputePeakMemorySimplifiedLayerNormalization");
  EXPECT_FALSE(record->has_gradient);
  EXPECT_TRUE(record->fusion_patterns.empty());
}

} // namespace
