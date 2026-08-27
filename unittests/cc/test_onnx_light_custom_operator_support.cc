// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/gradient/com_microsoft/gradients.h"
#include "onnx_light_cpu/patterns/com_microsoft/patterns.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"
#include "onnx_light_cpu/shapes/com_microsoft/shape_inference.h"

#include "onnx_core/builder/pattern_registry.h"
#include "onnx_core/gradient/gradient.h"
#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

namespace {

namespace builder_ns = ONNX_LIGHT_NAMESPACE::core::builder;
namespace grad_ns = ONNX_LIGHT_NAMESPACE::core::gradient;
namespace shapes_ns = ONNX_LIGHT_NAMESPACE::core::shapes;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

using ONNX_LIGHT_NAMESPACE::NodeProto;
using sym_ns::SymDim;
using sym_ns::SymTensor;
using sym_ns::TensorType;

NodeProto MakeNode(const char *op_type, const char *left, const char *right, const char *output) {
  NodeProto node;
  node.set_domain(onnx_light_cpu::kMicrosoftDomain);
  node.set_op_type(op_type);
  node.add_input(left);
  node.add_input(right);
  node.add_output(output);
  return node;
}

TEST(CustomOperatorSupport, ProvidesLightSchemas) {
  const auto schemas = onnx_light_cpu::GetMicrosoftOpSchemasWithHistory();
  ASSERT_EQ(schemas.size(), 2U);
  EXPECT_EQ(schemas[0].name(), "BiasGelu");
  EXPECT_EQ(schemas[1].name(), "CDist");
  for (const auto &schema : schemas) {
    EXPECT_EQ(schema.domain(), onnx_light_cpu::kMicrosoftDomain);
    EXPECT_EQ(schema.since_version(), 1);
    EXPECT_EQ(schema.node_determinism(),
              ONNX_LIGHT_NAMESPACE::core::schema::LightOpSchema::NodeDeterminism::Unknown);
  }
  const auto without_docs = onnx_light_cpu::GetMicrosoftOpSchemasWithHistory("CDist", false);
  ASSERT_EQ(without_docs.size(), 1U);
  EXPECT_TRUE(without_docs[0].doc().empty());
}

TEST(CustomOperatorSupport, ProvidesReadOnlyInventory) {
  const auto support = onnx_light_cpu::CollectOperatorSupport();
  ASSERT_EQ(support.size(), 2U);
  EXPECT_EQ(support[0].op_type, "BiasGelu");
  EXPECT_EQ(support[0].shape_inference_function, "onnx_light_cpu::ComputeShapeBiasGelu");
  EXPECT_EQ(support[0].peak_memory_function, "onnx_light_cpu::ComputePeakMemoryBiasGelu");
  EXPECT_EQ(support[0].fusion_patterns,
            std::vector<std::string>{"onnx_light_cpu::BiasGeluFusionPattern"});
  EXPECT_TRUE(support[0].has_gradient);
  EXPECT_EQ(support[1].op_type, "CDist");
}

TEST(CustomOperatorSupport, InfersCDistShapeAndConstraint) {
  shapes_ns::ShapesContext ctx;
  ctx.Set("A", SymTensor(nullptr, TensorType::kFloat, {SymDim("M"), SymDim("N")}));
  ctx.Set("B", SymTensor(nullptr, TensorType::kFloat, {SymDim("K"), SymDim("FEATURES")}));
  const NodeProto node = MakeNode("CDist", "A", "B", "C");
  onnx_light_cpu::ComputeShapeCDist(ctx, node);
  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Shape().Rank(), 2U);
  EXPECT_EQ(ctx.Get("C").Shape()[0], SymDim("M"));
  EXPECT_EQ(ctx.Get("C").Shape()[1], SymDim("K"));
  EXPECT_EQ(ctx.Constraints().size(), 1U);
}

TEST(CustomOperatorSupport, InfersBiasGeluShapeAndRejectsWrongBias) {
  shapes_ns::ShapesContext ctx;
  ctx.Set("A", SymTensor(nullptr, TensorType::kFloat, {SymDim(2), SymDim(3)}));
  ctx.Set("B", SymTensor(nullptr, TensorType::kFloat, {SymDim(3)}));
  const NodeProto node = MakeNode("BiasGelu", "A", "B", "C");
  onnx_light_cpu::ComputeShapeBiasGelu(ctx, node);
  EXPECT_EQ(ctx.Get("C").Shape(), ctx.Get("A").Shape());

  shapes_ns::ShapesContext invalid;
  invalid.Set("A", SymTensor(nullptr, TensorType::kFloat, {SymDim(2), SymDim(3)}));
  invalid.Set("B", SymTensor(nullptr, TensorType::kFloat, {SymDim(4)}));
  EXPECT_THROW(onnx_light_cpu::ComputeShapeBiasGelu(invalid, node), std::invalid_argument);
}

TEST(CustomOperatorSupport, RegistersShapeMemoryGradientAndPatternHooks) {
  onnx_light_cpu::RegisterMicrosoftShapeAndMemoryFunctions();
  EXPECT_NE(shapes_ns::DispatchTable().find("com.microsoft:CDist"),
            shapes_ns::DispatchTable().end());
  EXPECT_EQ(shapes_ns::ComputePeakMemory(onnx_light_cpu::kMicrosoftDomain, "BiasGelu",
                                         sym_ns::Device::kCPU, {}),
            0);

  grad_ns::GradRegistry gradients;
  onnx_light_cpu::RegisterCustomOperatorGradients(gradients);
  EXPECT_NE(gradients.find({onnx_light_cpu::kMicrosoftDomain, "CDist"}), gradients.end());
  EXPECT_NE(gradients.find({onnx_light_cpu::kMicrosoftDomain, "BiasGelu"}), gradients.end());

  onnx_light_cpu::RegisterCustomOperatorPatterns();
  const std::vector<std::string> patterns = builder_ns::RegisteredPatternNames();
  EXPECT_NE(std::find(patterns.begin(), patterns.end(), "MicrosoftCDist"), patterns.end());
  EXPECT_NE(std::find(patterns.begin(), patterns.end(), "MicrosoftBiasGelu"), patterns.end());
}

TEST(CustomOperatorSupport, BiasGeluGradientUsesOnlyStandardOnnxOperators) {
  grad_ns::GradRegistry gradients = grad_ns::DefaultGradRegistry();
  onnx_light_cpu::RegisterCustomOperatorGradients(gradients);
  const std::vector<NodeProto> nodes = {MakeNode("BiasGelu", "A", "B", "C")};
  const std::vector<std::string> inputs = {"A", "B"};
  const std::vector<std::string> xs = {"A", "B"};
  const std::vector<std::string> zs;
  const std::vector<ONNX_LIGHT_NAMESPACE::TensorProto> initializers;
  const auto gradient =
      grad_ns::GradientOfNodes(nodes, inputs, initializers, xs, "C", zs, gradients);

  bool has_range = false;
  bool has_safe_reduce = false;
  for (const auto &node : gradient.node()) {
    EXPECT_TRUE(node.domain().empty() || node.domain() == "ai.onnx");
    EXPECT_NE(node.op_type(), "BroadcastGradientArgs");
    has_range = has_range || node.op_type() == "Range";
    if (node.op_type() == "ReduceSum") {
      const auto *noop = ONNX_LIGHT_NAMESPACE::FindAttribute(node, "noop_with_empty_axes");
      has_safe_reduce = has_safe_reduce || (noop != nullptr && noop->i() == 1);
    }
  }
  EXPECT_TRUE(has_range);
  EXPECT_TRUE(has_safe_reduce);
}

} // namespace
