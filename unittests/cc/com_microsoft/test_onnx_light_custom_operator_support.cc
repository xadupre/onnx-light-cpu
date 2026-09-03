// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/gradient/com_microsoft/gradients.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/kernels/register_kernels.h"
#include "onnx_light_cpu/patterns/com_microsoft/patterns.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"
#include "onnx_light_cpu/shapes/com_microsoft/shape_inference.h"

#include "onnx_core/builder/pattern_registry.h"
#include "onnx_core/gradient/gradient.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace builder_ns = ONNX_LIGHT_NAMESPACE::core::builder;
namespace grad_ns = ONNX_LIGHT_NAMESPACE::core::gradient;
namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace shapes_ns = ONNX_LIGHT_NAMESPACE::core::shapes;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

using ONNX_LIGHT_NAMESPACE::FunctionProto;
using ONNX_LIGHT_NAMESPACE::GraphProto;
using ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::Tensor;
using rt_ns::Tensors;
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

NodeProto MakeGroupQueryAttentionNode(int64_t num_heads = 4, int64_t kv_num_heads = 2,
                                      int64_t causal = 1, std::optional<float> scale = 0.5f) {
  NodeProto node;
  node.set_domain(onnx_light_cpu::kMicrosoftDomain);
  node.set_op_type("GroupQueryAttention");
  node.add_input("Q");
  node.add_input("K");
  node.add_input("V");
  node.add_input("");
  node.add_input("");
  node.add_input("seqlens_k");
  node.add_input("total_sequence_length");
  node.add_output("Y");
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "num_heads", num_heads);
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "kv_num_heads", kv_num_heads);
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "causal", causal);
  if (scale.has_value()) {
    ONNX_LIGHT_NAMESPACE::AddAttribute(node, "scale", *scale);
  }
  return node;
}

FunctionProto MakeGroupQueryAttentionGradient(const NodeProto &node) {
  grad_ns::GradRegistry gradients = grad_ns::DefaultGradRegistry();
  onnx_light_cpu::RegisterCustomOperatorGradients(gradients);
  const std::vector<NodeProto> nodes = {node};
  const std::vector<std::string> inputs = {"Q", "K", "V", "seqlens_k", "total_sequence_length"};
  const std::vector<std::string> xs = {"Q", "K", "V"};
  const std::vector<std::string> zs;
  const std::vector<ONNX_LIGHT_NAMESPACE::TensorProto> initializers;
  return grad_ns::GradientOfNodes(nodes, inputs, initializers, xs, "Y", zs, gradients);
}

GraphProto MakeGraph(const FunctionProto &function) {
  GraphProto graph;
  graph.set_name("group_query_attention_gradient");
  for (const std::string &input : function.input()) {
    graph.add_input()->set_name(input);
  }
  for (const NodeProto &node : function.node()) {
    graph.ref_node().push_back(node);
  }
  for (const std::string &output : function.output()) {
    graph.add_output()->set_name(output);
  }
  return graph;
}

void RegisterExecutionKernels() {
  static const bool registered = [] {
    ONNX_LIGHT_NAMESPACE::onnx_kernels::RegisterKernelFunctions();
    onnx_light_cpu::RegisterAllKernels();
    return true;
  }();
  (void)registered;
}

Tensors RunGradient(const NodeProto &node, Tensor query, Tensor key, Tensor value,
                    Tensor output_grad) {
  RegisterExecutionKernels();
  const FunctionProto gradient = MakeGroupQueryAttentionGradient(node);
  const GraphProto graph = MakeGraph(gradient);
  rt_ns::RuntimeContext rt(rt_ns::KernelContext(rt_ns::DefaultOpset(23)));
  rt_ns::SubgraphSession session(rt, graph);
  return session.Run({{"Q", std::move(query)},
                      {"K", std::move(key)},
                      {"V", std::move(value)},
                      {"dy", std::move(output_grad)}},
                     rt);
}

float RunForwardObjective(const NodeProto &node, const std::vector<float> &query,
                          const std::vector<float> &key, const std::vector<float> &value,
                          const std::vector<float> &output_grad) {
  RegisterExecutionKernels();
  GraphProto graph;
  graph.set_name("group_query_attention_forward");
  for (const char *input : {"Q", "K", "V", "seqlens_k", "total_sequence_length"}) {
    graph.add_input()->set_name(input);
  }
  graph.ref_node().push_back(node);
  graph.add_output()->set_name("Y");

  rt_ns::RuntimeContext rt(rt_ns::KernelContext(rt_ns::DefaultOpset(23)));
  rt_ns::SubgraphSession session(rt, graph);
  Tensors outputs = session.Run(
      {{"Q", Tensor::FromFloat("Q", {1, 2, 4}, query)},
       {"K", Tensor::FromFloat("K", {1, 2, 2}, key)},
       {"V", Tensor::FromFloat("V", {1, 2, 2}, value)},
       {"seqlens_k", rt_ns::Tensor::FromInt32("seqlens_k", {1}, {1})},
       {"total_sequence_length", rt_ns::Tensor::FromInt32("total_sequence_length", {}, {2})}},
      rt);
  const float *output = outputs[0].AsFloat();
  float objective = 0.0f;
  for (size_t i = 0; i < output_grad.size(); ++i) {
    objective += output[i] * output_grad[i];
  }
  return objective;
}

TEST(CustomOperatorSupport, ProvidesLightSchemas) {
  const auto schemas = onnx_light_cpu::GetMicrosoftOpSchemasWithHistory();
  ASSERT_EQ(schemas.size(), 4U);
  EXPECT_EQ(schemas[0].name(), "BiasGelu");
  EXPECT_EQ(schemas[1].name(), "CDist");
  EXPECT_EQ(schemas[2].name(), "GroupQueryAttention");
  EXPECT_EQ(schemas[3].name(), "LinearAttention");
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
  ASSERT_EQ(support.size(), 4U);
  EXPECT_EQ(support[0].op_type, "BiasGelu");
  EXPECT_EQ(support[0].shape_inference_function, "onnx_light_cpu::ComputeShapeBiasGelu");
  EXPECT_EQ(support[0].peak_memory_function, "onnx_light_cpu::ComputePeakMemoryBiasGelu");
  EXPECT_EQ(support[0].fusion_patterns,
            std::vector<std::string>{"onnx_light_cpu::BiasGeluFusionPattern"});
  EXPECT_TRUE(support[0].has_gradient);
  EXPECT_EQ(support[1].op_type, "CDist");
  EXPECT_EQ(support[2].op_type, "GroupQueryAttention");
  EXPECT_EQ(support[2].shape_inference_function, "onnx_light_cpu::ComputeShapeGroupQueryAttention");
  EXPECT_EQ(support[2].peak_memory_function,
            "onnx_light_cpu::ComputePeakMemoryGroupQueryAttention");
  EXPECT_EQ(support[2].fusion_patterns,
            std::vector<std::string>{"onnx_light_cpu::GroupQueryAttentionFusionPattern"});
  EXPECT_TRUE(support[2].has_gradient);
  EXPECT_EQ(support[3].op_type, "LinearAttention");
  EXPECT_EQ(support[3].shape_inference_function, "onnx_light_cpu::ComputeShapeLinearAttention");
  EXPECT_EQ(support[3].peak_memory_function, "onnx_light_cpu::ComputePeakMemoryLinearAttention");
  EXPECT_EQ(support[3].fusion_patterns,
            std::vector<std::string>{"onnx_light_cpu::LinearAttentionFusionPattern"});
  EXPECT_FALSE(support[3].has_gradient);
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

TEST(CustomOperatorSupport, InfersGroupQueryAttentionShape) {
  shapes_ns::ShapesContext ctx;
  ctx.Set("Q", SymTensor(nullptr, TensorType::kFloat, {SymDim("B"), SymDim("S"), SymDim(32)}));
  ctx.Set("K", SymTensor(nullptr, TensorType::kFloat, {SymDim("B"), SymDim("S"), SymDim(16)}));
  ctx.Set("V", SymTensor(nullptr, TensorType::kFloat, {SymDim("B"), SymDim("S"), SymDim(16)}));
  onnx_light_cpu::ComputeShapeGroupQueryAttention(ctx, MakeGroupQueryAttentionNode());
  EXPECT_EQ(ctx.Get("Y").Shape(), ctx.Get("Q").Shape());
}

TEST(CustomOperatorSupport, InfersGroupQueryAttentionSymbolicPresentCacheShape) {
  shapes_ns::ShapesContext ctx;
  ctx.Set("Q", SymTensor(nullptr, TensorType::kFloat, {SymDim("B"), SymDim("S"), SymDim(32)}));
  ctx.Set("K", SymTensor(nullptr, TensorType::kFloat, {SymDim("BK"), SymDim("SK"), SymDim(16)}));
  ctx.Set("V", SymTensor(nullptr, TensorType::kFloat, {SymDim("BV"), SymDim("SV"), SymDim(16)}));
  ctx.Set("past_key", SymTensor(nullptr, TensorType::kFloat,
                                {SymDim("BP"), SymDim(2), SymDim("P"), SymDim(8)}));
  ctx.Set("past_value", SymTensor(nullptr, TensorType::kFloat,
                                  {SymDim("BPV"), SymDim(2), SymDim("PV"), SymDim(8)}));

  NodeProto node = MakeGroupQueryAttentionNode();
  node.ref_input()[3] = "past_key";
  node.ref_input()[4] = "past_value";
  node.add_output("present_key");
  node.add_output("present_value");
  onnx_light_cpu::ComputeShapeGroupQueryAttention(ctx, node);

  EXPECT_EQ(ctx.Get("Y").Shape(), sym_ns::SymShape({SymDim("B"), SymDim("S"), SymDim(32)}));
  EXPECT_EQ(ctx.Get("present_key").Shape(),
            sym_ns::SymShape({SymDim("B"), SymDim(2), SymDim("(SK)+(P)"), SymDim(8)}));
  EXPECT_EQ(ctx.Get("present_value").Shape(), ctx.Get("present_key").Shape());
  EXPECT_GE(ctx.Constraints().size(), 7U);
}

TEST(CustomOperatorSupport, InfersLinearAttentionInverseGroupingAndStateShape) {
  shapes_ns::ShapesContext ctx;
  ctx.Set("Q", SymTensor(nullptr, TensorType::kFloat, {SymDim("B"), SymDim("S"), SymDim(8)}));
  ctx.Set("K", SymTensor(nullptr, TensorType::kFloat, {SymDim("BK"), SymDim("SK"), SymDim(8)}));
  ctx.Set("V", SymTensor(nullptr, TensorType::kFloat, {SymDim("BV"), SymDim("SV"), SymDim(12)}));
  NodeProto node;
  node.set_domain(onnx_light_cpu::kMicrosoftDomain);
  node.set_op_type("LinearAttention");
  node.add_input("Q");
  node.add_input("K");
  node.add_input("V");
  node.add_output("Y");
  node.add_output("present_state");
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "q_num_heads", int64_t{1});
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "kv_num_heads", int64_t{2});
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "update_rule", std::string("linear"));
  onnx_light_cpu::ComputeShapeLinearAttention(ctx, node);
  EXPECT_EQ(ctx.Get("Y").Shape(), sym_ns::SymShape({SymDim("B"), SymDim("S"), SymDim(12)}));
  EXPECT_EQ(ctx.Get("present_state").Shape(),
            sym_ns::SymShape({SymDim("B"), SymDim(2), SymDim(8), SymDim(6)}));
  EXPECT_GE(ctx.Constraints().size(), 4U);
}

TEST(CustomOperatorSupport, RegistersShapeMemoryGradientAndPatternHooks) {
  onnx_light_cpu::RegisterMicrosoftShapeAndMemoryFunctions();
  EXPECT_NE(shapes_ns::DispatchTable().find("com.microsoft:CDist"),
            shapes_ns::DispatchTable().end());
  EXPECT_EQ(shapes_ns::ComputePeakMemory(onnx_light_cpu::kMicrosoftDomain, "BiasGelu",
                                         sym_ns::Device::kCPU, {}),
            0);
  EXPECT_EQ(shapes_ns::ComputePeakMemory(onnx_light_cpu::kMicrosoftDomain, "GroupQueryAttention",
                                         sym_ns::Device::kCPU, {}),
            0);
  EXPECT_EQ(shapes_ns::ComputePeakMemory(onnx_light_cpu::kMicrosoftDomain, "LinearAttention",
                                         sym_ns::Device::kCPU, {}),
            0);

  grad_ns::GradRegistry gradients;
  onnx_light_cpu::RegisterCustomOperatorGradients(gradients);
  EXPECT_NE(gradients.find({onnx_light_cpu::kMicrosoftDomain, "CDist"}), gradients.end());
  EXPECT_NE(gradients.find({onnx_light_cpu::kMicrosoftDomain, "BiasGelu"}), gradients.end());
  EXPECT_NE(gradients.find({onnx_light_cpu::kMicrosoftDomain, "GroupQueryAttention"}),
            gradients.end());
  EXPECT_EQ(gradients.find({onnx_light_cpu::kMicrosoftDomain, "LinearAttention"}), gradients.end());

  onnx_light_cpu::RegisterCustomOperatorPatterns();
  const std::vector<std::string> patterns = builder_ns::RegisteredPatternNames();
  EXPECT_NE(std::find(patterns.begin(), patterns.end(), "MicrosoftCDist"), patterns.end());
  EXPECT_NE(std::find(patterns.begin(), patterns.end(), "MicrosoftBiasGelu"), patterns.end());
  EXPECT_NE(std::find(patterns.begin(), patterns.end(), "MicrosoftGroupQueryAttention"),
            patterns.end());
  EXPECT_NE(std::find(patterns.begin(), patterns.end(), "MicrosoftLinearAttention"),
            patterns.end());
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

TEST(CustomOperatorSupport, GroupQueryAttentionGradientUsesStandardAttention) {
  const std::vector<std::optional<float>> scales = {0.5f, std::nullopt};
  for (const std::optional<float> scale : scales) {
    const FunctionProto gradient =
        MakeGroupQueryAttentionGradient(MakeGroupQueryAttentionNode(4, 2, 1, scale));
    bool has_attention = false;
    bool has_dynamic_scale = false;
    for (const NodeProto &node : gradient.node()) {
      EXPECT_TRUE(node.domain().empty() || node.domain() == "ai.onnx");
      has_attention = has_attention || node.op_type() == "Attention";
      has_dynamic_scale = has_dynamic_scale || node.op_type() == "Reciprocal";
    }
    EXPECT_TRUE(has_attention);
    EXPECT_EQ(has_dynamic_scale, !scale.has_value());
  }
}

TEST(CustomOperatorSupport, GroupQueryAttentionGradientMatchesFiniteDifferences) {
  const std::vector<float> base_query = {0.2f, -0.1f, 0.4f, 0.3f, -0.2f, 0.5f, 0.1f, -0.4f};
  const std::vector<float> base_key = {0.3f, -0.2f, -0.1f, 0.4f};
  const std::vector<float> base_value = {0.5f, -0.3f, 0.2f, 0.7f};
  const std::vector<float> output_grad = {0.2f, -0.4f, 0.3f, 0.1f, -0.2f, 0.5f, -0.1f, 0.4f};
  constexpr float epsilon = 1e-3f;
  const std::vector<std::optional<float>> scales = {0.6f, std::nullopt};

  for (const int64_t causal : {int64_t{0}, int64_t{1}}) {
    for (const std::optional<float> scale : scales) {
      SCOPED_TRACE(::testing::Message() << "causal=" << causal << ", scale="
                                        << (scale.has_value() ? "explicit" : "implicit"));
      const NodeProto node = MakeGroupQueryAttentionNode(2, 1, causal, scale);
      const Tensors gradients = RunGradient(node, Tensor::FromFloat("Q", {1, 2, 4}, base_query),
                                            Tensor::FromFloat("K", {1, 2, 2}, base_key),
                                            Tensor::FromFloat("V", {1, 2, 2}, base_value),
                                            Tensor::FromFloat("dy", {1, 2, 4}, output_grad));
      ASSERT_EQ(gradients.size(), 3U);

      const std::vector<std::vector<float>> bases = {base_query, base_key, base_value};
      for (size_t input_index = 0; input_index < bases.size(); ++input_index) {
        ASSERT_EQ(gradients[input_index].element_count(), bases[input_index].size());
        const float *actual = gradients[input_index].AsFloat();
        for (size_t element = 0; element < bases[input_index].size(); ++element) {
          std::vector<float> query = base_query;
          std::vector<float> key = base_key;
          std::vector<float> value = base_value;
          std::vector<float> *perturbed[] = {&query, &key, &value};
          (*perturbed[input_index])[element] += epsilon;
          const float plus = RunForwardObjective(node, query, key, value, output_grad);
          (*perturbed[input_index])[element] -= 2.0f * epsilon;
          const float minus = RunForwardObjective(node, query, key, value, output_grad);
          const float expected = (plus - minus) / (2.0f * epsilon);
          EXPECT_NEAR(actual[element], expected, 3e-3f)
              << "input=" << input_index << ", element=" << element;
        }
      }
    }
  }
}

TEST(CustomOperatorSupport, GroupQueryAttentionGradientRunsForFloat16AndBFloat16) {
  const std::vector<float> query = {0.2f, -0.1f, 0.4f, 0.3f, -0.2f, 0.5f, 0.1f, -0.4f};
  const std::vector<float> key = {0.3f, -0.2f, -0.1f, 0.4f};
  const std::vector<float> value = {0.5f, -0.3f, 0.2f, 0.7f};
  const std::vector<float> output_grad = {0.2f, -0.4f, 0.3f, 0.1f, -0.2f, 0.5f, -0.1f, 0.4f};

  for (const rt_ns::DataType type : {rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16}) {
    SCOPED_TRACE(static_cast<int>(type));
    const auto make_tensor = [&](const char *name, std::vector<int64_t> shape,
                                 const std::vector<float> &values) {
      return type == rt_ns::DataType::FLOAT16
                 ? rt_ns::MakeFloat16Tensor(name, std::move(shape), values)
                 : rt_ns::MakeBfloat16Tensor(name, std::move(shape), values);
    };
    const Tensors gradients =
        RunGradient(MakeGroupQueryAttentionNode(2, 1, 1, std::nullopt),
                    make_tensor("Q", {1, 2, 4}, query), make_tensor("K", {1, 2, 2}, key),
                    make_tensor("V", {1, 2, 2}, value), make_tensor("dy", {1, 2, 4}, output_grad));
    ASSERT_EQ(gradients.size(), 3U);
    for (size_t index = 0; index < gradients.size(); ++index) {
      EXPECT_EQ(gradients[index].data_type, static_cast<int32_t>(type));
      EXPECT_EQ(gradients[index].element_count(), index == 0 ? 8U : 4U);
      const auto *bits = reinterpret_cast<const uint16_t *>(gradients[index].bytes());
      for (size_t element = 0; element < gradients[index].element_count(); ++element) {
        const float value = type == rt_ns::DataType::FLOAT16
                                ? onnx_light_cpu::detail::Float16BitsToFloat(bits[element])
                                : onnx_light_cpu::detail::Bfloat16BitsToFloat(bits[element]);
        EXPECT_TRUE(std::isfinite(value));
      }
    }
  }
}

TEST(CustomOperatorSupport, GroupQueryAttentionGradientRejectsUnsupportedVariants) {
  NodeProto attention_bias = MakeGroupQueryAttentionNode();
  while (attention_bias.input_size() <= 10) {
    attention_bias.add_input("");
  }
  attention_bias.ref_input()[10] = "attention_bias";

  NodeProto softcap = MakeGroupQueryAttentionNode();
  ONNX_LIGHT_NAMESPACE::AddAttribute(softcap, "softcap", 1.0f);

  NodeProto cached = MakeGroupQueryAttentionNode();
  cached.ref_input()[3] = "past_key";

  for (const NodeProto &node :
       {attention_bias, softcap, cached, MakeGroupQueryAttentionNode(3, 2)}) {
    EXPECT_ANY_THROW((void)MakeGroupQueryAttentionGradient(node));
  }
}

} // namespace
