// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/com_microsoft/moe_kernel.h"
#include "onnx_light_cpu/kernels/math/sigmoid_softmax_kernel.h"
#include "onnx_light_cpu/patterns/com_microsoft/patterns.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"
#include "onnx_light_cpu/shapes/com_microsoft/shape_inference.h"

#include "onnx_core/builder/graph_builder.h"
#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/builder/pattern_registry.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"
#include "onnx_extensions/shapes/dispatch_table.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
namespace builder = ONNX_LIGHT_NAMESPACE::core::builder;
namespace rt = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym = ONNX_LIGHT_NAMESPACE::core::symbolic;
using ONNX_LIGHT_NAMESPACE::AddAttribute;
using ONNX_LIGHT_NAMESPACE::NodeProto;
using ONNX_LIGHT_NAMESPACE::TensorProto;
using Pattern = onnx_light_cpu::MoEFusionPattern;

struct Options {
  int64_t k = 2;
  bool normalize = true;
  std::string activation = "Relu";
  bool bias1 = true;
  bool bias2 = true;
  bool commute = false;
  bool symbolic = false;
  std::string runtime_parameter;
  sym::TensorType type = sym::TensorType::kFloat;
};

auto SchemaLookup(const std::string &op) {
  auto schemas = ONNX_LIGHT_NAMESPACE::onnx_op::GetAllOnnxOpSchemasWithHistory(op, false);
  const auto custom = onnx_light_cpu::GetMicrosoftOpSchemasWithHistory(op, false);
  schemas.insert(schemas.end(), custom.begin(), custom.end());
  return schemas;
}

void Initializer(builder::GraphBuilder &graph, const std::string &name,
                 const std::vector<int64_t> &values) {
  TensorProto tensor;
  tensor.set_name(name);
  tensor.set_data_type(TensorProto::DataType::INT64);
  tensor.add_dims(static_cast<int64_t>(values.size()));
  for (const auto value : values) {
    tensor.ref_int64_data().push_back(value);
  }
  graph.MakeInitializer(tensor);
}

std::unique_ptr<builder::GraphBuilder> MakeGraph(const Options &options = {}) {
  ONNX_LIGHT_NAMESPACE::onnx_kernels::RegisterKernelFunctions();
  ONNX_LIGHT_NAMESPACE::onnx_shapes::RegisterShapeFunctions();
  onnx_light_cpu::RegisterMicrosoftShapeAndMemoryFunctions();
  auto graph = std::make_unique<builder::GraphBuilder>("moe_pattern", SchemaLookup);
  graph->set_device(sym::Device::kCPU);
  graph->SetOpsetVersion("", 29);
  const sym::SymDim tokens = options.symbolic ? sym::SymDim("M") : sym::SymDim(3);
  graph->MakeInput("X", options.type, sym::SymShape({tokens, sym::SymDim(2)}));
  graph->MakeInput("logits", options.type, sym::SymShape({tokens, sym::SymDim(3)}));
  const auto parameter = [&](const std::string &name, const std::vector<int64_t> &shape,
                             const std::vector<float> &values) {
    if (options.runtime_parameter == name || options.type != sym::TensorType::kFloat) {
      sym::SymShape symbolic_shape;
      for (const auto dimension : shape) {
        symbolic_shape.PushBack(sym::SymDim(dimension));
      }
      graph->MakeInput(name, options.type, symbolic_shape);
    } else {
      graph->MakeInitializer(
          ONNX_LIGHT_NAMESPACE::MakeInitializer<float>(name.c_str(), shape, values));
    }
  };
  std::vector<float> w1(24), w2(24);
  for (std::size_t i = 0; i < w1.size(); ++i) {
    w1[i] = static_cast<float>(static_cast<int>(i % 7) - 2) * 0.17F;
    w2[i] = static_cast<float>(static_cast<int>(i % 11) - 4) * 0.23F;
  }
  parameter("W1", {3, 2, 4}, w1);
  parameter("W2", {3, 4, 2}, w2);
  if (options.bias1) {
    parameter("B1", {3, 4}, {0, 1, -1, 2, 3, -2, 1, 0, -1, 0, 2, 1});
  }
  if (options.bias2) {
    parameter("B2", {3, 2}, {1, -1, 2, 0, -3, 1});
  }
  Initializer(*graph, "k", {options.k});
  Initializer(*graph, "axis1", {1});
  Initializer(*graph, "last", {-1});
  Initializer(*graph, "flat_shape", {-1, 4});
  Initializer(*graph, "index_shape", {-1, 1});
  Initializer(*graph, "result_shape", {-1, options.k, 2});
  NodeProto spec;
  AddAttribute(spec, "axis", int64_t{-1});
  graph->MakeNode("Softmax", {"logits"}, {"probabilities"}, "", "", spec.attribute());
  graph->MakeNode("TopK", {"probabilities", "k"}, {"values", "indices"}, "", "", spec.attribute());
  auto weights = std::string("values");
  if (options.normalize) {
    NodeProto reduction;
    AddAttribute(reduction, "keepdims", int64_t{1});
    graph->MakeNode("ReduceSum", {"values", "axis1"}, {"denominator"}, "", "",
                    reduction.attribute());
    graph->MakeNode("Div", {"values", "denominator"}, {"normalized"});
    weights = "normalized";
  }
  graph->MakeNode("GroupedMatMul",
                  options.bias1 ? std::vector<std::string>{"X", "W1", "indices", "B1"}
                                : std::vector<std::string>{"X", "W1", "indices"},
                  {"fc1"});
  auto activated = std::string("fc1");
  if (options.activation == "SiLU") {
    graph->MakeNode("Sigmoid", {"fc1"}, {"sigmoid"});
    graph->MakeNode("Mul", {"sigmoid", "fc1"}, {"activated"});
    activated = "activated";
  } else if (options.activation == "GeluTanh") {
    NodeProto spec;
    AddAttribute(spec, "approximate", std::string("tanh"));
    graph->MakeNode("Gelu", {"fc1"}, {"activated"}, "", "", spec.attribute());
    activated = "activated";
  } else if (!options.activation.empty()) {
    graph->MakeNode(options.activation, {"fc1"}, {"activated"});
    activated = "activated";
  }
  graph->MakeNode("Reshape", {activated, "flat_shape"}, {"flat"});
  graph->MakeNode("Reshape", {"indices", "index_shape"}, {"flat_indices"});
  graph->MakeNode("GroupedMatMul", {"flat", "W2", "flat_indices", options.bias2 ? "B2" : ""},
                  {"fc2"});
  graph->MakeNode("Reshape", {"fc2", "result_shape"}, {"experts"});
  graph->MakeNode("Unsqueeze", {weights, "last"}, {"routing"});
  graph->MakeNode("Mul",
                  options.commute ? std::vector<std::string>{"experts", "routing"}
                                  : std::vector<std::string>{"routing", "experts"},
                  {"weighted"});
  NodeProto reduction;
  AddAttribute(reduction, "keepdims", int64_t{0});
  graph->MakeNode("ReduceSum", {"weighted", "axis1"}, {"Y"}, "", "", reduction.attribute());
  graph->MakeOutput("Y");
  return graph;
}

std::vector<std::unique_ptr<builder::PatternOptimization>> Patterns() {
  onnx_light_cpu::RegisterCustomOperatorPatterns();
  std::vector<std::unique_ptr<builder::PatternOptimization>> patterns;
  patterns.push_back(builder::CreateRegisteredPattern("MicrosoftMoE"));
  return patterns;
}

NodeProto &Node(builder::GraphBuilder &graph, std::string output) {
  for (const auto &node : graph.Nodes()) {
    if (std::find(node.output().begin(), node.output().end(), output) != node.output().end()) {
      return const_cast<NodeProto &>(node);
    }
  }
  throw std::runtime_error("missing output: " + output);
}

void Reject(builder::GraphBuilder &graph) {
  builder::GraphGraph optimizer(graph, Patterns());
  const Pattern pattern;
  EXPECT_EQ(pattern.Match(optimizer, Node(graph, "Y")).pattern, nullptr);
}

TEST(MoEFusion, RegistryAcceptsRoutingActivationsBiasesAndSymbolicTokens) {
  for (const int64_t k : {1, 2, 3}) {
    for (const bool normalize : {false, true}) {
      for (const std::string activation : {"", "Identity", "Relu", "GeluTanh"}) {
        for (const bool bias : {false, true}) {
          auto graph = MakeGraph({.k = k,
                                  .normalize = normalize,
                                  .activation = activation,
                                  .bias1 = bias,
                                  .bias2 = !bias,
                                  .commute = !normalize,
                                  .symbolic = true});
          builder::GraphGraph optimizer(*graph, Patterns());
          const Pattern pattern;
          const auto match = pattern.Match(optimizer, Node(*graph, "Y"));
          ASSERT_NE(match.pattern, nullptr) << match.ToString();
          optimizer.Optimize();
          const auto moe = Node(*graph, "Y");
          EXPECT_EQ(moe.op_type(), "MoE");
          EXPECT_EQ(moe.domain(), "com.microsoft");
          ASSERT_EQ(moe.input_size(), 6);
          EXPECT_EQ(moe.input(3).empty(), !bias);
          EXPECT_EQ(moe.input(5).empty(), bias);
          EXPECT_EQ(graph->OpsetVersion("com.microsoft"), 1);
        }
      }
    }
  }
}

TEST(MoEFusion, ApplyRevalidatesAndTransposesWeightsWithoutReorderingExperts) {
  auto graph = MakeGraph();
  graph->MakeOutput("X");
  graph->MakeOutput("W1");
  graph->MakeInput("W1_moe_transposed", sym::TensorType::kFloat, sym::SymShape({1}));
  builder::GraphGraph optimizer(*graph, Patterns());
  const Pattern pattern;
  const auto match = pattern.Match(optimizer, Node(*graph, "Y"));
  ASSERT_NE(match.pattern, nullptr) << match.ToString();
  const auto replacements = pattern.Apply(optimizer, match.nodes);
  ASSERT_EQ(replacements.size(), 3U);
  EXPECT_EQ(replacements[0].op_type(), "Transpose");
  EXPECT_EQ(replacements[0].input(0), "W1");
  EXPECT_NE(replacements[0].output(0), "W1_moe_transposed");
  EXPECT_EQ(replacements[1].op_type(), "Transpose");
  EXPECT_EQ(replacements[1].input(0), "W2");
  EXPECT_EQ(replacements[2].input(1), "probabilities");
  EXPECT_EQ(replacements[2].input(3), "B1");
  EXPECT_EQ(replacements[2].input(5), "B2");
  const auto *mode = ONNX_LIGHT_NAMESPACE::FindAttribute(replacements[2], "topk_tie_break_low");
  ASSERT_NE(mode, nullptr);
  EXPECT_EQ(mode->i(), 1);
  const auto *router =
      ONNX_LIGHT_NAMESPACE::FindAttribute(replacements[2], "router_probs_are_logits");
  ASSERT_NE(router, nullptr);
  EXPECT_EQ(router->i(), 0);
  auto invalid = match.nodes;
  std::swap(invalid[0], invalid[1]);
  EXPECT_THROW(pattern.Apply(optimizer, invalid), builder::BuilderError);
  AddAttribute(Node(*graph, "values"), "largest", int64_t{0});
  EXPECT_THROW(pattern.Apply(optimizer, match.nodes), builder::BuilderError);
}

TEST(MoEFusion, RejectsEveryEscapingIntermediateIncludingBothTopKOutputs) {
  for (const std::string output :
       {"values", "indices", "denominator", "normalized", "fc1", "activated", "flat",
        "flat_indices", "fc2", "experts", "routing", "weighted"}) {
    for (const bool graph_output : {false, true}) {
      SCOPED_TRACE(output);
      auto graph = MakeGraph();
      if (graph_output) {
        graph->MakeOutput(output);
      } else {
        graph->MakeNode("Identity", {output}, {"side"});
        graph->MakeOutput("side");
      }
      Reject(*graph);
    }
  }
}

TEST(MoEFusion, PreservesSharedSoftmaxOutput) {
  for (const bool graph_output : {false, true}) {
    auto graph = MakeGraph();
    if (graph_output) {
      graph->MakeOutput("probabilities");
    } else {
      graph->MakeNode("Identity", {"probabilities"}, {"side"});
      graph->MakeOutput("side");
    }
    builder::GraphGraph(*graph, Patterns()).Optimize();
    ASSERT_EQ(Node(*graph, "Y").op_type(), "MoE");
    EXPECT_EQ(Node(*graph, "Y").input(1), "probabilities");
    EXPECT_EQ(Node(*graph, "probabilities").op_type(), "Softmax");
  }
}

TEST(MoEFusion, AcceptsDefaultDomainAliasAndEquivalentAxes) {
  auto graph = MakeGraph();
  for (const auto &node : graph->Nodes()) {
    const_cast<NodeProto &>(node).set_domain("ai.onnx");
  }
  for (const std::string output : {"probabilities", "values"}) {
    auto &node = Node(*graph, output);
    node.clear_attribute();
    AddAttribute(node, "axis", int64_t{1});
  }
  builder::GraphGraph optimizer(*graph, Patterns());
  EXPECT_NE(Pattern().Match(optimizer, Node(*graph, "Y")).pattern, nullptr);
}

TEST(MoEFusion, RejectsCapturedIntermediates) {
  for (const std::string output : {"values", "indices", "fc1", "weighted"}) {
    auto model = MakeGraph()->ToModel();
    auto &capture = *model.mutable_graph()->add_node();
    capture.set_op_type("If");
    capture.add_input("condition");
    capture.add_output("captured");
    for (const char *name : {"then_branch", "else_branch"}) {
      auto &attribute = *capture.add_attribute();
      attribute.set_name(name);
      attribute.set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::AttributeType::GRAPH);
      auto &branch = *attribute.mutable_g();
      branch.set_name(name);
      branch.ref_node().push_back(ONNX_LIGHT_NAMESPACE::MakeNode("Identity", {output}, {"out"}));
      branch.add_output()->set_name("out");
    }
    auto &condition = *model.mutable_graph()->add_input();
    condition.set_name("condition");
    condition.mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::DataType::BOOL);
    condition.mutable_type()->mutable_tensor_type()->mutable_shape();
    builder::GraphBuilder graph(model, SchemaLookup);
    graph.set_device(sym::Device::kCPU);
    Reject(graph);
  }
}

TEST(MoEFusion, RejectsSiluWithoutAnEquivalentActivationContract) {
  auto isolated = MakeGraph({.activation = "SiLU"});
  Reject(*isolated);
  for (const bool shared : {false, true}) {
    auto graph = MakeGraph({.activation = "SiLU"});
    if (shared) {
      graph->MakeOutput("sigmoid");
    } else {
      Node(*graph, "sigmoid").ref_input()[0] = "X";
    }
    Reject(*graph);
  }
}

TEST(MoEFusion, RejectsRetainedInputsProducedByTheMatchedBlock) {
  auto graph = MakeGraph();
  Node(*graph, "fc1").ref_input()[0] = "values";
  Reject(*graph);
}

TEST(MoEFusion, RejectsRuntimeExpertWeightsAndBiases) {
  for (const std::string parameter : {"W1", "W2", "B1", "B2"}) {
    SCOPED_TRACE(parameter);
    auto graph = MakeGraph({.runtime_parameter = parameter});
    Reject(*graph);
  }
}

TEST(MoEFusion, RejectsGpuAndUnspecifiedTargets) {
  for (const auto device : {sym::Device::kGPU0, sym::Device::kUndefined}) {
    auto graph = MakeGraph();
    graph->set_device(device);
    Reject(*graph);
  }
}

TEST(MoEFusion, FoldsImmutableExpertLayoutsBeforeInference) {
  auto graph = MakeGraph();
  builder::GraphGraph(*graph, Patterns()).Optimize();
  ASSERT_EQ(Node(*graph, "Y").op_type(), "MoE");
  ASSERT_EQ(graph->Nodes().size(), 2U);
  EXPECT_EQ(Node(*graph, "Y").input(1), "probabilities");
  EXPECT_EQ(Node(*graph, "probabilities").op_type(), "Softmax");
}

TEST(MoEFusion, RejectsNonFloatUnsupportedOpsetsAndAlteredContracts) {
  for (const auto type :
       {sym::TensorType::kFloat16, sym::TensorType::kBfloat16, sym::TensorType::kDouble}) {
    auto graph = MakeGraph({.type = type});
    Reject(*graph);
  }
  {
    auto graph = MakeGraph();
    graph->SetOpsetVersion("com.microsoft", 2);
    Reject(*graph);
  }
  {
    auto graph = MakeGraph();
    graph->SetOpsetVersion("", 28);
    Reject(*graph);
  }
  for (const std::string output :
       {"probabilities", "values", "denominator", "normalized", "fc1", "activated", "flat",
        "flat_indices", "fc2", "experts", "routing", "weighted", "Y"}) {
    for (const bool overload : {false, true}) {
      auto graph = MakeGraph();
      auto &node = Node(*graph, output);
      if (overload) {
        node.set_overload("custom");
      } else {
        AddAttribute(node, "unexpected", int64_t{1});
      }
      Reject(*graph);
    }
    auto graph = MakeGraph();
    Node(*graph, output).set_domain("custom");
    Reject(*graph);
  }
  for (const std::string approximate : {"", "none"}) {
    auto graph = MakeGraph({.activation = "Gelu"});
    if (!approximate.empty()) {
      AddAttribute(Node(*graph, "activated"), "approximate", approximate);
    }
    Reject(*graph);
  }
  for (const std::string attribute : {"largest", "sorted", "axis"}) {
    auto graph = MakeGraph();
    auto &node = Node(*graph, "values");
    node.clear_attribute();
    AddAttribute(node, attribute.c_str(), int64_t{0});
    Reject(*graph);
  }
  {
    auto graph = MakeGraph();
    Node(*graph, "fc2").ref_input()[2] = "indices";
    Reject(*graph);
  }
  {
    auto graph = MakeGraph();
    Node(*graph, "experts").ref_input()[1] = "flat_shape";
    Reject(*graph);
  }
  for (const int64_t k : {0, -1, 4}) {
    auto graph = MakeGraph();
    builder::GraphGraph optimizer(*graph, Patterns());
    TensorProto constant;
    constant.set_data_type(TensorProto::DataType::INT64);
    constant.add_dims(1);
    constant.ref_int64_data().push_back(k);
    optimizer.SetComputedConstant("k", constant);
    EXPECT_EQ(Pattern().Match(optimizer, Node(*graph, "Y")).pattern, nullptr);
  }
}

TEST(MoEFusion, RejectsModelLocalFunctionsWithDefaultOverload) {
  for (const std::string op : {"Softmax", "TopK", "GroupedMatMul", "Relu", "Reshape", "Unsqueeze",
                               "Mul", "Div", "ReduceSum", "Transpose", "MoE"}) {
    auto graph = MakeGraph();
    auto function = std::make_shared<ONNX_LIGHT_NAMESPACE::FunctionProto>();
    function->set_name(op);
    function->set_domain(op == "MoE" ? "com.microsoft" : "");
    function->add_input("arg");
    function->add_output("result");
    function->add_opset("", 29);
    function->ref_node().push_back(ONNX_LIGHT_NAMESPACE::MakeNode("Identity", {"arg"}, {"result"}));
    // Modern builders reject builtin redefinitions themselves; inject a legacy
    // context to exercise the pattern's independent defensive check.
    using Functions = std::unordered_map<std::string, const ONNX_LIGHT_NAMESPACE::FunctionProto *>;
    const_cast<Functions &>(
        graph->Shapes()
            .LocalFunctions())[ONNX_LIGHT_NAMESPACE::core::shapes::ShapesContext::LocalFunctionKey(
        function->domain(), function->name(), "")] = function.get();
    Reject(*graph);
  }
}

rt::Tensors RunGraph(builder::GraphBuilder &graph,
                     std::vector<std::pair<std::string, rt::Tensor>> inputs,
                     bool require_cpu_softmax = false) {
  const auto model = graph.ToModel();
  const rt::OpsetImports opsets = {{"", 29}, {"ai.onnx", 29}, {"com.microsoft", 1}};
  rt::RuntimeContext runtime{rt::KernelContext(rt::DefaultOpset(29), nullptr, &opsets)};
  runtime.set_kernel_usage_enabled(require_cpu_softmax);
  rt::SubgraphSession session(model.graph(), &opsets);
  auto outputs = session.Run(std::move(inputs), runtime);
  if (require_cpu_softmax) {
    const auto usage = runtime.GetKernelUsage();
    EXPECT_NE(std::find(usage.begin(), usage.end(), onnx_light_cpu::SoftmaxKernel::kName),
              usage.end());
  }
  for (auto &output : outputs) {
    output = output.ToOwned();
  }
  return outputs;
}

rt::Tensors RunGraph(builder::GraphBuilder &graph, bool ties) {
  return RunGraph(graph,
                  {{"X", rt::Tensor::FromFloat("X", {3, 2}, {1, -2, 0.5F, 3, -1, 2})},
                   {"logits", rt::Tensor::FromFloat(
                                  "logits", {3, 3},
                                  ties ? std::vector<float>{0, 0, 0, 1, 1, -1, -1, 2, 2}
                                       : std::vector<float>{0.2F, -0.5F, 1, 2, -1, 0, -2, 1, 3})}});
}

void CheckNearTieRouting(bool cpu_softmax) {
  onnx_light_cpu::RegisterMoEKernel();
  for (const int64_t k : {1, 2, 3}) {
    for (const bool normalize : {false, true}) {
      SCOPED_TRACE(::testing::Message() << k << " " << normalize);
      auto graph = MakeGraph({.k = k, .normalize = normalize});
      if (cpu_softmax) {
        onnx_light_cpu::RegisterSoftmaxKernel();
      }
      const auto run = [&] {
        return RunGraph(
            *graph,
            {{"X", rt::Tensor::FromFloat("X", {3, 2}, {1, -2, 0.5F, 3, -1, 2})},
             {"logits",
              rt::Tensor::FromFloat("logits", {3, 3},
                                    {-5.960464477539063e-08F, 0, -0.017000000923871994F, 0,
                                     -5.960464477539063e-08F, -0.017000000923871994F, 0, 0, 0})}},
            cpu_softmax);
      };
      const auto original = run();
      builder::GraphGraph(*graph, Patterns()).Optimize();
      ASSERT_EQ(Node(*graph, "Y").op_type(), "MoE");
      const auto fused = run();
      ASSERT_EQ(original.size(), 1U);
      ASSERT_EQ(fused.size(), 1U);
      for (std::size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(original[0].AsFloat()[i], fused[0].AsFloat()[i], 2.0e-5F);
      }
    }
  }
}

TEST(MoEFusion, NearTieRoutingPreservesOriginalSoftmaxOrderAndSelectedExperts) {
  CheckNearTieRouting(false);
}

TEST(MoEFusion, NearTieRoutingPreservesRegisteredCpuSoftmaxArithmetic) {
  CheckNearTieRouting(true);
}

TEST(MoEFusion, PreservesNanFromSelectedExpertsWithUnderflowedRoutingWeights) {
  onnx_light_cpu::RegisterMoEKernel();
  for (const bool normalize : {false, true}) {
    auto model =
        MakeGraph({.normalize = normalize, .activation = "", .bias1 = false, .bias2 = false})
            ->ToModel();
    std::vector<float> w1(24, 0.0F), w2(24, 1.0F);
    w1[0] = 1.0F;
    w1[8] = 3.0e38F;
    for (auto &initializer : model.mutable_graph()->ref_initializer()) {
      if (initializer.name() == "W1") {
        initializer = ONNX_LIGHT_NAMESPACE::MakeInitializer<float>("W1", {3, 2, 4}, w1);
      } else if (initializer.name() == "W2") {
        initializer = ONNX_LIGHT_NAMESPACE::MakeInitializer<float>("W2", {3, 4, 2}, w2);
      }
    }
    builder::GraphBuilder graph(model, SchemaLookup);
    graph.set_device(sym::Device::kCPU);
    const auto run = [&] {
      return RunGraph(
          graph,
          {{"X", rt::Tensor::FromFloat("X", {3, 2}, {2, 0, 2, 0, 2, 0})},
           {"logits", rt::Tensor::FromFloat("logits", {3, 3},
                                            {0, -104, -200, 0, -104, -200, 0, -104, -200})}});
    };
    const auto original = run();
    builder::GraphGraph(graph, Patterns()).Optimize();
    ASSERT_EQ(Node(graph, "Y").op_type(), "MoE");
    const auto fused = run();
    ASSERT_EQ(original.size(), 1U);
    ASSERT_EQ(fused.size(), 1U);
    for (std::size_t i = 0; i < 6; ++i) {
      EXPECT_TRUE(std::isnan(original[0].AsFloat()[i]));
      EXPECT_TRUE(std::isnan(fused[0].AsFloat()[i]));
    }
  }
}

TEST(MoEFusion, UpstreamSubgraphSessionNumericalParityIncludingRouterTies) {
  onnx_light_cpu::RegisterMoEKernel();
  for (const int64_t k : {1, 2, 3}) {
    for (const bool normalize : {false, true}) {
      for (const std::string activation : {"", "Relu", "GeluTanh"}) {
        for (const bool biases : {false, true}) {
          for (const bool ties : {false, true}) {
            SCOPED_TRACE(::testing::Message() << k << " " << normalize << " " << activation << " "
                                              << biases << " " << ties);
            auto graph = MakeGraph({.k = k,
                                    .normalize = normalize,
                                    .activation = activation,
                                    .bias1 = biases,
                                    .bias2 = biases,
                                    .commute = true});
            const auto original = RunGraph(*graph, ties);
            builder::GraphGraph(*graph, Patterns()).Optimize();
            ASSERT_EQ(Node(*graph, "Y").op_type(), "MoE");
            const auto fused = RunGraph(*graph, ties);
            ASSERT_EQ(original.size(), 1U);
            ASSERT_EQ(fused.size(), 1U);
            EXPECT_EQ(original[0].shape, fused[0].shape);
            for (std::size_t i = 0; i < 6; ++i) {
              EXPECT_NEAR(original[0].AsFloat()[i], fused[0].AsFloat()[i], 2.0e-5F);
            }
          }
        }
      }
    }
  }
}

TEST(MoEFusion, InsertsAfterSharedComputedWeightsRatherThanBeforeTheRouter) {
  onnx_light_cpu::RegisterMoEKernel();
  auto model = MakeGraph()->ToModel();
  auto original = std::move(model.mutable_graph()->ref_node());
  for (auto &node : original) {
    if (node.op_type() == "GroupedMatMul") {
      node.ref_input()[1] = node.input(1) + "_computed";
    }
    const bool router = node.op_type() == "Softmax";
    model.mutable_graph()->ref_node().push_back(std::move(node));
    if (router) {
      for (const std::string weight : {"W1", "W2"}) {
        model.mutable_graph()->ref_node().push_back(
            ONNX_LIGHT_NAMESPACE::MakeNode("Identity", {weight}, {weight + "_computed"}));
      }
    }
  }
  builder::GraphBuilder graph(model, SchemaLookup);
  graph.set_device(sym::Device::kCPU);
  const auto before = RunGraph(graph, true);
  builder::GraphGraph optimizer(graph, Patterns());
  const Pattern pattern;
  const auto match = pattern.Match(optimizer, Node(graph, "Y"));
  ASSERT_NE(match.pattern, nullptr);
  EXPECT_EQ(match.insert_at, &Node(graph, "Y"));
  optimizer.Optimize();
  const auto after = RunGraph(graph, true);
  ASSERT_EQ(Node(graph, "Y").op_type(), "MoE");
  for (std::size_t i = 0; i < 6; ++i) {
    EXPECT_NEAR(before[0].AsFloat()[i], after[0].AsFloat()[i], 2.0e-5F);
  }
}
} // namespace
