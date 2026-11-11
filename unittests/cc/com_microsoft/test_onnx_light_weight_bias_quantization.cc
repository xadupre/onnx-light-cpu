// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/math/gemm_kernel.h"
#include "onnx_light_cpu/patterns/weight_bias_quantization.h"

#include "onnx_core/builder/graph_builder.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"
#include "onnx_extensions/shapes/dispatch_table.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
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
using DT = TensorProto::DataType;

struct Quantization {
  float scale = 0.5F;
  std::vector<int64_t> scale_shape;
  DT scale_type = DT::FLOAT;
  DT type = DT::INT8;
  DT zero_type = DT::UNDEFINED;
  int32_t zero = 0;
  bool omit_zero = false;
  bool empty_zero = false;
  bool dynamic_scale = false;
  bool dynamic_zero = false;
  std::vector<int64_t> zero_shape;
  NodeProto attributes;
};

struct Options {
  std::string op = "Conv";
  std::string domain;
  int rank = 3;
  int opset = 23;
  sym::Device device = sym::Device::kCPU;
  Quantization activation;
  Quantization output;
  Quantization weight_quantization;
  Quantization bias_quantization;
  std::vector<float> weight = {127, -64, 32, 0, 2, -2};
  std::vector<float> bias = {0.5F, -1.5F, 2.5F};
  std::vector<int32_t> integer_weight = {127, -64, 32, 0, 2, -2};
  std::vector<int32_t> integer_bias = {1, -3, 5};
  std::vector<int64_t> weight_shape;
  std::vector<int64_t> activation_shape;
  std::vector<int64_t> bias_shape = {3};
  DT float_type = DT::FLOAT;
  bool with_bias = true;
  bool weight_dq = false;
  bool bias_dq = false;
  bool dynamic_weight = false;
  bool dynamic_bias = false;
  bool default_weight = false;
  bool default_bias = false;
  bool computed_weight = false;
  bool raw_weight = false;
  bool activation_dq = true;
  bool output_q = true;
  std::string post_op;
  std::string expose;
  std::string branch;
  std::string capture;
  NodeProto attributes;

  Options() {
    weight_quantization.scale = 1.0F;
    bias_quantization.type = DT::INT32;
  }
};

TensorProto FloatTensor(const std::string &name, const std::vector<int64_t> &shape,
                        const std::vector<float> &values, DT type = DT::FLOAT, bool raw = false) {
  TensorProto tensor;
  tensor.set_name(name);
  tensor.set_data_type(type);
  for (auto dim : shape) {
    tensor.add_dims(dim);
  }
  for (float value : values) {
    if (type == DT::DOUBLE) {
      tensor.ref_double_data().push_back(value);
    } else if (raw) {
      const auto bits = std::bit_cast<uint32_t>(value);
      for (int byte = 0; byte < 4; ++byte) {
        tensor.ref_raw_data().push_back(static_cast<uint8_t>(bits >> (8 * byte)));
      }
    } else {
      tensor.ref_float_data().push_back(value);
    }
  }
  return tensor;
}

TensorProto IntegerTensor(const std::string &name, const std::vector<int64_t> &shape,
                          const std::vector<int32_t> &values, DT type = DT::INT8) {
  TensorProto tensor;
  tensor.set_name(name);
  tensor.set_data_type(type);
  for (auto dim : shape) {
    tensor.add_dims(dim);
  }
  for (int32_t value : values) {
    tensor.ref_int32_data().push_back(value);
  }
  return tensor;
}

sym::TensorType SymbolicType(DT type) { return sym::DataTypeToTensorType(type); }

sym::SymShape SymbolicShape(const std::vector<int64_t> &dimensions) {
  sym::SymShape shape;
  for (auto dimension : dimensions) {
    shape.PushBack(sym::SymDim(dimension));
  }
  return shape;
}

std::vector<std::string> QuantizationInputs(builder::GraphBuilder &graph, const std::string &input,
                                            const std::string &prefix,
                                            const Quantization &quantization) {
  const auto scale = prefix + "_scale";
  if (quantization.dynamic_scale) {
    graph.MakeInput(scale, SymbolicType(quantization.scale_type),
                    SymbolicShape(quantization.scale_shape));
  } else {
    graph.MakeInitializer(FloatTensor(scale, quantization.scale_shape, {quantization.scale},
                                      quantization.scale_type));
  }
  std::vector<std::string> inputs = {input, scale};
  if (quantization.empty_zero) {
    inputs.push_back("");
  } else if (!quantization.omit_zero) {
    const auto zero = prefix + "_zero";
    const auto type =
        quantization.zero_type == DT::UNDEFINED ? quantization.type : quantization.zero_type;
    if (quantization.dynamic_zero) {
      graph.MakeInput(zero, SymbolicType(type), SymbolicShape(quantization.zero_shape));
    } else {
      graph.MakeInitializer(
          IntegerTensor(zero, quantization.zero_shape, {quantization.zero}, type));
    }
    inputs.push_back(zero);
  }
  return inputs;
}

std::vector<int64_t> WeightShape(const Options &options) {
  if (!options.weight_shape.empty()) {
    return options.weight_shape;
  }
  if (options.op == "Gemm") {
    for (const auto &attribute : options.attributes.attribute()) {
      if (attribute.name() == "transB" && attribute.i() == 1) {
        return {3, 2};
      }
    }
    return {2, 3};
  }
  auto shape =
      options.op == "ConvTranspose" ? std::vector<int64_t>{2, 3} : std::vector<int64_t>{3, 2};
  shape.resize(options.rank, 1);
  return shape;
}

std::vector<int64_t> ActivationShape(const Options &options) {
  if (!options.activation_shape.empty()) {
    return options.activation_shape;
  }
  return options.op == "Gemm" ? std::vector<int64_t>{2, 2}
                              : (options.rank == 4 ? std::vector<int64_t>{1, 2, 2, 2}
                                                   : std::vector<int64_t>{1, 2, 4});
}

std::unique_ptr<builder::GraphBuilder> MakeGraph(const Options &options = {},
                                                 bool rejection_fixture = false) {
  ONNX_LIGHT_NAMESPACE::onnx_kernels::RegisterKernelFunctions();
  ONNX_LIGHT_NAMESPACE::onnx_shapes::RegisterShapeFunctions();
  builder::OperatorRegistry registry;
  if (rejection_fixture) {
    // Let malformed ranks/attribute lengths reach the stage's own checks rather
    // than failing earlier in built-in convolution shape inference.
    registry.RegisterShapeInference(
        options.domain, options.op, [](builder::ShapesContext &context, const NodeProto &node) {
          const auto &input = context.Get(node.input(0));
          context.Set(node.output(0), sym::SymTensor(nullptr, input.Dtype(), input.Shape()));
        });
  }
  auto graph = std::make_unique<builder::GraphBuilder>(
      "weight_bias_quantization", builder::GraphBuilder::SchemaLookupFn{},
      std::unordered_map<std::string, int>{}, registry);
  graph->SetOpsetVersion("", options.opset);
  if (!options.domain.empty()) {
    graph->SetOpsetVersion(options.domain, 1);
  }
  graph->set_device(options.device);
  graph->MakeInput("Aq", SymbolicType(options.activation.type),
                   SymbolicShape(ActivationShape(options)));
  if (options.activation_dq) {
    graph->MakeNode("DequantizeLinear", QuantizationInputs(*graph, "Aq", "a", options.activation),
                    {"A"}, "", "activation_dq", options.activation.attributes.attribute());
  } else {
    graph->MakeInput("A", SymbolicType(options.float_type),
                     SymbolicShape(ActivationShape(options)));
  }
  if (options.weight_dq) {
    if (options.dynamic_weight) {
      graph->MakeInput("Wq", SymbolicType(options.weight_quantization.type),
                       SymbolicShape(WeightShape(options)));
    } else {
      graph->MakeInitializer(IntegerTensor("Wq", WeightShape(options), options.integer_weight,
                                           options.weight_quantization.type));
    }
    graph->MakeNode("DequantizeLinear",
                    QuantizationInputs(*graph, "Wq", "w", options.weight_quantization), {"W"}, "",
                    "weight_dq", options.weight_quantization.attributes.attribute());
  } else if (options.dynamic_weight) {
    graph->MakeInput("W", SymbolicType(options.float_type), SymbolicShape(WeightShape(options)));
  } else {
    if (options.default_weight) {
      graph->MakeInput("W", SymbolicType(options.float_type), SymbolicShape(WeightShape(options)));
    }
    graph->MakeInitializer(FloatTensor(options.computed_weight ? "W0" : "W", WeightShape(options),
                                       options.weight, options.float_type, options.raw_weight));
    if (options.computed_weight) {
      graph->MakeNode("Identity", {"W0"}, {"W"});
    }
  }
  if (options.with_bias) {
    if (options.bias_dq) {
      graph->MakeInitializer(
          IntegerTensor("Bq", options.bias_shape, options.integer_bias, DT::INT32));
      graph->MakeNode("DequantizeLinear",
                      QuantizationInputs(*graph, "Bq", "b", options.bias_quantization), {"B"}, "",
                      "bias_dq", options.bias_quantization.attributes.attribute());
    } else if (options.dynamic_bias) {
      graph->MakeInput("B", SymbolicType(options.float_type), SymbolicShape(options.bias_shape));
    } else {
      if (options.default_bias) {
        graph->MakeInput("B", SymbolicType(options.float_type), SymbolicShape(options.bias_shape));
      }
      graph->MakeInitializer(
          FloatTensor("B", options.bias_shape, options.bias, options.float_type));
    }
  }
  graph->MakeNode(options.op,
                  options.with_bias ? std::vector<std::string>{"A", "W", "B"}
                                    : std::vector<std::string>{"A", "W"},
                  {"C"}, options.domain, "linear", options.attributes.attribute());
  std::string output = "C";
  if (!options.post_op.empty()) {
    if (options.post_op == "Clip") {
      graph->MakeInitializer(FloatTensor("clip_min", {}, {0}));
      graph->MakeInitializer(FloatTensor("clip_max", {}, {100}));
      graph->MakeNode("Clip", {"C", "clip_min", "clip_max"}, {"P"});
    } else {
      graph->MakeNode(options.post_op, {"C"}, {"P"});
    }
    output = "P";
  }
  if (options.output_q) {
    graph->MakeNode("QuantizeLinear", QuantizationInputs(*graph, output, "y", options.output),
                    {"Y"}, "", "output_q", options.output.attributes.attribute());
    output = "Y";
  }
  graph->MakeOutput(output);
  if (!options.expose.empty()) {
    graph->MakeOutput(options.expose);
  }
  if (!options.branch.empty()) {
    graph->MakeNode("Identity", {options.branch}, {"other"});
    graph->MakeOutput("other");
  }
  if (!options.capture.empty()) {
    graph->MakeInput("condition", sym::TensorType::kBool, SymbolicShape({}));
    auto conditional = ONNX_LIGHT_NAMESPACE::MakeNode("If", {"condition"}, {"captured_output"});
    for (const char *branch : {"then_branch", "else_branch"}) {
      auto *attribute = conditional.add_attribute();
      attribute->set_name(branch);
      attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::AttributeType::GRAPH);
      auto *subgraph = attribute->mutable_g();
      subgraph->set_name(branch);
      subgraph->ref_node().push_back(
          ONNX_LIGHT_NAMESPACE::MakeNode("Identity", {options.capture}, {"captured"}));
      subgraph->add_output()->set_name("captured");
    }
    graph->ImportNode(conditional);
    graph->MakeOutput("captured_output");
  }
  return graph;
}

const NodeProto *Producer(const builder::GraphBuilder &graph, const std::string &name) {
  for (const auto &node : graph.Nodes()) {
    for (const auto &output : node.output()) {
      if (output == name) {
        return &node;
      }
    }
  }
  return nullptr;
}

const NodeProto &Linear(const builder::GraphBuilder &graph) {
  for (const auto &node : graph.Nodes()) {
    if (node.op_type() == "Conv" || node.op_type() == "ConvTranspose" || node.op_type() == "Gemm") {
      return node;
    }
  }
  throw std::runtime_error("missing linear node");
}

const TensorProto &Initializer(const builder::GraphBuilder &graph, const std::string &name) {
  for (const auto &tensor : graph.Initializers()) {
    if (tensor.name() == name) {
      return tensor;
    }
  }
  throw std::runtime_error("missing initializer: " + name);
}

std::vector<int32_t> Integers(const TensorProto &proto) {
  const auto tensor = rt::TensorFromProto(proto);
  std::vector<int32_t> result;
  const auto count = tensor.size_bytes() / (proto.data_type() == DT::INT32 ? sizeof(int32_t) : 1);
  for (std::size_t i = 0; i < count; ++i) {
    if (proto.data_type() == DT::INT32) {
      result.push_back(tensor.AsInt32()[i]);
    } else if (proto.data_type() == DT::INT8) {
      result.push_back(static_cast<int8_t>(tensor.bytes()[i]));
    } else {
      result.push_back(tensor.bytes()[i]);
    }
  }
  return result;
}

void ExpectPrepared(builder::GraphBuilder &graph, const Options &options) {
  const auto original = Linear(graph);
  const auto activation = *Producer(graph, "A");
  const auto quantize = *Producer(graph, "Y");
  onnx_light_cpu::PrepareWeightBiasQuantization(graph);
  const auto &linear = Linear(graph);
  EXPECT_EQ(linear.op_type(), options.op);
  EXPECT_EQ(linear.domain(), original.domain());
  EXPECT_EQ(linear.name(), original.name());
  EXPECT_EQ(linear.output(0), original.output(0));
  EXPECT_EQ(linear.attribute().size(), original.attribute().size());
  for (std::size_t i = 0; i < original.attribute().size(); ++i) {
    EXPECT_EQ(linear.attribute()[i].SerializeAsString(),
              original.attribute()[i].SerializeAsString());
  }
  ASSERT_NE(Producer(graph, "A"), nullptr);
  EXPECT_EQ(Producer(graph, "A")->SerializeAsString(), activation.SerializeAsString());
  ASSERT_NE(Producer(graph, "Y"), nullptr);
  EXPECT_EQ(Producer(graph, "Y")->SerializeAsString(), quantize.SerializeAsString());
  const auto *weight_dq = Producer(graph, linear.input(1));
  ASSERT_NE(weight_dq, nullptr);
  ASSERT_EQ(weight_dq->op_type(), "DequantizeLinear");
  EXPECT_EQ(Initializer(graph, weight_dq->input(0)).data_type(), DT::INT8);
  EXPECT_EQ(Initializer(graph, weight_dq->input(1)).dims().size(), 0U);
  if (options.with_bias) {
    const auto *bias_dq = Producer(graph, linear.input(2));
    ASSERT_NE(bias_dq, nullptr);
    ASSERT_EQ(bias_dq->op_type(), "DequantizeLinear");
    EXPECT_EQ(Initializer(graph, bias_dq->input(0)).data_type(), DT::INT32);
    EXPECT_EQ(Initializer(graph, bias_dq->input(1)).dims().size(), 0U);
  }
  for (const auto &node : graph.Nodes()) {
    EXPECT_NE(node.op_type(), "Mul");
    EXPECT_NE(node.op_type(), "Div");
    EXPECT_NE(node.op_type(), "Round");
    EXPECT_NE(node.op_type(), "Cast");
    EXPECT_NE(node.op_type(), "Constant");
    if (node.op_type() == "QuantizeLinear") {
      EXPECT_EQ(node.output(0), "Y");
    }
  }
  const auto once = graph.ToGraph().SerializeAsString();
  onnx_light_cpu::PrepareWeightBiasQuantization(graph);
  EXPECT_EQ(graph.ToGraph().SerializeAsString(), once);
}

void ExpectRejected(const Options &options) {
  auto graph = MakeGraph(options, true);
  const auto before = graph->ToGraph().SerializeAsString();
  onnx_light_cpu::PrepareWeightBiasQuantization(*graph);
  EXPECT_EQ(graph->ToGraph().SerializeAsString(), before);
}

TEST(WeightBiasQuantization, PreparesAllOperatorsRanksBiasAndActivationPaths) {
  for (const std::string op : {"Conv", "ConvTranspose", "Gemm"}) {
    for (int rank : (op == "Gemm" ? std::vector<int>{2} : std::vector<int>{3, 4})) {
      for (bool bias : {false, true}) {
        for (const std::string post_op : {"", "Relu", "Clip"}) {
          SCOPED_TRACE(op + "/" + std::to_string(rank) + "/" + post_op);
          Options options;
          options.op = op;
          options.rank = rank;
          options.with_bias = bias;
          options.post_op = post_op;
          auto graph = MakeGraph(options);
          ExpectPrepared(*graph, options);
          EXPECT_EQ(graph->Outputs().size(), 1U);
          EXPECT_EQ(graph->Outputs()[0].name(), "Y");
        }
      }
    }
  }
}

TEST(WeightBiasQuantization, ExplicitStageDoesNotRunDuringOrdinaryConstantFolding) {
  auto graph = MakeGraph();
  builder::ConstantFoldingOptions folding;
  folding.excluded_ops.insert({"", "DequantizeLinear"});
  graph->ConstantFold(folding);
  EXPECT_EQ(Linear(*graph).input(1), "W");
  EXPECT_EQ(Linear(*graph).input(2), "B");
  EXPECT_EQ(Initializer(*graph, "W").data_type(), DT::FLOAT);
  EXPECT_EQ(Initializer(*graph, "B").data_type(), DT::FLOAT);
  ExpectPrepared(*graph, {});
}

TEST(WeightBiasQuantization, UsesScalarSymmetricWeightsAndTiesToEvenBias) {
  for (bool raw : {false, true}) {
    Options options;
    options.raw_weight = raw;
    options.weight = {127, -63.5F, 31.5F, -31.5F, 2.5F, -2.5F};
    options.bias = {0.25F, 0.75F, -1.25F};
    auto graph = MakeGraph(options);
    ExpectPrepared(*graph, options);
    const auto &linear = Linear(*graph);
    const auto &weight = *Producer(*graph, linear.input(1));
    const auto &bias = *Producer(*graph, linear.input(2));
    EXPECT_EQ(Integers(Initializer(*graph, weight.input(0))),
              (std::vector<int32_t>{127, -64, 32, -32, 2, -2}));
    EXPECT_EQ(Integers(Initializer(*graph, bias.input(0))), (std::vector<int32_t>{0, 2, -2}));
    EXPECT_FLOAT_EQ(rt::TensorFromProto(Initializer(*graph, weight.input(1))).AsFloat()[0], 1.0F);
    EXPECT_FLOAT_EQ(rt::TensorFromProto(Initializer(*graph, bias.input(1))).AsFloat()[0], 0.5F);
    ASSERT_EQ(weight.input().size(), 3U);
    EXPECT_EQ(Integers(Initializer(*graph, weight.input(2))), (std::vector<int32_t>{0}));
    if (bias.input().size() == 3U && !bias.input(2).empty()) {
      EXPECT_EQ(Integers(Initializer(*graph, bias.input(2))), (std::vector<int32_t>{0}));
    } else {
      EXPECT_EQ(bias.input().size(), 2U);
    }
  }
}

TEST(WeightBiasQuantization, DerivesWeightScaleFromAbsoluteMaximumIncludingNegativeMaximum) {
  for (float sign : {-1.0F, 1.0F}) {
    Options options;
    options.weight = {sign * 254, -127, 63.5F, 0, 5, -5};
    auto graph = MakeGraph(options);
    ExpectPrepared(*graph, options);
    const auto &dq = *Producer(*graph, Linear(*graph).input(1));
    EXPECT_FLOAT_EQ(rt::TensorFromProto(Initializer(*graph, dq.input(1))).AsFloat()[0], 2.0F);
    EXPECT_EQ(Integers(Initializer(*graph, dq.input(0))),
              (std::vector<int32_t>{static_cast<int32_t>(sign * 127), -64, 32, 0, 2, -2}));
  }
}

TEST(WeightBiasQuantization, AllZeroWeightsUseUnitScale) {
  Options options;
  options.weight = {0, -0.0F, 0, -0.0F, 0, 0};
  auto graph = MakeGraph(options);
  ExpectPrepared(*graph, options);
  const auto &dq = *Producer(*graph, Linear(*graph).input(1));
  EXPECT_FLOAT_EQ(rt::TensorFromProto(Initializer(*graph, dq.input(1))).AsFloat()[0], 1.0F);
  EXPECT_EQ(Integers(Initializer(*graph, dq.input(0))), (std::vector<int32_t>(6, 0)));
}

TEST(WeightBiasQuantization, AcceptsComputedConstantWeightsAndPreservesConstantDqBoundaries) {
  Options options;
  options.computed_weight = true;
  auto graph = MakeGraph(options);
  graph->MakeInitializer(IntegerTensor("unrelated_q", {2}, {-3, 7}));
  graph->MakeInitializer(FloatTensor("unrelated_scale", {}, {0.25F}));
  graph->MakeNode("DequantizeLinear", {"unrelated_q", "unrelated_scale"}, {"unrelated_float"});
  graph->MakeOutput("unrelated_float");
  const auto boundary = Producer(*graph, "unrelated_float")->SerializeAsString();
  ExpectPrepared(*graph, options);
  ASSERT_NE(Producer(*graph, "unrelated_float"), nullptr);
  EXPECT_EQ(Producer(*graph, "unrelated_float")->SerializeAsString(), boundary);

  options.bias_shape = {2};
  options.bias = {0.5F, -1.5F};
  ExpectRejected(options);
}

TEST(WeightBiasQuantization, AcceptsExistingSymmetricInt8WeightsWithFloatBiasOnly) {
  for (bool omit : {false, true}) {
    Options options;
    options.weight_dq = true;
    options.weight_quantization.omit_zero = omit;
    auto graph = MakeGraph(options);
    const auto weight = Producer(*graph, "W")->SerializeAsString();
    const auto integer_weight = Initializer(*graph, "Wq").SerializeAsString();
    ExpectPrepared(*graph, options);
    EXPECT_EQ(Producer(*graph, "W")->SerializeAsString(), weight);
    EXPECT_EQ(Initializer(*graph, "Wq").SerializeAsString(), integer_weight);
    options.with_bias = false;
    ExpectRejected(options);
  }
}

TEST(WeightBiasQuantization, AlreadyPreparedInputsAreNoOpAndAsymmetricPreparationIsRejected) {
  Options options;
  options.weight_dq = true;
  options.bias_dq = true;
  ExpectRejected(options);
  options.weight_dq = false;
  ExpectRejected(options);
}

TEST(WeightBiasQuantization, PreservesSharedAndPublicFloatInitializers) {
  for (const std::string op : {"Conv", "ConvTranspose", "Gemm"}) {
    Options options;
    options.op = op;
    auto graph = MakeGraph(options);
    graph->MakeOutput("W");
    graph->MakeOutput("B");
    graph->MakeInput("other_w", sym::TensorType::kFloat, SymbolicShape(WeightShape(options)));
    graph->MakeInput("other_b", sym::TensorType::kFloat, SymbolicShape(options.bias_shape));
    graph->MakeNode("Add", {"W", "other_w"}, {"shared_w"});
    graph->MakeNode("Add", {"B", "other_b"}, {"shared_b"});
    graph->MakeOutput("shared_w");
    graph->MakeOutput("shared_b");
    const auto weight = Initializer(*graph, "W").SerializeAsString();
    const auto bias = Initializer(*graph, "B").SerializeAsString();
    ExpectPrepared(*graph, options);
    EXPECT_EQ(Initializer(*graph, "W").SerializeAsString(), weight);
    EXPECT_EQ(Initializer(*graph, "B").SerializeAsString(), bias);
    EXPECT_NE(Linear(*graph).input(1), "W");
    EXPECT_NE(Linear(*graph).input(2), "B");
    ASSERT_NE(Producer(*graph, "shared_w"), nullptr);
    EXPECT_EQ(Producer(*graph, "shared_w")->input(0), "W");
    ASSERT_NE(Producer(*graph, "shared_b"), nullptr);
    EXPECT_EQ(Producer(*graph, "shared_b")->input(0), "B");
  }
}

TEST(WeightBiasQuantization, RejectsSharedOrPublicIntermediateOutputsButAllowsSharedActivation) {
  for (const std::string post_op : {"", "Relu", "Clip"}) {
    for (const std::string value :
         (post_op.empty() ? std::vector<std::string>{"C"} : std::vector<std::string>{"C", "P"})) {
      Options options;
      options.post_op = post_op;
      options.expose = value;
      ExpectRejected(options);
      options.expose.clear();
      options.branch = value;
      ExpectRejected(options);
    }
  }
  Options options;
  options.expose = "A";
  options.branch = "A";
  auto graph = MakeGraph(options);
  ExpectPrepared(*graph, options);
}

TEST(WeightBiasQuantization, RejectsSubgraphCapturedIntermediatesButPreservesCapturedConstants) {
  for (const std::string post_op : {"", "Relu", "Clip"}) {
    for (const std::string capture :
         (post_op.empty() ? std::vector<std::string>{"C"} : std::vector<std::string>{"C", "P"})) {
      Options options;
      options.post_op = post_op;
      options.capture = capture;
      ExpectRejected(options);
    }
  }
  for (const std::string capture : {"A", "W", "B"}) {
    Options options;
    options.capture = capture;
    auto graph = MakeGraph(options);
    const auto weight = Initializer(*graph, "W").SerializeAsString();
    const auto bias = Initializer(*graph, "B").SerializeAsString();
    ExpectPrepared(*graph, options);
    if (capture == "W") {
      EXPECT_EQ(Initializer(*graph, "W").SerializeAsString(), weight);
    }
    if (capture == "B") {
      EXPECT_EQ(Initializer(*graph, "B").SerializeAsString(), bias);
    }
  }
}

TEST(WeightBiasQuantization, RejectsMissingBoundariesAndUnsupportedPostOperations) {
  Options options;
  options.activation_dq = false;
  ExpectRejected(options);
  options = {};
  options.output_q = false;
  ExpectRejected(options);
  options = {};
  options.post_op = "Sigmoid";
  ExpectRejected(options);
}

TEST(WeightBiasQuantization, RequiresCpuAndDefaultOpsetAtLeast13) {
  Options options;
  options.device = sym::Device::kUndefined;
  ExpectRejected(options);
  options = {};
  options.opset = 12;
  ExpectRejected(options);
  options = {};
  options.domain = "custom.linear";
  ExpectRejected(options);
  options = {};
  for (int opset : {13, 18, 23}) {
    options.opset = opset;
    auto graph = MakeGraph(options);
    ExpectPrepared(*graph, options);
  }
}

TEST(WeightBiasQuantization, AcceptsSignedUnsignedAndOmittedScalarZeroPoints) {
  for (DT input_type : {DT::INT8, DT::UINT8}) {
    for (DT output_type : {DT::INT8, DT::UINT8}) {
      for (int omitted : {0, 1, 2}) {
        Options options;
        options.activation.type = input_type;
        options.activation.zero = input_type == DT::UINT8 ? 128 : -3;
        options.output.type = output_type;
        options.output.zero = output_type == DT::UINT8 ? 128 : -7;
        options.activation.omit_zero = omitted == 1;
        options.activation.empty_zero = omitted == 2;
        options.output.omit_zero = omitted == 1;
        options.output.empty_zero = omitted == 2;
        auto graph = MakeGraph(options);
        ExpectPrepared(*graph, options);
      }
    }
  }
}

TEST(WeightBiasQuantization, RejectsDynamicOrNonFloatConstants) {
  Options options;
  options.dynamic_weight = true;
  ExpectRejected(options);
  options = {};
  options.dynamic_bias = true;
  ExpectRejected(options);
  options = {};
  options.default_weight = true;
  ExpectRejected(options);
  options = {};
  options.default_bias = true;
  ExpectRejected(options);
  options = {};
  options.float_type = DT::DOUBLE;
  options.activation.scale_type = DT::DOUBLE;
  ExpectRejected(options);
  for (bool output : {false, true}) {
    for (int invalid : {0, 1, 2, 3, 4, 5}) {
      options = {};
      auto &q = output ? options.output : options.activation;
      switch (invalid) {
      case 0:
        q.dynamic_scale = true;
        break;
      case 1:
        q.dynamic_zero = true;
        break;
      case 2:
        q.scale_type = DT::DOUBLE;
        break;
      case 3:
        q.zero_type = output ? DT::INT32 : DT::UINT8;
        break;
      case 4:
        q.zero_shape = {1};
        break;
      case 5:
        q.type = DT::INT32;
        break;
      }
      ExpectRejected(options);
    }
  }
}

TEST(WeightBiasQuantization, RejectsPerAxisBlockedAndInvalidScalesAtEveryBoundary) {
  for (int boundary : {0, 1, 2}) {
    for (int invalid : {0, 1, 2, 3, 4, 5, 6, 7}) {
      Options options;
      options.weight_dq = boundary == 2;
      auto &q = boundary == 0 ? options.activation
                              : (boundary == 1 ? options.output : options.weight_quantization);
      switch (invalid) {
      case 0:
        q.scale = 0;
        break;
      case 1:
        q.scale = -1;
        break;
      case 2:
        q.scale = std::numeric_limits<float>::infinity();
        break;
      case 3:
        q.scale = std::numeric_limits<float>::quiet_NaN();
        break;
      case 4:
        q.scale_shape = {1};
        break;
      case 5:
        q.scale_shape = {1, 1};
        break;
      case 6:
        AddAttribute(q.attributes, "block_size", int64_t{2});
        break;
      case 7:
        q.zero_shape = {1};
        break;
      }
      SCOPED_TRACE(std::to_string(boundary) + "/" + std::to_string(invalid));
      ExpectRejected(options);
    }
  }
}

TEST(WeightBiasQuantization, ScalarAxisIsIgnoredAndZeroBlockSizeIsAccepted) {
  for (int64_t axis : {-1, 0, 1}) {
    Options options;
    AddAttribute(options.activation.attributes, "axis", axis);
    AddAttribute(options.output.attributes, "axis", axis);
    AddAttribute(options.activation.attributes, "block_size", int64_t{0});
    AddAttribute(options.output.attributes, "block_size", int64_t{0});
    auto graph = MakeGraph(options);
    ExpectPrepared(*graph, options);
  }
}

TEST(WeightBiasQuantization, ExistingWeightDqMustBeConstantInt8SymmetricScalar) {
  for (int invalid : {0, 1, 2, 3, 4, 5}) {
    Options options;
    options.weight_dq = true;
    switch (invalid) {
    case 0:
      options.weight_quantization.type = DT::UINT8;
      options.integer_weight = {127, 64, 32, 0, 2, 2};
      break;
    case 1:
      options.weight_quantization.zero = 1;
      break;
    case 2:
      options.weight_quantization.zero_type = DT::UINT8;
      break;
    case 3:
      options.weight_quantization.dynamic_scale = true;
      break;
    case 4:
      options.weight_quantization.dynamic_zero = true;
      break;
    case 5:
      options.dynamic_weight = true;
      break;
    }
    ExpectRejected(options);
  }
}

TEST(WeightBiasQuantization, RejectsNonfiniteWeightsBiasAndDegenerateDerivedScales) {
  for (float value :
       {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN()}) {
    for (bool bias : {false, true}) {
      Options options;
      (bias ? options.bias : options.weight)[1] = value;
      ExpectRejected(options);
    }
  }
  Options options;
  options.weight.assign(6, std::numeric_limits<float>::denorm_min());
  ExpectRejected(options);
  options.weight.assign(6, std::numeric_limits<float>::min());
  ExpectRejected(options);
  options = {};
  options.activation.scale = std::numeric_limits<float>::max();
  options.weight.assign(6, 254);
  ExpectRejected(options);
  options = {};
  options.activation.scale = std::numeric_limits<float>::denorm_min();
  options.weight.assign(6, 1);
  ExpectRejected(options);
}

TEST(WeightBiasQuantization, ChecksInt32RangeBeforeChangingEitherWeightOrBias) {
  Options options;
  options.activation.scale = 1;
  options.bias = {-2147483648.0F, 2147483520.0F, 0};
  auto graph = MakeGraph(options);
  ExpectPrepared(*graph, options);
  const auto &dq = *Producer(*graph, Linear(*graph).input(2));
  EXPECT_EQ(Integers(Initializer(*graph, dq.input(0))),
            (std::vector<int32_t>{std::numeric_limits<int32_t>::min(), 2147483520, 0}));
  for (float value : {2147483648.0F, -2147483904.0F, std::numeric_limits<float>::max()}) {
    options.bias[1] = value;
    ExpectRejected(options);
  }
}

TEST(WeightBiasQuantization, BiasUsesFloat32ScaleProductAndFloat32Division) {
  Options options;
  options.activation.scale = 0.1F;
  options.weight.assign(6, 381.0F);
  // float32(0.1 * 3) is 0.3000000119; float32(0.45 / product) is
  // 1.49999988, not the mathematical half tie 1.5.
  options.bias = {0.45F, -0.45F, 0.75F};
  auto graph = MakeGraph(options);
  ExpectPrepared(*graph, options);
  const auto &dq = *Producer(*graph, Linear(*graph).input(2));
  EXPECT_EQ(Integers(Initializer(*graph, dq.input(0))), (std::vector<int32_t>{1, -1, 2}));
  EXPECT_FLOAT_EQ(rt::TensorFromProto(Initializer(*graph, dq.input(1))).AsFloat()[0], 0.3F);
}

TEST(WeightBiasQuantization, UsesNearestEvenAndRestoresCallerRoundingMode) {
  const int previous = std::fegetround();
  ASSERT_NE(previous, -1);
  struct RestoreRounding {
    int mode;
    ~RestoreRounding() { std::fesetround(mode); }
  } restore{previous};
  for (int mode : {FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
    Options options;
    options.weight = {127, -63.5F, 31.5F, -31.5F, 2.5F, -2.5F};
    options.bias = {0.25F, 0.75F, -1.25F};
    auto graph = MakeGraph(options);
    ASSERT_EQ(std::fesetround(mode), 0);
    onnx_light_cpu::PrepareWeightBiasQuantization(*graph);
    EXPECT_EQ(std::fegetround(), mode);
    const auto *weight = Producer(*graph, Linear(*graph).input(1));
    const auto *bias = Producer(*graph, Linear(*graph).input(2));
    ASSERT_NE(weight, nullptr);
    ASSERT_NE(bias, nullptr);
    EXPECT_EQ(Integers(Initializer(*graph, weight->input(0))),
              (std::vector<int32_t>{127, -64, 32, -32, 2, -2}));
    EXPECT_EQ(Integers(Initializer(*graph, bias->input(0))), (std::vector<int32_t>{0, 2, -2}));
  }
}

TEST(WeightBiasQuantization, ValidatesConvolutionAttributesAndPreservesSupportedValues) {
  for (const std::string op : {"Conv", "ConvTranspose"}) {
    for (int rank : {3, 4}) {
      Options options;
      options.op = op;
      options.rank = rank;
      AddAttribute(options.attributes, "group", int64_t{1});
      AddAttribute(options.attributes, "kernel_shape", std::vector<int64_t>(rank - 2, 1));
      AddAttribute(options.attributes, "strides", std::vector<int64_t>(rank - 2, 2));
      AddAttribute(options.attributes, "dilations", std::vector<int64_t>(rank - 2, 2));
      AddAttribute(options.attributes, "pads", std::vector<int64_t>(2 * (rank - 2), 1));
      AddAttribute(options.attributes, "auto_pad", std::string("NOTSET"));
      auto graph = MakeGraph(options);
      ExpectPrepared(*graph, options);
    }
    for (int invalid = 0; invalid < 12; ++invalid) {
      Options options;
      options.op = op;
      switch (invalid) {
      case 0:
        AddAttribute(options.attributes, "group", int64_t{2});
        break;
      case 1:
        AddAttribute(options.attributes, "auto_pad", std::string("SAME_UPPER"));
        break;
      case 2:
        AddAttribute(options.attributes, "kernel_shape", std::vector<int64_t>{2});
        break;
      case 3:
        AddAttribute(options.attributes, "strides", std::vector<int64_t>{0});
        break;
      case 4:
        AddAttribute(options.attributes, "dilations", std::vector<int64_t>{-1});
        break;
      case 5:
        AddAttribute(options.attributes, "pads", std::vector<int64_t>{-1, 0});
        break;
      case 6:
        AddAttribute(options.attributes, "pads", std::vector<int64_t>{0});
        break;
      case 7:
        AddAttribute(options.attributes, "strides", std::vector<int64_t>{1, 1});
        break;
      case 8:
        AddAttribute(options.attributes, "dilations", std::vector<int64_t>{1, 1});
        break;
      case 9:
        AddAttribute(options.attributes, "kernel_shape", std::vector<int64_t>{1, 1});
        break;
      case 10:
        AddAttribute(options.attributes, "output_padding", std::vector<int64_t>{0});
        break;
      case 11:
        AddAttribute(options.attributes, "output_shape", std::vector<int64_t>{4});
        break;
      }
      SCOPED_TRACE(op + "/" + std::to_string(invalid));
      ExpectRejected(options);
    }
  }
}

TEST(WeightBiasQuantization, ValidatesGemmTransposeAlphaBetaAndBiasShape) {
  for (int64_t trans_a : {0, 1}) {
    for (int64_t trans_b : {0, 1}) {
      Options options;
      options.op = "Gemm";
      AddAttribute(options.attributes, "transA", trans_a);
      AddAttribute(options.attributes, "transB", trans_b);
      AddAttribute(options.attributes, "alpha", 1.0F);
      AddAttribute(options.attributes, "beta", 1.0F);
      auto graph = MakeGraph(options);
      ExpectPrepared(*graph, options);
    }
  }
  for (const std::string attribute : {"transA", "transB", "alpha", "beta"}) {
    Options options;
    options.op = "Gemm";
    if (attribute == "transA" || attribute == "transB") {
      AddAttribute(options.attributes, attribute.c_str(), int64_t{2});
    } else {
      AddAttribute(options.attributes, attribute.c_str(), 0.5F);
    }
    ExpectRejected(options);
  }
  for (const std::string op : {"Conv", "ConvTranspose", "Gemm"}) {
    Options options;
    options.op = op;
    options.bias_shape = {1, 3};
    ExpectRejected(options);
    options.bias_shape = {1};
    options.bias = {1};
    ExpectRejected(options);
  }
}

TEST(WeightBiasQuantization, RejectsUnsupportedRanks) {
  for (const std::string op : {"Conv", "ConvTranspose"}) {
    Options options;
    options.op = op;
    options.rank = 5;
    options.activation_shape = {1, 2, 1, 2, 2};
    ExpectRejected(options);
    options = {};
    options.op = op;
    options.activation_shape = {1, 2};
    ExpectRejected(options);
  }
  Options options;
  options.op = "Gemm";
  options.activation_shape = {1, 2, 2};
  ExpectRejected(options);
  options.activation_shape.clear();
  options.weight_shape = {1, 2, 3};
  ExpectRejected(options);
}

rt::Tensors RuntimeInputs(const Options &options, const std::vector<int8_t> &values) {
  if (options.activation.type == DT::UINT8) {
    std::vector<uint8_t> unsigned_values;
    for (int8_t value : values) {
      unsigned_values.push_back(
          static_cast<uint8_t>(static_cast<int>(value) + options.activation.zero));
    }
    return {rt::Tensor::FromUint8("Aq", ActivationShape(options), unsigned_values)};
  }
  return {rt::Tensor::FromInt8("Aq", ActivationShape(options), values)};
}

void ExpectSameRuntimeOutput(const rt::Tensors &actual, const rt::Tensors &expected,
                             int tolerance = 0) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    ASSERT_EQ(actual[i].data_type, expected[i].data_type);
    ASSERT_EQ(actual[i].shape, expected[i].shape);
    ASSERT_EQ(actual[i].size_bytes(), expected[i].size_bytes());
    for (std::size_t j = 0; j < actual[i].size_bytes(); ++j) {
      const int a = actual[i].data_type == rt::DataType::INT8
                        ? static_cast<int8_t>(actual[i].bytes()[j])
                        : actual[i].bytes()[j];
      const int b = expected[i].data_type == rt::DataType::INT8
                        ? static_cast<int8_t>(expected[i].bytes()[j])
                        : expected[i].bytes()[j];
      EXPECT_LE(std::abs(a - b), tolerance) << "output=" << i << " element=" << j;
    }
  }
}

TEST(WeightBiasQuantization, EndToEndMatchesIndependentInitializerGraphAndDisabledOriginal) {
  for (const std::string op : {"Conv", "ConvTranspose", "Gemm"}) {
    for (int rank : (op == "Gemm" ? std::vector<int>{2} : std::vector<int>{3, 4})) {
      for (const std::string post_op : {"", "Relu", "Clip"}) {
        for (bool inexact : {false, true}) {
          SCOPED_TRACE(op + "/" + std::to_string(rank) + "/" + post_op + "/" +
                       std::to_string(inexact));
          Options options;
          options.op = op;
          options.rank = rank;
          options.post_op = post_op;
          options.output.scale = 1;
          if (inexact) {
            options.weight = {127, -63.5F, 31.5F, 0, 2.5F, -2.5F};
            options.bias = {0.25F, 0.75F, -1.25F};
          }
          auto original = MakeGraph(options);
          auto expected_options = options;
          expected_options.weight_dq = true;
          expected_options.bias_dq = true;
          if (inexact) {
            expected_options.integer_weight = {127, -64, 32, 0, 2, -2};
            expected_options.integer_bias = {0, 2, -2};
          }
          auto expected = MakeGraph(expected_options);
          const auto inputs =
              RuntimeInputs(options, op == "Gemm" ? std::vector<int8_t>{-3, 1, 2, -1}
                                                  : std::vector<int8_t>{-3, 1, 2, -1, 2, -2, 1, 3});
          // Both reference executions use built-in kernels on manually supplied
          // initializers, independently of the stage's quantization/folding code.
          const auto disabled = rt::RunModel(original->ToModel(), inputs);
          const auto reference = rt::RunModel(expected->ToModel(), inputs);
          ExpectPrepared(*original, options);
          const auto enabled = rt::RunModel(original->ToModel(), inputs);
          ExpectSameRuntimeOutput(enabled, reference);
          // |activation| <= 1.5, reduction width 2, weight error <= .5,
          // bias error <= .25, and final Q rounding adds at most one unit:
          // ceil(2 * 1.5 * .5 + .25 + 1) = 3 output quantization units.
          ExpectSameRuntimeOutput(enabled, disabled, inexact ? 3 : 0);
        }
      }
    }
  }
}

TEST(WeightBiasQuantization, EndToEndSignedUnsignedZeroPointsAndFinalQuantizerSaturation) {
  for (const std::string op : {"Conv", "ConvTranspose", "Gemm"}) {
    for (DT input_type : {DT::INT8, DT::UINT8}) {
      for (DT output_type : {DT::INT8, DT::UINT8}) {
        Options options;
        options.op = op;
        options.activation.type = input_type;
        options.activation.zero = input_type == DT::UINT8 ? 128 : 0;
        options.output.type = output_type;
        options.output.zero = output_type == DT::UINT8 ? 128 : -3;
        options.output.scale = 0.125F;
        auto graph = MakeGraph(options);
        auto expected_options = options;
        expected_options.weight_dq = true;
        expected_options.bias_dq = true;
        auto expected = MakeGraph(expected_options);
        const auto inputs =
            RuntimeInputs(options, op == "Gemm" ? std::vector<int8_t>{-100, 100, 100, -100}
                                                : std::vector<int8_t>{-100, 100, -100, 100, 100,
                                                                      -100, 100, -100});
        const auto disabled = rt::RunModel(graph->ToModel(), inputs);
        const auto reference = rt::RunModel(expected->ToModel(), inputs);
        ExpectPrepared(*graph, options);
        const auto enabled = rt::RunModel(graph->ToModel(), inputs);
        ExpectSameRuntimeOutput(enabled, reference);
        ExpectSameRuntimeOutput(enabled, disabled);
        ASSERT_EQ(enabled.size(), 1U);
        const auto minimum = output_type == DT::INT8 ? uint8_t{128} : uint8_t{0};
        const auto maximum = output_type == DT::INT8 ? uint8_t{127} : uint8_t{255};
        const auto *begin = enabled[0].bytes();
        const auto *end = begin + enabled[0].size_bytes();
        EXPECT_NE(std::find(begin, end, minimum), end);
        EXPECT_NE(std::find(begin, end, maximum), end);
      }
    }
  }
}

TEST(WeightBiasQuantization, EndToEndGemmTransposeAndOptionalBiasUseIndependentReference) {
  for (int64_t trans_a : {0, 1}) {
    for (int64_t trans_b : {0, 1}) {
      for (bool bias : {false, true}) {
        Options options;
        options.op = "Gemm";
        options.with_bias = bias;
        AddAttribute(options.attributes, "transA", trans_a);
        AddAttribute(options.attributes, "transB", trans_b);
        auto graph = MakeGraph(options);
        auto expected_options = options;
        expected_options.weight_dq = true;
        expected_options.bias_dq = bias;
        auto expected = MakeGraph(expected_options);
        const auto inputs = RuntimeInputs(options, {-3, 1, 2, -1});
        const auto disabled = rt::RunModel(graph->ToModel(), inputs);
        const auto reference = rt::RunModel(expected->ToModel(), inputs);
        ExpectPrepared(*graph, options);
        const auto enabled = rt::RunModel(graph->ToModel(), inputs);
        ExpectSameRuntimeOutput(enabled, reference);
        ExpectSameRuntimeOutput(enabled, disabled);
      }
    }
  }
}

TEST(WeightBiasQuantization, EndToEndOmittedZeroPointsAndZeroWeights) {
  for (const std::string op : {"Conv", "ConvTranspose", "Gemm"}) {
    for (DT input_type : {DT::INT8, DT::UINT8}) {
      Options options;
      options.op = op;
      options.activation.type = input_type;
      options.activation.omit_zero = true;
      options.output.omit_zero = true;
      options.weight.assign(6, 0);
      auto graph = MakeGraph(options);
      auto expected_options = options;
      expected_options.weight_dq = true;
      expected_options.bias_dq = true;
      expected_options.integer_weight.assign(6, 0);
      expected_options.weight_quantization.omit_zero = true;
      auto expected = MakeGraph(expected_options);
      const auto inputs =
          RuntimeInputs(options, op == "Gemm" ? std::vector<int8_t>{1, 3, 2, 1}
                                              : std::vector<int8_t>{1, 3, 2, 1, 2, 2, 1, 3});
      const auto disabled = rt::RunModel(graph->ToModel(), inputs);
      const auto reference = rt::RunModel(expected->ToModel(), inputs);
      ExpectPrepared(*graph, options);
      const auto enabled = rt::RunModel(graph->ToModel(), inputs);
      ExpectSameRuntimeOutput(enabled, reference);
      ExpectSameRuntimeOutput(enabled, disabled);
    }
  }
}

TEST(WeightBiasQuantization, EndToEndAcceleratedCpuGemmMatchesIndependentReference) {
  Options options;
  options.op = "Gemm";
  options.output.scale = 1;
  options.weight = {127, -63.5F, 31.5F, 0, 2.5F, -2.5F};
  options.bias = {0.25F, 0.75F, -1.25F};
  AddAttribute(options.attributes, "transA", int64_t{1});
  AddAttribute(options.attributes, "transB", int64_t{1});
  auto graph = MakeGraph(options);
  auto expected_options = options;
  expected_options.weight_dq = true;
  expected_options.bias_dq = true;
  expected_options.integer_weight = {127, -64, 32, 0, 2, -2};
  expected_options.integer_bias = {0, 2, -2};
  auto expected = MakeGraph(expected_options);
  const auto inputs = RuntimeInputs(options, {-3, 1, 2, -1});
  const auto disabled = rt::RunModel(graph->ToModel(), inputs);
  const auto reference = rt::RunModel(expected->ToModel(), inputs);
  ExpectPrepared(*graph, options);

  onnx_light_cpu::RegisterGemmKernel();
  const auto model = graph->ToModel();
  const rt::OpsetImports opsets = {{"", 23}, {"ai.onnx", 23}};
  rt::RuntimeContext runtime{rt::KernelContext(rt::DefaultOpset(23), nullptr, &opsets)};
  runtime.set_kernel_usage_enabled(true);
  rt::SubgraphSession session(model.graph(), &opsets);
  const auto enabled = session.Run({{"Aq", inputs[0]}}, runtime);
  const auto usage = runtime.GetKernelUsage();
  EXPECT_NE(std::find(usage.begin(), usage.end(), onnx_light_cpu::GemmKernel::kName), usage.end());
  ExpectSameRuntimeOutput(enabled, reference);
  // Same bound as the non-accelerated inexact-weight cases above.
  ExpectSameRuntimeOutput(enabled, disabled, 3);
}
} // namespace
