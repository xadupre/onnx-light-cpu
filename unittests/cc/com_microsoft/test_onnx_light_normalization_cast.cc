// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/com_microsoft/naive_skip_simplified_layer_normalization_kernel.h"
#include "onnx_light_cpu/kernels/com_microsoft/skip_simplified_layer_normalization_kernel.h"
#include "onnx_light_cpu/kernels/tensor/cast_kernel.h"
#include "onnx_light_cpu/patterns/com_microsoft/patterns.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"
#include "onnx_light_cpu/shapes/com_microsoft/shape_inference.h"

#include "onnx_core/builder/graph_builder.h"
#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"
#include "onnx_extensions/shapes/dispatch_table.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {
namespace builder = ONNX_LIGHT_NAMESPACE::core::builder;
namespace rt = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym = ONNX_LIGHT_NAMESPACE::core::symbolic;
using ONNX_LIGHT_NAMESPACE::AddAttribute;
using ONNX_LIGHT_NAMESPACE::NodeProto;
using ONNX_LIGHT_NAMESPACE::TensorProto;
using Pattern = onnx_light_cpu::NormalizationCastEliminationPattern;
constexpr auto kKernel = onnx_light_cpu::NaiveSkipSimplifiedLayerNormalizationKernel::kName;

struct Options {
  sym::TensorType type = sym::TensorType::kFloat16;
  int width = 4;
  int skip_layout = 0;
  bool constant_weight = false;
  float weight = 1.0F;
  bool raw_weight = false;
  bool bias = false;
  bool statistics = false;
  bool residual = false;
  bool float_residual = false;
  bool shared_input = false;
  bool shared_output = false;
};

NodeProto CastSpec(TensorProto::DataType type) {
  NodeProto spec;
  AddAttribute(spec, "to", static_cast<int64_t>(type));
  return spec;
}

std::unique_ptr<builder::GraphBuilder> MakeGraph(const Options &options = {}) {
  ONNX_LIGHT_NAMESPACE::onnx_kernels::RegisterKernelFunctions();
  ONNX_LIGHT_NAMESPACE::onnx_shapes::RegisterShapeFunctions();
  onnx_light_cpu::RegisterMicrosoftShapeAndMemoryFunctions();
  auto graph =
      std::make_unique<builder::GraphBuilder>("normalization_cast", [](const std::string &op) {
        auto schemas = ONNX_LIGHT_NAMESPACE::onnx_op::GetAllOnnxOpSchemasWithHistory(op, false);
        const auto custom = onnx_light_cpu::GetMicrosoftOpSchemasWithHistory(op, false);
        schemas.insert(schemas.end(), custom.begin(), custom.end());
        return schemas;
      });
  graph->SetOpsetVersion("", 23);
  graph->SetOpsetVersion("com.microsoft", 1);
  graph->set_device(sym::Device::kCPU);
  const auto float_cast = CastSpec(TensorProto::DataType::FLOAT);
  const auto half_cast = CastSpec(TensorProto::DataType::FLOAT16);
  for (const char *name : {"X", "R", "W", "B"}) {
    if (std::string(name) == "W" && options.constant_weight) {
      TensorProto weight;
      weight.set_name("W32");
      weight.set_data_type(TensorProto::DataType::FLOAT);
      weight.add_dims(options.width);
      for (int i = 0; i < options.width; ++i) {
        if (options.raw_weight) {
          const auto bits = std::bit_cast<uint32_t>(options.weight);
          for (int byte = 0; byte < 4; ++byte) {
            weight.ref_raw_data().push_back(static_cast<uint8_t>(bits >> (8 * byte)));
          }
        } else {
          weight.ref_float_data().push_back(options.weight);
        }
      }
      graph->MakeInitializer(weight);
      continue;
    }
    if (std::string(name) == "B" && !options.bias) {
      continue;
    }
    auto shape = sym::SymShape({options.width});
    if (std::string(name) == "X") {
      shape = options.skip_layout == 0 ? sym::SymShape({2, options.width})
                                       : sym::SymShape({2, 2, options.width});
    } else if (std::string(name) == "R") {
      shape = options.skip_layout == 2 ? sym::SymShape({1, 2, options.width})
                                       : sym::SymShape({2, options.width});
    }
    graph->MakeInput(name, options.type, shape);
    graph->MakeNode("Cast", {name}, {std::string(name) + "32"}, "", "", float_cast.attribute());
  }
  NodeProto spec;
  AddAttribute(spec, "epsilon", 1.0e-5F);
  graph->MakeNode("SkipSimplifiedLayerNormalization",
                  options.bias ? std::vector<std::string>{"X32", "R32", "W32", "B32"}
                               : std::vector<std::string>{"X32", "R32", "W32"},
                  {"N", options.statistics ? "mean" : "", options.statistics ? "inverse" : "",
                   options.residual ? "sum" : ""},
                  "com.microsoft", "normalize", spec.attribute());
  graph->MakeNode("Cast", {"N"}, {"Y"}, "", "", half_cast.attribute());
  graph->MakeOutput("Y");
  if (options.residual) {
    if (options.float_residual) {
      graph->MakeOutput("sum");
    } else {
      graph->MakeNode("Cast", {"sum"}, {"S"}, "", "", half_cast.attribute());
      graph->MakeOutput("S");
    }
  }
  if (options.statistics) {
    graph->MakeOutput("mean");
    graph->MakeOutput("inverse");
  }
  if (options.shared_input) {
    graph->MakeOutput("X32");
  }
  if (options.shared_output) {
    graph->MakeOutput("N");
  }
  return graph;
}

std::vector<std::unique_ptr<builder::PatternOptimization>>
Patterns(const std::string &kernel = kKernel) {
  std::vector<std::unique_ptr<builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<Pattern>(kernel));
  return patterns;
}

const NodeProto &Normalization(const builder::GraphBuilder &graph) {
  for (const auto &node : graph.Nodes()) {
    if (node.op_type() == "SkipSimplifiedLayerNormalization") {
      return node;
    }
  }
  throw std::runtime_error("missing normalization");
}

std::vector<uint8_t> Bytes(const rt::Tensor &tensor) {
  return {tensor.bytes(), tensor.bytes() + tensor.size_bytes()};
}

TEST(NormalizationCast, RemovesCastsPreservesAttributesNamesAndOptionalOutputs) {
  for (bool constant : {false, true}) {
    for (bool raw : {false, true}) {
      for (bool optional : {false, true}) {
        auto graph = MakeGraph({.constant_weight = constant,
                                .raw_weight = raw,
                                .bias = optional,
                                .statistics = optional,
                                .residual = optional});
        builder::GraphGraph optimizer(*graph, Patterns());
        const auto original = Normalization(*graph);
        Pattern pattern(kKernel);
        const auto match = pattern.Match(optimizer, Normalization(*graph));
        ASSERT_NE(match.pattern, nullptr) << match.ToString();
        optimizer.Optimize();
        ASSERT_EQ(graph->Nodes().size(), 1U);
        const auto &node = graph->Nodes()[0];
        EXPECT_EQ(node.domain(), "com.microsoft");
        EXPECT_EQ(node.input(0), "X");
        EXPECT_EQ(node.input(1), "R");
        EXPECT_EQ(node.output(0), "Y");
        EXPECT_EQ(node.output(1), original.output(1));
        EXPECT_EQ(node.output(2), original.output(2));
        EXPECT_EQ(node.output(3), optional ? "S" : "");
        ASSERT_EQ(node.attribute().size(), 1U);
        EXPECT_EQ(node.attribute()[0].name(), "epsilon");
        EXPECT_FLOAT_EQ(node.attribute()[0].f(), 1.0e-5F);
        EXPECT_EQ(graph->GetShape("Y").Dtype(), sym::TensorType::kFloat16);
        if (constant) {
          ASSERT_EQ(graph->Initializers().size(), 1U);
          EXPECT_EQ(graph->Initializers()[0].data_type(), TensorProto::DataType::FLOAT16);
        }
      }
    }
  }
}

TEST(NormalizationCast, PreservesOtherInputCastConsumers) {
  auto graph = MakeGraph({.shared_input = true});
  builder::GraphGraph(*graph, Patterns()).Optimize();
  ASSERT_EQ(graph->Nodes().size(), 2U);
  EXPECT_EQ(graph->Nodes()[0].op_type(), "Cast");
  EXPECT_EQ(graph->Nodes()[0].output(0), "X32");
  EXPECT_EQ(Normalization(*graph).input(0), "X");
}

TEST(NormalizationCast, UnsupportedContractsAndTargetsLeaveGraphUnchanged) {
  for (auto device : {sym::Device::kCPU, sym::Device::kUndefined, sym::Device::kGPU0}) {
    for (const std::string kernel :
         {"", "unknown", onnx_light_cpu::SkipSimplifiedLayerNormalizationKernel::kName}) {
      auto graph = MakeGraph();
      graph->set_device(device);
      const auto before = graph->ToGraph().SerializeAsString();
      builder::GraphGraph(*graph, Patterns(kernel)).Optimize();
      EXPECT_EQ(graph->ToGraph().SerializeAsString(), before);
    }
  }
  for (auto device : {sym::Device::kUndefined, sym::Device::kGPU0}) {
    auto graph = MakeGraph();
    graph->set_device(device);
    builder::GraphGraph optimizer(*graph);
    Pattern pattern(kKernel);
    EXPECT_EQ(pattern.Match(optimizer, Normalization(*graph)).pattern, nullptr);
  }
}

TEST(NormalizationCast, RejectsRoundingTypeAndConsumerChanges) {
  for (const Options &options :
       {Options{.type = sym::TensorType::kBfloat16}, Options{.type = sym::TensorType::kDouble},
        Options{.constant_weight = true, .weight = 1.0001F},
        Options{.constant_weight = true, .weight = 1.0001F, .raw_weight = true},
        Options{.constant_weight = true, .weight = 70000.0F},
        Options{.constant_weight = true, .weight = std::numeric_limits<float>::infinity()},
        Options{.constant_weight = true, .weight = std::numeric_limits<float>::quiet_NaN()},
        Options{.residual = true, .float_residual = true}, Options{.shared_output = true}}) {
    auto graph = MakeGraph(options);
    const auto before = graph->ToGraph().SerializeAsString();
    builder::GraphGraph(*graph, Patterns()).Optimize();
    EXPECT_EQ(graph->ToGraph().SerializeAsString(), before);
  }
}

TEST(NormalizationCast, RejectsSchemaAttributesDomainsTypesAndShapes) {
  auto graph = MakeGraph();
  builder::GraphGraph optimizer(*graph);
  Pattern pattern(kKernel);
  const auto original = Normalization(*graph);
  for (const char *domain : {"", "ai.onnx", "other"}) {
    auto node = original;
    node.set_domain(domain);
    EXPECT_EQ(pattern.Match(optimizer, node).pattern, nullptr);
  }
  for (const char *op : {"SimplifiedLayerNormalization", "SkipLayerNormalization"}) {
    auto node = original;
    node.set_op_type(op);
    EXPECT_EQ(pattern.Match(optimizer, node).pattern, nullptr);
  }
  for (const char *attribute : {"stash_type", "axis", "unsupported"}) {
    auto node = original;
    AddAttribute(node, attribute, int64_t{1});
    EXPECT_EQ(pattern.Match(optimizer, node).pattern, nullptr);
  }
  for (std::size_t slot = 0; slot < 3; ++slot) {
    auto node = original;
    node.ref_input()[slot] = "X";
    EXPECT_EQ(pattern.Match(optimizer, node).pattern, nullptr);
  }
  auto node = original;
  node.ref_input()[2] = "X32";
  EXPECT_EQ(pattern.Match(optimizer, node).pattern, nullptr);
  node = original;
  node.ref_attribute()[0].set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::AttributeType::INT);
  EXPECT_EQ(pattern.Match(optimizer, node).pattern, nullptr);
  graph->SetOpsetVersion("com.microsoft", 2);
  EXPECT_EQ(pattern.Match(optimizer, original).pattern, nullptr);
}

TEST(NormalizationCast, OriginalAndRewrittenGraphsHaveIdenticalBytes) {
  onnx_light_cpu::RegisterCastKernel();
  onnx_light_cpu::RegisterNaiveSkipSimplifiedLayerNormalizationKernel();
  for (int width : {1, 7, 32, 33, 129}) {
    for (bool constant : {false, true}) {
      for (float magnitude : {0.0001F, 1.0F, 10000.0F}) {
        auto graph = MakeGraph({.width = width,
                                .constant_weight = constant,
                                .bias = true,
                                .statistics = true,
                                .residual = true});
        rt::Tensors inputs;
        std::vector<float> x(2 * width), residual(2 * width), weight(width, 1.0F),
            bias(width, 0.0003F);
        for (int i = 0; i < 2 * width; ++i) {
          x[i] = magnitude;
          residual[i] = (i % 3 == 0 ? 1.0F : -1.0F) * 0.0003F;
        }
        inputs.push_back(rt::MakeFloat16Tensor("X", {2, width}, x));
        inputs.push_back(rt::MakeFloat16Tensor("R", {2, width}, residual));
        inputs.push_back(rt::MakeFloat16Tensor("W", {width}, weight));
        inputs.push_back(rt::MakeFloat16Tensor("B", {width}, bias));
        const auto original = rt::RunModel(graph->ToModel(), inputs);
        builder::GraphGraph(*graph, Patterns()).Optimize();
        ASSERT_EQ(graph->Nodes().size(), 1U);
        const auto rewritten = rt::RunModel(graph->ToModel(), inputs);
        ASSERT_EQ(original.size(), rewritten.size());
        for (std::size_t i = 0; i < original.size(); ++i) {
          EXPECT_EQ(original[i].data_type, rewritten[i].data_type);
          EXPECT_EQ(original[i].shape, rewritten[i].shape);
          EXPECT_EQ(Bytes(original[i]), Bytes(rewritten[i]))
              << "width=" << width << " magnitude=" << magnitude << " output=" << i;
        }
      }
    }
  }
}

TEST(NormalizationCast, BroadcastResidualRetainsFloatAdditionAndHalfRounding) {
  onnx_light_cpu::RegisterCastKernel();
  onnx_light_cpu::RegisterNaiveSkipSimplifiedLayerNormalizationKernel();
  for (int layout : {1, 2}) {
    auto graph = MakeGraph({.skip_layout = layout,
                            .constant_weight = true,
                            .weight = -0.0F,
                            .raw_weight = true,
                            .bias = true,
                            .statistics = true,
                            .residual = true});
    const rt::Tensors inputs = {
        rt::MakeFloat16Tensor("X", {2, 2, 4},
                              {10000, 10000, 10000, 10000, 0, 0.0001F, -0.0001F, 0, 1, 1, 1, 1,
                               65504, 65504, -65504, -65504}),
        rt::MakeFloat16Tensor("R", layout == 1 ? rt::Shape{2, 4} : rt::Shape{1, 2, 4},
                              {0.0003F, -0.0003F, 0.0001F, -0.0001F, 65504, 65504, 65504, 65504}),
        rt::MakeFloat16Tensor("B", {4}, {0.0001F, -0.0001F, 0.0001F, -0.0001F})};
    const auto original = rt::RunModel(graph->ToModel(), inputs);
    builder::GraphGraph(*graph, Patterns()).Optimize();
    ASSERT_EQ(graph->Nodes().size(), 1U);
    const auto rewritten = rt::RunModel(graph->ToModel(), inputs);
    ASSERT_EQ(original.size(), rewritten.size());
    for (std::size_t i = 0; i < original.size(); ++i) {
      EXPECT_EQ(original[i].data_type, rewritten[i].data_type);
      EXPECT_EQ(original[i].shape, rewritten[i].shape);
      EXPECT_EQ(Bytes(original[i]), Bytes(rewritten[i]));
    }
  }
}
} // namespace
