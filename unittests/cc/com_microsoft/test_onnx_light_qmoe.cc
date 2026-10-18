// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_light_cpu/kernels/com_microsoft/qmoe_kernel.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
using onnx_light_cpu::QMoEKernel;
using ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::Tensor;

NodeProto MakeNode(int64_t bits = 4, int64_t block = 32, bool fc3 = false) {
  NodeProto node;
  node.set_domain("com.microsoft");
  node.set_op_type("QMoE");
  for (const char *name : {"X", "R", "W1", "S1", "", "W2", "S2", "", fc3 ? "W3" : "",
                           fc3 ? "S3" : "", "", "", "", ""}) {
    node.add_input(name);
  }
  node.add_output("Y");
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "expert_weight_bits", bits);
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "block_size", block);
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "activation_type", std::string("identity"));
  return node;
}

const rt_ns::KernelContext &Context() {
  static const rt_ns::KernelContext context{rt_ns::OpsetId("com.microsoft", 1)};
  return context;
}

struct Inputs {
  Tensor x = Tensor::FromFloat("", {1, 2}, {2.0f, -1.0f});
  Tensor router = Tensor::FromFloat("", {1, 1}, {0.0f});
  // Blockwise matrices are input-major, offset-eight INT4, low nibble first.
  Tensor w1 = Tensor::FromUint8("", {1, 2, 1}, {0xA9, 0xCB});
  Tensor s1 = Tensor::FromFloat("", {1, 2, 1}, {0.5f, 0.25f});
  Tensor w2 = Tensor::FromUint8("", {1, 2, 1}, {0xA9, 0xCB});
  Tensor s2 = Tensor::FromFloat("", {1, 2, 1}, {0.5f, 0.25f});

  Tensor Run(const QMoEKernel &kernel) const {
    return kernel(x, router, w1, s1, nullptr, w2, s2, nullptr);
  }
};

TEST(QMoERuntime, BlockWeightsHaveIndependentExpectedValues) {
  Inputs inputs;
  const Tensor output = inputs.Run(QMoEKernel(MakeNode(), Context()));
  // X * [[.5,1],[.75,1]] = [.25,1], followed by the same FC2.
  EXPECT_FLOAT_EQ(output.AsFloat()[0], 0.875f);
  EXPECT_FLOAT_EQ(output.AsFloat()[1], 1.25f);
}

TEST(QMoERuntime, PreservesThreeDimensionalInputShape) {
  Inputs inputs;
  inputs.x = Tensor::FromFloat("", {1, 1, 2}, {2.0f, -1.0f});
  const Tensor output = inputs.Run(QMoEKernel(MakeNode(), Context()));
  EXPECT_EQ(output.shape, (rt_ns::Shape{1, 1, 2}));
  EXPECT_FLOAT_EQ(output.AsFloat()[0], 0.875f);
  EXPECT_FLOAT_EQ(output.AsFloat()[1], 1.25f);
}

TEST(QMoERuntime, AcceptsIndependentFloatingScaleTypes) {
  Inputs inputs;
  inputs.s1 = rt_ns::MakeFloat16Tensor("", {1, 2, 1}, {0.5f, 0.25f});
  inputs.s2 = rt_ns::MakeBfloat16Tensor("", {1, 2, 1}, {0.5f, 0.25f});
  const Tensor output = inputs.Run(QMoEKernel(MakeNode(), Context()));
  EXPECT_FLOAT_EQ(output.AsFloat()[0], 0.875f);
  EXPECT_FLOAT_EQ(output.AsFloat()[1], 1.25f);
}

TEST(QMoERuntime, BlockScalesApplyToStoredRowsRatherThanOutputColumns) {
  for (const int64_t block : {16, 32, 64}) {
    Inputs inputs;
    inputs.s1 = Tensor::FromFloat("", {1, 2, 1}, {0.5f, 0.25f});
    inputs.s2 = Tensor::FromFloat("", {1, 2, 1}, {0.5f, 0.25f});
    const Tensor output = inputs.Run(QMoEKernel(MakeNode(4, block), Context()));
    // X * [[.5,1],[.75,1]] = [.25,1], followed by the same FC2.
    EXPECT_FLOAT_EQ(output.AsFloat()[0], 0.875f);
    EXPECT_FLOAT_EQ(output.AsFloat()[1], 1.25f);
  }
}

TEST(QMoERuntime, AcceptsCanonicalOutputMajorTensorScalesAndUnsignedInt8) {
  for (const int64_t bits : {4, 8}) {
    Inputs inputs;
    inputs.x = Tensor::FromFloat("", {1, 3}, {2, 4, 6});
    inputs.w1 = bits == 4 ? Tensor::FromUint8("", {1, 2, 2}, {0x89, 0x08, 0x98, 0x08})
                          : Tensor::FromUint8("", {1, 2, 3}, {129, 128, 128, 128, 129, 128});
    inputs.s1 = Tensor::FromFloat("", {1, 2}, {0.5f, 0.25f});
    inputs.w2 = bits == 4 ? Tensor::FromUint8("", {1, 3, 1}, {0x89, 0x98, 0x99})
                          : Tensor::FromUint8("", {1, 3, 2}, {129, 128, 128, 129, 129, 129});
    inputs.s2 = Tensor::FromFloat("", {1, 3}, {1, 2, 3});
    const Tensor output = inputs.Run(QMoEKernel(MakeNode(bits, 0), Context()));
    EXPECT_FLOAT_EQ(output.AsFloat()[0], 1.0f);
    EXPECT_FLOAT_EQ(output.AsFloat()[1], 2.0f);
    EXPECT_FLOAT_EQ(output.AsFloat()[2], 6.0f);
  }
}

TEST(QMoERuntime, RejectsLegacyColumnScaledTensorPacking) {
  Inputs inputs;
  inputs.s1 = Tensor::FromFloat("", {1, 2}, {0.5f, 0.25f});
  inputs.s2 = Tensor::FromFloat("", {1, 3}, {1, 1, 1});
  inputs.x = Tensor::FromFloat("", {1, 3}, {1, 2, 3});
  inputs.w1 = Tensor::FromUint8("", {1, 3, 1}, {0x99, 0x99, 0x99});
  inputs.w2 = Tensor::FromUint8("", {1, 2, 2}, {0x99, 0x09, 0x99, 0x09});
  EXPECT_THROW(inputs.Run(QMoEKernel(MakeNode(4, 0), Context())), std::invalid_argument);
}

rt_ns::Tensors RunGraph(const NodeProto &node) {
  static const bool registered = [] {
    onnx_light_cpu::RegisterQMoEKernel();
    return true;
  }();
  (void)registered;
  ONNX_LIGHT_NAMESPACE::GraphProto graph;
  graph.set_name("qmoe_runtime");
  *graph.add_node() = node;
  for (const char *name : {"X", "R", "W1", "S1", "W2", "S2", "zero_points"}) {
    graph.add_input()->set_name(name);
  }
  graph.add_output()->set_name("Y");
  Inputs inputs;
  rt_ns::RuntimeContext runtime{Context()};
  rt_ns::SubgraphSession session(graph);
  auto outputs = session.Run({{"X", std::move(inputs.x)},
                              {"R", std::move(inputs.router)},
                              {"W1", std::move(inputs.w1)},
                              {"S1", std::move(inputs.s1)},
                              {"W2", std::move(inputs.w2)},
                              {"S2", std::move(inputs.s2)},
                              {"zero_points", Tensor::FromUint8("", {1}, {8})}},
                             runtime);
  for (auto &output : outputs) {
    output = output.ToOwned();
  }
  return outputs;
}

TEST(QMoERuntime, DispatchesThroughRunWithOptionalInputHoles) {
  const auto outputs = RunGraph(MakeNode());
  ASSERT_EQ(outputs.size(), 1U);
  EXPECT_FLOAT_EQ(outputs[0].AsFloat()[0], 0.875f);
  EXPECT_FLOAT_EQ(outputs[0].AsFloat()[1], 1.25f);
}

TEST(QMoERuntime, EmptyTokensPreserveShapeAndType) {
  for (const int64_t block : {16, 32, 64}) {
    Inputs inputs;
    inputs.x = Tensor::FromFloat("", {0, 2}, {});
    inputs.router = Tensor::FromFloat("", {0, 1}, {});
    const Tensor output = inputs.Run(QMoEKernel(MakeNode(4, block), Context()));
    EXPECT_EQ(output.shape, (rt_ns::Shape{0, 2}));
    EXPECT_EQ(output.data_type, static_cast<int32_t>(rt_ns::DataType::FLOAT));
  }
}

TEST(QMoERuntime, RejectsInvalidQuantizationAndRoutingAttributes) {
  for (const auto &entry :
       std::vector<std::pair<std::string, int64_t>>{{"expert_weight_bits", 3},
                                                    {"expert_weight_bits", 16},
                                                    {"block_size", -1},
                                                    {"block_size", 1},
                                                    {"block_size", 15},
                                                    {"block_size", 24},
                                                    {"k", 0},
                                                    {"k", -1},
                                                    {"weights_prepacked", 1},
                                                    {"use_sparse_mixer", 1},
                                                    {"normalize_routing_weights", 2},
                                                    {"swiglu_fusion", 1}}) {
    SCOPED_TRACE(entry.first);
    NodeProto node = MakeNode();
    for (auto &attribute : node.ref_attribute()) {
      if (attribute.name() == entry.first) {
        attribute.set_i(entry.second);
      }
    }
    if (entry.first != "expert_weight_bits" && entry.first != "block_size") {
      ONNX_LIGHT_NAMESPACE::AddAttribute(node, entry.first.c_str(), entry.second);
    }
    EXPECT_THROW(QMoEKernel(node, Context()), std::invalid_argument);
  }
}

TEST(QMoERuntime, RejectsInvalidActivationParameters) {
  for (const auto &entry : std::vector<std::pair<std::string, float>>{
           {"activation_alpha", std::numeric_limits<float>::quiet_NaN()},
           {"activation_beta", std::numeric_limits<float>::infinity()},
           {"swiglu_limit", -1.0f}}) {
    NodeProto node = MakeNode();
    ONNX_LIGHT_NAMESPACE::AddAttribute(node, entry.first.c_str(), entry.second);
    EXPECT_THROW(QMoEKernel(node, Context()), std::invalid_argument);
  }
}

TEST(QMoERuntime, RejectsUnsupportedActivationAndExplicitZeroPoints) {
  NodeProto node = MakeNode();
  for (auto &attribute : node.ref_attribute()) {
    if (attribute.name() == "activation_type") {
      attribute.set_s("not_an_activation");
    }
  }
  EXPECT_THROW(QMoEKernel(node, Context()), std::invalid_argument);
  for (int slot : {11, 12, 13}) {
    node = MakeNode();
    node.ref_input()[slot] = "zero_points";
    try {
      RunGraph(node);
      FAIL() << "Expected explicit zero points to be rejected";
    } catch (const std::exception &error) {
      EXPECT_NE(std::string(error.what()).find("zero points"), std::string::npos);
    }
  }
}

TEST(QMoERuntime, RejectsInputRouterAndWeightContractViolations) {
  const QMoEKernel kernel(MakeNode(), Context());
  for (int scenario = 0; scenario < 15; ++scenario) {
    SCOPED_TRACE(scenario);
    Inputs inputs;
    switch (scenario) {
    case 0:
      inputs.x = Tensor::FromFloat("", {2}, {1, 2});
      break;
    case 1:
      inputs.router = Tensor::FromFloat("", {2, 1}, {0, 0});
      break;
    case 2:
      inputs.router = Tensor::FromFloat("", {1, 2}, {0, 0});
      break;
    case 3:
      inputs.w1 = Tensor::FromUint8("", {2, 1}, {1, 2});
      break;
    case 4:
      inputs.w1 = Tensor::FromFloat("", {1, 2, 1}, {1, 2});
      break;
    case 5:
      inputs.w1 = Tensor::FromUint8("", {1, 3, 1}, {1, 2, 3});
      break;
    case 6:
      inputs.s1 = Tensor::FromFloat("", {1, 3}, {1, 1, 1});
      break;
    case 7:
      inputs.s2 = Tensor::FromFloat("", {1, 1}, {1});
      break;
    case 8:
      inputs.router = Tensor::FromInt32("", {1, 1}, {0});
      break;
    case 9:
      inputs.x = Tensor::FromFloat("", {1, 1, 1, 2}, {1, 2});
      break;
    case 10:
      inputs.x = Tensor::FromFloat("", {1, 0}, {});
      break;
    case 11:
      inputs.router = Tensor::FromFloat("", {1}, {0});
      break;
    case 12:
      inputs.s1 = Tensor::FromInt32("", {1, 2}, {1, 1});
      break;
    case 13:
      inputs.w2 = Tensor::FromUint8("", {1, 0, 1}, {});
      break;
    case 14:
      inputs.router = Tensor::FromFloat("", {1, 0}, {});
      break;
    }
    EXPECT_THROW(inputs.Run(kernel), std::invalid_argument);
  }
}

TEST(QMoERuntime, RejectsMalformedBiasAndUnpairedThirdProjection) {
  Inputs inputs;
  const QMoEKernel kernel(MakeNode(), Context());
  const Tensor bias = Tensor::FromFloat("", {1, 3}, {0, 0, 0});
  EXPECT_THROW(
      kernel(inputs.x, inputs.router, inputs.w1, inputs.s1, &bias, inputs.w2, inputs.s2, nullptr),
      std::invalid_argument);
  EXPECT_THROW(kernel(inputs.x, inputs.router, inputs.w1, inputs.s1, nullptr, inputs.w2, inputs.s2,
                      nullptr, &inputs.w1),
               std::invalid_argument);
  EXPECT_THROW(kernel(inputs.x, inputs.router, inputs.w1, inputs.s1, nullptr, inputs.w2, inputs.s2,
                      nullptr, nullptr, &inputs.s1),
               std::invalid_argument);
}

TEST(QMoERuntime, RejectsTopKLargerThanExpertCountAndInvalidBlockScales) {
  Inputs inputs;
  NodeProto node = MakeNode();
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "k", int64_t{2});
  EXPECT_THROW(inputs.Run(QMoEKernel(node, Context())), std::invalid_argument);
  node = MakeNode(4, 32);
  inputs.s1 = Tensor::FromFloat("", {1, 2}, {0.5f, 0.25f});
  EXPECT_THROW(inputs.Run(QMoEKernel(node, Context())), std::invalid_argument);
}

TEST(QMoEDequantization, SimdLayoutsPreserveQuantizationAndTails) {
  constexpr std::size_t experts = 2;
  constexpr std::size_t inputs = 35;
  constexpr std::size_t outputs = 27;
  for (const std::size_t bits : {4U, 8U}) {
    for (const std::size_t block_size : {0U, 16U}) {
      for (const bool output_major : {false, true}) {
        for (const bool unsigned_int8 : {false, true}) {
          if (bits == 4 && unsigned_int8) {
            continue;
          }
          onnx_light_cpu::QMoEAttributes attributes;
          attributes.expert_weight_bits = bits;
          attributes.block_size = block_size;
          const auto factor = 8 / bits;
          const auto stored_rows = output_major ? outputs : inputs;
          const auto columns = output_major ? inputs : outputs;
          const auto packed = (columns + factor - 1) / factor;
          const auto blocks = block_size == 0 ? 1 : (columns + block_size - 1) / block_size;
          std::vector<std::uint8_t> data(experts * stored_rows * packed);
          std::vector<float> scales(experts * stored_rows * blocks);
          for (std::size_t row = 0; row < experts * stored_rows; ++row) {
            for (std::size_t block = 0; block < blocks; ++block) {
              scales[row * blocks + block] = 0.25f * static_cast<float>(row + block + 1);
            }
            for (std::size_t column = 0; column < columns; ++column) {
              const auto index = row * columns + column;
              const int quantized = bits == 4 ? static_cast<int>(index % 16) - 8
                                              : static_cast<int>((index * 7) % 256) - 128;
              const auto encoded = bits == 4       ? static_cast<std::uint8_t>(quantized + 8)
                                   : unsigned_int8 ? static_cast<std::uint8_t>(quantized + 128)
                                                   : static_cast<std::uint8_t>(quantized);
              if (bits == 4) {
                data[row * packed + column / 2] |=
                    static_cast<std::uint8_t>(encoded << (4 * (column % 2)));
              } else {
                data[row * packed + column] = encoded;
              }
            }
          }
          onnx_light_cpu::QMoEWeight weight{
              data.data(),  scales.data(),   onnx_light_cpu::MoEDataType::kFloat32,
              output_major, block_size == 0, unsigned_int8};
          std::vector<float> actual(experts * outputs * inputs);
          onnx_light_cpu::DequantizeQMoEWeights(weight, actual.data(), experts, inputs, outputs,
                                                attributes);
          for (std::size_t row = 0; row < experts * stored_rows; ++row) {
            const auto expert = row / stored_rows;
            const auto stored_row = row % stored_rows;
            for (std::size_t column = 0; column < columns; ++column) {
              const auto byte = data[row * packed + column / factor];
              const int quantized = bits == 4
                                        ? static_cast<int>((byte >> (4 * (column % 2))) & 15) - 8
                                    : unsigned_int8 ? static_cast<int>(byte) - 128
                                    : byte < 128    ? static_cast<int>(byte)
                                                    : static_cast<int>(byte) - 256;
              const auto block = block_size == 0 ? 0 : column / block_size;
              const auto scale = scales[row * blocks + block];
              const auto output_index =
                  output_major ? row * inputs + column
                               : expert * outputs * inputs + column * inputs + stored_row;
              EXPECT_FLOAT_EQ(actual[output_index], static_cast<float>(quantized) * scale);
            }
          }
        }
      }
    }
  }
}

} // namespace
