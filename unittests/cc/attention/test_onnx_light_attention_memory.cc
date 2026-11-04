// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/attention/attention_kernel.h"
#include "onnx_light_cpu/kernels/register_kernels.h"
#include "onnx_light_cpu/shapes/ai_onnx/shape_inference.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

namespace shapes_ns = ONNX_LIGHT_NAMESPACE::core::shapes;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;
using ONNX_LIGHT_NAMESPACE::NodeProto;
using sym_ns::SymDim;
using sym_ns::SymShape;

void IntAttribute(NodeProto &node, const char *name, std::int64_t value) {
  auto *attribute = node.add_attribute();
  attribute->set_name(name);
  attribute->set_i(value);
}

NodeProto AttentionNode(bool present = true, bool qk = false) {
  NodeProto node;
  node.set_op_type("Attention");
  for (const char *name : {"Q", "K", "V", "", "past_key", "past_value"}) {
    node.add_input(name);
  }
  for (const char *name :
       {"Y", present ? "present_key" : "", present ? "present_value" : "", qk ? "QK" : ""}) {
    node.add_output(name);
  }
  IntAttribute(node, "q_num_heads", 16);
  IntAttribute(node, "kv_num_heads", 4);
  IntAttribute(node, "is_causal", 1);
  return node;
}

std::vector<SymShape> PrefillShapes(std::int64_t length = 32000, std::int64_t past = 0) {
  return {SymShape({1, length, 2048}), SymShape({1, length, 512}),
          SymShape({1, length, 512}),  SymShape({}),
          SymShape({1, 4, past, 128}), SymShape({1, 4, past, 128})};
}

std::int64_t Scratch(const NodeProto &node, const std::vector<SymShape> &shapes) {
  return shapes_ns::ComputePeakMemory(sym_ns::Device::kCPU, node, shapes).worst_case.AsInt();
}

TEST(AttentionMemory, RegisteredStreamingPrefillIsBoundedWithPresentOutputs) {
  onnx_light_cpu::RegisterAllKernels();
  onnx_light_cpu::RegisterExperimentalShapeAndMemoryFunctions();
  const auto node = AttentionNode();
  const auto bytes = Scratch(node, PrefillShapes());
  EXPECT_GT(bytes, 0);
  EXPECT_LT(bytes, 256'000'000);
  EXPECT_EQ(bytes, Scratch(node, PrefillShapes(64000)));
  EXPECT_FALSE(shapes_ns::ComputePeakMemory(sym_ns::Device::kCPU, node, PrefillShapes())
                   .average_case.has_value());

  auto no_present = AttentionNode(false);
  EXPECT_EQ(bytes, Scratch(no_present, PrefillShapes()));
  no_present.set_domain("ai.onnx");
  EXPECT_EQ(bytes, Scratch(no_present, PrefillShapes()));
}

TEST(AttentionMemory, ExplicitMaskConversionIsNotErased) {
  onnx_light_cpu::RegisterExperimentalShapeAndMemoryFunctions();
  auto node = AttentionNode();
  auto shapes = PrefillShapes();
  const auto unmasked = Scratch(node, shapes);
  node.set_input(3, "mask");
  shapes[3] = SymShape({1, 1, 32000, 32000});
  // The original 2.048 GB FP16 mask remains an input, not scratch. Its FP32
  // conversion is additional scratch regardless of whether scores are streamed.
  EXPECT_EQ(Scratch(node, shapes) - unmasked, 4'096'000'000);
  shapes[3] = SymShape({1, 1, 1, 32000});
  EXPECT_EQ(Scratch(node, shapes) - unmasked, 128'000);
}

TEST(AttentionMemory, MaterializedQkAndFp64IncludeConversionsButNotDeclaredOutputs) {
  onnx_light_cpu::RegisterExperimentalShapeAndMemoryFunctions();
  constexpr std::int64_t dense = 65'536'000'000;
  const auto shapes = PrefillShapes();
  auto node = AttentionNode(true, true);
  const auto qk = Scratch(node, shapes);
  EXPECT_GE(qk, dense);
  // One FP32 score conversion buffer, not an additional declared FP16 QK output.
  EXPECT_LT(qk, dense + 1'000'000'000);
  for (std::int64_t mode = 0; mode < 4; ++mode) {
    auto mode_node = node;
    IntAttribute(mode_node, "qk_matmul_output_mode", mode);
    EXPECT_EQ(qk, Scratch(mode_node, shapes));
  }
  node = AttentionNode();
  IntAttribute(node, "softmax_precision", 11);
  EXPECT_GE(Scratch(node, shapes), 2 * dense);
  node = AttentionNode();
  IntAttribute(node, "softmax_precision", 1);
  EXPECT_EQ(Scratch(node, shapes), Scratch(AttentionNode(), shapes));
}

TEST(AttentionMemory, Rank4CacheAndUnknownDimensions) {
  onnx_light_cpu::RegisterExperimentalShapeAndMemoryFunctions();
  const auto node = AttentionNode();
  auto shapes = PrefillShapes(128, 4096);
  const auto cached = Scratch(node, shapes);
  EXPECT_GT(cached, 0);
  shapes[0] = SymShape({1, 16, 128, 128});
  shapes[1] = SymShape({1, 4, 128, 128});
  shapes[2] = shapes[1];
  EXPECT_GT(Scratch(node, shapes), 0);
  shapes[0] = SymShape({SymDim(1), SymDim(16), SymDim("sequence"), SymDim(128)});
  EXPECT_TRUE(shapes_ns::ComputePeakMemory(sym_ns::Device::kCPU, node, shapes).worst_case.IsExpr());
  EXPECT_TRUE(shapes_ns::ComputePeakMemory(sym_ns::Device::kCPU, node, {}).worst_case.IsExpr());
  // Shape-only queries cannot resolve rank-3 head attributes/output wiring.
  EXPECT_TRUE(shapes_ns::ComputePeakMemory("", "Attention", sym_ns::Device::kCPU, shapes)
                  .worst_case.IsExpr());
}

TEST(AttentionMemory, InvalidDimensionsAndOverflowDoNotBecomeZero) {
  onnx_light_cpu::RegisterExperimentalShapeAndMemoryFunctions();
  auto shapes = PrefillShapes();
  shapes[0] = SymShape({1, -1, 2048});
  EXPECT_THROW(Scratch(AttentionNode(), shapes), std::invalid_argument);
  auto node = AttentionNode();
  node.set_input(3, "mask");
  shapes = PrefillShapes();
  shapes[3] = SymShape({1, 1, std::numeric_limits<std::int64_t>::max(), 32000});
  EXPECT_THROW(Scratch(node, shapes), std::invalid_argument);
}

} // namespace
