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
  attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::INT);
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

std::int64_t Peak(const NodeProto &node, const std::vector<SymShape> &shapes) {
  return shapes_ns::ComputePeakMemory(sym_ns::Device::kCPU, node, shapes).worst_case.AsInt();
}

TEST(AttentionMemory, RegisteredStreamingPrefillIsBoundedWithPresentOutputs) {
  onnx_light_cpu::RegisterAllKernels();
  onnx_light_cpu::RegisterExperimentalShapeAndMemoryFunctions();
  const auto node = AttentionNode();
  const auto bytes = Peak(node, PrefillShapes());
  EXPECT_GT(bytes, 0);
  EXPECT_LT(bytes, 256'000'000);
  EXPECT_EQ(bytes, Peak(node, PrefillShapes(64000)));
  EXPECT_FALSE(shapes_ns::ComputePeakMemory(sym_ns::Device::kCPU, node, PrefillShapes())
                   .average_case.has_value());

  auto no_present = AttentionNode(false);
  EXPECT_EQ(bytes, Peak(no_present, PrefillShapes()));
  no_present.set_domain("ai.onnx");
  EXPECT_EQ(bytes, Peak(no_present, PrefillShapes()));
}

TEST(AttentionMemory, SingleKeyHeadSinkDoesNotReportZeroPeak) {
  onnx_light_cpu::RegisterExperimentalShapeAndMemoryFunctions();
  auto node = AttentionNode(false);
  auto shapes = PrefillShapes(1);
  const auto metadata_peak = Peak(node, shapes);
  EXPECT_GT(metadata_peak, 0);
  node.add_input("");
  node.add_input("head_sink");
  shapes.emplace_back(SymShape({}));
  shapes.emplace_back(SymShape({16}));
  EXPECT_EQ(Peak(node, shapes), metadata_peak + 64);
}

TEST(AttentionMemory, HeadSinkIsCountedAcrossExecutionPaths) {
  onnx_light_cpu::RegisterExperimentalShapeAndMemoryFunctions();
  for (bool present : {false, true}) {
    for (bool qk : {false, true}) {
      for (std::int64_t precision : {1, 11}) {
        for (std::int64_t length : {0, 1, 128}) {
          SCOPED_TRACE(::testing::Message()
                       << present << " " << qk << " " << precision << " " << length);
          auto node = AttentionNode(present, qk);
          IntAttribute(node, "softmax_precision", precision);
          auto shapes = PrefillShapes(length);
          const auto without_sink = Peak(node, shapes);
          node.add_input("");
          node.add_input("");
          shapes.emplace_back(SymShape({}));
          shapes.emplace_back(SymShape({16}));
          EXPECT_EQ(Peak(node, shapes), without_sink);
          node.ref_input()[7] = "head_sink";
          EXPECT_EQ(Peak(node, shapes), without_sink + 64);
          node.ref_input()[3] = "mask";
          shapes[3] = SymShape({1});
          EXPECT_EQ(Peak(node, shapes), without_sink + 64 + 4);
        }
      }
    }
  }
}

TEST(AttentionMemory, LargeHeadSinkIsNotCappedWithStreamingWorkers) {
  onnx_light_cpu::RegisterExperimentalShapeAndMemoryFunctions();
  constexpr std::int64_t heads = 1LL << 28;
  auto node = AttentionNode(false);
  node.ref_input()[4] = "";
  node.ref_input()[5] = "";
  std::vector<SymShape> shapes = {SymShape({1, 1024, 1, 1}), SymShape({1, 1, 1, 1}),
                                  SymShape({1, 1, 1, 1})};
  const auto capped_workers = Peak(node, shapes);
  shapes[0] = SymShape({1, heads, 1, 1});
  EXPECT_EQ(Peak(node, shapes), capped_workers);
  node.add_input("");
  node.add_input("head_sink");
  shapes.resize(7, SymShape({}));
  shapes.emplace_back(SymShape({heads}));
  EXPECT_EQ(Peak(node, shapes), capped_workers + heads * 4);
}

TEST(AttentionMemory, HeadSinkByteArithmeticIsChecked) {
  onnx_light_cpu::RegisterExperimentalShapeAndMemoryFunctions();
  auto node = AttentionNode(false);
  node.ref_input()[4] = "";
  node.ref_input()[5] = "";
  node.add_input("");
  node.add_input("");
  std::vector<SymShape> shapes = {SymShape({1, 1, 0, 1}), SymShape({1, 1, 0, 1}),
                                  SymShape({1, 1, 0, 1})};
  shapes.resize(8, SymShape({}));
  const auto size_limit = std::numeric_limits<std::size_t>::max() / sizeof(float);
  const auto check = [&](std::int64_t heads, bool mask) {
    shapes[0] = SymShape({1, heads, 0, 1});
    shapes[7] = SymShape({heads});
    node.ref_input()[3] = mask ? "mask" : "";
    shapes[3] = mask ? SymShape({1}) : SymShape({});
    node.ref_input()[7] = "";
    EXPECT_EQ(Peak(node, shapes), mask ? 4 : 0);
    node.ref_input()[7] = "head_sink";
    EXPECT_THROW(Peak(node, shapes), std::invalid_argument);
  };
  // Empty Q/K/V isolate the sink's byte multiplication and addition checks.
  check(static_cast<std::int64_t>(size_limit + 1), false);
  check(static_cast<std::int64_t>(size_limit), true);
  if (std::numeric_limits<std::size_t>::max() >
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    check(std::numeric_limits<std::int64_t>::max() / sizeof(float) + 1, false);
  }
}

TEST(AttentionMemory, ExplicitMaskConversionIsNotErased) {
  onnx_light_cpu::RegisterExperimentalShapeAndMemoryFunctions();
  auto node = AttentionNode();
  auto shapes = PrefillShapes();
  const auto unmasked = Peak(node, shapes);
  node.ref_input()[3] = "mask";
  shapes[3] = SymShape({1, 1, 32000, 32000});
  // The original 2.048 GB FP16 mask remains an input, not scratch. Its FP32
  // conversion is additional scratch regardless of whether scores are streamed.
  EXPECT_EQ(Peak(node, shapes) - unmasked, 4'096'000'000);
  shapes[3] = SymShape({1, 1, 1, 32000});
  EXPECT_EQ(Peak(node, shapes) - unmasked, 128'000);
}

TEST(AttentionMemory, MaterializedQkAndFp64IncludeConversionsButNotDeclaredOutputs) {
  onnx_light_cpu::RegisterExperimentalShapeAndMemoryFunctions();
  constexpr std::int64_t dense = 65'536'000'000;
  const auto shapes = PrefillShapes();
  auto node = AttentionNode(true, true);
  const auto qk = Peak(node, shapes);
  EXPECT_GE(qk, dense);
  // One FP32 score conversion buffer, not an additional declared FP16 QK output.
  EXPECT_LT(qk, dense + 1'000'000'000);
  for (std::int64_t mode = 0; mode < 4; ++mode) {
    auto mode_node = node;
    IntAttribute(mode_node, "qk_matmul_output_mode", mode);
    EXPECT_EQ(qk, Peak(mode_node, shapes));
  }
  node = AttentionNode();
  IntAttribute(node, "softmax_precision", 11);
  EXPECT_GE(Peak(node, shapes), 2 * dense);
  node = AttentionNode();
  IntAttribute(node, "softmax_precision", 1);
  EXPECT_EQ(Peak(node, shapes), Peak(AttentionNode(), shapes));
}

TEST(AttentionMemory, Rank4CacheAndUnknownDimensions) {
  onnx_light_cpu::RegisterExperimentalShapeAndMemoryFunctions();
  const auto node = AttentionNode();
  auto shapes = PrefillShapes(128, 4096);
  const auto cached = Peak(node, shapes);
  EXPECT_GT(cached, 0);
  shapes[0] = SymShape({1, 16, 128, 128});
  shapes[1] = SymShape({1, 4, 128, 128});
  shapes[2] = shapes[1];
  EXPECT_GT(Peak(node, shapes), 0);
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
  EXPECT_THROW(Peak(AttentionNode(), shapes), std::invalid_argument);
  auto node = AttentionNode();
  node.ref_input()[3] = "mask";
  shapes = PrefillShapes();
  shapes[3] = SymShape({1, 1, std::numeric_limits<std::int64_t>::max(), 32000});
  EXPECT_THROW(Peak(node, shapes), std::invalid_argument);
}

} // namespace
