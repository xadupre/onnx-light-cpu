// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/shapes/shapes_context.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_light_cpu/kernels/com_microsoft/attention_kernel.h"
#include "onnx_light_cpu/kernels/register_kernels.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"
#include "onnx_light_cpu/shapes/com_microsoft/shape_inference.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
namespace rt = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym = ONNX_LIGHT_NAMESPACE::core::symbolic;
using ONNX_LIGHT_NAMESPACE::NodeProto;
NodeProto Node(bool present = true, int64_t heads = 1) {
  NodeProto node;
  node.set_domain("com.microsoft");
  node.set_op_type("Attention");
  for (const char *name : {"x", "w", "", "", "", ""})
    node.add_input(name);
  node.add_output("y");
  if (present)
    node.add_output("present");
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "num_heads", heads);
  return node;
}
rt::RuntimeContext Runtime() {
  return rt::RuntimeContext{rt::KernelContext{rt::OpsetId("com.microsoft", 1)},
                            rt::TensorMap{{"x", rt::Tensor::FromFloat("x", {1, 2, 1}, {1, 2})},
                                          {"w", rt::Tensor::FromFloat("w", {1, 3}, {1, 1, 2})}}};
}
void Run(const NodeProto &node, rt::RuntimeContext &runtime) {
  onnx_light_cpu::MicrosoftAttentionKernel kernel(
      node, rt::KernelContext{rt::OpsetId("com.microsoft", 1)});
  kernel.Run(runtime);
}

TEST(MicrosoftAttentionKernel, ProjectionSoftmaxAndPackedPresent) {
  auto node = Node();
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "do_rotary", int64_t{0});
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "rotary_embedding_dim", int64_t{0});
  node.ref_input()[2] = "bias";
  auto runtime = Runtime();
  runtime.Set("bias", rt::Tensor::FromFloat("bias", {3}, {0, 0, 1}));
  Run(node, runtime);
  const auto &y = runtime.Get("y");
  ASSERT_EQ(y.shape, (rt::Shape{1, 2, 1}));
  EXPECT_NEAR(y.AsFloat()[0], 3 + 2 / (1 + std::exp(-1.f)), 1.e-5);
  EXPECT_NEAR(y.AsFloat()[1], 3 + 2 / (1 + std::exp(-2.f)), 1.e-5);
  const auto &present = runtime.Get("present");
  ASSERT_EQ(present.shape, (rt::Shape{2, 1, 1, 2, 1}));
  for (int i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(present.AsFloat()[i], (std::vector<float>{1, 2, 3, 5})[i]);
}

TEST(MicrosoftAttentionKernel, MasksLengthsStartsAndAllRawRanks) {
  for (int mode = 0; mode < 5; ++mode) {
    auto node = Node();
    node.ref_input()[3] = "mask";
    auto runtime = Runtime();
    if (mode == 0)
      runtime.Set("mask", rt::Tensor::FromInt32("mask", {1}, {1}));
    else if (mode == 1)
      runtime.Set("mask", rt::Tensor::FromInt32("mask", {2}, {2, 1}));
    else if (mode == 2)
      runtime.Set("mask", rt::Tensor::FromInt32("mask", {1, 2}, {1, 0}));
    else if (mode == 3)
      runtime.Set("mask", rt::Tensor::FromInt32("mask", {1, 2, 2}, {1, 0, 1, 0}));
    else
      runtime.Set("mask", rt::Tensor::FromInt32("mask", {1, 1, 3, 3}, {1, 0, 0, 1, 0, 0, 1, 0, 0}));
    Run(node, runtime);
    EXPECT_NEAR(runtime.Get("y").AsFloat()[0], mode == 1 ? 4 : 2, 1.e-5) << mode;
    EXPECT_NEAR(runtime.Get("y").AsFloat()[1], mode == 1 ? 4 : 2, 1.e-5) << mode;
  }
}

TEST(MicrosoftAttentionKernel, CausalCachedChunkAndFourDimensionalMaskOffset) {
  for (bool masked : {false, true}) {
    auto node = Node();
    node.ref_input()[4] = "past";
    ONNX_LIGHT_NAMESPACE::AddAttribute(node, "unidirectional", int64_t{masked ? 0 : 1});
    auto runtime = Runtime();
    runtime.Set("x", rt::Tensor::FromFloat("x", {1, 2, 1}, {0, 0}));
    runtime.Set("bias", rt::Tensor::FromFloat("bias", {3}, {0, 1, 2}));
    node.ref_input()[2] = "bias";
    runtime.Set("past", rt::Tensor::FromFloat("past", {2, 1, 1, 1, 1}, {7, 8}));
    if (masked) {
      node.ref_input()[3] = "mask";
      runtime.Set("mask", rt::Tensor::FromInt32("mask", {1, 1, 3, 3}, {1, 1, 1, 0, 1, 0, 0, 0, 1}));
    }
    Run(node, runtime);
    EXPECT_NEAR(runtime.Get("y").AsFloat()[0], masked ? 2 : 5, 1.e-5);
    EXPECT_NEAR(runtime.Get("y").AsFloat()[1], masked ? 2 : 4, 1.e-5);
    const std::vector<float> expected{7, 1, 1, 8, 2, 2};
    for (int i = 0; i < 6; ++i)
      EXPECT_FLOAT_EQ(runtime.Get("present").AsFloat()[i], expected[i]);
  }
}

TEST(MicrosoftAttentionKernel, UnequalValueWidthAndAttentionBias) {
  auto node = Node(false);
  node.ref_input()[5] = "ab";
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "qkv_hidden_sizes", std::vector<int64_t>{1, 1, 2});
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "scale", 0.f);
  auto runtime = Runtime();
  runtime.Set("w", rt::Tensor::FromFloat("w", {1, 4}, {1, 1, 2, 3}));
  runtime.Set("ab", rt::Tensor::FromFloat("ab", {1, 1, 2, 2}, {0, std::log(3.f), 0, 0}));
  Run(node, runtime);
  ASSERT_EQ(runtime.Get("y").shape, (rt::Shape{1, 2, 2}));
  const std::vector<float> expected{3.5f, 5.25f, 3, 4.5f};
  for (int i = 0; i < 4; ++i)
    EXPECT_NEAR(runtime.Get("y").AsFloat()[i], expected[i], 1.e-5);
}

TEST(MicrosoftAttentionKernel, MultiHeadProjectionStrides) {
  auto node = Node(true, 2);
  auto runtime = Runtime();
  runtime.Set("x", rt::Tensor::FromFloat("x", {2, 2, 2}, {0, 0, 0, 0, 0, 0, 0, 0}));
  runtime.Set("w", rt::Tensor::FromFloat("w", {2, 6}, std::vector<float>(12, 0)));
  node.ref_input()[2] = "bias";
  runtime.Set("bias", rt::Tensor::FromFloat("bias", {6}, {0, 0, 1, 2, 3, 4}));
  Run(node, runtime);
  for (int i = 0; i < 8; ++i)
    EXPECT_FLOAT_EQ(runtime.Get("y").AsFloat()[i], i % 2 ? 4 : 3);
  const auto &present = runtime.Get("present");
  ASSERT_EQ(present.shape, (rt::Shape{2, 2, 2, 2, 1}));
  const std::vector<float> expected{1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4};
  for (int i = 0; i < 16; ++i)
    EXPECT_FLOAT_EQ(present.AsFloat()[i], expected[i]);
}

TEST(MicrosoftAttentionKernel, EmptyQueriesPreservePast) {
  for (bool past : {false, true}) {
    auto node = Node();
    auto runtime = Runtime();
    runtime.Set("x", rt::Tensor::FromFloat("x", {1, 0, 1}, {}));
    if (past) {
      node.ref_input()[4] = "past";
      runtime.Set("past", rt::Tensor::FromFloat("past", {2, 1, 1, 1, 1}, {3, 4}));
    }
    Run(node, runtime);
    EXPECT_EQ(runtime.Get("y").shape, (rt::Shape{1, 0, 1}));
    EXPECT_EQ(runtime.Get("present").shape, (rt::Shape{2, 1, 1, past ? 1 : 0, 1}));
    if (past)
      EXPECT_FLOAT_EQ(runtime.Get("present").AsFloat()[1], 4);
  }
}

TEST(MicrosoftAttentionKernel, RejectsUnsupportedModesTypesAndShapes) {
  for (int mode = 0; mode < 11; ++mode) {
    auto node = Node();
    auto runtime = Runtime();
    if (mode == 0)
      ONNX_LIGHT_NAMESPACE::AddAttribute(node, "do_rotary", int64_t{1});
    if (mode == 1)
      ONNX_LIGHT_NAMESPACE::AddAttribute(node, "past_present_share_buffer", int64_t{1});
    if (mode == 2) {
      node.add_input("past_sequence_length");
      runtime.Set("past_sequence_length", rt::Tensor::FromInt32("past_sequence_length", {}, {0}));
    }
    if (mode == 3)
      runtime.Set("w", rt::Tensor::FromFloat("w", {1, 4}, {1, 1, 1, 1}));
    if (mode == 4)
      runtime.Set("w", rt::Tensor::FromFloat("w", {2, 3}, std::vector<float>(6, 1)));
    if (mode == 5)
      runtime.Set("x", rt::MakeFloat16Tensor("x", {1, 2, 1}, {1, 2}));
    if (mode == 6) {
      node.ref_input()[3] = "mask";
      runtime.Set("mask", rt::Tensor::FromInt32("mask", {5}, std::vector<int32_t>(5, 1)));
    }
    if (mode == 7) {
      node.ref_input()[3] = "mask";
      runtime.Set("mask", rt::Tensor::FromInt32("mask", {2}, {1, 2}));
    }
    if (mode == 8)
      ONNX_LIGHT_NAMESPACE::AddAttribute(node, "scale", std::numeric_limits<float>::infinity());
    if (mode == 9) {
      node = Node(false);
      node.ref_input()[4] = "past";
      runtime.Set("past", rt::Tensor::FromFloat("past", {2, 1, 1, 0, 1}, {}));
    }
    if (mode == 10)
      ONNX_LIGHT_NAMESPACE::AddAttribute(node, "rotary_embedding_dim", int64_t{16});
    EXPECT_THROW(Run(node, runtime), std::invalid_argument) << mode;
  }
}

TEST(MicrosoftAttentionKernel, EmptyBatch) {
  auto node = Node();
  auto runtime = Runtime();
  runtime.Set("x", rt::Tensor::FromFloat("x", {0, 2, 1}, {}));
  node.ref_input()[4] = "past";
  runtime.Set("past", rt::Tensor::FromFloat("past", {2, 0, 1, 2, 1}, {}));
  Run(node, runtime);
  EXPECT_EQ(runtime.Get("y").shape, (rt::Shape{0, 2, 1}));
  EXPECT_EQ(runtime.Get("present").shape, (rt::Shape{2, 0, 1, 4, 1}));
}

TEST(MicrosoftAttentionKernel, SingleTokenPrefillAndDecode) {
  for (bool cached : {false, true}) {
    auto node = Node();
    auto runtime = Runtime();
    runtime.Set("x", rt::Tensor::FromFloat("x", {1, 1, 1}, {2}));
    if (cached) {
      node.ref_input()[4] = "past";
      runtime.Set("past", rt::Tensor::FromFloat("past", {2, 1, 1, 1, 1}, {1, 2}));
    }
    Run(node, runtime);
    EXPECT_NEAR(runtime.Get("y").AsFloat()[0], cached ? 2 + 2 / (1 + std::exp(-2.f)) : 4, 1.e-5);
  }
}

TEST(MicrosoftAttentionKernel, FiniteFullyMaskedRowsAndCausalFilterOverride) {
  auto node = Node();
  auto runtime = Runtime();
  node.ref_input()[3] = "mask";
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "mask_filter_value", -1.f);
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "unidirectional", int64_t{1});
  runtime.Set("mask", rt::Tensor::FromInt32("mask", {1, 2}, {0, 0}));
  Run(node, runtime);
  EXPECT_NEAR(runtime.Get("y").AsFloat()[0], 2, 1.e-5);
  EXPECT_NEAR(runtime.Get("y").AsFloat()[1], 2 + 2 / (1 + std::exp(-2.f)), 1.e-5);
}

TEST(MicrosoftAttentionKernel, SchemaSymbolicShapeAndScratchMemory) {
  const auto schemas = onnx_light_cpu::GetMicrosoftOpSchemasWithHistory("Attention");
  ASSERT_EQ(schemas.size(), 1u);
  EXPECT_EQ(schemas[0].since_version(), 1);
  auto node = Node(true, 4);
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "do_rotary", int64_t{0});
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "rotary_embedding_dim", int64_t{0});
  ONNX_LIGHT_NAMESPACE::core::shapes::ShapesContext shapes;
  shapes.Set("x",
             sym::SymTensor(nullptr, sym::TensorType::kFloat,
                            sym::SymShape({sym::SymDim("B"), sym::SymDim("S"), sym::SymDim(32)})));
  shapes.Set("w", sym::SymTensor(nullptr, sym::TensorType::kFloat, sym::SymShape({32, 96})));
  onnx_light_cpu::ComputeShapeMicrosoftAttention(shapes, node);
  const auto &y = shapes.Get("y").Shape();
  EXPECT_EQ(y[0].ToString(), "B");
  EXPECT_EQ(y[1].ToString(), "S");
  EXPECT_EQ(y[2].AsInt(), 32);
  EXPECT_EQ(shapes.Get("present").Shape()[4].AsInt(), 8);
  EXPECT_GT(onnx_light_cpu::ComputePeakMemoryMicrosoftAttention(
                sym::Device::kCPU, node, {sym::SymShape({2, 3, 32}), sym::SymShape({32, 96})}),
            0);
  shapes.Set("w", sym::SymTensor(nullptr, sym::TensorType::kFloat, sym::SymShape({31, 96})));
  EXPECT_THROW(onnx_light_cpu::ComputeShapeMicrosoftAttention(shapes, node), std::invalid_argument);
}

} // namespace
