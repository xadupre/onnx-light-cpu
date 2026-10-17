// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/kernels/com_microsoft/multi_head_attention_kernel.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"
#include "onnx_light_cpu/shapes/com_microsoft/shape_inference.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/shapes/shapes_context.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
namespace rt = ONNX_LIGHT_NAMESPACE::core::runtime;
using ONNX_LIGHT_NAMESPACE::NodeProto;

NodeProto MakeNode(bool cache = false, std::int64_t heads = 1) {
  NodeProto node;
  node.set_domain("com.microsoft");
  node.set_op_type("MultiHeadAttention");
  for (const char *name : {"q", "k", "v", "", "", "", cache ? "pk" : "", cache ? "pv" : ""})
    node.add_input(name);
  node.add_output("out");
  if (cache) {
    node.add_output("present_k");
    node.add_output("present_v");
  }
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "num_heads", heads);
  return node;
}

TEST(MultiHeadAttentionKernel, SoftmaxScaleBiasCacheAndPresent) {
  auto node = MakeNode(true);
  node.ref_input()[3] = "bias";
  node.ref_input()[4] = "mask";
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "scale", 1.0f);
  rt::RuntimeContext runtime{rt::KernelContext{rt::OpsetId("com.microsoft", 1)}};
  runtime.Set("q", rt::Tensor::FromFloat("q", {1, 1, 1}, {1}));
  runtime.Set("k", rt::Tensor::FromFloat("k", {1, 1, 1}, {1}));
  runtime.Set("v", rt::Tensor::FromFloat("v", {1, 1, 1}, {4}));
  runtime.Set("bias", rt::Tensor::FromFloat("bias", {3}, {0, 1, 2}));
  runtime.Set("mask", rt::Tensor::FromInt32("mask", {1, 2}, {1, 1}));
  runtime.Set("pk", rt::Tensor::FromFloat("pk", {1, 1, 1, 1}, {0}));
  runtime.Set("pv", rt::Tensor::FromFloat("pv", {1, 1, 1, 1}, {2}));
  onnx_light_cpu::MultiHeadAttentionKernel kernel(
      node, rt::KernelContext{rt::OpsetId("com.microsoft", 1)});
  kernel.Run(runtime);
  const float weight = std::exp(2.f) / (1.f + std::exp(2.f));
  EXPECT_NEAR(runtime.Get("out").AsFloat()[0], 2.f * (1 - weight) + 6.f * weight, 1.e-5f);
  EXPECT_FLOAT_EQ(runtime.Get("present_k").AsFloat()[1], 2.f);
  EXPECT_FLOAT_EQ(runtime.Get("present_v").AsFloat()[1], 6.f);
}

TEST(MultiHeadAttentionKernel, RejectsUnsupportedSlotsAndPackedInput) {
  auto node = MakeNode();
  rt::RuntimeContext runtime{rt::KernelContext{rt::OpsetId("com.microsoft", 1)}};
  runtime.Set("q", rt::Tensor::FromFloat("q", {1, 1, 1}, {0}));
  runtime.Set("k", rt::Tensor::FromFloat("k", {1, 1, 1}, {0}));
  runtime.Set("v", rt::Tensor::FromFloat("v", {1, 1, 1}, {1}));
  node.add_output("present_key");
  node.add_output("present_value");
  node.add_output("qk");
  onnx_light_cpu::MultiHeadAttentionKernel qk(node,
                                              rt::KernelContext{rt::OpsetId("com.microsoft", 1)});
  EXPECT_THROW(qk.Run(runtime), std::invalid_argument);
  node = MakeNode();
  rt::RuntimeContext packed_runtime{rt::KernelContext{rt::OpsetId("com.microsoft", 1)}};
  packed_runtime.Set("q", rt::Tensor::FromFloat("q", {1, 1, 1}, {0}));
  packed_runtime.Set("k", rt::Tensor::FromFloat("k", {1, 1, 1, 1, 1}, {0}));
  packed_runtime.Set("v", rt::Tensor::FromFloat("v", {1, 1, 1}, {1}));
  onnx_light_cpu::MultiHeadAttentionKernel packed(
      node, rt::KernelContext{rt::OpsetId("com.microsoft", 1)});
  EXPECT_THROW(packed.Run(packed_runtime), std::invalid_argument);
  auto shared = MakeNode();
  shared.add_input("past_sequence_length");
  onnx_light_cpu::MultiHeadAttentionKernel shared_kernel(
      shared, rt::KernelContext{rt::OpsetId("com.microsoft", 1)});
  EXPECT_THROW(shared_kernel.Run(runtime), std::invalid_argument);
}

TEST(MultiHeadAttentionKernel, ZeroQueryLengthProducesEmptyOutput) {
  auto node = MakeNode();
  rt::RuntimeContext runtime{rt::KernelContext{rt::OpsetId("com.microsoft", 1)}};
  runtime.Set("q", rt::Tensor::FromFloat("q", {1, 0, 1}, {}));
  runtime.Set("k", rt::Tensor::FromFloat("k", {1, 1, 1}, {1}));
  runtime.Set("v", rt::Tensor::FromFloat("v", {1, 1, 1}, {2}));
  onnx_light_cpu::MultiHeadAttentionKernel kernel(
      node, rt::KernelContext{rt::OpsetId("com.microsoft", 1)});
  kernel.Run(runtime);
  EXPECT_EQ(runtime.Get("out").shape, (rt::Shape{1, 0, 1}));
}

TEST(MultiHeadAttentionKernel, LengthMaskAndSinglePresentOutput) {
  auto node = MakeNode();
  node.ref_input()[4] = "mask";
  node.add_output("present_k");
  rt::RuntimeContext runtime{rt::KernelContext{rt::OpsetId("com.microsoft", 1)}};
  runtime.Set("q", rt::Tensor::FromFloat("q", {1, 1, 1}, {0}));
  runtime.Set("k", rt::Tensor::FromFloat("k", {1, 2, 1}, {3, 4}));
  runtime.Set("v", rt::Tensor::FromFloat("v", {1, 2, 1}, {5, 9}));
  runtime.Set("mask", rt::Tensor::FromInt32("mask", {1}, {1}));
  onnx_light_cpu::MultiHeadAttentionKernel kernel(
      node, rt::KernelContext{rt::OpsetId("com.microsoft", 1)});
  kernel.Run(runtime);
  EXPECT_FLOAT_EQ(runtime.Get("out").AsFloat()[0], 5.f);
  EXPECT_EQ(runtime.Get("present_k").shape, (rt::Shape{1, 1, 2, 1}));
  EXPECT_FLOAT_EQ(runtime.Get("present_k").AsFloat()[1], 4.f);
}

TEST(MultiHeadAttentionKernel, WhisperCrossAttentionAllFloatingTypes) {
  for (const auto dtype : {rt::DataType::FLOAT, rt::DataType::FLOAT16, rt::DataType::BFLOAT16}) {
    auto make = [dtype](const char *name, rt::Shape shape, std::vector<float> values) {
      if (dtype == rt::DataType::FLOAT16)
        return rt::MakeFloat16Tensor(name, shape, values);
      if (dtype == rt::DataType::BFLOAT16)
        return rt::MakeBfloat16Tensor(name, shape, values);
      return rt::Tensor::FromFloat(name, shape, values);
    };
    auto node = MakeNode(false, 2);
    node.ref_input()[3] = "bias";
    node.ref_input()[5] = "attention_bias";
    rt::RuntimeContext runtime{rt::KernelContext{rt::OpsetId("com.microsoft", 1)}};
    runtime.Set("q", make("q", {1, 1, 2}, {0, 0}));
    runtime.Set("k", make("k", {1, 2, 2, 1}, {0, 0, 0, 0}));
    runtime.Set("v", make("v", {1, 2, 2, 2}, {2, 4, 6, 8, 10, 12, 14, 16}));
    runtime.Set("bias", make("bias", {8}, {0, 0, 10, 20, 100, 200, 300, 400}));
    runtime.Set("attention_bias", make("attention_bias", {1, 2, 1, 2}, {0, 0, 0, -10000}));
    onnx_light_cpu::MultiHeadAttentionKernel kernel(
        node, rt::KernelContext{rt::OpsetId("com.microsoft", 1)});
    kernel.Run(runtime);
    const auto &result = runtime.Get("out");
    const float expected[] = {4, 6, 10, 12};
    for (std::size_t i = 0; i < 4; ++i) {
      float actual = dtype == rt::DataType::FLOAT ? result.AsFloat()[i]
                     : dtype == rt::DataType::FLOAT16
                         ? onnx_light_cpu::detail::Float16BitsToFloat(
                               reinterpret_cast<const std::uint16_t *>(result.bytes())[i])
                         : onnx_light_cpu::detail::Bfloat16BitsToFloat(
                               reinterpret_cast<const std::uint16_t *>(result.bytes())[i]);
      EXPECT_NEAR(actual, expected[i], 0.01f);
    }
  }
}

TEST(MultiHeadAttentionKernel, SymbolicShapesAndScratchMemory) {
  namespace sym = ONNX_LIGHT_NAMESPACE::core::symbolic;
  ONNX_LIGHT_NAMESPACE::core::shapes::ShapesContext shapes;
  shapes.Set("q", sym::SymTensor(nullptr, sym::TensorType::kFloat,
                                 {sym::SymDim("B"), sym::SymDim("S"), sym::SymDim(8)}));
  shapes.Set("k", sym::SymTensor(nullptr, sym::TensorType::kFloat,
                                 {sym::SymDim("B"), sym::SymDim("L"), sym::SymDim(8)}));
  shapes.Set("v", sym::SymTensor(nullptr, sym::TensorType::kFloat,
                                 {sym::SymDim("B"), sym::SymDim("L"), sym::SymDim(12)}));
  shapes.Set("pk",
             sym::SymTensor(nullptr, sym::TensorType::kFloat,
                            {sym::SymDim("B"), sym::SymDim(2), sym::SymDim("P"), sym::SymDim(4)}));
  shapes.Set("pv",
             sym::SymTensor(nullptr, sym::TensorType::kFloat,
                            {sym::SymDim("B"), sym::SymDim(2), sym::SymDim("P"), sym::SymDim(6)}));
  auto node = MakeNode(true, 2);
  onnx_light_cpu::ComputeShapeMultiHeadAttention(shapes, node);
  EXPECT_EQ(shapes.Get("out").Shape()[2], sym::SymDim(12));
  EXPECT_EQ(shapes.Get("present_k").Shape()[2].ToString(), "(P)+(L)");
  EXPECT_EQ(shapes.Get("present_v").Shape()[3], sym::SymDim(6));
  const auto schemas = onnx_light_cpu::GetMicrosoftOpSchemasWithHistory("MultiHeadAttention");
  ASSERT_EQ(schemas.size(), 1U);
  EXPECT_EQ(schemas.front().max_output(), 3);
  EXPECT_GT(onnx_light_cpu::ComputePeakMemoryMultiHeadAttention(
                sym::Device::kCPU, {{sym::SymDim(2), sym::SymDim(4), sym::SymDim(8)},
                                    {sym::SymDim(2), sym::SymDim(5), sym::SymDim(8)}}),
            0);
}
} // namespace
