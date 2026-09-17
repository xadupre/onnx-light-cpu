// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/com_microsoft/matmul_nbits.h"
#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/kernels/com_microsoft/matmul_nbits_kernel.h"

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

struct InlineExecutor {
  std::int64_t dispatches = 0;
  std::int64_t blocks = 0;

  static void Run(void *context, std::int64_t num_blocks, void *task_context,
                  onnx_light_cpu::ExecutionBlockFn task) {
    auto &self = *static_cast<InlineExecutor *>(context);
    ++self.dispatches;
    self.blocks = num_blocks;
    for (std::int64_t block = 0; block < num_blocks; ++block) {
      task(task_context, block);
    }
  }
};

ONNX_LIGHT_NAMESPACE::NodeProto MakeNode(std::int64_t k, std::int64_t n, std::int64_t bits = 4,
                                         std::int64_t block_size = 32) {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_domain("com.microsoft");
  node.set_op_type("MatMulNBits");
  node.add_input("A");
  node.add_input("B");
  node.add_input("scales");
  node.add_output("Y");
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "K", k);
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "N", n);
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "bits", bits);
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "block_size", block_size);
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "accuracy_level", std::int64_t{4});
  return node;
}

TEST(MatMulNBitsKernel, DecodesInt4AndAppliesPerBlockScalesAndBias) {
  std::vector<std::uint8_t> packed(64, 0x88);
  std::fill(packed.begin() + 32, packed.end(), 0x99);
  packed[16] = 0x8a;
  packed[48] = 0x87;
  const std::vector<float> a_values(33, 1.0f);
  const rt_ns::Tensor a = rt_ns::Tensor::FromFloat("", {1, 33}, a_values);
  const rt_ns::Tensor b = rt_ns::Tensor::FromUint8("", {2, 2, 16}, packed);
  const rt_ns::Tensor scales = rt_ns::Tensor::FromFloat("", {2, 2}, {1.0f, 0.25f, 0.5f, 2.0f});
  const rt_ns::Tensor bias = rt_ns::Tensor::FromFloat("", {2}, {1.0f, -1.0f});
  const onnx_light_cpu::MatMulNBitsKernel kernel{
      MakeNode(33, 2), rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};
  const rt_ns::Tensor y = kernel(a, b, scales, &bias);
  ASSERT_EQ(y.shape, (rt_ns::Shape{1, 2}));
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 1.5f);
  EXPECT_FLOAT_EQ(y.AsFloat()[1], 13.0f);
}

TEST(MatMulNBitsKernel, PreservesLeadingDimensionsAndFlatScales) {
  const std::vector<std::uint8_t> packed(16, 0x99);
  const rt_ns::Tensor a = rt_ns::Tensor::FromFloat("", {2, 1, 32}, std::vector<float>(64, 1.0f));
  const rt_ns::Tensor b = rt_ns::Tensor::FromUint8("", {1, 1, 16}, packed);
  const rt_ns::Tensor scales = rt_ns::Tensor::FromFloat("", {1}, {0.5f});
  const onnx_light_cpu::MatMulNBitsKernel kernel{
      MakeNode(32, 1), rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};
  const rt_ns::Tensor y = kernel(a, b, scales);
  EXPECT_EQ(y.shape, (rt_ns::Shape{2, 1, 1}));
  for (std::size_t i = 0; i < y.element_count(); ++i) {
    EXPECT_FLOAT_EQ(y.AsFloat()[i], 16.0f);
  }
}

TEST(MatMulNBitsKernel, RejectsUnsupportedContractAndInvalidShapes) {
  EXPECT_THROW((onnx_light_cpu::MatMulNBitsKernel{
                   MakeNode(32, 1, 8), rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}}),
               std::invalid_argument);
  EXPECT_THROW(
      (onnx_light_cpu::MatMulNBitsKernel{MakeNode(32, 1, 4, 64),
                                         rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}}),
      std::invalid_argument);

  const onnx_light_cpu::MatMulNBitsKernel kernel{
      MakeNode(32, 1), rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};
  const rt_ns::Tensor b =
      rt_ns::Tensor::FromUint8("", {1, 1, 16}, std::vector<std::uint8_t>(16, 0x88));
  const rt_ns::Tensor scales = rt_ns::Tensor::FromFloat("", {1, 1}, {1.0f});
  EXPECT_THROW(kernel(rt_ns::Tensor::FromFloat("", {1, 31}, std::vector<float>(31)), b, scales),
               std::invalid_argument);
  EXPECT_THROW(kernel(rt_ns::Tensor::FromFloat("", {1, 32}, std::vector<float>(32)),
                      rt_ns::Tensor::FromUint8("", {1, 1, 15}, std::vector<std::uint8_t>(15, 0x88)),
                      scales),
               std::invalid_argument);
}

TEST(MatMulNBitsKernel, LowLevelRejectsUnsupportedBlockSize) {
  float value = 0.0f;
  const std::uint8_t packed = 0x88;
  EXPECT_THROW(
      onnx_light_cpu::MatMulNBitsFloat32(&value, &packed, &value, nullptr, &value, 1, 1, 1, 16),
      std::invalid_argument);
}

TEST(MatMulNBitsKernel, UsesRuntimeSchedulingAndSuppressesNestedParallelism) {
  const std::vector<float> a(32, 1.0f);
  const std::vector<std::uint8_t> packed(512 * 16, 0x88);
  const std::vector<float> scales(512, 1.0f);
  std::vector<float> output(512);
  InlineExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 8, &InlineExecutor::Run};
  onnx_light_cpu::ExecutionExecutorScope scope(&view);

  onnx_light_cpu::MatMulNBitsFloat32(a.data(), packed.data(), scales.data(), nullptr, output.data(),
                                     1, 32, 2, 32);
  EXPECT_EQ(executor.dispatches, 0);

  onnx_light_cpu::MatMulNBitsFloat32(a.data(), packed.data(), scales.data(), nullptr, output.data(),
                                     1, 32, 512, 32);
  EXPECT_EQ(executor.dispatches, 1);
  EXPECT_GT(executor.blocks, 1);

  executor = {};
  {
    onnx_light_cpu::detail::ExecutionRegionScope region;
    onnx_light_cpu::MatMulNBitsFloat32(a.data(), packed.data(), scales.data(), nullptr,
                                       output.data(), 1, 32, 512, 32);
  }
  EXPECT_EQ(executor.dispatches, 0);
}

} // namespace
