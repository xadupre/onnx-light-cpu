// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/com_microsoft/matmul_nbits.h"
#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/kernels/com_microsoft/matmul_nbits_kernel.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/tuning/kernel_tuning.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
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

rt_ns::Tensor MakeTensor(rt_ns::DataType data_type, const rt_ns::Shape &shape,
                         const std::vector<float> &values) {
  if (data_type == rt_ns::DataType::FLOAT16) {
    return rt_ns::MakeFloat16Tensor("", shape, values);
  }
  if (data_type == rt_ns::DataType::BFLOAT16) {
    return rt_ns::MakeBfloat16Tensor("", shape, values);
  }
  return rt_ns::Tensor::FromFloat("", shape, values);
}

float ReadValue(const rt_ns::Tensor &tensor, std::size_t index) {
  if (tensor.data_type == static_cast<std::int32_t>(rt_ns::DataType::FLOAT)) {
    return tensor.AsFloat()[index];
  }
  const auto value = reinterpret_cast<const std::uint16_t *>(tensor.bytes())[index];
  return tensor.data_type == static_cast<std::int32_t>(rt_ns::DataType::FLOAT16)
             ? onnx_light_cpu::detail::Float16BitsToFloat(value)
             : onnx_light_cpu::detail::Bfloat16BitsToFloat(value);
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

TEST(MatMulNBitsKernel, Accuracy4VnniMatchesBlockQuantizationAndRefreshesWeights) {
  if (!onnx_light_cpu::MatMulNBitsAccuracy4Float32Available()) {
    GTEST_SKIP() << "AVX-512 VNNI and AVX-512BW are required.";
  }
  constexpr std::size_t rows = 9;
  constexpr std::size_t k = 64;
  constexpr std::size_t n = 16;
  constexpr std::size_t blocks = k / 32;
  std::vector<float> a(rows * k), scales(n * blocks), bias(n);
  std::vector<std::uint8_t> packed(n * blocks * 16);
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<float>(static_cast<int>(i % 29) - 14) / 17.0f;
  }
  for (std::size_t i = 0; i < scales.size(); ++i) {
    scales[i] = static_cast<float>(i % 7 + 1) / 128.0f;
  }
  for (std::size_t i = 0; i < bias.size(); ++i) {
    bias[i] = static_cast<float>(i) / 32.0f;
  }
  for (std::size_t i = 0; i < packed.size(); ++i) {
    packed[i] = static_cast<std::uint8_t>(i * 37 + 11);
  }

  const rt_ns::Tensor input = rt_ns::Tensor::FromFloat("", {rows, k}, a);
  const rt_ns::Tensor scale_tensor = rt_ns::Tensor::FromFloat("", {n, blocks}, scales);
  const rt_ns::Tensor bias_tensor = rt_ns::Tensor::FromFloat("", {n}, bias);
  const onnx_light_cpu::MatMulNBitsKernel kernel{
      MakeNode(k, n), rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};

  auto check = [&](const std::vector<std::uint8_t> &weights) {
    const rt_ns::Tensor weight_tensor = rt_ns::Tensor::FromUint8("", {n, blocks, 16}, weights);
    const rt_ns::Tensor output = kernel(input, weight_tensor, scale_tensor, &bias_tensor);
    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t column = 0; column < n; ++column) {
        float expected = bias[column];
        for (std::size_t block = 0; block < blocks; ++block) {
          const float *activation = a.data() + row * k + block * 32;
          float maximum = 0.0f;
          for (std::size_t offset = 0; offset < 32; ++offset) {
            maximum = std::max(maximum, std::abs(activation[offset]));
          }
          const float activation_scale = maximum / 127.0f;
          std::int32_t dot = 0;
          for (std::size_t offset = 0; offset < 32; ++offset) {
            const auto quantized_activation = static_cast<std::int32_t>(
                std::nearbyint(activation[offset] * (maximum == 0.0f ? 0.0f : 127.0f / maximum)));
            const std::uint8_t byte = weights[(column * blocks + block) * 16 + offset / 2];
            const std::int32_t quantized_weight =
                static_cast<std::int32_t>((byte >> ((offset % 2) * 4)) & 15) - 8;
            dot += quantized_activation * quantized_weight;
          }
          expected += static_cast<float>(dot) * activation_scale * scales[column * blocks + block];
        }
        EXPECT_NEAR(output.AsFloat()[row * n + column], expected, 2e-6f);
      }
    }
  };

  check(packed);
  packed[0] ^= 0x0f;
  check(packed);
}

TEST(MatMulNBitsKernel, SupportsEveryNonDoubleFloatAndIntWidth) {
  for (const rt_ns::DataType data_type :
       {rt_ns::DataType::FLOAT, rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16}) {
    for (const std::int64_t bits : {2, 4, 8}) {
      const std::size_t blob_size = static_cast<std::size_t>(4 * bits);
      const std::uint8_t packed_value = bits == 2 ? 0xff : bits == 4 ? 0x99 : 0x81;
      const rt_ns::Tensor a = MakeTensor(data_type, {1, 33}, std::vector<float>(33, 1.0f));
      const rt_ns::Tensor b =
          rt_ns::Tensor::FromUint8("", {1, 2, static_cast<std::int64_t>(blob_size)},
                                   std::vector<std::uint8_t>(2 * blob_size, packed_value));
      const rt_ns::Tensor scales = MakeTensor(data_type, {1, 2}, {0.5f, 0.5f});
      const rt_ns::Tensor bias = MakeTensor(data_type, {1}, {0.25f});
      const onnx_light_cpu::MatMulNBitsKernel kernel{
          MakeNode(33, 1, bits), rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};
      const rt_ns::Tensor y = kernel(a, b, scales, &bias);
      EXPECT_EQ(y.data_type, static_cast<std::int32_t>(data_type));
      EXPECT_FLOAT_EQ(ReadValue(y, 0), 16.75f)
          << "type=" << static_cast<int>(data_type) << ", bits=" << bits;
    }
  }
}

TEST(MatMulNBitsKernel, RegistersTypeAndBitSpecificTuningSchemas) {
  onnx_light_cpu::MatMulNBitsKernel::RegisterTuningSchemas();
  for (const rt_ns::DataType data_type :
       {rt_ns::DataType::FLOAT, rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16}) {
    for (const std::int64_t bits : {2, 4, 8}) {
      const onnx_light_cpu::MatMulNBitsKernel kernel{
          MakeNode(32, 1, bits), rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};
      const auto key = kernel.TuningKey(static_cast<std::int32_t>(data_type));
      EXPECT_EQ(key.library, "onnx_light_cpu");
      EXPECT_EQ(key.kernel, "MatMulNBits");
      EXPECT_EQ(key.implementation, "packed_int" + std::to_string(bits));
      EXPECT_EQ(key.element_type, static_cast<std::int32_t>(data_type));
      EXPECT_EQ(key.tuning_abi, onnx_light_cpu::MatMulNBitsKernel::kTuningAbi);
      EXPECT_NE(rt_ns::GetKernelTuningRegistry().FindSchema(key), nullptr);
    }
  }
  const onnx_light_cpu::MatMulNBitsKernel kernel{
      MakeNode(32, 1), rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};
  EXPECT_TRUE(kernel.TuningKey(static_cast<std::int32_t>(rt_ns::DataType::DOUBLE)).library.empty());
}

TEST(MatMulNBitsKernel, RejectsUnsupportedContractAndInvalidShapes) {
  EXPECT_THROW((onnx_light_cpu::MatMulNBitsKernel{
                   MakeNode(32, 1, 3), rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}}),
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
  EXPECT_THROW(kernel(rt_ns::Tensor::FromDouble("", {1, 32}, std::vector<double>(32, 1.0)), b,
                      rt_ns::Tensor::FromDouble("", {1, 1}, {1.0})),
               std::invalid_argument);
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
