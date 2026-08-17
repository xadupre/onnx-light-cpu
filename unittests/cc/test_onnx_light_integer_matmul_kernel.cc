// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/math/integer_matmul_kernel.h"

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

rt_ns::KernelContext MakeCtx() { return rt_ns::KernelContext(rt_ns::OpsetId(std::string(), 18)); }

template <typename T>
rt_ns::Tensor Tensor(const char *name, const rt_ns::Shape &shape, std::vector<T> values) {
  return rt_ns::Tensor::From<T>(name, shape, values);
}

TEST(OnnxLightMatMulIntegerKernel, MixedSignedInputsAndPerAxisZeroPoints) {
  onnx_light_cpu::MatMulIntegerKernel kernel(MakeCtx());
  const auto a = Tensor<std::uint8_t>("a", {2, 3}, {3, 4, 5, 7, 8, 9});
  const auto b = Tensor<std::int8_t>("b", {3, 2}, {-2, 3, 4, -5, 6, 7});
  const auto a_zp = Tensor<std::uint8_t>("az", {2}, {3, 7});
  const auto b_zp = Tensor<std::int8_t>("bz", {2}, {-2, 3});

  const auto y = kernel(a, b, &a_zp, &b_zp);

  ASSERT_EQ(y.shape, (rt_ns::Shape{2, 2}));
  EXPECT_EQ(y.AsInt32()[0], 22);
  EXPECT_EQ(y.AsInt32()[1], 0);
  EXPECT_EQ(y.AsInt32()[2], 22);
  EXPECT_EQ(y.AsInt32()[3], 0);
}

TEST(OnnxLightMatMulIntegerKernel, BroadcastsBatchesAndSupportsVectors) {
  onnx_light_cpu::MatMulIntegerKernel kernel(MakeCtx());
  const auto a = Tensor<std::int8_t>("a", {2, 1, 1, 2}, {1, 2, 3, 4});
  const auto b = Tensor<std::int8_t>("b", {1, 3, 2, 1}, {1, 2, 3, 4, 5, 6});

  const auto y = kernel(a, b);

  ASSERT_EQ(y.shape, (rt_ns::Shape{2, 3, 1, 1}));
  const std::vector<int32_t> expected = {5, 11, 17, 11, 25, 39};
  EXPECT_EQ(std::vector<int32_t>(y.AsInt32(), y.AsInt32() + expected.size()), expected);

  const auto vector_a = Tensor<std::uint8_t>("va", {3}, {1, 2, 3});
  const auto vector_b = Tensor<std::uint8_t>("vb", {3}, {4, 5, 6});
  const auto dot = kernel(vector_a, vector_b);
  EXPECT_TRUE(dot.shape.empty());
  EXPECT_EQ(dot.AsInt32()[0], 32);
}

TEST(OnnxLightMatMulIntegerKernel, AccumulationWrapsModuloInt32) {
  onnx_light_cpu::MatMulIntegerKernel kernel(MakeCtx());
  constexpr std::size_t k = 40000;
  const auto a =
      Tensor<std::uint8_t>("a", {1, static_cast<int64_t>(k)}, std::vector<std::uint8_t>(k, 255));
  const auto b =
      Tensor<std::uint8_t>("b", {static_cast<int64_t>(k), 1}, std::vector<std::uint8_t>(k, 255));

  const auto y = kernel(a, b);

  const std::uint32_t wrapped = static_cast<std::uint32_t>(65025ULL * k);
  EXPECT_EQ(static_cast<std::uint32_t>(y.AsInt32()[0]), wrapped);
}

TEST(OnnxLightQLinearMatMulKernel, RequantizesRoundToEvenAndSaturates) {
  onnx_light_cpu::QLinearMatMulKernel kernel(MakeCtx());
  const auto a = Tensor<std::int8_t>("a", {1, 1}, {1});
  const auto b = Tensor<std::int8_t>("b", {1, 3}, {1, 3, -3});
  const auto scale = Tensor<float>("scale", {}, {0.5f});
  const auto one = Tensor<float>("one", {}, {1.0f});
  const auto zero = Tensor<std::int8_t>("zero", {}, {0});

  const auto y = kernel(a, scale, zero, b, one, zero, one, zero);

  ASSERT_EQ(y.shape, (rt_ns::Shape{1, 3}));
  EXPECT_EQ(y.AsInt8()[0], 0);
  EXPECT_EQ(y.AsInt8()[1], 2);
  EXPECT_EQ(y.AsInt8()[2], -2);

  const auto large_a = Tensor<std::uint8_t>("a", {1, 2}, {255, 255});
  const auto large_b = Tensor<std::uint8_t>("b", {2, 1}, {255, 255});
  const auto uzero = Tensor<std::uint8_t>("zero", {}, {0});
  const auto saturated = kernel(large_a, one, uzero, large_b, one, uzero, one, uzero);
  EXPECT_EQ(saturated.AsUint8()[0], std::numeric_limits<std::uint8_t>::max());

  const auto negative_a = Tensor<std::int8_t>("a", {1, 2}, {127, 127});
  const auto negative_b = Tensor<std::int8_t>("b", {2, 1}, {-128, -128});
  const auto negative = kernel(negative_a, one, zero, negative_b, one, zero, one, zero);
  EXPECT_EQ(negative.AsInt8()[0], std::numeric_limits<std::int8_t>::min());
}

TEST(OnnxLightQLinearMatMulKernel, RejectsInvalidScaleAndZeroPointTypes) {
  onnx_light_cpu::QLinearMatMulKernel kernel(MakeCtx());
  const auto a = Tensor<std::int8_t>("a", {1, 1}, {1});
  const auto b = Tensor<std::int8_t>("b", {1, 1}, {1});
  const auto zero = Tensor<std::int8_t>("zero", {}, {0});
  const auto uzero = Tensor<std::uint8_t>("uzero", {}, {0});
  const auto one = Tensor<float>("one", {}, {1.0f});
  const auto zero_scale = Tensor<float>("scale", {}, {0.0f});
  const auto negative_scale = Tensor<float>("scale", {}, {-1.0f});

  EXPECT_THROW(kernel(a, one, zero, b, one, zero, zero_scale, zero), std::invalid_argument);
  EXPECT_THROW(kernel(a, negative_scale, zero, b, one, zero, one, zero), std::invalid_argument);
  EXPECT_THROW(kernel(a, one, uzero, b, one, zero, one, zero), std::invalid_argument);
}

} // namespace
