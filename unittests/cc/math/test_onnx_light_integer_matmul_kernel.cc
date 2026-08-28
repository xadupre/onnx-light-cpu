// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/math/integer_matmul_kernel.h"

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
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

template <typename T> T RequantizeReference(int32_t accumulator, float scale, int32_t zero_point) {
  const double rounded =
      std::nearbyint(static_cast<double>(accumulator) * static_cast<double>(scale));
  const double shifted = rounded + static_cast<double>(zero_point);
  const double clamped = std::clamp(shifted, static_cast<double>(std::numeric_limits<T>::min()),
                                    static_cast<double>(std::numeric_limits<T>::max()));
  return static_cast<T>(clamped);
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

TEST(OnnxLightQLinearMatMulKernel, ContiguousPathMatchesReferenceAndBatchBroadcast) {
  onnx_light_cpu::QLinearMatMulKernel kernel(MakeCtx());
  const auto a = Tensor<std::int8_t>("a", {2, 3, 5},
                                     {1,  -2, 7,  12, -6, 3, 4,  -5, 9,  2, -8, 5,  6, -1, 10,
                                      -4, 11, -7, 8,  0,  2, -3, 14, -9, 1, 6,  -5, 4, 7,  -2});
  const auto b = Tensor<std::int8_t>(
      "b", {1, 5, 4}, {3, -4, 2, 5, -6, 1, 7, -3, 4, 8, -2, 6, -5, 9, 0, 1, 2, -7, 3, 4});
  const auto scale_a = Tensor<float>("a_scale", {}, {0.5f});
  const auto scale_b = Tensor<float>("b_scale", {}, {0.25f});
  const auto scale_y = Tensor<float>("y_scale", {}, {0.2f});
  const auto zp_a = Tensor<std::int8_t>("a_zero", {}, {2});
  const auto zp_b = Tensor<std::int8_t>("b_zero", {}, {-3});
  const auto zp_y = Tensor<std::int8_t>("y_zero", {}, {4});

  const auto y = kernel(a, scale_a, zp_a, b, scale_b, zp_b, scale_y, zp_y);
  ASSERT_EQ(y.shape, (rt_ns::Shape{2, 3, 4}));

  const float combined_scale = (0.5f * 0.25f) / 0.2f;
  std::vector<std::int8_t> expected(static_cast<std::size_t>(2 * 3 * 4));
  for (int64_t batch = 0; batch < 2; ++batch) {
    for (int64_t row = 0; row < 3; ++row) {
      for (int64_t column = 0; column < 4; ++column) {
        int32_t accumulator = 0;
        for (int64_t depth = 0; depth < 5; ++depth) {
          const int32_t av = static_cast<int32_t>(a.AsInt8()[batch * 15 + row * 5 + depth]);
          const int32_t bv = static_cast<int32_t>(b.AsInt8()[depth * 4 + column]);
          accumulator += (av - 2) * (bv + 3);
        }
        expected[static_cast<std::size_t>(batch * 12 + row * 4 + column)] =
            RequantizeReference<std::int8_t>(accumulator, combined_scale, 4);
      }
    }
  }
  EXPECT_EQ(std::vector<std::int8_t>(y.AsInt8(), y.AsInt8() + expected.size()), expected);
}

} // namespace
