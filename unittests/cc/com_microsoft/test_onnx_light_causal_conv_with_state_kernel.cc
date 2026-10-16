// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/kernels/com_microsoft/causal_conv_with_state_kernel.h"

#include "onnx_core/runtime/kernels/cast_helper.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

rt_ns::KernelContext Context() { return rt_ns::KernelContext(rt_ns::OpsetId("com.microsoft", 1)); }

TEST(CausalConvWithState, PrefillAndDecodeKeepIndependentState) {
  onnx_light_cpu::CausalConvWithStateKernel kernel(Context());
  const auto weight = rt_ns::Tensor::FromFloat("", {2, 1, 3}, {1, 2, 3, 1, 1, 1});
  const auto bias = rt_ns::Tensor::FromFloat("", {2}, {1, -1});
  const auto input = rt_ns::Tensor::FromFloat("", {1, 2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
  auto [output, state] = kernel(input, weight, &bias);
  EXPECT_EQ(std::vector<float>(output.AsFloat(), output.AsFloat() + 8),
            (std::vector<float>{4, 9, 15, 21, 4, 10, 17, 20}));
  EXPECT_EQ(std::vector<float>(state.AsFloat(), state.AsFloat() + 4),
            (std::vector<float>{3, 4, 7, 8}));

  const auto token = rt_ns::Tensor::FromFloat("", {1, 2, 1}, {5, 9});
  auto [next_output, next_state] = kernel(token, weight, &bias, &state);
  EXPECT_FLOAT_EQ(next_output.AsFloat()[0], 27);
  EXPECT_FLOAT_EQ(next_output.AsFloat()[1], 23);
  EXPECT_EQ(std::vector<float>(next_state.AsFloat(), next_state.AsFloat() + 4),
            (std::vector<float>{4, 5, 8, 9}));
  auto [again, again_state] = kernel(token, weight, &bias, &next_state);
  EXPECT_FLOAT_EQ(again.AsFloat()[0], 30);
  EXPECT_FLOAT_EQ(again.AsFloat()[1], 25);
  auto [independent, independent_state] = kernel(token, weight, &bias);
  EXPECT_FLOAT_EQ(independent.AsFloat()[0], 16);
  EXPECT_FLOAT_EQ(independent.AsFloat()[1], 8);
  EXPECT_FLOAT_EQ(state.AsFloat()[0], 3);
}

TEST(CausalConvWithState, ShortSequenceDilationAndChannelsLast) {
  onnx_light_cpu::CausalConvWithStateKernel kernel(Context());
  const auto weight = rt_ns::Tensor::FromFloat("", {2, 1, 3}, {1, 1, 1, 1, 1, 1});
  const auto input = rt_ns::Tensor::FromFloat("", {1, 2, 2}, {1, 10, 2, 20});
  const auto past = rt_ns::Tensor::FromFloat("", {1, 4, 2}, {3, 30, 4, 40, 5, 50, 6, 60});
  auto [output, state] = kernel(input, weight, nullptr, &past, "none", true, 2);
  EXPECT_EQ(std::vector<float>(output.AsFloat(), output.AsFloat() + 4),
            (std::vector<float>{9, 90, 12, 120}));
  EXPECT_EQ(std::vector<float>(state.AsFloat(), state.AsFloat() + 8),
            (std::vector<float>{5, 50, 6, 60, 1, 10, 2, 20}));
  auto [fresh, fresh_state] = kernel(input, weight, nullptr, nullptr, "none", true, 2);
  EXPECT_FLOAT_EQ(fresh.AsFloat()[0], 1);
  EXPECT_FLOAT_EQ(fresh.AsFloat()[2], 2);
}

TEST(CausalConvWithState, EmptySequencePreservesPastAndUnitKernelHasEmptyState) {
  onnx_light_cpu::CausalConvWithStateKernel kernel(Context());
  const auto empty = rt_ns::Tensor::FromFloat("", {1, 1, 0}, {});
  const auto weight = rt_ns::Tensor::FromFloat("", {1, 1, 3}, {1, 1, 1});
  const auto past = rt_ns::Tensor::FromFloat("", {1, 1, 2}, {2, 3});
  auto [output, state] = kernel(empty, weight, nullptr, &past);
  EXPECT_EQ(output.element_count(), 0);
  ASSERT_EQ(state.element_count(), 2);
  EXPECT_FLOAT_EQ(state.AsFloat()[0], 2);
  EXPECT_FLOAT_EQ(state.AsFloat()[1], 3);
  const auto one = rt_ns::Tensor::FromFloat("", {1, 1, 1}, {4});
  const auto pointwise = rt_ns::Tensor::FromFloat("", {1, 1, 1}, {2});
  auto [value, zero_state] = kernel(one, pointwise);
  EXPECT_FLOAT_EQ(value.AsFloat()[0], 8);
  EXPECT_EQ(zero_state.element_count(), 0);
}

TEST(CausalConvWithState, HalfTypesAndSilu) {
  onnx_light_cpu::CausalConvWithStateKernel kernel(Context());
  for (bool bf16 : {false, true}) {
    auto input = bf16 ? rt_ns::MakeBfloat16Tensor("", {1, 1, 2}, {1, 2})
                      : rt_ns::MakeFloat16Tensor("", {1, 1, 2}, {1, 2});
    auto weight = bf16 ? rt_ns::MakeBfloat16Tensor("", {1, 1, 2}, {1, 1})
                       : rt_ns::MakeFloat16Tensor("", {1, 1, 2}, {1, 1});
    auto [output, state] = kernel(input, weight, nullptr, nullptr, "silu");
    const auto *values = reinterpret_cast<const std::uint16_t *>(output.bytes());
    auto read = [bf16](std::uint16_t value) {
      return bf16 ? onnx_light_cpu::detail::Bfloat16BitsToFloat(value)
                  : onnx_light_cpu::detail::Float16BitsToFloat(value);
    };
    EXPECT_NEAR(read(values[0]), 1.0f / (1.0f + std::exp(-1.0f)), 0.02f);
    EXPECT_NEAR(read(values[1]), 3.0f / (1.0f + std::exp(-3.0f)), 0.02f);
    EXPECT_FLOAT_EQ(read(reinterpret_cast<const std::uint16_t *>(state.bytes())[0]), 2);
  }
}

TEST(CausalConvWithState, RejectsInvalidGeometryAndAttributes) {
  onnx_light_cpu::CausalConvWithStateKernel kernel(Context());
  const auto input = rt_ns::Tensor::FromFloat("", {1, 2, 1}, {1, 2});
  const auto weight = rt_ns::Tensor::FromFloat("", {2, 1, 2}, {1, 1, 1, 1});
  const auto wrong_weight = rt_ns::Tensor::FromFloat("", {1, 1, 2}, {1, 1});
  const auto wrong_state = rt_ns::Tensor::FromFloat("", {1, 2, 2}, {1, 2, 3, 4});
  EXPECT_THROW((void)kernel(input, wrong_weight), std::invalid_argument);
  EXPECT_THROW((void)kernel(input, weight, nullptr, &wrong_state), std::invalid_argument);
  EXPECT_THROW((void)kernel(input, weight, nullptr, nullptr, "relu"), std::invalid_argument);
  EXPECT_THROW((void)kernel(input, weight, nullptr, nullptr, "none", false, 0),
               std::invalid_argument);
}

} // namespace
