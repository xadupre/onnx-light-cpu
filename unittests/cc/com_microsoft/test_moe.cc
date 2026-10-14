// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/com_microsoft/moe.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace onnx_light_cpu {
namespace {

TEST(MoE, SelectsTopExpertsAndNormalizes) {
  const float input[] = {1.0f, 2.0f};
  const float router[] = {0.0f, 1.0f, 2.0f};
  const float fc1[] = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
  const float fc2[] = {1.0f, 2.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  float output[2] = {};
  MoEAttributes attributes;
  attributes.activation = "identity";
  attributes.k = 2;
  attributes.normalize = true;
  ComputeMoE(input, router, fc1, nullptr, fc2, nullptr, output, 1, 2, 3, 1, false, attributes);
  const float top = std::exp(2.0f) / (std::exp(1.0f) + std::exp(2.0f));
  EXPECT_NEAR(output[0], 1.0f + 2.0f * top, 1e-5f);
  EXPECT_NEAR(output[1], 2.0f + 2.0f * top, 1e-5f);
  attributes.normalize = false;
  ComputeMoE(input, router, fc1, nullptr, fc2, nullptr, output, 1, 2, 3, 1, false, attributes);
  EXPECT_LT(output[0], 1.0f + 2.0f * top);
}

TEST(MoE, InterleavedSwiGLUAndFloat16Output) {
  const float input[] = {2.0f};
  const float router[] = {0.0f};
  const float fc1[] = {1.0f, 2.0f};
  const float fc2[] = {3.0f};
  float output[] = {0.0f};
  MoEAttributes attributes;
  attributes.activation = "swiglu";
  attributes.limit = 1.0f;
  attributes.alpha = 2.0f;
  attributes.beta = 0.5f;
  ComputeMoE(input, router, fc1, nullptr, fc2, nullptr, output, 1, 1, 1, 1, false, attributes);
  EXPECT_NEAR(output[0], 3.0f * (1.0f / (1.0f + std::exp(-2.0f))) * 1.5f, 1e-5f);

  const std::uint16_t half_input[] = {detail::FloatToFloat16Bits(2.0f)};
  const std::uint16_t half_router[] = {0};
  const std::uint16_t half_fc1[] = {detail::FloatToFloat16Bits(1.0f),
                                    detail::FloatToFloat16Bits(2.0f)};
  const std::uint16_t half_fc2[] = {detail::FloatToFloat16Bits(3.0f)};
  std::uint16_t half_output[] = {0};
  ComputeMoE(half_input, half_router, half_fc1, nullptr, half_fc2, nullptr, half_output, 1, 1, 1, 1,
             true, attributes);
  EXPECT_NEAR(detail::Float16BitsToFloat(half_output[0]), output[0], 0.01f);
}

TEST(MoE, EmptyTokensDoNotReadRouter) {
  MoEAttributes attributes;
  float output = 42.0f;
  ComputeMoE(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &output, 0, 2, 1, 1, false,
             attributes);
  EXPECT_FLOAT_EQ(output, 42.0f);
}

} // namespace
} // namespace onnx_light_cpu
