// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/attention/linear_attention.h"

#include <gtest/gtest.h>

#include <array>

namespace {

TEST(LinearAttention, GqaSharesStateAcrossQueryHeads) {
  onnx_light_cpu::LinearAttentionParameters parameters;
  parameters.batch_size = 1;
  parameters.sequence_length = 1;
  parameters.query_heads = 2;
  parameters.key_value_heads = 1;
  parameters.key_head_size = 2;
  parameters.value_head_size = 1;
  parameters.rule = onnx_light_cpu::LinearAttentionRule::kLinear;
  parameters.scale = 1.0f;

  const std::array<float, 4> query{2.0f, 3.0f, 1.0f, 2.0f};
  const std::array<float, 2> key{4.0f, 5.0f};
  const std::array<float, 1> value{6.0f};
  std::array<float, 2> state{};
  std::array<float, 2> output{};

  onnx_light_cpu::LinearAttentionFloat32(parameters, query.data(), key.data(), value.data(),
                                         nullptr, nullptr, state.data(), output.data());

  EXPECT_EQ(state, (std::array<float, 2>{24.0f, 30.0f}));
  EXPECT_EQ(output, (std::array<float, 2>{138.0f, 84.0f}));
}

} // namespace
