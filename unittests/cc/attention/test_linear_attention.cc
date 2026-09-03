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

// Differential coverage for the com.microsoft-only extensions: inverse
// grouping (key_value_heads > query_heads, several state heads share one
// query head) and key-head sharing (key_heads < key_value_heads, several
// state heads share one physical key head). Expected numbers are computed by
// hand from the linear-attention recurrence (state += outer(key, value);
// output = state^T @ query) independently of ProcessHead's internal
// decomposition, so these tests would catch a regression in either grouping
// direction without exercising the same code path as the implementation.
TEST(LinearAttention, InverseGroupingSharesQueryAcrossStateHeads) {
  onnx_light_cpu::LinearAttentionParameters parameters;
  parameters.batch_size = 1;
  parameters.sequence_length = 1;
  parameters.query_heads = 1;     // Hq
  parameters.key_value_heads = 2; // Hkv > Hq: inverse grouping.
  parameters.key_head_size = 2;
  parameters.value_head_size = 1;
  parameters.rule = onnx_light_cpu::LinearAttentionRule::kLinear;
  parameters.scale = 1.0f;

  const std::array<float, 2> query{2.0f, 3.0f};
  // Key/value hold one physical head per state head (Hk defaults to Hkv).
  const std::array<float, 4> key{1.0f, 1.0f, 2.0f, 1.0f};
  const std::array<float, 2> value{5.0f, 4.0f};
  std::array<float, 4> state{};
  std::array<float, 2> output{};

  onnx_light_cpu::LinearAttentionFloat32(parameters, query.data(), key.data(), value.data(),
                                         nullptr, nullptr, state.data(), output.data());

  // state_head 0: state = key[0:2] * value[0] = [5, 5]; output = 5*2+5*3=25.
  // state_head 1: state = key[2:4] * value[1] = [8, 4]; output = 8*2+4*3=28.
  EXPECT_EQ(state, (std::array<float, 4>{5.0f, 5.0f, 8.0f, 4.0f}));
  EXPECT_EQ(output, (std::array<float, 2>{25.0f, 28.0f}));
}

TEST(LinearAttention, KeyHeadSharingBroadcastsPhysicalKeyAcrossStateHeads) {
  onnx_light_cpu::LinearAttentionParameters parameters;
  parameters.batch_size = 1;
  parameters.sequence_length = 1;
  parameters.query_heads = 2;     // Hq == Hkv: standard (1:1) query grouping.
  parameters.key_value_heads = 2; // Hkv
  parameters.key_heads = 1;       // Hk < Hkv: both state heads share one key.
  parameters.key_head_size = 2;
  parameters.value_head_size = 1;
  parameters.rule = onnx_light_cpu::LinearAttentionRule::kLinear;
  parameters.scale = 1.0f;

  const std::array<float, 4> query{1.0f, 2.0f, 2.0f, 1.0f};
  const std::array<float, 2> key{3.0f, 2.0f}; // One physical head, shared by both state heads.
  const std::array<float, 2> value{4.0f, 5.0f};
  std::array<float, 4> state{};
  std::array<float, 2> output{};

  onnx_light_cpu::LinearAttentionFloat32(parameters, query.data(), key.data(), value.data(),
                                         nullptr, nullptr, state.data(), output.data());

  // state_head 0: state = key * value[0] = [12, 8]; output = 12*1+8*2=28.
  // state_head 1: state = key * value[1] = [15, 10]; output = 15*2+10*1=40.
  EXPECT_EQ(state, (std::array<float, 4>{12.0f, 8.0f, 15.0f, 10.0f}));
  EXPECT_EQ(output, (std::array<float, 2>{28.0f, 40.0f}));
}

} // namespace
