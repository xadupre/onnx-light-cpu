// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/com_microsoft/linear_attention_kernel.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

using onnx_light_cpu::MicrosoftLinearAttentionKernel;
using rt_ns::Tensor;

TEST(MicrosoftLinearAttentionKernel, SupportsInverseGroupingAndSharedKeyHeads) {
  const MicrosoftLinearAttentionKernel kernel{rt_ns::KernelContext{rt_ns::DefaultOpset(27)}};
  MicrosoftLinearAttentionKernel::Attributes attributes;
  attributes.update_rule = "linear";
  attributes.query_heads = 1;
  attributes.key_value_heads = 2;
  attributes.scale = 1.0f;

  const Tensor query = Tensor::FromFloat("query", {1, 1, 2}, {2.0f, 3.0f});
  const Tensor key = Tensor::FromFloat("key", {1, 1, 2}, {1.0f, 1.0f});
  const Tensor value = Tensor::FromFloat("value", {1, 1, 2}, {5.0f, 4.0f});
  const auto result = kernel(query, key, value, attributes);

  EXPECT_EQ(result.output.shape, (rt_ns::Shape{1, 1, 2}));
  EXPECT_EQ(result.present_state.shape, (rt_ns::Shape{1, 2, 2, 1}));
  EXPECT_EQ(std::vector<float>(result.output.AsFloat(), result.output.AsFloat() + 2),
            (std::vector<float>{25.0f, 20.0f}));
}

TEST(MicrosoftLinearAttentionKernel, RejectsUnsupportedStateWindow) {
  const MicrosoftLinearAttentionKernel kernel{rt_ns::KernelContext{rt_ns::DefaultOpset(27)}};
  MicrosoftLinearAttentionKernel::Attributes attributes;
  attributes.update_rule = "linear";
  attributes.query_heads = 1;
  attributes.key_value_heads = 1;
  attributes.state_window = 8;
  const Tensor input = Tensor::FromFloat("input", {1, 1, 1}, {1.0f});
  EXPECT_THROW((void)kernel(input, input, input, attributes), std::invalid_argument);
}

TEST(MicrosoftLinearAttentionKernel, RejectsNonFloatActivations) {
  const MicrosoftLinearAttentionKernel kernel{rt_ns::KernelContext{rt_ns::DefaultOpset(27)}};
  MicrosoftLinearAttentionKernel::Attributes attributes;
  attributes.update_rule = "linear";
  attributes.query_heads = 1;
  attributes.key_value_heads = 1;
  const Tensor input = rt_ns::MakeFloat16Tensor("input", {1, 1, 1}, {1.0f});
  EXPECT_THROW((void)kernel(input, input, input, attributes), std::invalid_argument);
}

} // namespace
