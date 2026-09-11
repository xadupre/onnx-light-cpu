// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/com_microsoft/linear_attention_kernel.h"
#include "onnx_light_cpu/kernels/com_microsoft/naive_linear_attention_kernel.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"
#include "onnx_light_cpu/kernels/register_kernels.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/random.h"
#include "onnx_core/runtime/kernels/tensor_compare.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

using onnx_light_cpu::MicrosoftLinearAttentionKernel;
using onnx_light_cpu::NaiveMicrosoftLinearAttentionKernel;
using rt_ns::Tensor;

void ExpectParity(const Tensor &actual, const Tensor &expected) {
  // FLOAT backend tolerances: atol=1e-7, rtol=1e-3.
  const auto comparison = rt_ns::CompareTensors(actual, expected, 1.e-3, 1.e-7);
  EXPECT_TRUE(comparison.close) << comparison.message;
}

TEST(MicrosoftLinearAttentionKernel, NaiveReferenceCoversMicrosoftContract) {
  const rt_ns::KernelContext ctx{rt_ns::OpsetId("com.microsoft", 1)};
  const MicrosoftLinearAttentionKernel optimized{ctx};
  const NaiveMicrosoftLinearAttentionKernel naive{ctx};
  for (const std::string rule : {"linear", "gated", "delta", "gated_delta"}) {
    for (const auto heads :
         {std::array<std::int64_t, 3>{2, 2, 2}, {4, 2, 2}, {1, 2, 2}, {2, 1, 2}, {1, 1, 2}}) {
      for (const std::int64_t sequence : {0, 1, 3}) {
        for (const bool with_past : {false, true}) {
          for (const bool per_dimension : {false, true}) {
            for (const bool shared_beta : {false, true}) {
              for (const float scale : {0.0f, 0.7f}) {
                SCOPED_TRACE(::testing::Message()
                             << rule << " heads=" << heads[0] << "," << heads[1] << "," << heads[2]
                             << " sequence=" << sequence << " past=" << with_past
                             << " per_dimension=" << per_dimension << " shared_beta=" << shared_beta
                             << " scale=" << scale);
                auto make_tensor = [](const rt_ns::Shape &shape, std::uint64_t seed) {
                  auto values = rt_ns::Randn<float>(shape, seed);
                  for (auto &value : values) {
                    value *= 0.1f;
                  }
                  return Tensor::FromFloat("", shape, values);
                };
                const Tensor query = make_tensor({2, sequence, heads[0] * 4}, 1);
                const Tensor key = make_tensor({2, sequence, heads[1] * 4}, 2);
                const Tensor value = make_tensor({2, sequence, heads[2] * 3}, 3);
                const Tensor past = make_tensor({2, heads[2], 4, 3}, 4);
                const Tensor decay =
                    make_tensor({2, sequence, heads[2] * (per_dimension ? 4 : 1)}, 5);
                const Tensor beta = make_tensor({2, sequence, shared_beta ? 1 : heads[2]}, 6);
                MicrosoftLinearAttentionKernel::Attributes attributes;
                attributes.update_rule = rule;
                attributes.query_heads = heads[0];
                attributes.key_value_heads = heads[2];
                attributes.scale = scale;
                attributes.chunk_size = 1;
                // Microsoft accepts and ignores well-shaped gates unused by the rule.
                const auto expected = naive(query, key, value, attributes,
                                            with_past ? &past : nullptr, &decay, &beta);
                const auto actual = optimized(query, key, value, attributes,
                                              with_past ? &past : nullptr, &decay, &beta);
                ExpectParity(actual.output, expected.output);
                ExpectParity(actual.present_state, expected.present_state);
                ExpectParity(past, make_tensor({2, heads[2], 4, 3}, 4));
              }
            }
          }
        }
      }
    }
  }
}

TEST(MicrosoftLinearAttentionKernel, NaiveAndDefaultDispatchAreIndependent) {
  using onnx_light_cpu::MicrosoftKernelImplementation;
  using ONNX_LIGHT_NAMESPACE::AddAttribute;
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_domain("com.microsoft");
  node.set_op_type("LinearAttention");
  for (const auto *name : {"query", "key", "value", "", "decay", "beta"}) {
    node.add_input(name);
  }
  node.add_output("output");
  node.add_output("present_state");
  AddAttribute(node, "q_num_heads", std::int64_t{1});
  AddAttribute(node, "kv_num_heads", std::int64_t{2});
  AddAttribute(node, "chunk_size", std::int64_t{1});
  // Omitted update_rule and scale must use the Microsoft defaults.
  rt_ns::RuntimeContext runtime{rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};
  runtime.Set("query", Tensor::FromFloat("", {1, 1, 2}, {0.2f, 0.3f}));
  runtime.Set("key", Tensor::FromFloat("", {1, 1, 2}, {0.1f, 0.4f}));
  runtime.Set("value", Tensor::FromFloat("", {1, 1, 2}, {0.5f, 0.6f}));
  runtime.Set("decay", Tensor::FromFloat("", {1, 1, 2}, {-0.1f, -0.2f}));
  runtime.Set("beta", Tensor::FromFloat("", {1, 1, 1}, {0.5f}));

  onnx_light_cpu::RegisterAllKernels();
  auto optimized = rt_ns::KernelDispatchTable().at("com.microsoft:LinearAttention")(node, runtime);
  ASSERT_NE(dynamic_cast<MicrosoftLinearAttentionKernel *>(optimized.get()), nullptr);
  onnx_light_cpu::RegisterKernelForSession(runtime, "com.microsoft", "LinearAttention", true,
                                           MicrosoftKernelImplementation::NAIVE);
  onnx_light_cpu::ClearUsedKernelNames();
  runtime.custom_kernels().at("com.microsoft:LinearAttention")(node, runtime);
  EXPECT_EQ(onnx_light_cpu::UsedKernelNames(),
            (std::vector<std::string>{NaiveMicrosoftLinearAttentionKernel::kName}));
  const Tensor reference_output = runtime.Get("output");
  const Tensor reference_state = runtime.Get("present_state");

  onnx_light_cpu::ClearUsedKernelNames();
  optimized->Run(runtime);
  EXPECT_EQ(onnx_light_cpu::UsedKernelNames(),
            (std::vector<std::string>{MicrosoftLinearAttentionKernel::kName}));
  ExpectParity(runtime.Get("output"), reference_output);
  ExpectParity(runtime.Get("present_state"), reference_state);
  ASSERT_TRUE(onnx_light_cpu::RegisterKernelGlobal("com.microsoft", "LinearAttention"));
  auto global = rt_ns::KernelDispatchTable().at("com.microsoft:LinearAttention")(node, runtime);
  EXPECT_NE(dynamic_cast<MicrosoftLinearAttentionKernel *>(global.get()), nullptr);
}

TEST(MicrosoftLinearAttentionKernel, NaiveRejectsUnsupportedInputsAndOutputs) {
  const rt_ns::KernelContext ctx{rt_ns::OpsetId("com.microsoft", 1)};
  NaiveMicrosoftLinearAttentionKernel naive{ctx};
  NaiveMicrosoftLinearAttentionKernel::Attributes attributes;
  attributes.update_rule = "linear";
  attributes.query_heads = 1;
  attributes.key_value_heads = 1;
  const Tensor input = Tensor::FromFloat("", {1, 1, 1}, {1.0f});
  attributes.state_window = 8;
  EXPECT_THROW((void)naive(input, input, input, attributes), std::invalid_argument);
  attributes.state_window = 0;
  const Tensor half = rt_ns::MakeFloat16Tensor("", {1, 1, 1}, {1.0f});
  EXPECT_THROW((void)naive(half, half, half, attributes), std::invalid_argument);
  attributes.update_rule = "gated_delta";
  EXPECT_THROW((void)naive(input, input, input, attributes), std::invalid_argument);

  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.add_input("query");
  node.add_input("key");
  node.add_input("value");
  node.add_output("output");
  naive.set_node(node);
  rt_ns::RuntimeContext runtime{ctx};
  EXPECT_THROW(naive.Run(runtime), std::invalid_argument);
  node.add_output("");
  EXPECT_THROW(naive.Run(runtime), std::invalid_argument);
}

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
