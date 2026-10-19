// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "ml_cpu_compute/execution.h"
#include "ml_cpu_compute/math/half_conversion.h"
#include "onnx_light_cpu/kernels/com_microsoft/causal_conv_with_state_kernel.h"
#include "onnx_light_cpu/kernels/com_microsoft/naive_causal_conv_with_state_kernel.h"
#include "onnx_light_cpu/kernels/register_kernels.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/random.h"
#include "onnx_core/runtime/kernels/tensor_compare.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

rt_ns::KernelContext Context() { return rt_ns::KernelContext(rt_ns::OpsetId("com.microsoft", 1)); }

rt_ns::Tensor MakeTensor(rt_ns::DataType dtype, const rt_ns::Shape &shape, std::uint64_t seed) {
  const auto values = rt_ns::Randn<float>(shape, seed);
  if (dtype == rt_ns::DataType::FLOAT16) {
    return rt_ns::MakeFloat16Tensor("", shape, values);
  }
  if (dtype == rt_ns::DataType::BFLOAT16) {
    return rt_ns::MakeBfloat16Tensor("", shape, values);
  }
  return rt_ns::Tensor::FromFloat("", shape, values);
}

void ExpectParity(const rt_ns::Tensor &actual, const rt_ns::Tensor &expected) {
  const bool is_float = actual.data_type == static_cast<int>(rt_ns::DataType::FLOAT);
  const auto comparison =
      rt_ns::CompareTensors(actual, expected, is_float ? 1.e-3 : 2.e-2, is_float ? 1.e-7 : 1.e-5);
  EXPECT_TRUE(comparison.close) << comparison.message;
}

TEST(CausalConvWithState, NaiveReferenceCoversTypesLayoutsAndStateBoundaries) {
  const onnx_light_cpu::CausalConvWithStateKernel kernel(Context());
  const onnx_light_cpu::NaiveCausalConvWithStateKernel naive(Context());
  for (const auto dtype :
       {rt_ns::DataType::FLOAT, rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16}) {
    for (const bool channels_last : {false, true}) {
      for (const std::int64_t channels : {0, 1, 3, 17}) {
        for (const std::int64_t length : {0, 1, 2, 7, 16, 17}) {
          for (const std::int64_t width : {1, 3}) {
            for (const std::int64_t dilation : {1, 2}) {
              for (const std::string activation : {"none", "silu", "swish"}) {
                for (const bool with_past : {false, true}) {
                  for (const bool with_bias : {false, true}) {
                    SCOPED_TRACE(::testing::Message()
                                 << static_cast<int>(dtype) << " last=" << channels_last
                                 << " C=" << channels << " L=" << length << " K=" << width
                                 << " dilation=" << dilation << " " << activation
                                 << " past=" << with_past << " bias=" << with_bias);
                    const std::int64_t pad = (width - 1) * dilation;
                    const rt_ns::Shape shape = channels_last ? rt_ns::Shape{2, length, channels, 1}
                                                             : rt_ns::Shape{2, channels, length};
                    const rt_ns::Shape state_shape = channels_last
                                                         ? rt_ns::Shape{2, pad, channels, 1}
                                                         : rt_ns::Shape{2, channels, pad};
                    const auto input = MakeTensor(dtype, shape, 1);
                    const auto weight = MakeTensor(dtype, {channels, 1, width}, 2);
                    const auto bias = MakeTensor(dtype, {channels}, 3);
                    const auto past = MakeTensor(dtype, state_shape, 4);
                    const auto expected =
                        naive(input, weight, with_bias ? &bias : nullptr,
                              with_past ? &past : nullptr, activation, channels_last, dilation);
                    const auto actual =
                        kernel(input, weight, with_bias ? &bias : nullptr,
                               with_past ? &past : nullptr, activation, channels_last, dilation);
                    ExpectParity(actual.first, expected.first);
                    ExpectParity(actual.second, expected.second);
                    ExpectParity(past, MakeTensor(dtype, state_shape, 4));
                    if (!channels_last && dilation == 1 && dtype != rt_ns::DataType::BFLOAT16) {
                      using Reference =
                          ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::CausalConvWithState;
                      const Reference reference(Context());
                      Reference::Attributes attributes;
                      attributes.activation = activation;
                      const auto upstream =
                          reference(input, weight, with_bias ? bias : rt_ns::Tensor{},
                                    with_past ? past : rt_ns::Tensor{}, attributes);
                      ExpectParity(expected.first, upstream.first);
                      ExpectParity(expected.second, upstream.second);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

struct InlineExecutor {
  std::int64_t calls = 0;
  std::int64_t blocks = 0;

  static void Run(void *context, std::int64_t count, void *task_context,
                  ml_cpu_compute::ExecutionBlockFn task) {
    auto &self = *static_cast<InlineExecutor *>(context);
    ++self.calls;
    self.blocks += count;
    for (std::int64_t block = 0; block < count; ++block) {
      task(task_context, block);
    }
  }
};

TEST(CausalConvWithState, SchedulingAccountsForConvolutionAndStateCopies) {
  InlineExecutor executor;
  const ml_cpu_compute::ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run};
  const ml_cpu_compute::ExecutionExecutorScope scope(&view);
  const onnx_light_cpu::CausalConvWithStateKernel kernel(Context());
  const onnx_light_cpu::NaiveCausalConvWithStateKernel naive(Context());
  const auto check = [&](std::int64_t channels, std::int64_t length, std::int64_t width) {
    const auto input = MakeTensor(rt_ns::DataType::FLOAT, {1, channels, length}, 1);
    const auto weight = MakeTensor(rt_ns::DataType::FLOAT, {channels, 1, width}, 2);
    const auto expected = naive(input, weight);
    const auto actual = kernel(input, weight);
    ExpectParity(actual.first, expected.first);
    ExpectParity(actual.second, expected.second);
  };
  check(64, 1, 4);
  EXPECT_EQ(executor.calls, 0);
  check(64, 512, 4);
  EXPECT_EQ(executor.calls, 1);
  EXPECT_GT(executor.blocks, 1);
  executor.calls = executor.blocks = 0;
  check(64, 0, 1025);
  EXPECT_EQ(executor.calls, 1);
  EXPECT_GT(executor.blocks, 1);
  executor.calls = executor.blocks = 0;
  {
    ml_cpu_compute::detail::ExecutionRegionScope nested;
    check(64, 512, 4);
  }
  EXPECT_EQ(executor.calls, 0);
}

TEST(CausalConvWithState, NaiveAndProductionDispatchAreIndependent) {
  using onnx_light_cpu::MicrosoftKernelImplementation;
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_domain("com.microsoft");
  node.set_op_type("CausalConvWithState");
  node.add_input("X");
  node.add_input("W");
  node.add_input("");
  node.add_input("past");
  node.add_output("Y");
  node.add_output("state");
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "activation", std::string("swish"));
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "channels_last", std::int64_t{1});
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "dilation", std::int64_t{2});
  rt_ns::RuntimeContext runtime{Context()};
  runtime.Set("X", MakeTensor(rt_ns::DataType::BFLOAT16, {2, 1, 3}, 1));
  runtime.Set("W", MakeTensor(rt_ns::DataType::BFLOAT16, {3, 1, 3}, 2));
  runtime.Set("past", MakeTensor(rt_ns::DataType::BFLOAT16, {2, 4, 3}, 3));
  onnx_light_cpu::RegisterAllKernels();
  auto optimized = rt_ns::KernelDispatchTable().at("com.microsoft:CausalConvWithState")(
      node, runtime, runtime.kernel_ctx());
  ASSERT_NE(dynamic_cast<onnx_light_cpu::CausalConvWithStateKernel *>(optimized.get()), nullptr);
  onnx_light_cpu::RegisterKernelForSession(runtime, "com.microsoft", "CausalConvWithState", true,
                                           MicrosoftKernelImplementation::NAIVE);
  auto naive = runtime.custom_kernels().at("com.microsoft:CausalConvWithState")(
      node, runtime, runtime.kernel_ctx());
  runtime.set_kernel_usage_enabled(true);
  naive->Run(runtime);
  EXPECT_EQ(runtime.GetKernelUsage(),
            (std::vector<std::string>{onnx_light_cpu::NaiveCausalConvWithStateKernel::kName}));
  const auto expected_output = runtime.Get("Y").ToOwned();
  const auto expected_state = runtime.Get("state").ToOwned();
  runtime.ClearKernelUsage();
  optimized->Run(runtime);
  EXPECT_EQ(runtime.GetKernelUsage(),
            (std::vector<std::string>{onnx_light_cpu::CausalConvWithStateKernel::kName}));
  ExpectParity(runtime.Get("Y"), expected_output);
  ExpectParity(runtime.Get("state"), expected_state);
}

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
      return bf16 ? ml_cpu_compute::detail::Bfloat16BitsToFloat(value)
                  : ml_cpu_compute::detail::Float16BitsToFloat(value);
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

TEST(CausalConvWithState, RejectsUndersizedBorrowedBuffers) {
  const onnx_light_cpu::CausalConvWithStateKernel kernel(Context());
  const onnx_light_cpu::NaiveCausalConvWithStateKernel naive(Context());
  const std::vector<float> storage{1, 2};
  const auto *bytes = reinterpret_cast<const std::uint8_t *>(storage.data());
  const auto input =
      rt_ns::Tensor::Borrow("", rt_ns::DataType::FLOAT, {1, 1, 2}, bytes, sizeof(float));
  const auto valid_input = rt_ns::Tensor::FromFloat("", {1, 1, 2}, {1, 2});
  const auto valid_weight = rt_ns::Tensor::FromFloat("", {1, 1, 1}, {1});
  const auto undersized_weight =
      rt_ns::Tensor::Borrow("", rt_ns::DataType::FLOAT, {1, 1, 2}, bytes, sizeof(float));
  const auto wider_weight = rt_ns::Tensor::FromFloat("", {1, 1, 2}, {1, 2});
  const auto undersized_bias = rt_ns::Tensor::Borrow("", rt_ns::DataType::FLOAT, {1}, bytes, 0);
  const auto undersized_past =
      rt_ns::Tensor::Borrow("", rt_ns::DataType::FLOAT, {1, 1, 1}, bytes, 0);
  const auto check = [&](const auto &implementation) {
    EXPECT_THROW((void)implementation(input, valid_weight), std::invalid_argument);
    EXPECT_THROW((void)implementation(valid_input, undersized_weight), std::invalid_argument);
    EXPECT_THROW((void)implementation(valid_input, wider_weight, &undersized_bias),
                 std::invalid_argument);
    EXPECT_THROW((void)implementation(valid_input, wider_weight, nullptr, &undersized_past),
                 std::invalid_argument);
  };
  check(kernel);
  check(naive);
}

} // namespace
