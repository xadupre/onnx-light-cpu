// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/com_microsoft/include_com_microsoft_cases.h"
#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"
#include "onnx_light_cpu/kernels/com_microsoft/causal_conv_with_state_kernel.h"
#include "onnx_light_cpu/kernels/com_microsoft/naive_causal_conv_with_state_kernel.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/cast_helper.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace onnx_light_cpu::backend_test {
namespace {

namespace bt_ns = ONNX_LIGHT_NAMESPACE::core::backend_test;
namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

} // namespace

void RegisterCpuCausalConvWithStateCases(std::vector<TestCase> &registry, TestMode mode) {
  const rt_ns::OpsetId opset(kMicrosoftDomain, 1);
  if (mode == TestMode::BENCHMARK) {
    for (const std::int64_t length : {1, 128, 513}) {
      for (const auto dtype :
           {rt_ns::DataType::FLOAT, rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16}) {
        for (const bool channels_last : {false, true}) {
          constexpr std::int64_t channels = 64;
          const rt_ns::Shape shape =
              channels_last ? rt_ns::Shape{1, length, channels} : rt_ns::Shape{1, channels, length};
          const rt_ns::Shape state_shape =
              channels_last ? rt_ns::Shape{1, 2, channels} : rt_ns::Shape{1, channels, 2};
          ONNX_LIGHT_NAMESPACE::NodeProto node;
          node.set_domain(kMicrosoftDomain);
          node.set_op_type("CausalConvWithState");
          node.add_input("input");
          node.add_input("weight");
          node.add_output("output");
          node.add_output("present_state");
          auto *attr = node.add_attribute();
          attr->set_name("channels_last");
          attr->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::INT);
          attr->set_i(channels_last);
          const std::string name = "test_cpu_causal_conv_c64_l" + std::to_string(length) + "_" +
                                   DataTypeSuffix(dtype) + (channels_last ? "_last" : "_first") +
                                   "_benchmark";
          bt_ns::Expect(registry, node, name, {rt_ns::DefaultOpset(13), opset},
                        {channels * length, channels * 3}, {channels * length, channels * 2},
                        [=](bool generate_expected_outputs) -> bt_ns::IoData {
                          auto input = MakeBenchmarkTensor(dtype, shape, 1);
                          auto weight = MakeBenchmarkTensor(dtype, {channels, 1, 3}, 2);
                          if (!generate_expected_outputs) {
                            return bt_ns::IoData{
                                {std::move(input), std::move(weight)}, {}, {}, false};
                          }
                          const CausalConvWithStateKernel kernel{rt_ns::KernelContext{opset}};
                          auto [output, state] =
                              kernel(input, weight, nullptr, nullptr, "none", channels_last);
                          return bt_ns::IoData{{std::move(input), std::move(weight)},
                                               {std::move(output), std::move(state)}};
                        },
                        "backend-test", bt_ns::TestCaseTag::AI_RT,
                        {bt_ns::TensorTypeSpec(static_cast<std::int32_t>(dtype), shape),
                         bt_ns::TensorTypeSpec(static_cast<std::int32_t>(dtype), state_shape)});
        }
      }
    }
    return;
  }
  for (const auto dtype :
       {rt_ns::DataType::FLOAT, rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16}) {
    for (const bool with_past : {false, true}) {
      for (const bool channels_last : {false, true}) {
        const std::int64_t length = 3;
        ONNX_LIGHT_NAMESPACE::NodeProto node;
        node.set_domain(kMicrosoftDomain);
        node.set_op_type("CausalConvWithState");
        node.add_input("input");
        node.add_input("weight");
        node.add_input(with_past ? "bias" : "");
        node.add_input(with_past ? "past_state" : "");
        node.add_output("output");
        node.add_output("present_state");
        if (channels_last) {
          auto *attr = node.add_attribute();
          attr->set_name("channels_last");
          attr->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::INT);
          attr->set_i(1);
        }
        const std::string name = std::string("test_cpu_causal_conv_") + DataTypeSuffix(dtype) +
                                 (with_past ? "_past" : "_zero") +
                                 (channels_last ? "_last" : "_first");
        bt_ns::Expect(
            registry, node, name, {rt_ns::DefaultOpset(13), opset},
            [=]() -> bt_ns::IoData {
              const rt_ns::Shape shape =
                  channels_last ? rt_ns::Shape{1, length, 1} : rt_ns::Shape{1, 1, length};
              const rt_ns::Shape state_shape =
                  channels_last ? rt_ns::Shape{1, 2, 1} : rt_ns::Shape{1, 1, 2};
              std::vector<float> values(static_cast<std::size_t>(length));
              for (std::int64_t i = 0; i < length; ++i) {
                values[static_cast<std::size_t>(i)] = static_cast<float>(i % 4 + 1);
              }
              auto x = MakeTensor(dtype, shape, values);
              auto w = MakeTensor(dtype, {1, 1, 3}, {1, 2, 3});
              auto bias = MakeTensor(dtype, {1}, {0});
              auto past = MakeTensor(dtype, state_shape, {4, 5});
              const NaiveCausalConvWithStateKernel reference{rt_ns::KernelContext{opset}};
              auto [expected, state] =
                  reference(x, w, with_past ? &bias : nullptr, with_past ? &past : nullptr, "none",
                            channels_last);
              std::vector<rt_ns::Tensor> inputs;
              inputs.push_back(x);
              inputs.push_back(w);
              if (with_past) {
                inputs.push_back(std::move(bias));
                inputs.push_back(std::move(past));
              }
              return bt_ns::IoData{std::move(inputs), {std::move(expected), std::move(state)}};
            },
            "backend-test", bt_ns::TestCaseTag::AI_RT);
      }
    }
  }
}

} // namespace onnx_light_cpu::backend_test
