// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/com_microsoft/include_com_microsoft_cases.h"
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

rt_ns::Tensor MakeTensor(rt_ns::DataType dtype, const rt_ns::Shape &shape,
                         const std::vector<float> &values) {
  if (dtype == rt_ns::DataType::FLOAT16) {
    return rt_ns::MakeFloat16Tensor("", shape, values);
  }
  if (dtype == rt_ns::DataType::BFLOAT16) {
    return rt_ns::MakeBfloat16Tensor("", shape, values);
  }
  return rt_ns::Tensor::FromFloat("", shape, values);
}

} // namespace

void RegisterCpuCausalConvWithStateCases(std::vector<TestCase> &registry, TestMode mode) {
  const rt_ns::OpsetId opset(kMicrosoftDomain, 1);
  for (const auto dtype :
       {rt_ns::DataType::FLOAT, rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16}) {
    for (const bool with_past : {false, true}) {
      for (const bool channels_last : {false, true}) {
        if (mode == TestMode::BENCHMARK && (with_past || channels_last)) {
          continue;
        }
        const std::int64_t length = mode == TestMode::BENCHMARK ? 128 : 3;
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
        const std::string name = "test_cpu_causal_conv_" + std::to_string(static_cast<int>(dtype)) +
                                 (with_past ? "_past" : "_zero") +
                                 (channels_last ? "_last" : "_first") +
                                 (mode == TestMode::BENCHMARK ? "_benchmark" : "");
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
              std::vector<float> expected(values.size());
              for (std::size_t i = 0; i < values.size(); ++i) {
                const float previous = i > 0 ? values[i - 1] : (with_past ? 5.0f : 0.0f);
                const float older =
                    i > 1 ? values[i - 2] : (with_past ? (i == 0 ? 4.0f : 5.0f) : 0.0f);
                expected[i] = older + 2.0f * previous + 3.0f * values[i];
              }
              const std::vector<float> state = {values[values.size() - 2], values.back()};
              std::vector<rt_ns::Tensor> inputs;
              inputs.push_back(x);
              inputs.push_back(w);
              if (with_past) {
                inputs.push_back(MakeTensor(dtype, {1}, {0}));
                inputs.push_back(MakeTensor(dtype, state_shape, {4, 5}));
              }
              return bt_ns::IoData{
                  std::move(inputs),
                  {MakeTensor(dtype, shape, expected), MakeTensor(dtype, state_shape, state)}};
            },
            "backend-test", bt_ns::TestCaseTag::AI_RT);
      }
    }
  }
}

} // namespace onnx_light_cpu::backend_test
