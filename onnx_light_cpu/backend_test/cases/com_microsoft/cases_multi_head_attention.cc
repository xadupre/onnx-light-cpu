// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/com_microsoft/include_com_microsoft_cases.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/cast_helper.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace onnx_light_cpu::backend_test {
namespace {
namespace bt_ns = ONNX_LIGHT_NAMESPACE::core::backend_test;
namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
using ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::DataType;
using rt_ns::Tensor;

Tensor Floats(DataType dtype, const std::string &name, rt_ns::Shape shape,
              const std::vector<float> &values) {
  if (dtype == DataType::FLOAT16)
    return rt_ns::MakeFloat16Tensor(name, shape, values);
  if (dtype == DataType::BFLOAT16)
    return rt_ns::MakeBfloat16Tensor(name, shape, values);
  return Tensor::FromFloat(name, shape, values);
}

void IntAttribute(NodeProto &node, const char *name, int64_t value) {
  auto *attribute = node.add_attribute();
  attribute->set_name(name);
  attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::INT);
  attribute->set_i(value);
}
} // namespace

void RegisterCpuMultiHeadAttentionCases(std::vector<TestCase> &registry, TestMode mode) {
  if (mode == TestMode::BENCHMARK)
    return;
  for (const auto dtype : {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16}) {
    for (const bool cross : {false, true}) {
      for (const bool cached : {false, true}) {
        if (cross && cached)
          continue;
        NodeProto node;
        node.set_domain(kMicrosoftDomain);
        node.set_op_type("MultiHeadAttention");
        for (const char *name : {"query", "key", "value", "", "mask", "attention_bias",
                                 cached ? "past_key" : "", cached ? "past_value" : ""})
          node.add_input(name);
        node.add_output("output");
        if (cached) {
          node.add_output("present_key");
          node.add_output("present_value");
        }
        IntAttribute(node, "num_heads", 1);
        if (cached)
          IntAttribute(node, "unidirectional", 1);
        const std::string name = "test_cpu_multi_head_attention_" +
                                 std::string(cross    ? "cross_"
                                             : cached ? "cache_"
                                                      : "self_") +
                                 (dtype == DataType::FLOAT     ? "float"
                                  : dtype == DataType::FLOAT16 ? "float16"
                                                               : "bfloat16");
        const auto opset = rt_ns::OpsetId(kMicrosoftDomain, 1);
        bt_ns::Expect(
            registry, node, name, {rt_ns::DefaultOpset(13), opset},
            [=]() -> bt_ns::IoData {
              Tensor q = Floats(dtype, "query", {1, 2, 2}, {0, 0, 0, 0});
              Tensor k =
                  Floats(dtype, "key", cross ? rt_ns::Shape{1, 1, 2, 2} : rt_ns::Shape{1, 2, 2},
                         {0, 0, 0, 0});
              Tensor v =
                  Floats(dtype, "value", cross ? rt_ns::Shape{1, 1, 2, 2} : rt_ns::Shape{1, 2, 2},
                         {2, 4, 6, 8});
              Tensor mask = Tensor::FromInt32("mask", {1, cached ? 3 : 2},
                                              cached ? std::vector<int32_t>{1, 0, 0}
                                                     : std::vector<int32_t>{1, 0});
              Tensor attention_bias = Floats(dtype, "attention_bias", {1, 1, 2, cached ? 3 : 2},
                                             cached ? std::vector<float>{0, 0, 0, 0, 0, 0}
                                                    : std::vector<float>{0, 0, 0, 0});
              std::vector<Tensor> inputs{q, k, v, mask, attention_bias};
              std::vector<Tensor> outputs;
              if (cached) {
                inputs.push_back(Floats(dtype, "past_key", {1, 1, 1, 2}, {0, 0}));
                inputs.push_back(Floats(dtype, "past_value", {1, 1, 1, 2}, {10, 12}));
                outputs.push_back(Floats(dtype, "output", {1, 2, 2}, {10, 12, 10, 12}));
                outputs.push_back(Floats(dtype, "present_key", {1, 1, 3, 2}, {0, 0, 0, 0, 0, 0}));
                outputs.push_back(
                    Floats(dtype, "present_value", {1, 1, 3, 2}, {10, 12, 2, 4, 6, 8}));
              } else {
                outputs.push_back(Floats(dtype, "output", {1, 2, 2}, {2, 4, 2, 4}));
              }
              return bt_ns::IoData{std::move(inputs), std::move(outputs)};
            },
            "backend-test", bt_ns::TestCaseTag::AI_RT);
      }
    }
  }
}
} // namespace onnx_light_cpu::backend_test
