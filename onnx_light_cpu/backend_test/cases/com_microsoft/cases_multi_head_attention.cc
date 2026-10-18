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

void FloatAttribute(NodeProto &node, const char *name, float value) {
  auto *attribute = node.add_attribute();
  attribute->set_name(name);
  attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::FLOAT);
  attribute->set_f(value);
}

const char *TypeName(DataType dtype) {
  return dtype == DataType::FLOAT ? "float" : dtype == DataType::FLOAT16 ? "float16" : "bfloat16";
}

void RegisterNontrivialCase(std::vector<TestCase> &registry, DataType dtype, bool empty_past) {
  NodeProto node;
  node.set_domain(kMicrosoftDomain);
  node.set_op_type("MultiHeadAttention");
  for (const char *name : {"query", "key", "value", "bias", "mask", "attention_bias",
                           empty_past ? "past_key" : "", empty_past ? "past_value" : ""})
    node.add_input(name);
  node.add_output("output");
  if (empty_past) {
    node.add_output("present_key");
    node.add_output("present_value");
  }
  IntAttribute(node, "num_heads", 1);
  FloatAttribute(node, "scale", 0.5f);
  const std::string name = std::string("test_cpu_multi_head_attention_nontrivial_") +
                           (empty_past ? "empty_past_" : "prefill_") + TypeName(dtype);
  bt_ns::Expect(
      registry, node, name, {rt_ns::DefaultOpset(13), rt_ns::OpsetId(kMicrosoftDomain, 1)},
      [=]() -> bt_ns::IoData {
        std::vector<Tensor> inputs{
            Floats(dtype, "query", {2, 2, 1}, {2, 0, -2, 0}),
            Floats(dtype, "key", {2, 2, 1}, {1, 0, 1, 0}),
            Floats(dtype, "value", {2, 2, 1}, {2, 6, 3, 7}),
            Floats(dtype, "bias", {3}, {1, -1, 2}),
            Tensor::FromInt32("mask", {2, 2, 2}, {1, 1, 1, 1, 1, 1, 1, 1}),
            Floats(dtype, "attention_bias", {1, 1, 2, 2}, {0, 0.5f, 0, 0.5f})};
        if (empty_past) {
          inputs.push_back(Floats(dtype, "past_key", {2, 1, 0, 1}, {}));
          inputs.push_back(Floats(dtype, "past_value", {2, 1, 0, 1}, {}));
        }
        // Analytical softmax logits for the four (batch, query) rows are
        // [0,-1], [0,0], [0,1], [0,0]. K/V projection bias applies to
        // current values, including when the supplied past has length zero.
        const float probability = 1.0f / (1.0f + std::exp(1.0f));
        std::vector<Tensor> outputs{
            Floats(dtype, "output", {2, 2, 1}, {4 + 4 * probability, 6, 9 - 4 * probability, 7})};
        if (empty_past) {
          outputs.push_back(Floats(dtype, "present_key", {2, 1, 2, 1}, {0, -1, 0, -1}));
          outputs.push_back(Floats(dtype, "present_value", {2, 1, 2, 1}, {4, 8, 5, 9}));
        }
        return bt_ns::IoData{std::move(inputs), std::move(outputs)};
      },
      "backend-test", bt_ns::TestCaseTag::AI_RT);
}

void RegisterDecodeCase(std::vector<TestCase> &registry, DataType dtype) {
  NodeProto node;
  node.set_domain(kMicrosoftDomain);
  node.set_op_type("MultiHeadAttention");
  for (const char *name :
       {"query", "key", "value", "bias", "mask", "attention_bias", "past_key", "past_value"})
    node.add_input(name);
  for (const char *name : {"output", "present_key", "present_value"})
    node.add_output(name);
  IntAttribute(node, "num_heads", 1);
  IntAttribute(node, "unidirectional", 1);
  FloatAttribute(node, "scale", 0.5f);
  bt_ns::Expect(
      registry, node, std::string("test_cpu_multi_head_attention_decode_") + TypeName(dtype),
      {rt_ns::DefaultOpset(13), rt_ns::OpsetId(kMicrosoftDomain, 1)},
      [=]() -> bt_ns::IoData {
        std::vector<Tensor> inputs{Floats(dtype, "query", {1, 1, 1}, {1}),
                                   Floats(dtype, "key", {1, 1, 1}, {2}),
                                   Floats(dtype, "value", {1, 1, 1}, {3}),
                                   Floats(dtype, "bias", {3}, {1, 1, 1}),
                                   Tensor::FromInt32("mask", {1}, {2}),
                                   Floats(dtype, "attention_bias", {1, 1, 1, 2}, {1, -1}),
                                   Floats(dtype, "past_key", {1, 1, 1, 1}, {-1}),
                                   Floats(dtype, "past_value", {1, 1, 1, 1}, {5})};
        // Past/current logits are [0,2], so the output is the weighted
        // average of past value 5 and projected current value 4.
        std::vector<Tensor> outputs{
            Floats(dtype, "output", {1, 1, 1}, {4 + 1.0f / (1.0f + std::exp(2.0f))}),
            Floats(dtype, "present_key", {1, 1, 2, 1}, {-1, 3}),
            Floats(dtype, "present_value", {1, 1, 2, 1}, {5, 4})};
        return bt_ns::IoData{std::move(inputs), std::move(outputs)};
      },
      "backend-test", bt_ns::TestCaseTag::AI_RT);
}

void RegisterCrossCase(std::vector<TestCase> &registry, DataType dtype) {
  NodeProto node;
  node.set_domain(kMicrosoftDomain);
  node.set_op_type("MultiHeadAttention");
  for (const char *name : {"query", "key", "value", "bias", "", "attention_bias"})
    node.add_input(name);
  node.add_output("output");
  IntAttribute(node, "num_heads", 2);
  FloatAttribute(node, "scale", 0.5f);
  bt_ns::Expect(
      registry, node,
      std::string("test_cpu_multi_head_attention_whisper_scores_") + TypeName(dtype),
      {rt_ns::DefaultOpset(13), rt_ns::OpsetId(kMicrosoftDomain, 1)},
      [=]() -> bt_ns::IoData {
        std::vector<Tensor> inputs{Floats(dtype, "query", {1, 1, 2}, {1, -1}),
                                   Floats(dtype, "key", {1, 2, 2, 1}, {0, 2, 0, 2}),
                                   Floats(dtype, "value", {1, 2, 2, 1}, {2, 6, 3, 9}),
                                   Floats(dtype, "bias", {6}, {1, 0, 100, 100, 100, 100}),
                                   Floats(dtype, "attention_bias", {1, 2, 1, 2}, {0, -1, 0, 1})};
        // Rank-4 K/V ignore their projection-bias segments. Head logits
        // are [0,1] and [0,0], respectively.
        const float probability = 1.0f / (1.0f + std::exp(1.0f));
        std::vector<Tensor> outputs{Floats(dtype, "output", {1, 1, 2}, {6 - 4 * probability, 6})};
        return bt_ns::IoData{std::move(inputs), std::move(outputs)};
      },
      "backend-test", bt_ns::TestCaseTag::AI_RT);
}
} // namespace

void RegisterCpuMultiHeadAttentionCases(std::vector<TestCase> &registry, TestMode mode) {
  if (mode == TestMode::BENCHMARK)
    return;
  for (const auto dtype : {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16}) {
    RegisterNontrivialCase(registry, dtype, false);
    RegisterNontrivialCase(registry, dtype, true);
    RegisterDecodeCase(registry, dtype);
    RegisterCrossCase(registry, dtype);
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
