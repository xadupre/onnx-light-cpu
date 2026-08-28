// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/com_microsoft/include_com_microsoft_cases.h"

#include "onnx_light_cpu/kernels/attention/attention_kernel.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/kernels/random.h"

#include <cstdint>
#include <vector>

namespace onnx_light_cpu::backend_test {

namespace {

namespace bt_ns = ONNX_LIGHT_NAMESPACE::core::backend_test;
namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

using bt_ns::Expect;
using bt_ns::IoData;
using bt_ns::TestCase;
using bt_ns::TestMode;
using ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::DefaultOpset;
using rt_ns::KernelContext;
using rt_ns::OpsetId;
using rt_ns::Tensor;

void AddIntAttribute(NodeProto &node, const char *name, std::int64_t value) {
  auto *attribute = node.add_attribute();
  attribute->set_name(name);
  attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::AttributeType::INT);
  attribute->set_i(value);
}

NodeProto MakeGroupQueryAttentionNode() {
  NodeProto node;
  node.set_op_type("GroupQueryAttention");
  node.set_domain(kMicrosoftDomain);
  node.add_input("query");
  node.add_input("key");
  node.add_input("value");
  node.add_input("");
  node.add_input("");
  node.add_input("seqlens_k");
  node.add_input("total_sequence_length");
  node.add_output("output");
  AddIntAttribute(node, "num_heads", 4);
  AddIntAttribute(node, "kv_num_heads", 2);
  return node;
}

NodeProto MakeAttentionNode() {
  NodeProto node;
  node.set_op_type("Attention");
  node.add_input("query");
  node.add_input("key");
  node.add_input("value");
  node.add_output("output");
  AddIntAttribute(node, "q_num_heads", 4);
  AddIntAttribute(node, "kv_num_heads", 2);
  AddIntAttribute(node, "is_causal", 1);
  return node;
}

} // namespace

void RegisterCpuGroupQueryAttentionCases(std::vector<TestCase> &registry, TestMode mode) {
  if (mode == TestMode::BENCHMARK) {
    return;
  }
  const OpsetId microsoft_opset(kMicrosoftDomain, 1);
  Expect(registry, MakeGroupQueryAttentionNode(), "test_cpu_group_query_attention_gqa_causal",
         {DefaultOpset(23), microsoft_opset}, [=]() -> IoData {
           Tensor query = Tensor::FromFloat("", {2, 3, 32}, rt_ns::Randn<float>({2, 3, 32}, 41));
           Tensor key = Tensor::FromFloat("", {2, 3, 16}, rt_ns::Randn<float>({2, 3, 16}, 42));
           Tensor value = Tensor::FromFloat("", {2, 3, 16}, rt_ns::Randn<float>({2, 3, 16}, 43));
           Tensor seqlens_k = Tensor::FromInt32("", {2}, {2, 2});
           Tensor total_sequence_length = Tensor::FromInt32("", {}, {3});
           AttentionKernel attention{KernelContext{DefaultOpset(23)}};
           Tensor output = attention(MakeAttentionNode(), query, key, value, nullptr);
           return IoData{{std::move(query), std::move(key), std::move(value), std::move(seqlens_k),
                          std::move(total_sequence_length)},
                         {std::move(output)}};
         });
}

} // namespace onnx_light_cpu::backend_test
