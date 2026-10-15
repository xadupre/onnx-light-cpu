// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/com_microsoft/include_com_microsoft_cases.h"
#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"

#include "onnx_light_cpu/kernels/com_microsoft/moe_kernel.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace onnx_light_cpu::backend_test {
namespace {

namespace bt_ns = ONNX_LIGHT_NAMESPACE::core::backend_test;
namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
using rt_ns::DataType;
using rt_ns::Tensor;

ONNX_LIGHT_NAMESPACE::NodeProto MakeNode(bool swiglu, bool normalize, int64_t k) {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_domain(kMicrosoftDomain);
  node.set_op_type("MoE");
  for (const char *name : {"input", "router_probs", "fc1_experts_weights", "fc1_experts_bias",
                           "fc2_experts_weights", "fc2_experts_bias", "", ""}) {
    node.add_input(name);
  }
  node.add_output("output");
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "k", k);
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "activation_type",
                                     std::string(swiglu ? "swiglu" : "relu"));
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "swiglu_fusion", std::int64_t{swiglu ? 1 : 0});
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "normalize_routing_weights",
                                     std::int64_t{normalize ? 1 : 0});
  if (swiglu) {
    ONNX_LIGHT_NAMESPACE::AddAttribute(node, "activation_alpha", 1.702f);
    ONNX_LIGHT_NAMESPACE::AddAttribute(node, "activation_beta", 1.0f);
    ONNX_LIGHT_NAMESPACE::AddAttribute(node, "swiglu_limit", 7.0f);
  }
  return node;
}

bt_ns::IoData BuildData(int64_t tokens, int64_t hidden, int64_t intermediate, int64_t experts,
                        DataType type, bool swiglu, bool normalize, int64_t k,
                        bool generate_expected_outputs) {
  std::vector<Tensor> inputs{
      MakeBenchmarkTensor(type, {tokens, hidden}, 1001),
      MakeBenchmarkTensor(type, {tokens, experts}, 1002),
      MakeBenchmarkTensor(type, {experts, intermediate * (swiglu ? 2 : 1), hidden}, 1003),
      MakeBenchmarkTensor(type, {experts, intermediate * (swiglu ? 2 : 1)}, 1004),
      MakeBenchmarkTensor(type, {experts, hidden, intermediate}, 1005),
      MakeBenchmarkTensor(type, {experts, hidden}, 1006),
  };
  if (!generate_expected_outputs) {
    return bt_ns::IoData{std::move(inputs), {}, {}, false};
  }
  const rt_ns::KernelContext ctx{rt_ns::OpsetId(kMicrosoftDomain, 1)};
  const MoEKernel kernel{MakeNode(swiglu, normalize, k), ctx};
  Tensor output = kernel(inputs[0], inputs[1], inputs[2], &inputs[3], inputs[4], &inputs[5]);
  return bt_ns::IoData{std::move(inputs), {std::move(output)}};
}

} // namespace

void RegisterCpuMoECases(std::vector<TestCase> &registry, TestMode mode) {
  const bool benchmark = mode == TestMode::BENCHMARK;
  for (const DataType type : {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16}) {
    for (const bool swiglu : {false, true}) {
      for (const bool normalize : {false, true}) {
        const auto tokens = benchmark ? 8 : 2;
        const auto hidden = benchmark ? 128 : 4;
        const auto intermediate = benchmark ? 256 : 3;
        const auto experts = benchmark ? 8 : 3;
        const int64_t k = normalize ? 2 : 1;
        const std::string name = "test_cpu_moe_" + std::string(swiglu ? "swiglu" : "relu") +
                                 (normalize ? "_normalized_" : "_unnormalized_") +
                                 DataTypeSuffix(type) + (benchmark ? "_benchmark" : "");
        const auto node = MakeNode(swiglu, normalize, k);
        const std::vector<rt_ns::OpsetId> opsets = {rt_ns::DefaultOpset(23),
                                                    rt_ns::OpsetId(kMicrosoftDomain, 1)};
        if (benchmark) {
          const auto fc1_width = intermediate * (swiglu ? 2 : 1);
          bt_ns::Expect(registry, node, name, opsets,
                        {tokens * hidden, tokens * experts, experts * fc1_width * hidden,
                         experts * fc1_width, experts * hidden * intermediate, experts * hidden},
                        {tokens * hidden},
                        [=](bool generate_expected_outputs) {
                          return BuildData(tokens, hidden, intermediate, experts, type, swiglu,
                                           normalize, k, generate_expected_outputs);
                        },
                        "backend-test", bt_ns::TestCaseTag::AI_RT,
                        {bt_ns::TensorTypeSpec(static_cast<std::int32_t>(type), {tokens, hidden})});
        } else {
          bt_ns::Expect(
              registry, node, name, opsets,
              [=]() {
                return BuildData(tokens, hidden, intermediate, experts, type, swiglu, normalize, k,
                                 true);
              },
              "backend-test", bt_ns::TestCaseTag::AI_RT);
        }
        registry.back().rtol = type == DataType::BFLOAT16  ? 3e-2
                               : type == DataType::FLOAT16 ? 5e-3
                                                           : 1e-5;
        registry.back().atol = type == DataType::BFLOAT16  ? 3e-2
                               : type == DataType::FLOAT16 ? 5e-3
                                                           : 1e-6;
      }
    }
  }
}

} // namespace onnx_light_cpu::backend_test
