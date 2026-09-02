// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/attention/include_attention_cases.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/kernels/random.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
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
using rt_ns::DataType;
using rt_ns::DefaultOpset;
using rt_ns::KernelContext;
using rt_ns::OpsetId;
using rt_ns::Shape;
using rt_ns::Tensor;

struct LinearAttentionCase {
  const char *name;
  std::int64_t batch;
  std::int64_t sequence;
  std::int64_t query_heads;
  std::int64_t key_value_heads;
  std::int64_t key_head_size;
  std::int64_t value_head_size;
  const char *rule;
  bool with_past;
  bool decay_per_dimension;
  bool beta_shared;
  DataType data_type;
  std::uint64_t seed;
};

void AddIntAttribute(NodeProto &node, const char *name, std::int64_t value) {
  auto *attribute = node.add_attribute();
  attribute->set_name(name);
  attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::AttributeType::INT);
  attribute->set_i(value);
}

void AddFloatAttribute(NodeProto &node, const char *name, float value) {
  auto *attribute = node.add_attribute();
  attribute->set_name(name);
  attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::AttributeType::FLOAT);
  attribute->set_f(value);
}

void AddStringAttribute(NodeProto &node, const char *name, const std::string &value) {
  auto *attribute = node.add_attribute();
  attribute->set_name(name);
  attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::AttributeType::STRING);
  attribute->set_s(value);
}

bool UsesDecay(const std::string &rule) { return rule == "gated" || rule == "gated_delta"; }

bool UsesBeta(const std::string &rule) { return rule == "delta" || rule == "gated_delta"; }

std::int64_t Count(const Shape &shape) {
  std::int64_t count = 1;
  for (std::int64_t dimension : shape) {
    count *= dimension;
  }
  return count;
}

Tensor MakeScaledRandomTensor(DataType data_type, const Shape &shape, std::uint64_t seed,
                              float scale) {
  std::vector<float> values = rt_ns::Randn<float>(shape, seed);
  for (float &value : values) {
    value *= scale;
  }
  return MakeTensor(data_type, shape, values);
}

NodeProto MakeLinearAttentionNode(const LinearAttentionCase &test_case) {
  NodeProto node;
  node.set_op_type("LinearAttention");
  node.add_input("query");
  node.add_input("key");
  node.add_input("value");
  if (test_case.with_past || UsesDecay(test_case.rule) || UsesBeta(test_case.rule)) {
    node.add_input(test_case.with_past ? "past_state" : "");
  }
  if (UsesDecay(test_case.rule) || UsesBeta(test_case.rule)) {
    node.add_input(UsesDecay(test_case.rule) ? "decay" : "");
  }
  if (UsesBeta(test_case.rule)) {
    node.add_input("beta");
  }
  node.add_output("output");
  node.add_output("present_state");
  AddStringAttribute(node, "update_rule", test_case.rule);
  AddIntAttribute(node, "q_num_heads", test_case.query_heads);
  AddIntAttribute(node, "kv_num_heads", test_case.key_value_heads);
  AddFloatAttribute(node, "scale", 1.0f);
  return node;
}

IoData MakeLinearAttentionData(const LinearAttentionCase &test_case, bool generate_expected_outputs,
                               const KernelContext &context) {
  const Shape query_shape{test_case.batch, test_case.sequence,
                          test_case.query_heads * test_case.key_head_size};
  const Shape key_shape{test_case.batch, test_case.sequence,
                        test_case.key_value_heads * test_case.key_head_size};
  const Shape value_shape{test_case.batch, test_case.sequence,
                          test_case.key_value_heads * test_case.value_head_size};
  const Shape state_shape{test_case.batch, test_case.key_value_heads, test_case.key_head_size,
                          test_case.value_head_size};
  Tensor query = MakeScaledRandomTensor(test_case.data_type, query_shape, test_case.seed, 0.05f);
  Tensor key = MakeScaledRandomTensor(test_case.data_type, key_shape, test_case.seed + 1, 0.05f);
  Tensor value =
      MakeScaledRandomTensor(test_case.data_type, value_shape, test_case.seed + 2, 0.25f);
  std::optional<Tensor> past;
  std::optional<Tensor> decay;
  std::optional<Tensor> beta;
  if (test_case.with_past) {
    past.emplace(
        MakeScaledRandomTensor(test_case.data_type, state_shape, test_case.seed + 3, 0.02f));
  }
  if (UsesDecay(test_case.rule)) {
    const Shape decay_shape{test_case.batch, test_case.sequence,
                            test_case.decay_per_dimension
                                ? test_case.key_value_heads * test_case.key_head_size
                                : test_case.key_value_heads};
    decay.emplace(
        MakeTensor(test_case.data_type, decay_shape,
                   std::vector<float>(static_cast<std::size_t>(Count(decay_shape)), -0.01f)));
  }
  if (UsesBeta(test_case.rule)) {
    const Shape beta_shape{test_case.batch, test_case.sequence,
                           test_case.beta_shared ? 1 : test_case.key_value_heads};
    beta.emplace(MakeTensor(test_case.data_type, beta_shape,
                            std::vector<float>(static_cast<std::size_t>(Count(beta_shape)), 0.5f)));
  }

  std::vector<Tensor> inputs;
  inputs.reserve(6);
  inputs.push_back(std::move(query));
  inputs.push_back(std::move(key));
  inputs.push_back(std::move(value));
  if (past.has_value()) {
    inputs.push_back(std::move(*past));
  }
  if (decay.has_value()) {
    inputs.push_back(std::move(*decay));
  }
  if (beta.has_value()) {
    inputs.push_back(std::move(*beta));
  }
  if (!generate_expected_outputs) {
    return IoData{std::move(inputs), {}, {}, false};
  }

  const Tensor *past_ptr = test_case.with_past ? &inputs[3] : nullptr;
  std::size_t optional_index = test_case.with_past ? 4 : 3;
  const Tensor *decay_ptr = UsesDecay(test_case.rule) ? &inputs[optional_index++] : nullptr;
  const Tensor *beta_ptr = UsesBeta(test_case.rule) ? &inputs[optional_index] : nullptr;
  ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::LinearAttention::Attributes attributes;
  attributes.update_rule = test_case.rule;
  attributes.has_scale = true;
  attributes.scale = 1.0f;
  attributes.q_num_heads = test_case.query_heads;
  attributes.kv_num_heads = test_case.key_value_heads;
  const ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::LinearAttention kernel{context};
  auto result = kernel(inputs[0], inputs[1], inputs[2], attributes, past_ptr, decay_ptr, beta_ptr);
  return IoData{std::move(inputs), {std::move(result.output), std::move(result.present_state)}};
}

void RegisterLinearAttentionCase(std::vector<TestCase> &registry,
                                 const LinearAttentionCase &test_case, bool benchmark) {
  const OpsetId opset = DefaultOpset(27);
  const Shape query_shape{test_case.batch, test_case.sequence,
                          test_case.query_heads * test_case.key_head_size};
  const Shape key_shape{test_case.batch, test_case.sequence,
                        test_case.key_value_heads * test_case.key_head_size};
  const Shape value_shape{test_case.batch, test_case.sequence,
                          test_case.key_value_heads * test_case.value_head_size};
  const Shape state_shape{test_case.batch, test_case.key_value_heads, test_case.key_head_size,
                          test_case.value_head_size};
  const Shape output_shape{test_case.batch, test_case.sequence,
                           test_case.query_heads * test_case.value_head_size};
  std::vector<std::int64_t> input_counts{Count(query_shape), Count(key_shape), Count(value_shape)};
  if (test_case.with_past) {
    input_counts.push_back(Count(state_shape));
  }
  if (UsesDecay(test_case.rule)) {
    input_counts.push_back(test_case.batch * test_case.sequence *
                           (test_case.decay_per_dimension
                                ? test_case.key_value_heads * test_case.key_head_size
                                : test_case.key_value_heads));
  }
  if (UsesBeta(test_case.rule)) {
    input_counts.push_back(test_case.batch * test_case.sequence *
                           (test_case.beta_shared ? 1 : test_case.key_value_heads));
  }

  const std::string name = "test_cpu_linear_attention_" + std::string(test_case.name) + "_" +
                           DataTypeSuffix(test_case.data_type) + (benchmark ? "_benchmark" : "");
  const NodeProto node = MakeLinearAttentionNode(test_case);
  Expect(registry, node, name, {opset}, input_counts, {Count(output_shape), Count(state_shape)},
         [test_case](bool generate_expected_outputs) {
           const KernelContext context{DefaultOpset(27)};
           return MakeLinearAttentionData(test_case, generate_expected_outputs, context);
         },
         "backend-test", bt_ns::TestCaseTag::NONE,
         {bt_ns::TensorTypeSpec(static_cast<std::int32_t>(test_case.data_type), output_shape),
          bt_ns::TensorTypeSpec(static_cast<std::int32_t>(test_case.data_type), state_shape)});
  if (test_case.data_type != DataType::FLOAT) {
    registry.back().rtol = 2.0e-2;
    registry.back().atol = 2.0e-2;
  }
}

} // namespace

void RegisterCpuLinearAttentionCases(std::vector<TestCase> &registry, TestMode mode) {
  if (mode == TestMode::BENCHMARK) {
    for (const LinearAttentionCase &test_case :
         {LinearAttentionCase{"qwen3_5_decode_t1_h16_d128_gated_delta_past", 1, 1, 16, 16, 128, 128,
                              "gated_delta", true, false, false, DataType::FLOAT, 2701},
          LinearAttentionCase{"qwen3_5_prefill_t128_h16_d128_gated_delta", 1, 128, 16, 16, 128, 128,
                              "gated_delta", false, false, false, DataType::FLOAT, 2711},
          LinearAttentionCase{"qwen3_5_prefill_t512_h16_d128_gated_delta", 1, 512, 16, 16, 128, 128,
                              "gated_delta", false, false, false, DataType::FLOAT, 2721},
          LinearAttentionCase{"qwen3_5_decode_t1_h16_d128_gated_delta_past", 1, 1, 16, 16, 128, 128,
                              "gated_delta", true, false, false, DataType::BFLOAT16, 2741},
          LinearAttentionCase{"qwen3_5_prefill_t128_h16_d128_gated_delta", 1, 128, 16, 16, 128, 128,
                              "gated_delta", false, false, false, DataType::BFLOAT16, 2751}}) {
      RegisterLinearAttentionCase(registry, test_case, true);
    }
    return;
  }

  for (const LinearAttentionCase &test_case :
       {LinearAttentionCase{"linear_mha", 1, 3, 2, 2, 4, 3, "linear", false, false, false,
                            DataType::FLOAT, 2601},
        LinearAttentionCase{"gated_per_dimension", 1, 3, 2, 2, 4, 3, "gated", false, true, false,
                            DataType::FLOAT, 2611},
        LinearAttentionCase{"delta_shared_beta", 2, 2, 2, 2, 4, 3, "delta", false, false, true,
                            DataType::FLOAT, 2621},
        LinearAttentionCase{"gqa_with_past", 1, 3, 4, 2, 4, 3, "gated_delta", true, false, false,
                            DataType::FLOAT, 2631},
        LinearAttentionCase{"gated_delta", 1, 3, 2, 2, 4, 3, "gated_delta", true, false, false,
                            DataType::FLOAT16, 2651},
        LinearAttentionCase{"gated_delta", 1, 3, 2, 2, 4, 3, "gated_delta", true, false, false,
                            DataType::BFLOAT16, 2661}}) {
    RegisterLinearAttentionCase(registry, test_case, false);
  }
}

} // namespace onnx_light_cpu::backend_test
