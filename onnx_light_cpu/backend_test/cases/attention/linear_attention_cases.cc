// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/attention/linear_attention_cases.h"

#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"
#include "onnx_light_cpu/kernels/com_microsoft/naive_linear_attention_kernel.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/random.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <algorithm>
#include <cstddef>
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
using ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::DataType;
using rt_ns::DefaultOpset;
using rt_ns::KernelContext;
using rt_ns::OpsetId;
using rt_ns::Shape;
using rt_ns::Tensor;

bool UsesDecay(const std::string &rule) { return rule == "gated" || rule == "gated_delta"; }
bool UsesBeta(const std::string &rule) { return rule == "delta" || rule == "gated_delta"; }

Tensor MakeScaledRandomTensor(DataType data_type, const Shape &shape, std::uint64_t seed,
                              float scale) {
  std::vector<float> values = rt_ns::Randn<float>(shape, seed);
  for (float &value : values) {
    value *= scale;
  }
  return MakeTensor(data_type, shape, values);
}

NodeProto MakeLinearAttentionNode(const LinearAttentionCase &test_case,
                                  LinearAttentionCaseContract contract) {
  NodeProto node;
  node.set_op_type("LinearAttention");
  if (contract == LinearAttentionCaseContract::kMicrosoft) {
    node.set_domain(kMicrosoftDomain);
  }
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
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "update_rule", std::string(test_case.rule));
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "q_num_heads", test_case.query_heads);
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "kv_num_heads", test_case.key_value_heads);
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "scale", 1.0f);
  return node;
}

std::vector<Tensor> MicrosoftReference(const LinearAttentionCase &test_case, const Tensor &query,
                                       const Tensor &key, const Tensor &value, const Tensor *past,
                                       const Tensor *decay, const Tensor *beta) {
  NaiveMicrosoftLinearAttentionKernel::Attributes attributes;
  attributes.update_rule = test_case.rule;
  attributes.scale = 1.0f;
  attributes.query_heads = test_case.query_heads;
  attributes.key_value_heads = test_case.key_value_heads;
  const KernelContext ctx{OpsetId(kMicrosoftDomain, 1)};
  const NaiveMicrosoftLinearAttentionKernel kernel{ctx};
  auto result = kernel(query, key, value, attributes, past, decay, beta);
  return {std::move(result.output), std::move(result.present_state)};
}

std::vector<Tensor> OnnxReference(const LinearAttentionCase &test_case, const Tensor &query,
                                  const Tensor &key, const Tensor &value, const Tensor *past,
                                  const Tensor *decay, const Tensor *beta) {
  ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::LinearAttention::Attributes attributes;
  attributes.update_rule = test_case.rule;
  attributes.has_scale = true;
  attributes.scale = 1.0f;
  attributes.q_num_heads = test_case.query_heads;
  attributes.kv_num_heads = test_case.key_value_heads;
  const ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::LinearAttention kernel{
      KernelContext{DefaultOpset(27)}};
  auto result = kernel(query, key, value, attributes, past, decay, beta);
  return {std::move(result.output), std::move(result.present_state)};
}

IoData MakeLinearAttentionData(const LinearAttentionCase &test_case,
                               LinearAttentionCaseContract contract,
                               bool generate_expected_outputs) {
  const Shape query_shape{test_case.batch, test_case.sequence,
                          test_case.query_heads * test_case.key_head_size};
  const Shape key_shape{test_case.batch, test_case.sequence,
                        test_case.key_heads * test_case.key_head_size};
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
                   std::vector<float>(static_cast<std::size_t>(decay_shape.product()), -0.01f)));
  }
  if (UsesBeta(test_case.rule)) {
    const Shape beta_shape{test_case.batch, test_case.sequence,
                           test_case.beta_shared ? 1 : test_case.key_value_heads};
    beta.emplace(
        MakeTensor(test_case.data_type, beta_shape,
                   std::vector<float>(static_cast<std::size_t>(beta_shape.product()), 0.5f)));
  }

  std::vector<Tensor> outputs;
  if (generate_expected_outputs) {
    const Tensor *past_ptr = past ? &*past : nullptr;
    const Tensor *decay_ptr = decay ? &*decay : nullptr;
    const Tensor *beta_ptr = beta ? &*beta : nullptr;
    outputs = contract == LinearAttentionCaseContract::kMicrosoft
                  ? MicrosoftReference(test_case, query, key, value, past_ptr, decay_ptr, beta_ptr)
                  : OnnxReference(test_case, query, key, value, past_ptr, decay_ptr, beta_ptr);
  }

  std::vector<Tensor> inputs;
  inputs.reserve(6);
  inputs.push_back(std::move(query));
  inputs.push_back(std::move(key));
  inputs.push_back(std::move(value));
  if (past) {
    inputs.push_back(std::move(*past));
  }
  if (decay) {
    inputs.push_back(std::move(*decay));
  }
  if (beta) {
    inputs.push_back(std::move(*beta));
  }
  return IoData{std::move(inputs), std::move(outputs), {}, generate_expected_outputs};
}

} // namespace

void RegisterLinearAttentionCase(std::vector<bt_ns::TestCase> &registry,
                                 const LinearAttentionCase &test_case,
                                 LinearAttentionCaseContract contract, bool benchmark) {
  const Shape query_shape{test_case.batch, test_case.sequence,
                          test_case.query_heads * test_case.key_head_size};
  const Shape key_shape{test_case.batch, test_case.sequence,
                        test_case.key_heads * test_case.key_head_size};
  const Shape value_shape{test_case.batch, test_case.sequence,
                          test_case.key_value_heads * test_case.value_head_size};
  const Shape state_shape{test_case.batch, test_case.key_value_heads, test_case.key_head_size,
                          test_case.value_head_size};
  const Shape output_shape{test_case.batch, test_case.sequence,
                           std::max(test_case.query_heads, test_case.key_value_heads) *
                               test_case.value_head_size};
  std::vector<std::int64_t> input_counts{query_shape.product(), key_shape.product(),
                                         value_shape.product()};
  if (test_case.with_past) {
    input_counts.push_back(state_shape.product());
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

  const bool microsoft = contract == LinearAttentionCaseContract::kMicrosoft;
  const OpsetId operator_opset = microsoft ? OpsetId(kMicrosoftDomain, 1) : DefaultOpset(27);
  const std::string prefix =
      microsoft ? "test_cpu_microsoft_linear_attention_" : "test_cpu_linear_attention_";
  const std::string name = prefix + test_case.name + "_" + DataTypeSuffix(test_case.data_type) +
                           (benchmark ? "_benchmark" : "");
  std::vector<OpsetId> opsets{DefaultOpset(27)};
  if (microsoft) {
    opsets.push_back(operator_opset);
  }
  Expect(registry, MakeLinearAttentionNode(test_case, contract), name, opsets, input_counts,
         {output_shape.product(), state_shape.product()},
         [test_case, contract](bool generate_expected_outputs) {
           return MakeLinearAttentionData(test_case, contract, generate_expected_outputs);
         },
         "backend-test", microsoft ? bt_ns::TestCaseTag::AI_RT : bt_ns::TestCaseTag::NONE,
         {bt_ns::TensorTypeSpec(static_cast<std::int32_t>(test_case.data_type), output_shape),
          bt_ns::TensorTypeSpec(static_cast<std::int32_t>(test_case.data_type), state_shape)});
  if (test_case.data_type != DataType::FLOAT) {
    registry.back().rtol = 2.0e-2;
    registry.back().atol = 2.0e-2;
  }
}

} // namespace onnx_light_cpu::backend_test
