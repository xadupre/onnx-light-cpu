// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/com_microsoft/include_com_microsoft_cases.h"
#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/random.h"
#include "onnx_proto/onnx_helper.h"

#include <algorithm>
#include <cmath>
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
using rt_ns::DefaultOpset;
using rt_ns::OpsetId;
using rt_ns::Shape;
using rt_ns::Tensor;

struct Case {
  const char *name;
  const char *rule;
  std::int64_t batch;
  std::int64_t sequence;
  std::int64_t query_heads;
  std::int64_t key_heads;
  std::int64_t value_heads;
  std::int64_t key_head_size;
  std::int64_t value_head_size;
  bool with_past;
};

bool UsesDecay(const std::string &rule) { return rule == "gated" || rule == "gated_delta"; }
bool UsesBeta(const std::string &rule) { return rule == "delta" || rule == "gated_delta"; }

std::int64_t Count(const Shape &shape) {
  std::int64_t count = 1;
  for (std::int64_t dimension : shape) {
    count *= dimension;
  }
  return count;
}

NodeProto MakeNode(const Case &test_case) {
  NodeProto node;
  node.set_domain(kMicrosoftDomain);
  node.set_op_type("LinearAttention");
  for (const char *input :
       {"query", "key", "value", test_case.with_past ? "past_state" : "",
        UsesDecay(test_case.rule) ? "decay" : "", UsesBeta(test_case.rule) ? "beta" : ""}) {
    node.add_input(input);
  }
  node.add_output("output");
  node.add_output("present_state");
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "update_rule", std::string(test_case.rule));
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "q_num_heads", test_case.query_heads);
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "kv_num_heads", test_case.value_heads);
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "scale", 1.0f);
  return node;
}

std::vector<Tensor> ReferenceOutputs(const Case &test_case, const Tensor &query, const Tensor &key,
                                     const Tensor &value, const Tensor *past, const Tensor *decay,
                                     const Tensor *beta) {
  const std::size_t batch = static_cast<std::size_t>(test_case.batch);
  const std::size_t sequence = static_cast<std::size_t>(test_case.sequence);
  const std::size_t hq = static_cast<std::size_t>(test_case.query_heads);
  const std::size_t hk = static_cast<std::size_t>(test_case.key_heads);
  const std::size_t hv = static_cast<std::size_t>(test_case.value_heads);
  const std::size_t dk = static_cast<std::size_t>(test_case.key_head_size);
  const std::size_t dv = static_cast<std::size_t>(test_case.value_head_size);
  const std::size_t output_heads = std::max(hq, hv);
  std::vector<float> state(batch * hv * dk * dv, 0.0f);
  if (past != nullptr && !state.empty()) {
    std::copy(past->AsFloat(), past->AsFloat() + state.size(), state.begin());
  }
  std::vector<float> output(batch * sequence * output_heads * dv);
  std::vector<float> retrieved(dv);
  for (std::size_t b = 0; b < batch; ++b) {
    for (std::size_t h = 0; h < hv; ++h) {
      float *head_state = state.data() + (b * hv + h) * dk * dv;
      const std::size_t key_head = h / (hv / hk);
      for (std::size_t t = 0; t < sequence; ++t) {
        const std::size_t token = b * sequence + t;
        const float *token_key = key.AsFloat() + (token * hk + key_head) * dk;
        const float *token_value = value.AsFloat() + (token * hv + h) * dv;
        if (decay != nullptr) {
          const float gate = std::exp(decay->AsFloat()[token * hv + h]);
          for (std::size_t i = 0; i < dk * dv; ++i) {
            head_state[i] *= gate;
          }
        }
        const float *update = token_value;
        if (beta != nullptr) {
          std::fill(retrieved.begin(), retrieved.end(), 0.0f);
          for (std::size_t i = 0; i < dk; ++i) {
            for (std::size_t j = 0; j < dv; ++j) {
              retrieved[j] += token_key[i] * head_state[i * dv + j];
            }
          }
          const float rate = beta->AsFloat()[token * hv + h];
          for (std::size_t j = 0; j < dv; ++j) {
            retrieved[j] = rate * (token_value[j] - retrieved[j]);
          }
          update = retrieved.data();
        }
        for (std::size_t i = 0; i < dk; ++i) {
          for (std::size_t j = 0; j < dv; ++j) {
            head_state[i * dv + j] += token_key[i] * update[j];
          }
        }
        const std::size_t first_query = hq >= hv ? h * (hq / hv) : h / (hv / hq);
        const std::size_t output_count = hq >= hv ? hq / hv : 1;
        const std::size_t first_output = hq >= hv ? first_query : h;
        for (std::size_t o = 0; o < output_count; ++o) {
          const float *token_query = query.AsFloat() + (token * hq + first_query + o) * dk;
          float *token_output = output.data() + (token * output_heads + first_output + o) * dv;
          for (std::size_t j = 0; j < dv; ++j) {
            token_output[j] = 0.0f;
            for (std::size_t i = 0; i < dk; ++i) {
              token_output[j] += token_query[i] * head_state[i * dv + j];
            }
          }
        }
      }
    }
  }
  return {Tensor::FromFloat(
              "output",
              {test_case.batch, test_case.sequence, static_cast<std::int64_t>(output_heads * dv)},
              output),
          Tensor::FromFloat("present_state",
                            {test_case.batch, test_case.value_heads, test_case.key_head_size,
                             test_case.value_head_size},
                            state)};
}

IoData MakeData(const Case &test_case, bool expected) {
  const Shape query_shape{test_case.batch, test_case.sequence,
                          test_case.query_heads * test_case.key_head_size};
  const Shape key_shape{test_case.batch, test_case.sequence,
                        test_case.key_heads * test_case.key_head_size};
  const Shape value_shape{test_case.batch, test_case.sequence,
                          test_case.value_heads * test_case.value_head_size};
  const Shape state_shape{test_case.batch, test_case.value_heads, test_case.key_head_size,
                          test_case.value_head_size};
  Tensor query = MakeBenchmarkTensor(rt_ns::DataType::FLOAT, query_shape, 4101);
  Tensor key = MakeBenchmarkTensor(rt_ns::DataType::FLOAT, key_shape, 4102);
  Tensor value = MakeBenchmarkTensor(rt_ns::DataType::FLOAT, value_shape, 4103);
  std::optional<Tensor> past;
  std::optional<Tensor> decay;
  std::optional<Tensor> beta;
  if (test_case.with_past) {
    past.emplace(MakeBenchmarkTensor(rt_ns::DataType::FLOAT, state_shape, 4104));
  }
  if (UsesDecay(test_case.rule)) {
    decay.emplace(Tensor::FromFloat(
        "decay", {test_case.batch, test_case.sequence, test_case.value_heads},
        std::vector<float>(
            static_cast<std::size_t>(test_case.batch * test_case.sequence * test_case.value_heads),
            -0.01f)));
  }
  if (UsesBeta(test_case.rule)) {
    beta.emplace(Tensor::FromFloat(
        "beta", {test_case.batch, test_case.sequence, test_case.value_heads},
        std::vector<float>(
            static_cast<std::size_t>(test_case.batch * test_case.sequence * test_case.value_heads),
            0.5f)));
  }

  std::vector<Tensor> inputs;
  inputs.push_back(query);
  inputs.push_back(key);
  inputs.push_back(value);
  if (past.has_value()) {
    inputs.push_back(*past);
  }
  if (decay.has_value()) {
    inputs.push_back(*decay);
  }
  if (beta.has_value()) {
    inputs.push_back(*beta);
  }
  if (!expected) {
    return IoData{std::move(inputs), {}, {}, false};
  }
  return IoData{std::move(inputs),
                ReferenceOutputs(test_case, query, key, value, past ? &*past : nullptr,
                                 decay ? &*decay : nullptr, beta ? &*beta : nullptr)};
}

void AddCase(std::vector<TestCase> &registry, const Case &test_case, bool benchmark) {
  const OpsetId opset(kMicrosoftDomain, 1);
  const Shape query_shape{test_case.batch, test_case.sequence,
                          test_case.query_heads * test_case.key_head_size};
  const Shape key_shape{test_case.batch, test_case.sequence,
                        test_case.key_heads * test_case.key_head_size};
  const Shape value_shape{test_case.batch, test_case.sequence,
                          test_case.value_heads * test_case.value_head_size};
  const Shape state_shape{test_case.batch, test_case.value_heads, test_case.key_head_size,
                          test_case.value_head_size};
  const Shape output_shape{test_case.batch, test_case.sequence,
                           std::max(test_case.query_heads, test_case.value_heads) *
                               test_case.value_head_size};
  std::vector<std::int64_t> input_counts{Count(query_shape), Count(key_shape), Count(value_shape)};
  if (test_case.with_past) {
    input_counts.push_back(Count(state_shape));
  }
  if (UsesDecay(test_case.rule)) {
    input_counts.push_back(test_case.batch * test_case.sequence * test_case.value_heads);
  }
  if (UsesBeta(test_case.rule)) {
    input_counts.push_back(test_case.batch * test_case.sequence * test_case.value_heads);
  }
  Expect(registry, MakeNode(test_case),
         "test_cpu_microsoft_linear_attention_" + std::string(test_case.name) + "_float32" +
             (benchmark ? "_benchmark" : ""),
         {DefaultOpset(27), opset}, input_counts, {Count(output_shape), Count(state_shape)},
         [=](bool expected) { return MakeData(test_case, expected); }, "backend-test",
         bt_ns::TestCaseTag::AI_RT,
         {bt_ns::TensorTypeSpec(static_cast<std::int32_t>(rt_ns::DataType::FLOAT), output_shape),
          bt_ns::TensorTypeSpec(static_cast<std::int32_t>(rt_ns::DataType::FLOAT), state_shape)});
}

} // namespace

void RegisterCpuMicrosoftLinearAttentionCases(std::vector<TestCase> &registry, TestMode mode) {
  if (mode == TestMode::BENCHMARK) {
    AddCase(registry, {"decode_h16_d128", "gated_delta", 1, 1, 16, 16, 16, 128, 128, true}, true);
    return;
  }
  for (const Case &test_case : {Case{"linear", "linear", 1, 3, 2, 2, 2, 4, 3, false},
                                Case{"gated", "gated", 1, 3, 2, 2, 2, 4, 3, false},
                                Case{"delta", "delta", 1, 3, 2, 2, 2, 4, 3, false},
                                Case{"gated_delta_past", "gated_delta", 1, 3, 2, 2, 2, 4, 3, true},
                                Case{"inverse_grouping", "linear", 1, 2, 1, 2, 2, 4, 3, false},
                                Case{"shared_key_head", "linear", 1, 2, 2, 1, 2, 4, 3, false},
                                Case{"empty_sequence", "linear", 1, 0, 2, 2, 2, 4, 3, false}}) {
    AddCase(registry, test_case, false);
  }
}

} // namespace onnx_light_cpu::backend_test
