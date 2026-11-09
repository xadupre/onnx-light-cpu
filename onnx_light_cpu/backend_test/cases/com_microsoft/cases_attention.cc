// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_light_cpu/backend_test/cases/com_microsoft/include_com_microsoft_cases.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"
#include "onnx_proto/onnx_helper.h"

#include <cmath>
#include <string>
#include <vector>

namespace onnx_light_cpu::backend_test {
namespace {
namespace bt = ONNX_LIGHT_NAMESPACE::core::backend_test;
namespace rt = ONNX_LIGHT_NAMESPACE::core::runtime;
using ONNX_LIGHT_NAMESPACE::NodeProto;
using rt::Tensor;
NodeProto Node(int64_t heads, bool present = true) {
  NodeProto node;
  node.set_domain(kMicrosoftDomain);
  node.set_op_type("Attention");
  for (const char *name : {"x", "w", "", "", "", ""})
    node.add_input(name);
  node.add_output("y");
  if (present)
    node.add_output("present");
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "num_heads", heads);
  return node;
}
} // namespace

void RegisterCpuMicrosoftAttentionCases(std::vector<TestCase> &registry, TestMode mode) {
  for (int variant = 0; variant < 10; ++variant) {
    auto node = Node(1);
    const bool causal = variant == 1 || variant == 7;
    if (causal)
      ONNX_LIGHT_NAMESPACE::AddAttribute(node, "unidirectional", int64_t{1});
    if (variant >= 2 && variant <= 6)
      node.ref_input()[3] = "mask";
    if (variant == 7)
      node.ref_input()[4] = "past";
    const int64_t s = variant == 8 ? 0 : 2;
    const int64_t b = variant == 9 ? 0 : 1;
    bt::Expect(
        registry, node, "test_cpu_ms_attention_float_mode_" + std::to_string(variant),
        {rt::DefaultOpset(13), rt::OpsetId(kMicrosoftDomain, 1)},
        [=]() -> bt::IoData {
          std::vector<Tensor> inputs{
              Tensor::FromFloat("x", {b, s, 1},
                                b && s ? std::vector<float>{1, 2} : std::vector<float>{}),
              Tensor::FromFloat("w", {1, 3}, {1, 1, 2})};
          if (variant == 2)
            inputs.push_back(Tensor::FromInt32("mask", {1}, {1}));
          if (variant == 3)
            inputs.push_back(Tensor::FromInt32("mask", {2}, {2, 1}));
          if (variant == 4)
            inputs.push_back(Tensor::FromInt32("mask", {1, 2}, {1, 0}));
          if (variant == 5)
            inputs.push_back(Tensor::FromInt32("mask", {1, 2, 2}, {1, 0, 1, 0}));
          if (variant == 6)
            inputs.push_back(Tensor::FromInt32("mask", {1, 1, 2, 2}, {1, 0, 1, 0}));
          if (variant == 7)
            inputs.push_back(Tensor::FromFloat("past", {2, 1, 1, 1, 1}, {0, 8}));
          std::vector<float> y;
          if (b && s) {
            y = {2 + 2 / (1 + std::exp(-1.f)), 2 + 2 / (1 + std::exp(-2.f))};
            if (variant == 1)
              y[0] = 2;
            if (variant >= 2 && variant <= 6)
              y = variant == 3 ? std::vector<float>{4, 4} : std::vector<float>{2, 2};
            if (variant == 7)
              y = {(8 + 2 * std::exp(1.f)) / (1 + std::exp(1.f)),
                   (8 + 2 * std::exp(2.f) + 4 * std::exp(4.f)) /
                       (1 + std::exp(2.f) + std::exp(4.f))};
          }
          const auto present = variant == 7 ? std::vector<float>{0, 1, 2, 8, 2, 4}
                               : b && s     ? std::vector<float>{1, 2, 2, 4}
                                            : std::vector<float>{};
          return {
              std::move(inputs),
              {Tensor::FromFloat("y", {b, s, 1}, y),
               Tensor::FromFloat("present", {2, b, 1, s + (variant == 7 ? 1 : 0), 1}, present)}};
        },
        "backend-test", bt::TestCaseTag::AI_RT);
  }
  for (bool unequal : {false, true}) {
    auto node = Node(1, !unequal);
    node.ref_input()[2] = "bias";
    node.ref_input()[5] = "attention_bias";
    const int64_t width = unequal ? 4 : 3;
    if (unequal) {
      ONNX_LIGHT_NAMESPACE::AddAttribute(node, "qkv_hidden_sizes", std::vector<int64_t>{1, 1, 2});
    }
    ONNX_LIGHT_NAMESPACE::AddAttribute(node, "scale", 0.f);
    bt::Expect(
        registry, node,
        std::string("test_cpu_ms_attention_bias_float_") + (unequal ? "unequal_v" : "equal"),
        {rt::DefaultOpset(13), rt::OpsetId(kMicrosoftDomain, 1)},
        [=]() -> bt::IoData {
          std::vector<float> bias(width, 0.f);
          bias[2] = 1;
          if (unequal)
            bias[3] = 2;
          const float first = 3 * std::exp(1.f) / (1 + 3 * std::exp(1.f));
          const float second = 1 / (1 + std::exp(-2.f));
          std::vector<Tensor> outputs{Tensor::FromFloat(
              "y", {1, 2, unequal ? 2 : 1},
              unequal
                  ? std::vector<float>{3 + 2 * first, 5 + 3 * first, 3 + 2 * second, 5 + 3 * second}
                  : std::vector<float>{3 + 2 * first, 3 + 2 * second})};
          if (!unequal)
            outputs.push_back(Tensor::FromFloat("present", {2, 1, 1, 2, 1}, {1, 2, 3, 5}));
          return {{Tensor::FromFloat("x", {1, 2, 1}, {1, 2}),
                   Tensor::FromFloat("w", {1, width},
                                     unequal ? std::vector<float>{1, 1, 2, 3}
                                             : std::vector<float>{1, 1, 2}),
                   Tensor::FromFloat("bias", {width}, bias),
                   Tensor::FromFloat("attention_bias", {1, 1, 2, 2}, {0, std::log(3.f), 0, 0})},
                  std::move(outputs)};
        },
        "backend-test", bt::TestCaseTag::AI_RT);
  }
  const std::vector<int64_t> whisper_lengths =
      mode == TestMode::BENCHMARK ? std::vector<int64_t>{128, 512} : std::vector<int64_t>{8, 32};
  for (int64_t s : whisper_lengths) {
    auto node = Node(4);
    node.ref_input()[2] = "bias";
    const std::string case_name =
        mode == TestMode::BENCHMARK
            ? "test_cpu_ms_attention_whisper_s" + std::to_string(s) + "_float32_benchmark"
            : "test_cpu_ms_attention_whisper_float_s" + std::to_string(s);
    bt::Expect(
        registry, node, case_name, {rt::DefaultOpset(13), rt::OpsetId(kMicrosoftDomain, 1)},
        [=]() -> bt::IoData {
          std::vector<float> bias(96, 0.f);
          for (int i = 64; i < 96; ++i)
            bias[i] = float(i - 64) / 32;
          std::vector<float> y(s * 32), present(2 * s * 32, 0.f);
          for (int64_t i = 0; i < s * 32; ++i) {
            y[i] = bias[64 + i % 32];
            present[s * 32 + i] = bias[64 + (i / (s * 8)) * 8 + i % 8];
          }
          return {{Tensor::FromFloat("x", {1, s, 32}, std::vector<float>(s * 32, 1.f)),
                   Tensor::FromFloat("w", {32, 96}, std::vector<float>(32 * 96, 0.f)),
                   Tensor::FromFloat("bias", {96}, bias)},
                  {Tensor::FromFloat("y", {1, s, 32}, y),
                   Tensor::FromFloat("present", {2, 1, 4, s, 8}, present)}};
        },
        "backend-test", bt::TestCaseTag::AI_RT);
  }
}
} // namespace onnx_light_cpu::backend_test
