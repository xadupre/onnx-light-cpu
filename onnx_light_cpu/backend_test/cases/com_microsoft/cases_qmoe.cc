// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/com_microsoft/include_com_microsoft_cases.h"
#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_proto/onnx_helper.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace onnx_light_cpu::backend_test {
namespace {

namespace bt_ns = ONNX_LIGHT_NAMESPACE::core::backend_test;
namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
using rt_ns::DataType;
using rt_ns::Tensor;

struct Configuration {
  DataType type;
  int64_t bits;
  int64_t block;
  std::string activation;
  bool normalize;
  bool separate;
  int64_t tokens = 3;
  int64_t hidden = 5;
  int64_t intermediate = 3;
  int64_t experts = 3;
  int64_t topk = 2;
  bool random_weights = false;
  float alpha = 1.0f;
  float beta = 0.0f;
  float limit = std::numeric_limits<float>::infinity();
};

Configuration DecoderConfiguration() {
  Configuration c{DataType::FLOAT, 4, 32, "swiglu", true, false};
  c.hidden = 35;
  c.intermediate = 33;
  c.random_weights = true;
  return c;
}

ONNX_LIGHT_NAMESPACE::NodeProto MakeQMoENode(const Configuration &c) {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_domain(kMicrosoftDomain);
  node.set_op_type("QMoE");
  for (const char *name :
       {"input", "router", "w1", "s1", "b1", "w2", "s2", "b2", c.separate ? "w3" : "",
        c.separate ? "s3" : "", c.separate ? "b3" : "", "", "", ""}) {
    node.add_input(name);
  }
  node.add_output("output");
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "expert_weight_bits", c.bits);
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "block_size", c.block);
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "activation_type", c.activation);
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "k", c.topk);
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "normalize_routing_weights",
                                     int64_t{c.normalize ? 1 : 0});
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "swiglu_fusion",
                                     int64_t{c.activation == "swiglu" ? 1 : 0});
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "activation_alpha", c.alpha);
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "activation_beta", c.beta);
  if (std::isfinite(c.limit)) {
    ONNX_LIGHT_NAMESPACE::AddAttribute(node, "swiglu_limit", c.limit);
  }
  return node;
}

struct Projection {
  Tensor weights;
  Tensor scales;
  std::vector<float> dequantized;
};

Projection MakeProjection(const Configuration &c, int64_t rows, int64_t columns, int seed) {
  // mbext 68a6860cc8fe2c3381f5248d215b7217a36822e3, builders/base.py:
  // separate experts quantize transposed linear weights; fused SwiGLU keeps
  // output-major weights. Block quantization groups the last stored axis.
  // Tensor-level fixtures use the canonical CPU output-major, row-scaled
  // contract, not TensorRT-LLM CUDA-preprocessed legacy producer tensors.
  const bool output_major = c.block == 0 || c.activation == "swiglu";
  const int64_t stored_rows = output_major ? rows : columns;
  const int64_t stored_columns = output_major ? columns : rows;
  const int64_t bytes = c.bits == 4 ? (stored_columns + 1) / 2 : stored_columns;
  const int64_t blocks = c.block == 0 ? 1 : (stored_columns + c.block - 1) / c.block;
  std::vector<uint8_t> packed(c.experts * stored_rows * bytes);
  std::vector<float> scales(c.experts * stored_rows * blocks);
  std::vector<float> decoded(c.experts * rows * columns);
  std::mt19937 generator(0x514d4f45U + static_cast<uint32_t>(seed));
  std::uniform_int_distribution<int> quantized(c.bits == 4 ? -7 : -127, c.bits == 4 ? 7 : 127);
  std::uniform_int_distribution<int> scale(1, 3);
  for (size_t i = 0; i < scales.size(); ++i) {
    scales[i] =
        (c.random_weights ? scale(generator) : 1 + (i + seed) % 3) / (c.bits == 4 ? 16.0f : 256.0f);
  }
  for (int64_t e = 0; e < c.experts; ++e) {
    for (int64_t r = 0; r < rows; ++r) {
      for (int64_t k = 0; k < columns; ++k) {
        const int value =
            c.random_weights
                ? quantized(generator)
                : static_cast<int>((e * 11 + r * 7 + k * 3 + seed) % (c.bits == 4 ? 15 : 255)) -
                      (c.bits == 4 ? 7 : 127);
        const int64_t inner = output_major ? k : r;
        const int64_t index =
            (e * stored_rows + (output_major ? r : k)) * bytes + (c.bits == 4 ? inner / 2 : inner);
        if (c.bits == 4) {
          packed[index] |= static_cast<uint8_t>((value + 8) << (4 * (inner % 2)));
        } else {
          packed[index] = static_cast<uint8_t>(c.block == 0 ? value + 128 : value);
        }
      }
      // Decode the serialized bytes independently of the QMoE kernel.
      for (int64_t k = 0; k < columns; ++k) {
        const int64_t inner = output_major ? k : r;
        const int64_t outer = output_major ? r : k;
        const uint8_t byte =
            packed[(e * stored_rows + outer) * bytes + (c.bits == 4 ? inner / 2 : inner)];
        const int nibble = (byte >> (4 * (inner % 2))) & 15;
        const int value = c.bits == 4    ? nibble - 8
                          : c.block == 0 ? static_cast<int>(byte) - 128
                                         : (byte >= 128 ? static_cast<int>(byte) - 256 : byte);
        decoded[(e * rows + r) * columns + k] =
            value * scales[c.block == 0 ? e * stored_rows + outer
                                        : (e * stored_rows + outer) * blocks + inner / c.block];
      }
    }
  }
  const rt_ns::Shape scale_shape = c.block == 0 ? rt_ns::Shape{c.experts, stored_rows}
                                                : rt_ns::Shape{c.experts, stored_rows, blocks};
  return {Tensor::FromUint8("", {c.experts, stored_rows, bytes}, packed),
          MakeTensor(c.type, scale_shape, scales), std::move(decoded)};
}

float Activate(float value, const std::string &activation) {
  if (activation == "relu") {
    return std::max(value, 0.0f);
  }
  if (activation == "silu" || activation == "swiglu") {
    return value / (1.0f + std::exp(-value));
  }
  if (activation == "gelu") {
    return 0.5f * value *
           (1.0f + std::tanh(std::sqrt(2.0f / std::acos(-1.0f)) *
                             (value + 0.044715f * value * value * value)));
  }
  return value;
}

bt_ns::IoData MakeQMoEData(const Configuration &c, bool expected, bool floating_weights = false) {
  const int64_t width = c.intermediate * (c.activation == "swiglu" ? 2 : 1);
  auto p1 = MakeProjection(c, width, c.hidden, 1);
  auto p2 = MakeProjection(c, c.hidden, c.intermediate, 5);
  auto p3 = MakeProjection(c, c.intermediate, c.hidden, 9);
  std::vector<float> x(c.tokens * c.hidden), router(c.tokens * c.experts);
  std::vector<float> b1(c.experts * width), b2(c.experts * c.hidden);
  std::vector<float> b3(c.experts * c.intermediate);
  std::mt19937 generator(0x514d4f45U);
  std::uniform_int_distribution<int> bias(-3, 3);
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = (static_cast<int>(i % 11) - 5) / 8.0f;
  }
  for (int64_t t = 0; t < c.tokens; ++t) {
    for (int64_t e = 0; e < c.experts; ++e) {
      router[t * c.experts + e] = static_cast<float>((e + t) % c.experts) / 2.0f;
    }
  }
  for (size_t i = 0; i < b1.size(); ++i) {
    b1[i] = (c.random_weights ? bias(generator) : static_cast<int>(i % 5) - 2) / 16.0f;
  }
  for (size_t i = 0; i < b2.size(); ++i) {
    b2[i] = (c.random_weights ? bias(generator) : static_cast<int>(i % 7) - 3) / 32.0f;
  }
  for (size_t i = 0; i < b3.size(); ++i) {
    b3[i] = (c.random_weights ? bias(generator) : static_cast<int>(i % 3) - 1) / 8.0f;
  }
  std::vector<Tensor> inputs;
  inputs.push_back(MakeTensor(c.type, {c.tokens, c.hidden}, x));
  inputs.push_back(MakeTensor(c.type, {c.tokens, c.experts}, router));
  if (floating_weights) {
    inputs.push_back(MakeTensor(c.type, {c.experts, width, c.hidden}, p1.dequantized));
  } else {
    inputs.push_back(std::move(p1.weights));
    inputs.push_back(std::move(p1.scales));
  }
  inputs.push_back(MakeTensor(c.type, {c.experts, width}, b1));
  if (floating_weights) {
    inputs.push_back(MakeTensor(c.type, {c.experts, c.hidden, c.intermediate}, p2.dequantized));
  } else {
    inputs.push_back(std::move(p2.weights));
    inputs.push_back(std::move(p2.scales));
  }
  inputs.push_back(MakeTensor(c.type, {c.experts, c.hidden}, b2));
  if (c.separate) {
    inputs.push_back(std::move(p3.weights));
    inputs.push_back(std::move(p3.scales));
    inputs.push_back(MakeTensor(c.type, {c.experts, c.intermediate}, b3));
  }
  if (!expected) {
    return bt_ns::IoData{std::move(inputs), {}, {}, false};
  }
  std::vector<float> output(c.tokens * c.hidden);
  for (int64_t t = 0; t < c.tokens; ++t) {
    std::vector<int64_t> order(c.experts);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
      return router[t * c.experts + a] > router[t * c.experts + b];
    });
    const float maximum = router[t * c.experts + order[0]];
    float denominator = 0.0f;
    for (int64_t j = 0; j < (c.normalize ? c.topk : c.experts); ++j) {
      denominator += std::exp(router[t * c.experts + order[j]] - maximum);
    }
    for (int64_t j = 0; j < c.topk; ++j) {
      const int64_t e = order[j];
      const float route = std::exp(router[t * c.experts + e] - maximum) / denominator;
      std::vector<float> fc1(width), activated(c.intermediate);
      for (int64_t r = 0; r < width; ++r) {
        fc1[r] = b1[e * width + r];
        for (int64_t k = 0; k < c.hidden; ++k) {
          fc1[r] += x[t * c.hidden + k] * p1.dequantized[(e * width + r) * c.hidden + k];
        }
      }
      for (int64_t r = 0; r < c.intermediate; ++r) {
        activated[r] = Activate(fc1[c.activation == "swiglu" ? 2 * r : r], c.activation);
        if (c.activation == "swiglu") {
          const float gate = std::min(fc1[2 * r], c.limit);
          const float linear = std::clamp(fc1[2 * r + 1], -c.limit, c.limit);
          activated[r] = gate / (1.0f + std::exp(-c.alpha * gate)) * (linear + c.beta);
        } else if (c.separate) {
          float up = b3[e * c.intermediate + r];
          for (int64_t k = 0; k < c.hidden; ++k) {
            up += x[t * c.hidden + k] * p3.dequantized[(e * c.intermediate + r) * c.hidden + k];
          }
          activated[r] *= up;
        }
      }
      for (int64_t r = 0; r < c.hidden; ++r) {
        float value = b2[e * c.hidden + r];
        for (int64_t k = 0; k < c.intermediate; ++k) {
          value += activated[k] * p2.dequantized[(e * c.hidden + r) * c.intermediate + k];
        }
        output[t * c.hidden + r] += route * value;
      }
    }
  }
  return bt_ns::IoData{std::move(inputs), {MakeTensor(c.type, {c.tokens, c.hidden}, output)}};
}

} // namespace

void RegisterCpuQMoECases(std::vector<TestCase> &registry, TestMode mode) {
  const bool benchmark = mode == TestMode::BENCHMARK;
  const std::vector<rt_ns::OpsetId> opsets = {rt_ns::DefaultOpset(23),
                                              rt_ns::OpsetId(kMicrosoftDomain, 1)};
  for (DataType type : {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16}) {
    for (int64_t bits : {4, 8}) {
      for (int64_t block : {0, 32}) {
        for (const std::string activation : {"identity", "relu", "gelu", "silu", "swiglu"}) {
          for (bool normalize : {false, true}) {
            Configuration c{type, bits, block, activation, normalize, activation != "swiglu"};
            if (block != 0) {
              c.hidden = 35;
              c.intermediate = 33;
            }
            const std::string name = "test_cpu_qmoe_bits" + std::to_string(bits) + "_block" +
                                     std::to_string(block) + "_" + activation +
                                     (normalize ? "_normalized_" : "_unnormalized_") +
                                     DataTypeSuffix(type) + (benchmark ? "_benchmark" : "");
            if (benchmark) {
              c.tokens = 8;
              c.hidden = 64;
              c.intermediate = 96;
              const auto packed_size = [c](int64_t outputs, int64_t inputs) {
                const bool output_major = c.block == 0 || c.activation == "swiglu";
                const int64_t last = output_major ? inputs : outputs;
                const int64_t first = output_major ? outputs : inputs;
                return c.experts * first * (c.bits == 4 ? (last + 1) / 2 : last);
              };
              const auto scale_size = [c](int64_t outputs, int64_t inputs) {
                const bool output_major = c.block == 0 || c.activation == "swiglu";
                const int64_t last = output_major ? inputs : outputs;
                const int64_t first = output_major ? outputs : inputs;
                return c.experts * first * (c.block == 0 ? 1 : (last + c.block - 1) / c.block);
              };
              const int64_t width = c.intermediate * (c.activation == "swiglu" ? 2 : 1);
              std::vector<int64_t> sizes{c.tokens * c.hidden,
                                         c.tokens * c.experts,
                                         packed_size(width, c.hidden),
                                         scale_size(width, c.hidden),
                                         c.experts * width,
                                         packed_size(c.hidden, c.intermediate),
                                         scale_size(c.hidden, c.intermediate),
                                         c.experts * c.hidden};
              if (c.separate) {
                sizes.push_back(packed_size(c.intermediate, c.hidden));
                sizes.push_back(scale_size(c.intermediate, c.hidden));
                sizes.push_back(c.experts * c.intermediate);
              }
              bt_ns::Expect(
                  registry, MakeQMoENode(c), name, opsets, sizes, {c.tokens * c.hidden},
                  [c](bool expected) { return MakeQMoEData(c, expected); }, "backend-test",
                  bt_ns::TestCaseTag::AI_RT,
                  {bt_ns::TensorTypeSpec(static_cast<int32_t>(type), {c.tokens, c.hidden})});
            } else {
              bt_ns::Expect(
                  registry, MakeQMoENode(c), name, opsets, [c]() { return MakeQMoEData(c, true); },
                  "backend-test", bt_ns::TestCaseTag::AI_RT);
            }
            registry.back().rtol = type == DataType::BFLOAT16  ? 3e-2
                                   : type == DataType::FLOAT16 ? 1e-3
                                                               : 1e-5;
            registry.back().atol = type == DataType::BFLOAT16 ? 3e-3 : 1e-6;
          }
        }
      }
    }
  }
  if (!benchmark) {
    for (DataType type : {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16}) {
      Configuration clipped{type, 4, 32, "swiglu", true, false};
      clipped.alpha = 1.702f;
      clipped.beta = 1.0f;
      clipped.limit = 0.25f;
      bt_ns::Expect(
          registry, MakeQMoENode(clipped),
          "test_cpu_qmoe_swiglu_clipped_" + std::string(DataTypeSuffix(type)), opsets,
          [clipped]() { return MakeQMoEData(clipped, true); }, "backend-test",
          bt_ns::TestCaseTag::AI_RT);
      registry.back().rtol = type == DataType::BFLOAT16 ? 3e-2 : 1e-3;
      registry.back().atol = type == DataType::BFLOAT16 ? 3e-3 : 1e-6;
      for (int64_t bits : {4, 8}) {
        for (int64_t block : {16, 64, 128}) {
          Configuration boundary{type, bits, block, "relu", true, true};
          boundary.hidden = 131;
          boundary.intermediate = 67;
          bt_ns::Expect(
              registry, MakeQMoENode(boundary),
              "test_cpu_qmoe_partial_blocks_bits" + std::to_string(bits) + "_block" +
                  std::to_string(block) + "_" + DataTypeSuffix(type),
              opsets, [boundary]() { return MakeQMoEData(boundary, true); }, "backend-test",
              bt_ns::TestCaseTag::AI_RT);
          registry.back().rtol = type == DataType::BFLOAT16 ? 3e-2 : 1e-3;
          registry.back().atol = type == DataType::BFLOAT16 ? 3e-3 : 1e-6;
        }
        for (int64_t block : {0, 32}) {
          for (int64_t tokens : {0, 1}) {
            Configuration c{type, bits, block, "relu", false, true};
            c.tokens = tokens;
            c.topk = 1;
            c.experts = 1;
            const std::string name =
                "test_cpu_qmoe_bits" + std::to_string(bits) + "_block" + std::to_string(block) +
                (tokens == 0 ? "_empty_" : "_single_expert_") + DataTypeSuffix(type);
            bt_ns::Expect(
                registry, MakeQMoENode(c), name, opsets, [c]() { return MakeQMoEData(c, true); },
                "backend-test", bt_ns::TestCaseTag::AI_RT);
          }
        }
      }
    }
    const Configuration c = DecoderConfiguration();
    bt_ns::Expect(
        registry, MakeQMoENode(c), "test_cpu_qmoe_decoder_random_int4_float32", opsets,
        [c]() { return MakeQMoEData(c, true); }, "backend-test", bt_ns::TestCaseTag::AI_RT);
    registry.back().rtol = 1e-5;
    registry.back().atol = 1e-6;
  }
}

void RegisterCpuQMoEReferenceCases(std::vector<TestCase> &registry, TestMode mode) {
  if (mode != TestMode::TEST) {
    return;
  }
  const Configuration c = DecoderConfiguration();
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_domain(kMicrosoftDomain);
  node.set_op_type("MoE");
  for (const char *name : {"input", "router", "w1", "b1", "w2", "b2", "", ""}) {
    node.add_input(name);
  }
  node.add_output("output");
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "activation_type", std::string("swiglu"));
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "swiglu_fusion", int64_t{1});
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "normalize_routing_weights", int64_t{1});
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "k", c.topk);
  bt_ns::Expect(
      registry, node, "test_cpu_moe_decoder_random_int4_float32",
      {rt_ns::DefaultOpset(23), rt_ns::OpsetId(kMicrosoftDomain, 1)},
      [c]() { return MakeQMoEData(c, true, true); }, "backend-test", bt_ns::TestCaseTag::AI_RT);
  registry.back().rtol = 1e-5;
  registry.back().atol = 1e-6;
}

} // namespace onnx_light_cpu::backend_test
