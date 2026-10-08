// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/attention/include_attention_cases.h"
#include "onnx_light_cpu/backend_test/cases/com_microsoft/include_com_microsoft_cases.h"
#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"
#include "onnx_light_cpu/kernels/attention/rotary_embedding_kernel.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
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

NodeProto MakeRotaryEmbeddingNode(bool microsoft, bool with_positions, bool interleaved,
                                  std::int64_t rotary_dim, std::int64_t num_heads) {
  NodeProto node;
  node.set_op_type("RotaryEmbedding");
  if (microsoft) {
    node.set_domain(kMicrosoftDomain);
  }
  node.add_input("X");
  if (microsoft) {
    node.add_input("position_ids");
    node.add_input("cos_cache");
    node.add_input("sin_cache");
  } else {
    node.add_input("cos_cache");
    node.add_input("sin_cache");
    if (with_positions) {
      node.add_input("position_ids");
    }
  }
  node.add_output("Y");
  if (interleaved) {
    ONNX_LIGHT_NAMESPACE::AddAttribute(node, "interleaved", std::int64_t{1});
  }
  if (rotary_dim != 0) {
    ONNX_LIGHT_NAMESPACE::AddAttribute(node, "rotary_embedding_dim", rotary_dim);
  }
  if (num_heads != 0) {
    ONNX_LIGHT_NAMESPACE::AddAttribute(node, "num_heads", num_heads);
  }
  return node;
}

std::vector<OpsetId> RotaryEmbeddingOpsets(bool microsoft) {
  std::vector<OpsetId> opsets{DefaultOpset(23)};
  if (microsoft) {
    opsets.emplace_back(kMicrosoftDomain, 1);
  }
  return opsets;
}

void RegisterCorrectnessCase(std::vector<TestCase> &registry, bool microsoft, DataType data_type,
                             const std::string &variant, const Shape &input_shape,
                             const std::vector<float> &input_values, const Shape &cache_shape,
                             const std::vector<float> &cos_values,
                             const std::vector<float> &sin_values, const Shape &position_shape,
                             const std::vector<std::int64_t> &position_values,
                             const std::vector<float> &expected_values, bool interleaved,
                             std::int64_t rotary_dim, std::int64_t num_heads) {
  const bool with_positions = !position_shape.empty() || !position_values.empty();
  const std::string name = std::string(microsoft ? "test_cpu_microsoft_rotary_embedding_"
                                                 : "test_cpu_rotary_embedding_") +
                           variant + "_" + DataTypeSuffix(data_type);
  const NodeProto node =
      MakeRotaryEmbeddingNode(microsoft, with_positions, interleaved, rotary_dim, num_heads);
  Expect(
      registry, node, name, RotaryEmbeddingOpsets(microsoft),
      [=]() -> IoData {
        Tensor input = MakeTensor(data_type, input_shape, input_values);
        Tensor cos = MakeTensor(data_type, cache_shape, cos_values);
        Tensor sin = MakeTensor(data_type, cache_shape, sin_values);
        Tensor expected = MakeTensor(data_type, input_shape, expected_values);
        if (!with_positions) {
          return IoData{{std::move(input), std::move(cos), std::move(sin)}, {std::move(expected)}};
        }
        Tensor positions = Tensor::FromInt64("", position_shape, position_values);
        if (microsoft) {
          return IoData{{std::move(input), std::move(positions), std::move(cos), std::move(sin)},
                        {std::move(expected)}};
        }
        return IoData{{std::move(input), std::move(cos), std::move(sin), std::move(positions)},
                      {std::move(expected)}};
      },
      "backend-test", microsoft ? bt_ns::TestCaseTag::AI_RT : bt_ns::TestCaseTag::NONE);
}

void RegisterBenchmarkCase(std::vector<TestCase> &registry, bool microsoft, DataType data_type) {
  constexpr std::int64_t kBatch = 1;
  constexpr std::int64_t kSequence = 128;
  constexpr std::int64_t kHeads = 8;
  constexpr std::int64_t kHeadSize = 64;
  const Shape input_shape{kBatch, kSequence, kHeads * kHeadSize};
  const Shape cache_shape{kSequence, kHeadSize / 2};
  const Shape position_shape{kBatch, kSequence};
  const std::int64_t input_count = input_shape.product();
  const std::int64_t cache_count = cache_shape.product();
  std::vector<std::int64_t> input_counts;
  if (microsoft) {
    input_counts = {input_count, position_shape.product(), cache_count, cache_count};
  } else {
    input_counts = {input_count, cache_count, cache_count, position_shape.product()};
  }
  const std::string name = std::string(microsoft ? "test_cpu_microsoft_rotary_embedding_"
                                                 : "test_cpu_rotary_embedding_") +
                           "b1_s128_h8_d64_" + DataTypeSuffix(data_type) + "_benchmark";
  const NodeProto node = MakeRotaryEmbeddingNode(microsoft, true, false, 0, kHeads);
  Expect(registry, node, name, RotaryEmbeddingOpsets(microsoft), input_counts, {input_count},
         [=](bool generate_expected_outputs) -> IoData {
           Tensor input = MakeBenchmarkTensor(data_type, input_shape, 7001);
           Tensor cos = MakeBenchmarkTensor(data_type, cache_shape, 7002);
           Tensor sin = MakeBenchmarkTensor(data_type, cache_shape, 7003);
           std::vector<std::int64_t> position_values(static_cast<std::size_t>(kSequence));
           for (std::int64_t i = 0; i < kSequence; ++i) {
             position_values[static_cast<std::size_t>(i)] = i;
           }
           Tensor positions = Tensor::FromInt64("", position_shape, std::move(position_values));
           std::vector<Tensor> outputs;
           if (generate_expected_outputs) {
             const RotaryEmbeddingKernel kernel{
                 KernelContext{microsoft ? OpsetId(kMicrosoftDomain, 1) : DefaultOpset(23)}};
             outputs.push_back(
                 kernel(input, cos, sin, &positions, false, 0, kHeads, microsoft, nullptr));
           }
           std::vector<Tensor> inputs;
           if (microsoft) {
             inputs = {std::move(input), std::move(positions), std::move(cos), std::move(sin)};
           } else {
             inputs = {std::move(input), std::move(cos), std::move(sin), std::move(positions)};
           }
           return IoData{std::move(inputs), std::move(outputs), {}, generate_expected_outputs};
         },
         "backend-test", microsoft ? bt_ns::TestCaseTag::AI_RT : bt_ns::TestCaseTag::NONE,
         {bt_ns::TensorTypeSpec(static_cast<std::int32_t>(data_type), input_shape)});
}

void RegisterDomainCases(std::vector<TestCase> &registry, TestMode mode, bool microsoft) {
  for (const DataType data_type : {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16}) {
    if (mode == TestMode::BENCHMARK) {
      RegisterBenchmarkCase(registry, microsoft, data_type);
      continue;
    }
    if (microsoft) {
      RegisterCorrectnessCase(registry, true, data_type, "partial_full_width_cache", {1, 2, 4},
                              {1, 2, 3, 4, 5, 6, 7, 8}, {3, 2}, {1, 1, 0, 1, -1, 0},
                              {0, 0, 1, 0, 0, 1}, {1}, {1}, {-2, 1, 3, 4, -5, -6, 7, 8}, false, 2,
                              1);
      RegisterCorrectnessCase(registry, true, data_type, "interleaved_scalar_offset", {1, 1, 1, 4},
                              {1, 2, 3, 4}, {2, 2}, {1, 1, 0, 1}, {0, 0, 1, 0}, {1}, {1},
                              {-2, 1, 3, 4}, true, 0, 0);
    } else {
      RegisterCorrectnessCase(registry, false, data_type, "split_position_ids", {1, 2, 4},
                              {1, 2, 3, 4, 5, 6, 7, 8}, {3, 2}, {1, 1, 0, 1, -1, 0},
                              {0, 0, 1, 0, 0, 1}, {1, 2}, {1, 2}, {-3, 2, 1, 4, -5, -8, -7, 6},
                              false, 4, 1);
      RegisterCorrectnessCase(registry, false, data_type, "interleaved_implicit_positions",
                              {1, 1, 1, 4}, {1, 2, 3, 4}, {1, 1, 2}, {0, 1}, {1, 0}, {}, {},
                              {-2, 1, 3, 4}, true, 0, 0);
    }
  }
}

} // namespace

void RegisterCpuRotaryEmbeddingCases(std::vector<TestCase> &registry, TestMode mode) {
  RegisterDomainCases(registry, mode, false);
}

void RegisterCpuMicrosoftRotaryEmbeddingCases(std::vector<TestCase> &registry, TestMode mode) {
  RegisterDomainCases(registry, mode, true);
}

} // namespace onnx_light_cpu::backend_test
