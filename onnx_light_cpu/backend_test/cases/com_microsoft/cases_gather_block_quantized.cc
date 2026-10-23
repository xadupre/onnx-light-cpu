// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/com_microsoft/include_com_microsoft_cases.h"
#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_proto/onnx_helper.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace onnx_light_cpu::backend_test {
namespace {

namespace bt_ns = ONNX_LIGHT_NAMESPACE::core::backend_test;
using bt_ns::Expect;
using bt_ns::IoData;
using rt_ns::DataType;
using rt_ns::DefaultOpset;
using rt_ns::OpsetId;
using rt_ns::Shape;
using rt_ns::Tensor;

ONNX_LIGHT_NAMESPACE::NodeProto MakeGatherBlockQuantizedNode(std::int64_t bits,
                                                             bool with_zero_points,
                                                             std::int64_t quantize_axis = 1,
                                                             std::int64_t block_size = 16,
                                                             bool defaults = false) {
  auto node = MakeNode("GatherBlockQuantized", {"data", "indices", "scales"}, {"output"});
  node.set_domain(kMicrosoftDomain);
  if (with_zero_points) {
    node.add_input("zero_points");
  }
  if (!defaults) {
    ONNX_LIGHT_NAMESPACE::AddAttribute(node, "bits", bits);
    ONNX_LIGHT_NAMESPACE::AddAttribute(node, "block_size", block_size);
    ONNX_LIGHT_NAMESPACE::AddAttribute(node, "gather_axis", std::int64_t{0});
    ONNX_LIGHT_NAMESPACE::AddAttribute(node, "quantize_axis", quantize_axis);
  }
  return node;
}

std::vector<float> Runs(std::initializer_list<std::pair<std::size_t, float>> runs) {
  std::vector<float> values;
  for (const auto &run : runs) {
    values.insert(values.end(), run.first, run.second);
  }
  return values;
}

void RegisterTwoBitCase(std::vector<TestCase> &registry) {
  const auto build_data = []() -> IoData {
    std::vector<std::uint8_t> packed(9, 0xff);
    packed.insert(packed.end(), 9, 0x55);
    // Three groups occupy one byte per row; the fourth code is padding, not the next row.
    return IoData{{Tensor::FromUint8("data", {2, 9}, packed),
                   Tensor::FromInt32("indices", {2, 2}, {1, -2, 2, -3}),
                   Tensor::FromFloat("scales", {2, 3}, {1, 2, 4, 0.5f, 1, 2}),
                   Tensor::FromUint8("zero_points", {2, 1}, {0xf9, 0xe4})},
                  {Tensor::FromFloat(
                      "output", {2, 2, 36},
                      Runs({{16, 0.5f}, {16, 0}, {4, -2}, {16, 2}, {16, 2}, {4, 0}, {72, 0}}))}};
  };
  Expect(registry, MakeGatherBlockQuantizedNode(2, true),
         "test_cpu_gather_block_quantized_bits2_float_indices32_multidim_oob_padded_zp",
         {DefaultOpset(26), OpsetId(kMicrosoftDomain, 1)}, build_data, "backend-test",
         bt_ns::TestCaseTag::AI_RT);
}

void RegisterFourBitScalarCase(std::vector<TestCase> &registry) {
  const auto build_data = []() -> IoData {
    std::vector<std::uint8_t> packed;
    for (std::uint8_t byte : {0x33, 0x55, 0x77, 0x99}) {
      packed.insert(packed.end(), 18, byte);
    }
    return IoData{
        {Tensor::FromUint8("data", {2, 2, 18}, packed), Tensor::FromInt64("indices", {}, {-1}),
         MakeTensor(DataType::FLOAT16, {2, 2, 3}, {1, 2, 4, 0.5f, 1, 2, 2, 0.5f, 1, 1, 2, 0.5f}),
         Tensor::FromUint8("zero_points", {2, 2, 2},
                           {0x21, 0xf3, 0x32, 0xf4, 0x43, 0xf5, 0x54, 0xf6})},
        {MakeTensor(DataType::FLOAT16, {2, 36},
                    Runs({{16, 8}, {16, 1.5f}, {4, 2}, {16, 5}, {16, 8}, {4, 1.5f}}))}};
  };
  Expect(registry, MakeGatherBlockQuantizedNode(4, true, -1),
         "test_cpu_gather_block_quantized_bits4_float16_indices64_scalar_rank3_padded_zp",
         {DefaultOpset(26), OpsetId(kMicrosoftDomain, 1)}, build_data, "backend-test",
         bt_ns::TestCaseTag::AI_RT);
}

void RegisterEightBitCase(std::vector<TestCase> &registry) {
  const auto build_data = []() -> IoData {
    std::vector<std::uint8_t> packed(20, 130);
    packed.insert(packed.end(), 20, 126);
    return IoData{{Tensor::FromUint8("data", {2, 20}, packed),
                   Tensor::FromInt64("indices", {5}, {1, 0, -1, 2, -3}),
                   Tensor::FromFloat("scales", {2, 2}, {1, 2, 0.5f, 4}),
                   Tensor::FromUint8("zero_points", {2, 2}, {127, 129, 125, 128})},
                  {Tensor::FromFloat(
                      "output", {5, 20},
                      Runs({{16, 0.5f}, {4, -8}, {16, 3}, {4, 2}, {16, 0.5f}, {4, -8}, {40, 0}}))}};
  };
  Expect(registry, MakeGatherBlockQuantizedNode(8, true),
         "test_cpu_gather_block_quantized_bits8_float_indices64_negative_oob_partial_block",
         {DefaultOpset(26), OpsetId(kMicrosoftDomain, 1)}, build_data, "backend-test",
         bt_ns::TestCaseTag::AI_RT);
}

void RegisterDefaultsCase(std::vector<TestCase> &registry) {
  const auto build_data = []() -> IoData {
    return IoData{
        {Tensor::FromUint8("data", {2, 4}, {0xbb, 0xbb, 0xbb, 0xbb, 0x55, 0x55, 0x55, 0x55}),
         Tensor::FromInt32("indices", {2}, {-2, -1}),
         MakeTensor(DataType::FLOAT16, {2, 1}, {0.5f, 2})},
        {MakeTensor(DataType::FLOAT16, {2, 8}, Runs({{8, 1.5f}, {8, -6}}))}};
  };
  Expect(registry, MakeGatherBlockQuantizedNode(4, false, 1, 128, true),
         "test_cpu_gather_block_quantized_defaults_float16_indices32_midpoint",
         {DefaultOpset(26), OpsetId(kMicrosoftDomain, 1)}, build_data, "backend-test",
         bt_ns::TestCaseTag::AI_RT);
}

void RegisterBenchmark(std::vector<TestCase> &registry, std::int64_t bits, DataType data_type) {
  constexpr std::int64_t rows = 4096;
  constexpr std::int64_t width = 256;
  constexpr std::int64_t count = 16;
  const std::int64_t packed_width = width * bits / 8;
  const auto build_data = [=](bool generate_expected_outputs) -> IoData {
    const auto code = (1U << (bits - 1)) + 1;
    std::uint8_t byte = 0;
    for (std::int64_t shift = 0; shift < 8; shift += bits) {
      byte |= static_cast<std::uint8_t>(code << shift);
    }
    std::vector<float> scales;
    scales.reserve(rows * 2);
    for (std::int64_t row = 0; row < rows; ++row) {
      scales.push_back(0.5f);
      scales.push_back(2);
    }
    std::vector<std::int64_t> indices;
    for (std::int64_t i = 0; i < count; ++i) {
      indices.push_back((i * 257 + 17) % rows);
    }
    std::vector<Tensor> inputs;
    inputs.push_back(Tensor::FromUint8("data", {rows, packed_width},
                                       std::vector<std::uint8_t>(rows * packed_width, byte)));
    inputs.push_back(Tensor::FromInt64("indices", {count}, indices));
    inputs.push_back(MakeTensor(data_type, {rows, 2}, scales));
    if (!generate_expected_outputs) {
      return IoData{std::move(inputs), {}, {}, false};
    }
    std::vector<float> expected;
    expected.reserve(count * width);
    for (std::int64_t i = 0; i < count; ++i) {
      expected.insert(expected.end(), 128, 0.5f);
      expected.insert(expected.end(), 128, 2);
    }
    return IoData{std::move(inputs), {MakeTensor(data_type, {count, width}, expected)}};
  };
  const std::string name = "test_cpu_gather_block_quantized_embedding4096x256_ids16_bits" +
                           std::to_string(bits) +
                           (data_type == DataType::FLOAT16 ? "_scalesfp16_uint8_benchmark"
                                                           : "_scalesfp32_uint8_benchmark");
  Expect(registry, MakeGatherBlockQuantizedNode(bits, false, 1, 128), name,
         {DefaultOpset(26), OpsetId(kMicrosoftDomain, 1)}, {rows * packed_width, count, rows * 2},
         {count * width}, build_data, "backend-test", bt_ns::TestCaseTag::AI_RT,
         {bt_ns::TensorTypeSpec(static_cast<std::int32_t>(data_type), Shape{count, width})});
}

} // namespace

void RegisterCpuGatherBlockQuantizedCases(std::vector<TestCase> &registry, TestMode mode) {
  if (mode == TestMode::BENCHMARK) {
    RegisterBenchmark(registry, 2, DataType::FLOAT);
    RegisterBenchmark(registry, 4, DataType::FLOAT16);
    RegisterBenchmark(registry, 8, DataType::FLOAT);
    return;
  }
  RegisterTwoBitCase(registry);
  RegisterFourBitScalarCase(registry);
  RegisterEightBitCase(registry);
  RegisterDefaultsCase(registry);
}

} // namespace onnx_light_cpu::backend_test
