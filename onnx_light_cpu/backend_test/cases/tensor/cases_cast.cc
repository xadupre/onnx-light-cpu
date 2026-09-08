// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/tensor/include_tensor_cases.h"

#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace onnx_light_cpu::backend_test {
namespace {

namespace bt_ns = ONNX_LIGHT_NAMESPACE::core::backend_test;
using rt_ns::DataType;
using rt_ns::Shape;
using rt_ns::Tensor;

struct CastShape {
  const char *name;
  Shape data;
  bool sequence_lengths = false;
};

const char *CastTypeSuffix(DataType type) {
  switch (type) {
  case DataType::BOOL:
    return "bool";
  case DataType::UINT16:
    return "uint16";
  default:
    return DataTypeSuffix(type);
  }
}

Tensor MakeCastInput(DataType type, const CastShape &shape) {
  const auto count = static_cast<std::size_t>(shape.data.product());
  if (shape.sequence_lengths) {
    constexpr std::array<std::int64_t, 8> lengths{40960, 0,          1,          2048,
                                                  -1,    2147483647, 2147483648, 4294967295};
    std::vector<std::int64_t> values(count);
    for (std::size_t i = 0; i < count; ++i) {
      values[i] = lengths[i % lengths.size()];
    }
    return Tensor::FromInt64("data", shape.data, values);
  }
  if (type == DataType::BOOL) {
    std::vector<std::uint8_t> values(count);
    for (std::size_t i = 0; i < count; ++i) {
      values[i] = static_cast<std::uint8_t>(i % 3 != 0);
    }
    return Tensor::FromBool("data", shape.data, values);
  }
  if (type == DataType::UINT16) {
    std::vector<std::uint16_t> values(count);
    for (std::size_t i = 0; i < count; ++i) {
      values[i] = static_cast<std::uint16_t>((i * 127 + 19) % 65536);
    }
    return Tensor::FromUint16("data", shape.data, values);
  }
  return MakeBenchmarkTensor(type, shape.data, 1327);
}

void RegisterCastCase(std::vector<TestCase> &registry, const CastShape &shape, DataType from,
                      DataType to, bool benchmark) {
  const auto opset = rt_ns::DefaultOpset(shape.sequence_lengths ? 26 : 13);
  auto node = MakeNode("Cast", {"data"}, {"output"});
  auto *attribute = node.add_attribute();
  attribute->set_name("to");
  attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::INT);
  attribute->set_i(static_cast<std::int32_t>(to));
  const std::string name = "test_cpu_cast_" + std::string(shape.name) + "_" + CastTypeSuffix(from) +
                           "_to_" + CastTypeSuffix(to) + (benchmark ? "_benchmark" : "");
  const auto build = [shape, from, to, opset](bool expected) -> bt_ns::IoData {
    Tensor data = MakeCastInput(from, shape);
    if (!expected) {
      return bt_ns::IoData{{std::move(data)}, {}, {}, false};
    }
    const ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::Cast reference(rt_ns::KernelContext{opset});
    Tensor output = reference(data, static_cast<std::int32_t>(to));
    return bt_ns::IoData{{std::move(data)}, {std::move(output)}};
  };
  const std::vector<std::int64_t> counts{shape.data.product()};
  if (benchmark) {
    bt_ns::Expect(registry, std::move(node), name, {opset}, counts, counts, build, "backend-test",
                  bt_ns::TestCaseTag::NONE,
                  {bt_ns::TensorTypeSpec(static_cast<std::int32_t>(to), shape.data)});
  } else {
    bt_ns::Expect(registry, std::move(node), name, {opset}, counts, counts,
                  [build]() { return build(true); });
  }
}

} // namespace

void RegisterCpuCastCases(std::vector<TestCase> &registry, TestMode mode) {
  if (mode == TestMode::BENCHMARK) {
    for (const CastShape &shape : {
             CastShape{"scalar", {}},
             CastShape{"small", {16}},
             CastShape{"medium", {1024}},
             CastShape{"tail", {65537}},
             CastShape{"large", {1048576}},
             CastShape{"matrix", {256, 4096}},
         }) {
      for (const auto &[from, to] : {
               std::pair{DataType::INT64, DataType::INT32},
               std::pair{DataType::INT32, DataType::INT64},
               std::pair{DataType::FLOAT, DataType::FLOAT16},
               std::pair{DataType::FLOAT16, DataType::FLOAT},
               std::pair{DataType::FLOAT, DataType::BFLOAT16},
               std::pair{DataType::BFLOAT16, DataType::FLOAT},
               std::pair{DataType::FLOAT, DataType::DOUBLE},
               std::pair{DataType::DOUBLE, DataType::FLOAT},
               std::pair{DataType::FLOAT, DataType::INT32},
               std::pair{DataType::INT32, DataType::FLOAT},
               std::pair{DataType::FLOAT, DataType::INT64},
               std::pair{DataType::INT64, DataType::FLOAT},
               std::pair{DataType::FLOAT, DataType::INT8},
               std::pair{DataType::INT8, DataType::FLOAT},
               std::pair{DataType::FLOAT, DataType::UINT8},
               std::pair{DataType::UINT8, DataType::FLOAT},
               std::pair{DataType::FLOAT, DataType::BOOL},
               std::pair{DataType::BOOL, DataType::FLOAT},
               std::pair{DataType::INT64, DataType::INT8},
               std::pair{DataType::INT16, DataType::INT32},
               std::pair{DataType::FLOAT, DataType::FLOAT},
               std::pair{DataType::INT64, DataType::INT64},
           }) {
        RegisterCastCase(registry, shape, from, to, true);
      }
    }
    return;
  }
  constexpr std::array<DataType, 11> types{DataType::FLOAT,    DataType::DOUBLE, DataType::FLOAT16,
                                           DataType::BFLOAT16, DataType::INT8,   DataType::UINT8,
                                           DataType::INT16,    DataType::UINT16, DataType::INT32,
                                           DataType::INT64,    DataType::BOOL};
  for (const CastShape &shape : {
           CastShape{"scalar", {}},
           CastShape{"empty", {2, 0, 5}},
           CastShape{"vector", {17}},
           CastShape{"multidim", {2, 3, 5}},
           CastShape{"tail", {257}},
       }) {
    for (DataType from : types) {
      for (DataType to : types) {
        RegisterCastCase(registry, shape, from, to, false);
      }
    }
  }
  RegisterCastCase(registry, {"qwen_length", {}, true}, DataType::INT64, DataType::INT32, false);
  RegisterCastCase(registry, {"qwen_lengths", {8}, true}, DataType::INT64, DataType::INT32, false);
}

} // namespace onnx_light_cpu::backend_test
