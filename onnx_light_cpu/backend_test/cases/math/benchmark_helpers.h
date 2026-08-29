// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/random.h"
#include "onnx_core/runtime/memory/simple_tensor.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace onnx_light_cpu::backend_test {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

inline const char *DataTypeSuffix(rt_ns::DataType data_type) {
  switch (data_type) {
  case rt_ns::DataType::FLOAT:
    return "float32";
  case rt_ns::DataType::DOUBLE:
    return "float64";
  case rt_ns::DataType::INT8:
    return "int8";
  case rt_ns::DataType::UINT8:
    return "uint8";
  case rt_ns::DataType::INT16:
    return "int16";
  case rt_ns::DataType::INT32:
    return "int32";
  case rt_ns::DataType::INT64:
    return "int64";
  case rt_ns::DataType::FLOAT16:
    return "float16";
  case rt_ns::DataType::BFLOAT16:
    return "bfloat16";
  default:
    throw std::invalid_argument("Unsupported benchmark tensor data type.");
  }
}

inline rt_ns::Tensor MakeBenchmarkTensor(rt_ns::DataType data_type, const rt_ns::Shape &shape,
                                         std::uint64_t seed, bool positive = false) {
  std::vector<float> values = rt_ns::Randn<float>(shape, seed);
  if (positive) {
    for (float &value : values) {
      value = std::abs(value) + 0.01f;
    }
  }
  switch (data_type) {
  case rt_ns::DataType::FLOAT:
    return rt_ns::Tensor::FromFloat("", shape, values);
  case rt_ns::DataType::DOUBLE:
    return rt_ns::Tensor::FromDouble("", shape, std::vector<double>(values.begin(), values.end()));
  case rt_ns::DataType::INT8: {
    std::vector<std::int8_t> converted(values.size());
    for (std::size_t i = 0; i < converted.size(); ++i) {
      converted[i] = static_cast<std::int8_t>(static_cast<int>((i + seed) % 31) - 15);
    }
    return rt_ns::Tensor::FromInt8("", shape, converted);
  }
  case rt_ns::DataType::UINT8: {
    std::vector<std::uint8_t> converted(values.size());
    for (std::size_t i = 0; i < converted.size(); ++i) {
      converted[i] = static_cast<std::uint8_t>((i + seed) % 31);
    }
    return rt_ns::Tensor::FromUint8("", shape, converted);
  }
  case rt_ns::DataType::INT16: {
    std::vector<std::int16_t> converted(values.size());
    for (std::size_t i = 0; i < converted.size(); ++i) {
      converted[i] = static_cast<std::int16_t>(static_cast<int>((i + seed) % 1001) - 500);
    }
    return rt_ns::Tensor::FromInt16("", shape, converted);
  }
  case rt_ns::DataType::INT32: {
    std::vector<std::int32_t> converted(values.size());
    for (std::size_t i = 0; i < converted.size(); ++i) {
      converted[i] = static_cast<std::int32_t>(static_cast<int>((i + seed) % 1001) - 500);
    }
    return rt_ns::Tensor::FromInt32("", shape, converted);
  }
  case rt_ns::DataType::INT64: {
    std::vector<std::int64_t> converted(values.size());
    for (std::size_t i = 0; i < converted.size(); ++i) {
      converted[i] = static_cast<std::int64_t>((i + seed) % 1001) - 500;
    }
    return rt_ns::Tensor::FromInt64("", shape, converted);
  }
  case rt_ns::DataType::FLOAT16:
    return rt_ns::MakeFloat16Tensor("", shape, values);
  case rt_ns::DataType::BFLOAT16:
    return rt_ns::MakeBfloat16Tensor("", shape, values);
  default:
    throw std::invalid_argument("Unsupported benchmark tensor data type.");
  }
}

template <typename KernelFactory>
void RegisterUnaryBenchmark(
    std::vector<ONNX_LIGHT_NAMESPACE::core::backend_test::TestCase> &registry,
    const std::string &op_type, KernelFactory kernel_factory, const rt_ns::OpsetId &opset,
    rt_ns::DataType data_type, std::int64_t size) {
  namespace bt_ns = ONNX_LIGHT_NAMESPACE::core::backend_test;
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type(op_type);
  node.add_input("x");
  node.add_output("y");
  std::string op_tag = op_type;
  for (char &character : op_tag) {
    if (character >= 'A' && character <= 'Z') {
      character = static_cast<char>(character - 'A' + 'a');
    }
  }
  const std::string name = "test_cpu_" + op_tag + "_n" + std::to_string(size) + "_" +
                           DataTypeSuffix(data_type) + "_benchmark";
  bt_ns::Expect(
      registry, std::move(node), name, {opset}, {size}, {size},
      [kernel_factory, op_type, data_type, size](bool generate_expected_outputs) -> bt_ns::IoData {
        rt_ns::Tensor x = MakeBenchmarkTensor(data_type, {size}, 987654321ULL, op_type == "Log");
        if (!generate_expected_outputs) {
          return bt_ns::IoData{{std::move(x)}, {}, false};
        }
        rt_ns::Tensor y = kernel_factory()(x);
        return bt_ns::IoData{{std::move(x)}, {std::move(y)}};
      });
}

} // namespace onnx_light_cpu::backend_test
