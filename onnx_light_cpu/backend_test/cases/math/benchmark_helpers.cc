// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/random.h"

#include <cmath>
#include <stdexcept>

namespace onnx_light_cpu::backend_test {

ONNX_LIGHT_NAMESPACE::NodeProto MakeNode(std::string_view op_type,
                                         std::initializer_list<const char *> inputs,
                                         std::initializer_list<const char *> outputs) {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type(std::string(op_type));
  for (const char *input : inputs) {
    node.add_input(input);
  }
  for (const char *output : outputs) {
    node.add_output(output);
  }
  return node;
}

const char *DataTypeSuffix(rt_ns::DataType data_type) {
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

rt_ns::Tensor MakeTensor(rt_ns::DataType data_type, const rt_ns::Shape &shape,
                         const std::vector<float> &values) {
  switch (data_type) {
  case rt_ns::DataType::FLOAT:
    return rt_ns::Tensor::FromFloat("", shape, values);
  case rt_ns::DataType::DOUBLE:
    return rt_ns::Tensor::FromDouble("", shape, std::vector<double>(values.begin(), values.end()));
  case rt_ns::DataType::FLOAT16:
    return rt_ns::MakeFloat16Tensor("", shape, values);
  case rt_ns::DataType::BFLOAT16:
    return rt_ns::MakeBfloat16Tensor("", shape, values);
  default:
    throw std::invalid_argument("Backend test tensor requires a floating-point data type.");
  }
}

rt_ns::Tensor MakeBenchmarkTensor(rt_ns::DataType data_type, const rt_ns::Shape &shape,
                                  std::uint64_t seed, bool positive) {
  std::vector<float> values = rt_ns::Randn<float>(shape, seed);
  if (positive) {
    for (float &value : values) {
      value = std::abs(value) + 0.01f;
    }
  }
  switch (data_type) {
  case rt_ns::DataType::FLOAT:
  case rt_ns::DataType::DOUBLE:
  case rt_ns::DataType::FLOAT16:
  case rt_ns::DataType::BFLOAT16:
    return MakeTensor(data_type, shape, values);
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
  default:
    throw std::invalid_argument("Unsupported benchmark tensor data type.");
  }
}

} // namespace onnx_light_cpu::backend_test
