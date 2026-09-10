// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/elementwise/include_elementwise_cases.h"

#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/memory/simple_tensor.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace onnx_light_cpu::backend_test {
namespace {

namespace bt_ns = ONNX_LIGHT_NAMESPACE::core::backend_test;
namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

using bt_ns::Expect;
using bt_ns::IoData;
using bt_ns::TestCase;
using bt_ns::TestMode;
using rt_ns::DataType;
using rt_ns::DefaultOpset;
using rt_ns::OpsetId;
using rt_ns::Tensor;

std::string Lowercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::size_t ElementCount(const rt_ns::Shape &shape) {
  std::size_t count = 1;
  for (std::int64_t dim : shape) {
    count *= static_cast<std::size_t>(std::max<std::int64_t>(dim, 0));
  }
  return count;
}

rt_ns::Shape BroadcastShape(const std::vector<rt_ns::Shape> &shapes) {
  size_t rank = 0;
  for (const rt_ns::Shape &shape : shapes) {
    rank = std::max(rank, shape.size());
  }
  std::vector<std::int64_t> output(rank, 1);
  for (const rt_ns::Shape &shape : shapes) {
    for (size_t axis = 0; axis < rank; ++axis) {
      const std::int64_t dim = axis < rank - shape.size() ? 1 : shape[axis - (rank - shape.size())];
      if (output[axis] != dim && output[axis] != 1 && dim != 1) {
        throw std::invalid_argument("incompatible broadcast shapes");
      }
      output[axis] = std::max(output[axis], dim);
    }
  }
  return rt_ns::Shape(std::move(output));
}

std::vector<std::int64_t> RowMajorStrides(const rt_ns::Shape &shape) {
  std::vector<std::int64_t> strides(shape.size(), 1);
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(shape.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] =
        strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
  }
  return strides;
}

std::vector<std::int64_t> UnravelIndex(std::size_t index, const rt_ns::Shape &shape,
                                       const std::vector<std::int64_t> &strides) {
  std::vector<std::int64_t> coords(shape.size(), 0);
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    if (shape[axis] == 0) {
      continue;
    }
    const auto stride = static_cast<std::size_t>(strides[axis]);
    coords[axis] = static_cast<std::int64_t>(index / stride);
    index %= stride;
  }
  return coords;
}

template <typename T>
std::vector<T> ComputeVariadicReference(std::string_view op_type,
                                        const std::vector<rt_ns::Shape> &shapes,
                                        const std::vector<std::vector<T>> &inputs) {
  const rt_ns::Shape output_shape = BroadcastShape(shapes);
  const std::vector<std::int64_t> output_strides = RowMajorStrides(output_shape);
  std::vector<std::vector<std::int64_t>> input_strides;
  input_strides.reserve(shapes.size());
  for (const auto &shape : shapes) {
    input_strides.push_back(RowMajorStrides(shape));
  }
  std::vector<T> output(ElementCount(output_shape));
  for (size_t index = 0; index < output.size(); ++index) {
    const std::vector<std::int64_t> out_coords = UnravelIndex(index, output_shape, output_strides);
    auto value = inputs.front()[0];
    bool first = true;
    for (size_t input_index = 0; input_index < inputs.size(); ++input_index) {
      const rt_ns::Shape &shape = shapes[input_index];
      const auto &strides = input_strides[input_index];
      std::size_t linear = 0;
      const size_t offset = output_shape.size() - shape.size();
      for (size_t axis = 0; axis < shape.size(); ++axis) {
        const std::int64_t coord = shape[axis] == 1 ? 0 : out_coords[offset + axis];
        linear += static_cast<size_t>(coord * strides[axis]);
      }
      const T current = inputs[input_index][linear];
      if (op_type == "Sum" || op_type == "Mean") {
        if (first) {
          value = current;
          first = false;
        } else {
          value = static_cast<T>(value + current);
        }
      } else if (op_type == "Min") {
        if (first) {
          value = current;
          first = false;
        } else {
          value = std::min(value, current);
        }
      } else if (op_type == "Max") {
        if (first) {
          value = current;
          first = false;
        } else {
          value = std::max(value, current);
        }
      } else {
        throw std::invalid_argument("unsupported variadic op");
      }
    }
    if (op_type == "Mean") {
      value = static_cast<T>(value / static_cast<T>(inputs.size()));
    }
    output[index] = value;
  }
  return output;
}

template <typename T> Tensor MakeTensor(const rt_ns::Shape &shape, const std::vector<T> &values) {
  return Tensor::From<T>("", shape, values);
}

ONNX_LIGHT_NAMESPACE::NodeProto MakeVariadicNode(std::string_view op_type,
                                                 std::size_t input_count) {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type(std::string(op_type));
  for (std::size_t i = 0; i < input_count; ++i) {
    node.add_input("x" + std::to_string(i));
  }
  node.add_output("y");
  return node;
}

template <typename T>
void RegisterVariadicCase(std::vector<TestCase> &registry, std::string_view op_type,
                          const std::string &name_suffix, const OpsetId &opset,
                          const std::vector<rt_ns::Shape> &shapes,
                          const std::vector<std::vector<T>> &values,
                          const rt_ns::Shape &expected_shape) {
  const std::string lower_name =
      "test_cpu_" + Lowercase(std::string(op_type)).append("_").append(name_suffix);
  Expect(registry, MakeVariadicNode(op_type, shapes.size()), lower_name, {opset}, [=]() -> IoData {
    std::vector<Tensor> inputs;
    inputs.reserve(shapes.size());
    for (size_t i = 0; i < shapes.size(); ++i) {
      inputs.push_back(MakeTensor<T>(shapes[i], values[i]));
    }
    const std::vector<T> expected_values = ComputeVariadicReference<T>(op_type, shapes, values);
    const Tensor expected = Tensor::From<T>("", expected_shape, expected_values);
    return IoData{std::move(inputs), {expected}};
  });
}

void RegisterVariadicBenchmark(std::vector<TestCase> &registry, std::string_view op_type,
                               const std::string &name_suffix, DataType type,
                               const rt_ns::Shape &shape) {
  const std::int64_t count = static_cast<std::int64_t>(ElementCount(shape));
  const std::string name = "test_cpu_" +
                           Lowercase(std::string(op_type)).append("_").append(name_suffix) + "_" +
                           DataTypeSuffix(type) + "_benchmark";
  Expect(registry, MakeNode(op_type, {"x0", "x1", "x2"}, {"y"}), name, {DefaultOpset(13)},
         {count, count, count}, {count},
         [=](bool generate_expected_outputs) -> IoData {
           Tensor x0 = MakeBenchmarkTensor(type, shape, 2001);
           Tensor x1 = MakeBenchmarkTensor(type, shape, 2002);
           Tensor x2 = MakeBenchmarkTensor(type, shape, 2003);
           if (!generate_expected_outputs) {
             return IoData{{std::move(x0), std::move(x1), std::move(x2)}, {}, {}, false};
           }
           if (type == DataType::FLOAT) {
             const std::vector<float> expected = ComputeVariadicReference<float>(
                 op_type, {shape, shape, shape},
                 {std::vector<float>(x0.AsFloat(), x0.AsFloat() + count),
                  std::vector<float>(x1.AsFloat(), x1.AsFloat() + count),
                  std::vector<float>(x2.AsFloat(), x2.AsFloat() + count)});
             return IoData{{std::move(x0), std::move(x1), std::move(x2)},
                           {Tensor::FromFloat("", shape, expected)}};
           }
           const std::vector<double> expected = ComputeVariadicReference<double>(
               op_type, {shape, shape, shape},
               {std::vector<double>(x0.AsDouble(), x0.AsDouble() + count),
                std::vector<double>(x1.AsDouble(), x1.AsDouble() + count),
                std::vector<double>(x2.AsDouble(), x2.AsDouble() + count)});
           return IoData{{std::move(x0), std::move(x1), std::move(x2)},
                         {Tensor::FromDouble("", shape, expected)}};
         },
         "backend-test", bt_ns::TestCaseTag::NONE,
         {bt_ns::TensorTypeSpec(static_cast<std::int32_t>(type), shape)});
}

} // namespace

void RegisterCpuVariadicCases(std::vector<TestCase> &registry, const std::string &op_type,
                              TestMode mode) {
  const OpsetId opset = DefaultOpset(13);
  if (mode == TestMode::BENCHMARK) {
    const rt_ns::Shape benchmark_shape = {1, 4096};
    if (op_type == "Sum" || op_type == "Mean") {
      for (DataType type : {DataType::FLOAT, DataType::DOUBLE}) {
        RegisterVariadicBenchmark(registry, op_type, "n4096_3inputs", type, benchmark_shape);
      }
    } else {
      RegisterVariadicBenchmark(registry, op_type, "n4096_3inputs", DataType::FLOAT,
                                benchmark_shape);
    }
    return;
  }

  if (op_type == "Sum") {
    RegisterVariadicCase<float>(registry, op_type, "broadcast3_float32", opset,
                                {{2, 1}, {1, 3}, {}},
                                {{1.0f, 2.0f}, {10.0f, 20.0f, 30.0f}, {100.0f}}, {2, 3});
    RegisterVariadicCase<double>(registry, op_type, "arity4_float64", opset,
                                 {{2, 2}, {2, 2}, {2, 2}, {2, 2}},
                                 {{1.0, 2.0, 3.0, 4.0},
                                  {5.0, 6.0, 7.0, 8.0},
                                  {9.0, 10.0, 11.0, 12.0},
                                  {13.0, 14.0, 15.0, 16.0}},
                                 {2, 2});
    return;
  }
  if (op_type == "Mean") {
    RegisterVariadicCase<float>(registry, op_type, "broadcast3_float32", opset,
                                {{2, 1}, {1, 3}, {}}, {{1.0f, 4.0f}, {10.0f, 20.0f, 30.0f}, {1.0f}},
                                {2, 3});
    RegisterVariadicCase<double>(registry, op_type, "arity4_float64", opset, {{2}, {2}, {2}, {2}},
                                 {{2.0, 6.0}, {4.0, 10.0}, {8.0, 14.0}, {10.0, 18.0}}, {2});
    return;
  }
  if (op_type == "Min") {
    RegisterVariadicCase<float>(registry, op_type, "broadcast3_float32", opset,
                                {{2, 1}, {1, 3}, {}}, {{5.0f, 1.0f}, {10.0f, -2.0f, 30.0f}, {3.0f}},
                                {2, 3});
    RegisterVariadicCase<double>(registry, op_type, "arity4_float64", opset,
                                 {{2, 2}, {2, 2}, {2, 2}, {2, 2}},
                                 {{5.0, 100.0, 7.0, -2.0},
                                  {4.0, 6.0, 9.0, 0.0},
                                  {-8.0, 40.0, 3.0, -7.0},
                                  {1.0, 2.0, 30.0, -1.0}},
                                 {2, 2});
    return;
  }
  if (op_type == "Max") {
    RegisterVariadicCase<float>(registry, op_type, "broadcast3_float32", opset,
                                {{2, 1}, {1, 3}, {}}, {{5.0f, 1.0f}, {10.0f, -2.0f, 30.0f}, {3.0f}},
                                {2, 3});
    RegisterVariadicCase<double>(registry, op_type, "arity4_float64", opset, {{2}, {2}, {2}, {2}},
                                 {{5.0, 100.0}, {4.0, 600.0}, {8.0, 40.0}, {1.0, 2.0}}, {2});
    return;
  }
  throw std::invalid_argument("Unsupported variadic op type for backend registration.");
}

} // namespace onnx_light_cpu::backend_test
