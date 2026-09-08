// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"
#include "onnx_light_cpu/backend_test/cases/tensor/include_tensor_cases.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace onnx_light_cpu::backend_test {
namespace {

namespace bt_ns = ONNX_LIGHT_NAMESPACE::core::backend_test;
using rt_ns::DataType;
using rt_ns::Shape;
using rt_ns::Tensor;

struct ConcatShape {
  const char *name;
  std::vector<Shape> inputs;
  std::int64_t axis;
};

void RegisterConcatCase(std::vector<TestCase> &registry, const ConcatShape &shape,
                        DataType data_type, bool benchmark) {
  const auto opset = rt_ns::DefaultOpset(13);
  auto node = MakeNode("Concat", {}, {"output"});
  auto *attribute = node.add_attribute();
  attribute->set_name("axis");
  attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::INT);
  attribute->set_i(shape.axis);
  const std::int64_t axis =
      shape.axis < 0 ? shape.axis + static_cast<std::int64_t>(shape.inputs.front().size())
                     : shape.axis;
  Shape output_shape = shape.inputs.front();
  output_shape[axis] = 0;
  std::vector<std::int64_t> input_sizes;
  for (std::size_t i = 0; i < shape.inputs.size(); ++i) {
    node.add_input("input" + std::to_string(i));
    input_sizes.push_back(shape.inputs[i].product());
    output_shape[axis] += shape.inputs[i][axis];
  }
  const std::string name = "test_cpu_concat_" + std::string(shape.name) + "_" +
                           DataTypeSuffix(data_type) + (benchmark ? "_benchmark" : "");
  const auto build = [shape, data_type, opset, output_shape](bool expected) -> bt_ns::IoData {
    rt_ns::Tensors inputs;
    for (std::size_t i = 0; i < shape.inputs.size(); ++i) {
      inputs.push_back(MakeBenchmarkTensor(data_type, shape.inputs[i], 947 + i));
    }
    if (!expected) {
      return bt_ns::IoData{std::move(inputs), {}, {}, false};
    }
    const ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::Concat reference(rt_ns::KernelContext{opset});
    Tensor output = reference(inputs, shape.axis);
    if (output.shape != output_shape) {
      throw std::logic_error("Concat backend case has incorrect output shape metadata.");
    }
    return bt_ns::IoData{std::move(inputs), {std::move(output)}};
  };
  const std::vector<std::int64_t> output_sizes{output_shape.product()};
  if (benchmark) {
    bt_ns::Expect(registry, std::move(node), name, {opset}, input_sizes, output_sizes, build,
                  "backend-test", bt_ns::TestCaseTag::NONE,
                  {bt_ns::TensorTypeSpec(static_cast<std::int32_t>(data_type), output_shape)});
  } else {
    bt_ns::Expect(registry, std::move(node), name, {opset}, input_sizes, output_sizes,
                  [build]() { return build(true); });
  }
}

} // namespace

void RegisterCpuConcatCases(std::vector<TestCase> &registry, TestMode mode) {
  if (mode == TestMode::BENCHMARK) {
    for (const ConcatShape &shape : {
             ConcatShape{"axis0", {{1024, 512}, {1024, 512}}, 0},
             ConcatShape{"last_axis", {{1024, 256}, {1024, 768}}, -1},
             ConcatShape{"middle_axis", {{16, 128, 64}, {16, 256, 64}}, 1},
             ConcatShape{"tail", {{257, 31}, {257, 34}}, 1},
             ConcatShape{"narrow", std::vector<Shape>(4, Shape{65536, 1}), 1},
             ConcatShape{"many_inputs", std::vector<Shape>(32, Shape{128, 16}), 1},
             ConcatShape{"uneven", {{512, 1}, {512, 63}, {512, 256}}, 1},
             ConcatShape{"single", {{1024, 1024}}, 0},
             ConcatShape{"empty_input", {{1024, 0}, {1024, 512}, {1024, 0}}, 1},
             ConcatShape{"vector", {{65535}, {65536}, {1}}, 0},
         }) {
      for (const DataType data_type : {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16,
                                       DataType::BFLOAT16, DataType::INT8, DataType::INT64}) {
        RegisterConcatCase(registry, shape, data_type, true);
      }
    }
    return;
  }
  for (const ConcatShape &shape : {
           ConcatShape{"axis0", {{1, 3}, {2, 3}}, 0},
           ConcatShape{"negative_axis", {{2, 1}, {2, 3}}, -1},
           ConcatShape{"middle_axis", {{2, 1, 3}, {2, 2, 3}}, 1},
           ConcatShape{"vector", {{1}, {2}, {0}}, 0},
           ConcatShape{"single", {{3, 4}}, 0},
           ConcatShape{"empty_input", {{2, 0}, {2, 3}, {2, 0}}, 1},
           ConcatShape{"all_empty_axis", {{2, 0}, {2, 0}}, 1},
           ConcatShape{"empty_outer", {{0, 2}, {0, 3}}, 1},
           ConcatShape{"empty_inner", {{2, 0}, {3, 0}}, 0},
           ConcatShape{"many_inputs", std::vector<Shape>(17, Shape{2, 1}), 1},
           ConcatShape{"high_rank", {{1, 2, 1, 3}, {1, 2, 2, 3}}, 2},
       }) {
    for (const DataType data_type :
         {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16, DataType::INT8,
          DataType::UINT8, DataType::INT16, DataType::INT32, DataType::INT64}) {
      RegisterConcatCase(registry, shape, data_type, false);
    }
  }
}

} // namespace onnx_light_cpu::backend_test
