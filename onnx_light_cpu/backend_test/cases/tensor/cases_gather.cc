// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/tensor/include_tensor_cases.h"

#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

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

struct GatherShape {
  const char *name;
  Shape data;
  Shape indices;
  std::int64_t axis;
};

Tensor MakeIndices(const Shape &shape, std::int64_t axis_size, DataType type) {
  std::vector<std::int64_t> values(static_cast<std::size_t>(shape.product()));
  for (std::size_t i = 0; i < values.size(); ++i) {
    const std::int64_t positive =
        i % 5 == 0 ? 0 : (static_cast<std::int64_t>(i) * 53 + 7) % axis_size;
    values[i] = i % 2 == 0 ? positive : positive - axis_size;
  }
  if (type == DataType::INT32) {
    std::vector<std::int32_t> converted;
    converted.reserve(values.size());
    for (std::int64_t value : values) {
      converted.push_back(static_cast<std::int32_t>(value));
    }
    return Tensor::FromInt32("indices", shape, converted);
  }
  return Tensor::FromInt64("indices", shape, values);
}

void RegisterGatherCase(std::vector<TestCase> &registry, const GatherShape &shape,
                        DataType data_type, DataType index_type, bool benchmark) {
  const auto opset = rt_ns::DefaultOpset(13);
  auto node = MakeNode("Gather", {"data", "indices"}, {"output"});
  auto *attribute = node.add_attribute();
  attribute->set_name("axis");
  attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::INT);
  attribute->set_i(shape.axis);
  const std::int64_t axis =
      shape.axis < 0 ? shape.axis + static_cast<std::int64_t>(shape.data.size()) : shape.axis;
  Shape output_shape;
  output_shape.insert(output_shape.end(), shape.data.begin(), shape.data.begin() + axis);
  output_shape.insert(output_shape.end(), shape.indices.begin(), shape.indices.end());
  output_shape.insert(output_shape.end(), shape.data.begin() + axis + 1, shape.data.end());
  const std::string name = "test_cpu_gather_" + std::string(shape.name) +
                           (index_type == DataType::INT32 ? "_indices32_" : "_indices64_") +
                           DataTypeSuffix(data_type) + (benchmark ? "_benchmark" : "");
  const auto generate = [shape, axis, data_type, index_type,
                         opset](bool expected) -> bt_ns::IoData {
    Tensor data = MakeBenchmarkTensor(data_type, shape.data, 719);
    Tensor indices = MakeIndices(shape.indices, shape.data[axis], index_type);
    if (!expected) {
      return bt_ns::IoData{{std::move(data), std::move(indices)}, {}, {}, false};
    }
    const ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::Gather reference(rt_ns::KernelContext{opset});
    Tensor output = reference(data, indices, shape.axis);
    return bt_ns::IoData{{std::move(data), std::move(indices)}, {std::move(output)}};
  };
  const std::vector<std::int64_t> input_sizes = {shape.data.product(), shape.indices.product()};
  const std::vector<std::int64_t> output_sizes = {output_shape.product()};
  if (benchmark) {
    bt_ns::Expect(registry, std::move(node), name, {opset}, input_sizes, output_sizes, generate,
                  "backend-test", bt_ns::TestCaseTag::NONE,
                  {bt_ns::TensorTypeSpec(static_cast<std::int32_t>(data_type), output_shape)});
  } else {
    bt_ns::Expect(registry, std::move(node), name, {opset}, input_sizes, output_sizes,
                  [generate]() { return generate(true); });
  }
}

} // namespace

void RegisterCpuGatherCases(std::vector<TestCase> &registry, TestMode mode) {
  if (mode == TestMode::BENCHMARK) {
    for (const GatherShape &shape : {
             GatherShape{"embedding_single", {4096, 128}, {}, 0},
             GatherShape{"embedding_small", {1024, 64}, {16}, 0},
             GatherShape{"embedding_tail", {1024, 65}, {31}, 0},
             GatherShape{"embedding_large", {8192, 256}, {2048}, 0},
             GatherShape{"embedding_multidim", {4096, 128}, {8, 32}, 0},
             GatherShape{"middle_axis", {32, 256, 64}, {127}, 1},
             GatherShape{"last_axis", {1024, 1024}, {255}, -1},
             GatherShape{"vector_random", {1048576}, {65535}, 0},
             GatherShape{"large_slice", {4, 1048576}, {2}, 0},
         }) {
      for (const DataType data_type : {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16,
                                       DataType::BFLOAT16, DataType::INT8, DataType::INT64}) {
        for (const DataType index_type : {DataType::INT32, DataType::INT64}) {
          RegisterGatherCase(registry, shape, data_type, index_type, true);
        }
      }
    }
    return;
  }

  for (const GatherShape &shape : {
           GatherShape{"vector", {7}, {3}, 0},
           GatherShape{"scalar_index", {3, 4}, {}, 0},
           GatherShape{"multidim_indices", {3, 4}, {2, 2}, 0},
           GatherShape{"middle_axis", {2, 3, 4}, {2, 1}, 1},
           GatherShape{"negative_axis", {2, 3, 4}, {2}, -1},
           GatherShape{"empty_indices", {2, 3, 4}, {0}, 1},
           GatherShape{"empty_axis", {2, 0, 4}, {0}, 1},
           GatherShape{"empty_outer", {0, 3, 4}, {2}, 1},
           GatherShape{"empty_inner", {2, 3, 0}, {2}, 1},
       }) {
    for (const DataType data_type :
         {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16, DataType::INT8,
          DataType::UINT8, DataType::INT16, DataType::INT32, DataType::INT64}) {
      for (const DataType index_type : {DataType::INT32, DataType::INT64}) {
        RegisterGatherCase(registry, shape, data_type, index_type, false);
      }
    }
  }
}

} // namespace onnx_light_cpu::backend_test
