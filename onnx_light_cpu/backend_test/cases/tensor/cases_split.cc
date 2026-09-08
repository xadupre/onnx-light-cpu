// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/tensor/include_tensor_cases.h"

#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include <cstdint>
#include <span>
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
using Ints = std::vector<std::int64_t>;

enum class SplitForm { kTensor, kAttribute, kImplicit, kNumOutputs };

struct SplitShape {
  const char *name;
  Shape data;
  std::int64_t axis;
  Ints sizes;
  SplitForm form = SplitForm::kTensor;
  std::int64_t opset = 18;
};

void AddInt(ONNX_LIGHT_NAMESPACE::NodeProto &node, const char *name, std::int64_t value) {
  auto *attribute = node.add_attribute();
  attribute->set_name(name);
  attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::INT);
  attribute->set_i(value);
}

void RegisterSplitCase(std::vector<TestCase> &registry, const SplitShape &shape, DataType data_type,
                       bool benchmark) {
  const auto opset = rt_ns::DefaultOpset(shape.opset);
  auto node = MakeNode("Split", {"data"}, {});
  AddInt(node, "axis", shape.axis);
  const auto rank = static_cast<std::int64_t>(shape.data.size());
  const auto axis = static_cast<std::size_t>(shape.axis < 0 ? shape.axis + rank : shape.axis);
  std::vector<std::int64_t> input_sizes{shape.data.product()};
  std::vector<std::int64_t> output_sizes;
  std::vector<Shape> output_shapes;
  std::vector<bt_ns::TypeSpec> output_types;
  for (std::size_t i = 0; i < shape.sizes.size(); ++i) {
    node.add_output("output_" + std::to_string(i));
    Shape output = shape.data;
    output[axis] = shape.sizes[i];
    output_sizes.push_back(output.product());
    output_types.push_back(bt_ns::TensorTypeSpec(static_cast<std::int32_t>(data_type), output));
    output_shapes.push_back(std::move(output));
  }
  if (shape.form == SplitForm::kTensor) {
    node.add_input("split");
    input_sizes.push_back(static_cast<std::int64_t>(shape.sizes.size()));
  } else if (shape.form == SplitForm::kAttribute) {
    auto *attribute = node.add_attribute();
    attribute->set_name("split");
    attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::INTS);
    for (std::int64_t size : shape.sizes) {
      attribute->add_ints(size);
    }
  } else if (shape.form == SplitForm::kNumOutputs) {
    AddInt(node, "num_outputs", static_cast<std::int64_t>(shape.sizes.size()));
  }
  const std::string name = "test_cpu_split_" + std::string(shape.name) + "_" +
                           DataTypeSuffix(data_type) + (benchmark ? "_benchmark" : "");
  const auto build = [shape, data_type, opset, output_shapes](bool expected) -> bt_ns::IoData {
    Tensor data = MakeBenchmarkTensor(data_type, shape.data, 941);
    std::vector<Tensor> inputs{data};
    if (shape.form == SplitForm::kTensor) {
      inputs.push_back(
          Tensor::FromInt64("split", {static_cast<std::int64_t>(shape.sizes.size())}, shape.sizes));
    }
    if (!expected) {
      return bt_ns::IoData{std::move(inputs), {}, {}, false};
    }
    const ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::Split reference(rt_ns::KernelContext{opset});
    const bool implicit =
        shape.form == SplitForm::kImplicit || shape.form == SplitForm::kNumOutputs;
    auto outputs =
        reference(data, shape.axis, implicit ? std::span<const std::int64_t>{} : shape.sizes,
                  implicit ? static_cast<std::int64_t>(shape.sizes.size()) : 0);
    if (outputs.size() != output_shapes.size()) {
      throw std::logic_error("Split backend case has incorrect output count metadata.");
    }
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      if (outputs[i].shape != output_shapes[i]) {
        throw std::logic_error("Split backend case has incorrect output shape metadata.");
      }
    }
    return bt_ns::IoData{std::move(inputs), std::move(outputs)};
  };
  if (benchmark) {
    bt_ns::Expect(registry, std::move(node), name, {opset}, input_sizes, output_sizes, build,
                  "backend-test", bt_ns::TestCaseTag::NONE, output_types);
  } else {
    bt_ns::Expect(registry, std::move(node), name, {opset}, input_sizes, output_sizes,
                  [build]() { return build(true); });
  }
}

} // namespace

void RegisterCpuSplitCases(std::vector<TestCase> &registry, TestMode mode) {
  if (mode == TestMode::BENCHMARK) {
    for (const SplitShape &shape : {
             SplitShape{"qkv_decode", {1, 4096}, -1, {2048, 1024, 1024}},
             SplitShape{"qkv_small", {8, 4096}, -1, {2048, 1024, 1024}},
             SplitShape{"qkv_prefill", {256, 4096}, -1, {2048, 1024, 1024}},
             SplitShape{"qkv_batched", {2, 16, 4096}, -1, {2048, 1024, 1024}},
             SplitShape{"first_axis", {2048, 512}, 0, {512, 1024, 512}},
             SplitShape{"middle_axis", {64, 128, 128}, 1, {32, 64, 32}},
             SplitShape{"last_axis", {1024, 1024}, -1, {128, 768, 128}},
             SplitShape{"vector", {1048576}, 0, {262144, 524288, 262144}},
             SplitShape{"narrow", {65536, 4}, -1, {1, 1, 1, 1}},
             SplitShape{"many_outputs", {1024, 1024}, -1, Ints(32, 32)},
             SplitShape{"tail", {257, 65}, -1, {17, 31, 17}},
             SplitShape{"uneven", {1024, 1025}, -1, {342, 342, 341}, SplitForm::kNumOutputs},
         }) {
      for (const DataType data_type : {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16,
                                       DataType::BFLOAT16, DataType::INT8, DataType::INT64}) {
        RegisterSplitCase(registry, shape, data_type, true);
      }
    }
    return;
  }
  for (const SplitShape &shape : {
           SplitShape{"qkv_decode", {1, 4096}, -1, {2048, 1024, 1024}},
           SplitShape{"qkv_prefill", {7, 4096}, -1, {2048, 1024, 1024}},
           SplitShape{"qkv_batched", {2, 3, 4096}, -1, {2048, 1024, 1024}},
           SplitShape{"first_axis", {7, 3, 5}, 0, {2, 4, 1}},
           SplitShape{"middle_axis", {3, 7, 5}, 1, {2, 4, 1}},
           SplitShape{"negative_axis", {3, 7, 5}, -2, {2, 4, 1}},
           SplitShape{"last_axis", {3, 17}, -1, {1, 7, 9}},
           SplitShape{"zero_pieces", {3, 7}, -1, {0, 2, 0, 5, 0}},
           SplitShape{"empty_outer", {0, 7, 5}, 1, {2, 4, 1}},
           SplitShape{"empty_inner", {3, 7, 0}, 1, {2, 4, 1}},
           SplitShape{"empty_axis", {3, 0, 5}, 1, {0, 0}},
           SplitShape{"identity", {3, 17}, 1, {17}},
           SplitShape{"vector", {17}, 0, {1, 7, 9}},
           SplitShape{"uneven", {3, 7}, -1, {3, 3, 1}, SplitForm::kNumOutputs},
           SplitShape{"num_outputs_equal", {3, 6}, -1, {2, 2, 2}, SplitForm::kNumOutputs},
           SplitShape{"implicit_v13", {3, 6}, -1, {2, 2, 2}, SplitForm::kImplicit, 13},
           SplitShape{"tensor_v13", {3, 7}, -1, {2, 4, 1}, SplitForm::kTensor, 13},
           SplitShape{"attributes_v11", {3, 7}, -1, {2, 4, 1}, SplitForm::kAttribute, 11},
           SplitShape{"attributes_v2", {7, 3}, 0, {2, 4, 1}, SplitForm::kAttribute, 2},
           SplitShape{"implicit_v11", {3, 6}, -1, {2, 2, 2}, SplitForm::kImplicit, 11},
       }) {
    for (const DataType data_type :
         {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16, DataType::INT8,
          DataType::UINT8, DataType::INT16, DataType::INT32, DataType::INT64}) {
      if (shape.opset < 13 && data_type == DataType::BFLOAT16) {
        continue;
      }
      RegisterSplitCase(registry, shape, data_type, false);
    }
  }
}

} // namespace onnx_light_cpu::backend_test
