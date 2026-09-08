// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/tensor/include_tensor_cases.h"

#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include <cstdint>
#include <limits>
#include <optional>
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

struct SliceShape {
  const char *name;
  Shape data;
  Ints starts;
  Ints ends;
  std::optional<Ints> axes;
  std::optional<Ints> steps;
  Shape output;
  bool legacy = false;
};

Tensor MakeParameter(const char *name, const Ints &values, DataType type) {
  const Shape shape{static_cast<std::int64_t>(values.size())};
  if (type == DataType::INT32) {
    std::vector<std::int32_t> converted;
    converted.reserve(values.size());
    for (std::int64_t value : values) {
      converted.push_back(static_cast<std::int32_t>(value));
    }
    return Tensor::FromInt32(name, shape, converted);
  }
  return Tensor::FromInt64(name, shape, values);
}

void AddInts(ONNX_LIGHT_NAMESPACE::NodeProto &node, const char *name, const Ints &values) {
  auto *attribute = node.add_attribute();
  attribute->set_name(name);
  attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::INTS);
  for (std::int64_t value : values) {
    attribute->add_ints(value);
  }
}

void RegisterSliceCase(std::vector<TestCase> &registry, const SliceShape &shape, DataType data_type,
                       DataType parameter_type, bool benchmark) {
  const auto opset = rt_ns::DefaultOpset(shape.legacy ? 1 : 13);
  auto node = MakeNode("Slice", {"data"}, {"output"});
  std::vector<std::int64_t> input_sizes{shape.data.product()};
  if (shape.legacy) {
    AddInts(node, "starts", shape.starts);
    AddInts(node, "ends", shape.ends);
    if (shape.axes) {
      AddInts(node, "axes", *shape.axes);
    }
  } else {
    node.add_input("starts");
    node.add_input("ends");
    input_sizes.push_back(static_cast<std::int64_t>(shape.starts.size()));
    input_sizes.push_back(static_cast<std::int64_t>(shape.ends.size()));
    if (shape.axes || shape.steps) {
      node.add_input(shape.axes ? "axes" : "");
    }
    if (shape.axes) {
      input_sizes.push_back(static_cast<std::int64_t>(shape.axes->size()));
    }
    if (shape.steps) {
      node.add_input("steps");
      input_sizes.push_back(static_cast<std::int64_t>(shape.steps->size()));
    }
  }
  const std::string parameters =
      shape.legacy ? "_attributes_"
                   : (parameter_type == DataType::INT32 ? "_params32_" : "_params64_");
  const std::string name = "test_cpu_slice_" + std::string(shape.name) + parameters +
                           DataTypeSuffix(data_type) + (benchmark ? "_benchmark" : "");
  const auto build = [shape, data_type, parameter_type, opset](bool expected) -> bt_ns::IoData {
    Tensor data = MakeBenchmarkTensor(data_type, shape.data, 823);
    Tensor starts = MakeParameter("starts", shape.starts, parameter_type);
    Tensor ends = MakeParameter("ends", shape.ends, parameter_type);
    std::optional<Tensor> axes;
    std::optional<Tensor> steps;
    if (shape.axes) {
      axes = MakeParameter("axes", *shape.axes, parameter_type);
    }
    if (shape.steps) {
      steps = MakeParameter("steps", *shape.steps, parameter_type);
    }
    std::vector<Tensor> inputs{data};
    if (!shape.legacy) {
      inputs.push_back(starts);
      inputs.push_back(ends);
      if (axes) {
        inputs.push_back(*axes);
      }
      if (steps) {
        inputs.push_back(*steps);
      }
    }
    if (!expected) {
      return bt_ns::IoData{std::move(inputs), {}, {}, false};
    }
    const ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::Slice reference(rt_ns::KernelContext{opset});
    Tensor output =
        reference(data, starts, ends, axes ? &*axes : nullptr, steps ? &*steps : nullptr);
    if (output.shape != shape.output) {
      throw std::logic_error("Slice backend case has incorrect output shape metadata.");
    }
    return bt_ns::IoData{std::move(inputs), {std::move(output)}};
  };
  const std::vector<std::int64_t> output_sizes{shape.output.product()};
  if (benchmark) {
    bt_ns::Expect(registry, std::move(node), name, {opset}, input_sizes, output_sizes, build,
                  "backend-test", bt_ns::TestCaseTag::NONE,
                  {bt_ns::TensorTypeSpec(static_cast<std::int32_t>(data_type), shape.output)});
  } else {
    bt_ns::Expect(registry, std::move(node), name, {opset}, input_sizes, output_sizes,
                  [build]() { return build(true); });
  }
}

} // namespace

void RegisterCpuSliceCases(std::vector<TestCase> &registry, TestMode mode) {
  constexpr std::int64_t minimum = std::numeric_limits<std::int32_t>::min();
  if (mode == TestMode::BENCHMARK) {
    for (const SliceShape &shape : {
             SliceShape{
                 "contiguous", {2048, 512}, {512}, {1536}, Ints{0}, std::nullopt, {1024, 512}},
             SliceShape{
                 "inner_crop", {1024, 1024}, {17}, {1009}, Ints{1}, std::nullopt, {1024, 992}},
             SliceShape{"inner_stride", {1024, 1024}, {0}, {1024}, Ints{1}, Ints{2}, {1024, 512}},
             SliceShape{
                 "reverse_inner", {1024, 1024}, {1023}, {minimum}, Ints{1}, Ints{-1}, {1024, 1024}},
             SliceShape{"outer_stride", {4096, 128}, {0}, {4096}, Ints{0}, Ints{3}, {1366, 128}},
             SliceShape{
                 "reverse_outer", {4096, 128}, {4095}, {minimum}, Ints{0}, Ints{-1}, {4096, 128}},
             SliceShape{"multi_axis",
                        {64, 128, 128},
                        {1, 3, 1},
                        {63, 127, 128},
                        Ints{0, 1, 2},
                        Ints{2, 2, 3},
                        {31, 62, 43}},
             SliceShape{"tail", {257, 65}, {1}, {64}, Ints{1}, Ints{2}, {257, 32}},
             SliceShape{"small", {8, 17}, {1}, {16}, Ints{1}, std::nullopt, {8, 15}},
             SliceShape{
                 "reverse_vector", {1048576}, {1048575}, {minimum}, Ints{0}, Ints{-1}, {1048576}},
         }) {
      for (const DataType data_type : {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16,
                                       DataType::BFLOAT16, DataType::INT8, DataType::INT64}) {
        for (const DataType parameter_type : {DataType::INT32, DataType::INT64}) {
          RegisterSliceCase(registry, shape, data_type, parameter_type, true);
        }
      }
    }
    return;
  }
  for (const SliceShape &shape : {
           SliceShape{"default_axes", {4, 5, 6}, {1}, {3}, std::nullopt, std::nullopt, {2, 5, 6}},
           SliceShape{"inner_crop", {3, 17}, {1}, {16}, Ints{1}, std::nullopt, {3, 15}},
           SliceShape{"negative_axis", {3, 7}, {-6}, {-1}, Ints{-1}, Ints{2}, {3, 3}},
           SliceShape{"reverse", {2, 7}, {6}, {minimum}, Ints{-1}, Ints{-1}, {2, 7}},
           SliceShape{"multi_axis", {4, 5, 6}, {1, 0}, {4, 6}, Ints{0, 2}, Ints{2, 2}, {2, 5, 3}},
           SliceShape{"unsorted_axes",
                      {4, 5, 6},
                      {5, 0},
                      {minimum, 4},
                      Ints{2, 0},
                      Ints{-2, 2},
                      {2, 5, 3}},
           SliceShape{"clipped_bounds", {2, 5}, {-100}, {100}, Ints{1}, Ints{1}, {2, 5}},
           SliceShape{"empty_range", {2, 5}, {4}, {1}, Ints{1}, Ints{1}, {2, 0}},
           SliceShape{"empty_data", {2, 0, 5}, {0}, {2}, Ints{0}, std::nullopt, {2, 0, 5}},
           SliceShape{
               "omitted_axes_with_steps", {6, 4}, {5}, {minimum}, std::nullopt, Ints{-2}, {3, 4}},
           SliceShape{"identity", {3, 4}, {}, {}, std::nullopt, std::nullopt, {3, 4}},
           SliceShape{"reverse_empty", {2, 0}, {-1}, {minimum}, Ints{1}, Ints{-1}, {2, 0}},
           SliceShape{
               "legacy_default_axes", {4, 5}, {1}, {3}, std::nullopt, std::nullopt, {2, 5}, true},
           SliceShape{"legacy_axes", {4, 5}, {1}, {4}, Ints{1}, std::nullopt, {4, 3}, true},
       }) {
    for (const DataType data_type :
         {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16, DataType::INT8,
          DataType::UINT8, DataType::INT16, DataType::INT32, DataType::INT64}) {
      if (shape.legacy && data_type == DataType::BFLOAT16) {
        continue;
      }
      for (const DataType parameter_type : {DataType::INT32, DataType::INT64}) {
        if (shape.legacy && parameter_type == DataType::INT32) {
          continue;
        }
        RegisterSliceCase(registry, shape, data_type, parameter_type, false);
      }
    }
  }
}

} // namespace onnx_light_cpu::backend_test
