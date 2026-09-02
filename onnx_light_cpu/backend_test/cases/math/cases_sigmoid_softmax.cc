// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"
#include "onnx_light_cpu/backend_test/cases/math/include_math_cases.h"

#include "onnx_light_cpu/kernels/math/sigmoid_softmax_kernel.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace onnx_light_cpu::backend_test {

namespace {

namespace bt_ns = ONNX_LIGHT_NAMESPACE::core::backend_test;
namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

using bt_ns::Expect;
using bt_ns::IoData;
using bt_ns::TestCase;
using bt_ns::TestMode;
using ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::DataType;
using rt_ns::DefaultOpset;
using rt_ns::KernelContext;
using rt_ns::OpsetId;
using rt_ns::Tensor;

NodeProto MakeNode(const char *op_type, std::int64_t axis = 0, bool has_axis = false) {
  NodeProto node;
  node.set_op_type(op_type);
  node.add_input("x");
  node.add_output("y");
  if (has_axis) {
    auto *attribute = node.add_attribute();
    attribute->set_name("axis");
    attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::INT);
    attribute->set_i(axis);
  }
  return node;
}

Tensor MakeTensor(DataType type, const rt_ns::Shape &shape, const std::vector<float> &values) {
  switch (type) {
  case DataType::FLOAT:
    return Tensor::FromFloat("", shape, values);
  case DataType::DOUBLE:
    return Tensor::FromDouble("", shape, std::vector<double>(values.begin(), values.end()));
  case DataType::FLOAT16:
    return rt_ns::MakeFloat16Tensor("", shape, values);
  case DataType::BFLOAT16:
    return rt_ns::MakeBfloat16Tensor("", shape, values);
  default:
    throw std::invalid_argument("Unsupported sigmoid/softmax test data type.");
  }
}

void SetTolerance(std::vector<TestCase> &registry, DataType type) {
  registry.back().rtol = type == DataType::FLOAT16 || type == DataType::BFLOAT16 ? 2.0e-2 : 2.0e-5;
  registry.back().atol = type == DataType::FLOAT16 || type == DataType::BFLOAT16 ? 2.0e-2 : 2.0e-6;
}

void RegisterSoftmaxBenchmark(std::vector<TestCase> &registry, const OpsetId &opset,
                              DataType data_type, std::int64_t rows, std::int64_t columns) {
  NodeProto node = MakeNode("Softmax", -1, true);
  const std::int64_t count = rows * columns;
  const std::string name = "test_cpu_softmax_" + std::to_string(rows) + "x" +
                           std::to_string(columns) + "_" + DataTypeSuffix(data_type) + "_benchmark";
  Expect(registry, std::move(node), name, {opset}, {count}, {count},
         [=](bool generate_expected_outputs) -> IoData {
           Tensor x = MakeBenchmarkTensor(data_type, {rows, columns}, 987654321ULL);
           if (!generate_expected_outputs) {
             return IoData{{std::move(x)}, {}, {}, false};
           }
           const SoftmaxKernel kernel{KernelContext{opset}};
           Tensor y = kernel(x, -1);
           return IoData{{std::move(x)}, {std::move(y)}};
         },
         "backend-test", bt_ns::TestCaseTag::NONE,
         {bt_ns::TensorTypeSpec(static_cast<std::int32_t>(data_type), {rows, columns})});
}

} // namespace

void RegisterCpuSigmoidCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);
  if (mode == TestMode::BENCHMARK) {
    for (const std::int64_t size : {1024, 32768, 65535, 65536, 131072, 1048576, 4194304}) {
      for (const DataType data_type :
           {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16}) {
        RegisterUnaryBenchmark(
            registry, "Sigmoid",
            [opset] { return onnx_light_cpu::SigmoidKernel{KernelContext{opset}}; }, opset,
            data_type, size);
      }
    }
    return;
  }

  const rt_ns::Shape shape = {2, 3};
  const std::vector<float> input = {-4.0F, -1.0F, 0.0F, 1.0F, 2.0F, 4.0F};
  const std::vector<float> expected = {0.01798621F, 0.26894143F, 0.5F,
                                       0.73105860F, 0.88079708F, 0.98201376F};
  for (const DataType data_type :
       {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16}) {
    const std::string name = "test_cpu_sigmoid_" + std::string(DataTypeSuffix(data_type));
    Expect(registry, MakeNode("Sigmoid"), name, {opset}, [=]() -> IoData {
      return IoData{{MakeTensor(data_type, shape, input)},
                    {MakeTensor(data_type, shape, expected)}};
    });
    SetTolerance(registry, data_type);
  }
}

void RegisterCpuSoftmaxCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);
  if (mode == TestMode::BENCHMARK) {
    constexpr std::array<std::pair<std::int64_t, std::int64_t>, 3> shapes = {
        std::pair<std::int64_t, std::int64_t>{1, 1024}, {32, 1024}, {1024, 1024}};
    for (const auto [rows, columns] : shapes) {
      for (const DataType data_type :
           {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16}) {
        RegisterSoftmaxBenchmark(registry, opset, data_type, rows, columns);
      }
    }
    return;
  }

  const rt_ns::Shape shape = {2, 3};
  const std::vector<float> input = {1.0F, 2.0F, 3.0F, -1.0F, 0.0F, 1.0F};
  const std::vector<float> expected = {0.09003057F, 0.24472848F, 0.66524094F,
                                       0.09003057F, 0.24472848F, 0.66524094F};
  for (const DataType data_type :
       {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16}) {
    const std::string name = "test_cpu_softmax_" + std::string(DataTypeSuffix(data_type));
    Expect(registry, MakeNode("Softmax"), name, {opset}, [=]() -> IoData {
      return IoData{{MakeTensor(data_type, shape, input)},
                    {MakeTensor(data_type, shape, expected)}};
    });
    SetTolerance(registry, data_type);
  }

  const std::vector<float> axis_zero_expected = {0.88079708F, 0.88079708F, 0.88079708F,
                                                 0.11920292F, 0.11920292F, 0.11920292F};
  Expect(registry, MakeNode("Softmax", 0, true), "test_cpu_softmax_axis_zero", {opset},
         [=]() -> IoData {
           return IoData{{Tensor::FromFloat("", shape, input)},
                         {Tensor::FromFloat("", shape, axis_zero_expected)}};
         });
  SetTolerance(registry, DataType::FLOAT);

  const OpsetId legacy_opset = DefaultOpset(12);
  const rt_ns::Shape legacy_shape = {2, 2, 2};
  const std::vector<float> legacy_input = {1.0F, 2.0F, 3.0F, 4.0F, 4.0F, 3.0F, 2.0F, 1.0F};
  const std::vector<float> legacy_expected = {0.03205860F, 0.08714432F, 0.23688282F, 0.64391428F,
                                              0.64391428F, 0.23688282F, 0.08714432F, 0.03205860F};
  Expect(registry, MakeNode("Softmax", 1, true), "test_cpu_softmax_opset12_axis", {legacy_opset},
         [=]() -> IoData {
           return IoData{{Tensor::FromFloat("", legacy_shape, legacy_input)},
                         {Tensor::FromFloat("", legacy_shape, legacy_expected)}};
         });
  SetTolerance(registry, DataType::FLOAT);
}

} // namespace onnx_light_cpu::backend_test
