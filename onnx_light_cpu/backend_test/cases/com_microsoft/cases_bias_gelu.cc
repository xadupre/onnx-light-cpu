// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/com_microsoft/include_com_microsoft_cases.h"
#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"

#include "onnx_light_cpu/kernels/com_microsoft/bias_gelu_kernel.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace onnx_light_cpu::backend_test {

namespace {

namespace bt_ns = ONNX_LIGHT_NAMESPACE::core::backend_test;
namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

using bt_ns::Expect;
using bt_ns::IoData;
using ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::DataType;
using rt_ns::DefaultOpset;
using rt_ns::KernelContext;
using rt_ns::MakeBfloat16Tensor;
using rt_ns::MakeFloat16Tensor;
using rt_ns::OpsetId;
using rt_ns::Tensor;

// Builds a single ``com.microsoft`` ``BiasGelu`` node reading ``A``/``B`` and
// writing ``C``, per the ONNX Runtime contrib op contract (``B`` broadcasts
// over ``A``'s last dimension).
NodeProto MakeBiasGeluNode() {
  NodeProto node;
  node.set_op_type("BiasGelu");
  node.set_domain(kMicrosoftDomain);
  node.add_input("A");
  node.add_input("B");
  node.add_output("C");
  return node;
}

} // namespace

// BiasGelu — covers every element type ``BiasGeluKernel`` implements:
// float32, float64, float16, bfloat16.
void RegisterCpuBiasGeluCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId microsoft_opset(kMicrosoftDomain, 1);
  if (mode == TestMode::BENCHMARK) {
    struct Shape {
      std::int64_t outer;
      std::int64_t inner;
    };
    for (const Shape &shape : {Shape{4096, 256}, Shape{1024, 1024}, Shape{256, 4096}}) {
      for (const DataType data_type :
           {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16}) {
        const std::int64_t size = shape.outer * shape.inner;
        const std::string name = "test_cpu_biasgelu_o" + std::to_string(shape.outer) + "_i" +
                                 std::to_string(shape.inner) + "_" + DataTypeSuffix(data_type) +
                                 "_benchmark";
        Expect(registry, MakeBiasGeluNode(), name, {DefaultOpset(13), microsoft_opset},
               {size, shape.inner}, {size},
               [=](bool generate_expected_outputs) -> IoData {
                 Tensor a =
                     MakeBenchmarkTensor(data_type, {shape.outer, shape.inner}, 987654321ULL);
                 Tensor b = MakeBenchmarkTensor(data_type, {shape.inner}, 135792468ULL);
                 if (!generate_expected_outputs) {
                   return IoData{{std::move(a), std::move(b)}, {}, {}, false};
                 }
                 const BiasGeluKernel kernel{KernelContext{microsoft_opset}};
                 Tensor c = kernel(a, b);
                 return IoData{{std::move(a), std::move(b)}, {std::move(c)}};
               },
               "backend-test", "",
               {bt_ns::TensorTypeSpec(static_cast<std::int32_t>(data_type),
                                      {shape.outer, shape.inner})});
      }
    }
    return;
  }

  const BiasGeluKernel kernel{KernelContext{microsoft_opset}};
  const OpsetId opset = microsoft_opset;

  Expect(registry, MakeBiasGeluNode(), "test_cpu_biasgelu_float32", {DefaultOpset(13), opset},
         [=]() -> IoData {
           Tensor a = Tensor::FromFloat("", {2, 3}, {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 3.0f});
           Tensor b = Tensor::FromFloat("", {3}, {0.1f, -0.2f, 0.3f});
           return IoData{{a, b}, {kernel(a, b)}};
         });

  Expect(registry, MakeBiasGeluNode(), "test_cpu_biasgelu_float64", {DefaultOpset(13), opset},
         [=]() -> IoData {
           Tensor a = Tensor::FromDouble("", {2, 4}, {-3.0, -1.5, 0.25, 1.0, 2.0, -0.5, 0.75, 4.0});
           Tensor b = Tensor::FromDouble("", {4}, {0.5, -0.5, 0.25, -0.25});
           return IoData{{a, b}, {kernel(a, b)}};
         });

  Expect(registry, MakeBiasGeluNode(), "test_cpu_biasgelu_float16", {DefaultOpset(13), opset},
         [=]() -> IoData {
           Tensor a = MakeFloat16Tensor("", {2, 3}, {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 3.0f});
           Tensor b = MakeFloat16Tensor("", {3}, {0.1f, -0.2f, 0.3f});
           return IoData{{a, b}, {kernel(a, b)}};
         });

  Expect(registry, MakeBiasGeluNode(), "test_cpu_biasgelu_bfloat16", {DefaultOpset(13), opset},
         [=]() -> IoData {
           Tensor a = MakeBfloat16Tensor("", {2, 3}, {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 3.0f});
           Tensor b = MakeBfloat16Tensor("", {3}, {0.1f, -0.2f, 0.3f});
           return IoData{{a, b}, {kernel(a, b)}};
         });
}

} // namespace onnx_light_cpu::backend_test
