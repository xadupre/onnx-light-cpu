// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"
#include "onnx_light_cpu/backend_test/com_microsoft/include_com_microsoft_cases.h"

#include "onnx_light_cpu/kernels/com_microsoft/cdist_kernel.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/backend_test/expect.h"
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
using ONNX_LIGHT_NAMESPACE::AttributeProto;
using ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::DataType;
using rt_ns::DefaultOpset;
using rt_ns::KernelContext;
using rt_ns::OpsetId;
using rt_ns::Tensor;

// Builds a single ``com.microsoft`` ``CDist`` node reading ``A``/``B`` and
// writing ``C``, with the ``metric`` attribute set when non-empty (mirroring
// the default omission ONNX Runtime allows for ``"sqeuclidean"``).
NodeProto MakeCDistNode(const std::string &metric) {
  NodeProto node;
  node.set_op_type("CDist");
  node.set_domain(kMicrosoftDomain);
  node.add_input("A");
  node.add_input("B");
  node.add_output("C");
  if (!metric.empty()) {
    AttributeProto *attribute = node.add_attribute();
    attribute->set_name("metric");
    attribute->set_type(AttributeProto::STRING);
    attribute->set_s(metric);
  }
  return node;
}

} // namespace

// CDist — covers both element types (float32, float64) and both accepted
// metrics (the implicit "sqeuclidean" default and explicit "euclidean").
void RegisterCpuCDistCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId microsoft_opset(kMicrosoftDomain, 1);
  const CDistKernel kernel{KernelContext{microsoft_opset}};

  if (mode == TestMode::BENCHMARK) {
    struct Shape {
      std::int64_t m;
      std::int64_t k;
      std::int64_t n;
    };
    for (const Shape &shape : {Shape{64, 64, 64}, Shape{256, 128, 64}, Shape{512, 512, 128}}) {
      for (const DataType data_type : {DataType::FLOAT, DataType::DOUBLE}) {
        for (const std::string &metric : {std::string("sqeuclidean"), std::string("euclidean")}) {
          const std::string name = "test_cpu_cdist_m" + std::to_string(shape.m) + "_k" +
                                   std::to_string(shape.k) + "_n" + std::to_string(shape.n) + "_" +
                                   metric + "_" + DataTypeSuffix(data_type) + "_benchmark";
          Expect(registry, MakeCDistNode(metric), name, {DefaultOpset(13), microsoft_opset},
                 {shape.m * shape.n, shape.k * shape.n}, {shape.m * shape.k}, [=]() -> IoData {
                   Tensor a = MakeBenchmarkTensor(data_type, {shape.m, shape.n}, 987654321ULL);
                   Tensor b = MakeBenchmarkTensor(data_type, {shape.k, shape.n}, 246813579ULL);
                   Tensor c = kernel(a, b, metric);
                   return IoData{{std::move(a), std::move(b)}, {std::move(c)}};
                 });
        }
      }
    }
    return;
  }

  // Default metric ("sqeuclidean" applies when the attribute is omitted).
  Expect(registry, MakeCDistNode(""), "test_cpu_cdist_default_metric_float32",
         {DefaultOpset(13), microsoft_opset}, [=]() -> IoData {
           Tensor a = Tensor::FromFloat("", {3, 2}, {0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 2.0f});
           Tensor b = Tensor::FromFloat("", {2, 2}, {0.0f, 0.0f, 1.0f, 0.0f});
           return IoData{{a, b}, {kernel(a, b, "sqeuclidean")}};
         });

  Expect(registry, MakeCDistNode("sqeuclidean"), "test_cpu_cdist_sqeuclidean_float64",
         {DefaultOpset(13), microsoft_opset}, [=]() -> IoData {
           Tensor a = Tensor::FromDouble("", {3, 2}, {0.0, 0.0, 1.0, 1.0, 2.0, 2.0});
           Tensor b = Tensor::FromDouble("", {2, 2}, {0.0, 0.0, 1.0, 0.0});
           return IoData{{a, b}, {kernel(a, b, "sqeuclidean")}};
         });

  Expect(registry, MakeCDistNode("euclidean"), "test_cpu_cdist_euclidean_float32",
         {DefaultOpset(13), microsoft_opset}, [=]() -> IoData {
           Tensor a = Tensor::FromFloat("", {3, 2}, {0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 2.0f});
           Tensor b = Tensor::FromFloat("", {2, 2}, {0.0f, 0.0f, 1.0f, 0.0f});
           return IoData{{a, b}, {kernel(a, b, "euclidean")}};
         });

  Expect(registry, MakeCDistNode("euclidean"), "test_cpu_cdist_euclidean_float64",
         {DefaultOpset(13), microsoft_opset}, [=]() -> IoData {
           Tensor a = Tensor::FromDouble(
               "", {4, 3}, {1.0, 2.0, 3.0, -1.0, 0.5, 2.5, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0});
           Tensor b = Tensor::FromDouble("", {2, 3}, {0.0, 0.0, 0.0, 1.0, 1.0, 1.0});
           return IoData{{a, b}, {kernel(a, b, "euclidean")}};
         });
}

} // namespace onnx_light_cpu::backend_test
