// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/math/include_math_cases.h"

#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"
#include "onnx_light_cpu/kernels/math/integer_matmul_kernel.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/kernel_context.h"

#include <array>
#include <cstdint>
#include <string>
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

struct MatMulShape {
  const char *name;
  std::int64_t m;
  std::int64_t n;
  std::int64_t k;
};

NodeProto MakeMatMulIntegerNode() {
  NodeProto node;
  node.set_op_type("MatMulInteger");
  node.add_input("A");
  node.add_input("B");
  node.add_output("Y");
  return node;
}

void RegisterMatMulIntegerCase(std::vector<TestCase> &registry,
                               const onnx_light_cpu::MatMulIntegerKernel &kernel,
                               const OpsetId &opset, const MatMulShape &shape, DataType a_type,
                               DataType b_type, bool benchmark) {
  const std::string name = "test_cpu_matmulinteger_" + std::string(shape.name) + "_" +
                           DataTypeSuffix(a_type) + "x" + DataTypeSuffix(b_type) +
                           (benchmark ? "_benchmark" : "");
  const std::int64_t a_count = shape.m * shape.k;
  const std::int64_t b_count = shape.k * shape.n;
  const std::int64_t y_count = shape.m * shape.n;
  if (benchmark) {
    Expect(registry, MakeMatMulIntegerNode(), name, {opset}, {a_count, b_count}, {y_count},
           [opset, shape, a_type, b_type](bool generate_expected_outputs) -> IoData {
             Tensor a = MakeBenchmarkTensor(a_type, {shape.m, shape.k}, 433);
             Tensor b = MakeBenchmarkTensor(b_type, {shape.k, shape.n}, 434);
             if (!generate_expected_outputs) {
               return IoData{{std::move(a), std::move(b)}, {}, false};
             }
             const onnx_light_cpu::MatMulIntegerKernel kernel{KernelContext{opset}};
             Tensor y = kernel(a, b);
             return IoData{{std::move(a), std::move(b)}, {std::move(y)}};
           });
    return;
  }
  Expect(registry, MakeMatMulIntegerNode(), name, {opset}, {a_count, b_count}, {y_count},
         [kernel, shape, a_type, b_type]() -> IoData {
           Tensor a = MakeBenchmarkTensor(a_type, {shape.m, shape.k}, 433);
           Tensor b = MakeBenchmarkTensor(b_type, {shape.k, shape.n}, 434);
           return IoData{{std::move(a), std::move(b)}, {kernel(a, b)}};
         });
}

} // namespace

void RegisterCpuMatMulIntegerCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(10);
  constexpr std::array<DataType, 2> data_types = {DataType::INT8, DataType::UINT8};
  if (mode == TestMode::BENCHMARK) {
    constexpr std::array<MatMulShape, 5> shapes = {
        MatMulShape{"square_64", 64, 64, 64},     MatMulShape{"square_128", 128, 128, 128},
        MatMulShape{"square_512", 512, 512, 512}, MatMulShape{"skinny_m", 1, 4096, 4096},
        MatMulShape{"large_k", 32, 32, 8192},
    };
    for (const MatMulShape &shape : shapes) {
      for (DataType a_type : data_types) {
        for (DataType b_type : data_types) {
          RegisterMatMulIntegerCase(registry, kernel, opset, shape, a_type, b_type, true);
        }
      }
    }
    return;
  }
  const onnx_light_cpu::MatMulIntegerKernel kernel{KernelContext{opset}};
  for (DataType a_type : data_types) {
    for (DataType b_type : data_types) {
      RegisterMatMulIntegerCase(registry, kernel, opset, {"small", 2, 3, 4}, a_type, b_type, false);
    }
  }
}

} // namespace onnx_light_cpu::backend_test
