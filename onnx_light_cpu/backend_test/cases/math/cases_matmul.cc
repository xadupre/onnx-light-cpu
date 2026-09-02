// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/math/include_math_cases.h"

#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"
#include "onnx_light_cpu/kernels/math/matmul_kernel.h"

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
  bool rank3 = false;
};

void RegisterMatMulCase(std::vector<TestCase> &registry, const OpsetId &opset,
                        const MatMulShape &shape, DataType data_type, bool benchmark) {
  const std::string name = "test_cpu_matmul_" + std::string(shape.name) + "_" +
                           DataTypeSuffix(data_type) + (benchmark ? "_benchmark" : "");
  const std::int64_t a_count = shape.m * shape.k;
  const std::int64_t b_count = shape.k * shape.n;
  const std::int64_t y_count = shape.m * shape.n;
  if (benchmark) {
    Expect(registry, MakeNode("MatMul", {"A", "B"}, {"Y"}), name, {opset}, {a_count, b_count},
           {y_count},
           [opset, shape, data_type](bool generate_expected_outputs) -> IoData {
             const rt_ns::Shape a_shape =
                 shape.rank3 ? rt_ns::Shape{1, shape.m, shape.k} : rt_ns::Shape{shape.m, shape.k};
             Tensor a = MakeBenchmarkTensor(data_type, a_shape, 433);
             Tensor b = MakeBenchmarkTensor(data_type, {shape.k, shape.n}, 434);
             if (!generate_expected_outputs) {
               return IoData{{std::move(a), std::move(b)}, {}, {}, false};
             }
             const onnx_light_cpu::MatMulKernel benchmark_kernel{KernelContext{opset}};
             Tensor y = benchmark_kernel(a, b);
             return IoData{{std::move(a), std::move(b)}, {std::move(y)}};
           },
           "backend-test", bt_ns::TestCaseTag::NONE,
           {bt_ns::TensorTypeSpec(static_cast<std::int32_t>(data_type),
                                  shape.rank3 ? rt_ns::Shape{1, shape.m, shape.n}
                                              : rt_ns::Shape{shape.m, shape.n})});
    return;
  }
  Expect(registry, MakeNode("MatMul", {"A", "B"}, {"Y"}), name, {opset}, {a_count, b_count},
         {y_count}, [opset, shape, data_type]() -> IoData {
           const rt_ns::Shape a_shape =
               shape.rank3 ? rt_ns::Shape{1, shape.m, shape.k} : rt_ns::Shape{shape.m, shape.k};
           Tensor a = MakeBenchmarkTensor(data_type, a_shape, 433);
           Tensor b = MakeBenchmarkTensor(data_type, {shape.k, shape.n}, 434);
           Tensor y = onnx_light_cpu::MatMulKernel{KernelContext{opset}}(a, b);
           return IoData{{std::move(a), std::move(b)}, {std::move(y)}};
         });
}

} // namespace

void RegisterCpuMatMulCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);
  const std::array<DataType, 4> data_types = {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16,
                                              DataType::BFLOAT16};
  if (mode == TestMode::BENCHMARK) {
    constexpr std::array<MatMulShape, 7> shapes = {
        MatMulShape{"square_64", 64, 64, 64},     MatMulShape{"square_128", 128, 128, 128},
        MatMulShape{"square_512", 512, 512, 512}, MatMulShape{"square_1024", 1024, 1024, 1024},
        MatMulShape{"skinny_m", 1, 4096, 4096},   MatMulShape{"skinny_n", 4096, 1, 4096},
        MatMulShape{"large_k", 32, 32, 8192},
    };
    for (const MatMulShape &shape : shapes) {
      for (DataType data_type : data_types) {
        RegisterMatMulCase(registry, opset, shape, data_type, true);
      }
    }
    for (const MatMulShape &shape :
         {MatMulShape{"llm_qwen3_8b_qkv_m1_k4096_n6144", 1, 6144, 4096, true},
          MatMulShape{"llm_qwen3_8b_o_proj_m1_k4096_n4096", 1, 4096, 4096, true},
          MatMulShape{"llm_qwen3_8b_gate_up_m1_k4096_n12288", 1, 12288, 4096, true},
          MatMulShape{"llm_qwen3_8b_down_m1_k12288_n4096", 1, 4096, 12288, true},
          MatMulShape{"llm_qwen3_8b_lm_head_m1_k4096_n151936", 1, 151936, 4096, true}}) {
      RegisterMatMulCase(registry, opset, shape, DataType::FLOAT16, true);
    }
    return;
  }
  for (DataType data_type : data_types) {
    RegisterMatMulCase(registry, opset, {"small", 2, 3, 4}, data_type, false);
  }
}

} // namespace onnx_light_cpu::backend_test
