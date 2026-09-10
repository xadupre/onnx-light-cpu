// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/math/include_math_cases.h"

#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"
#include "onnx_light_cpu/kernels/math/integer_matmul_kernel.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

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
using rt_ns::Shape;
using rt_ns::Tensor;

struct MatMulShape {
  const char *name;
  std::int64_t m;
  std::int64_t n;
  std::int64_t k;
};

void RegisterMatMulIntegerCase(std::vector<TestCase> &registry, const OpsetId &opset,
                               const MatMulShape &shape, DataType a_type, DataType b_type,
                               bool benchmark) {
  const std::string name = "test_cpu_matmulinteger_" + std::string(shape.name) + "_" +
                           DataTypeSuffix(a_type) + "x" + DataTypeSuffix(b_type) +
                           (benchmark ? "_benchmark" : "");
  const std::int64_t a_count = shape.m * shape.k;
  const std::int64_t b_count = shape.k * shape.n;
  const std::int64_t y_count = shape.m * shape.n;
  if (benchmark) {
    Expect(registry, MakeNode("MatMulInteger", {"A", "B"}, {"Y"}), name, {opset},
           {a_count, b_count}, {y_count},
           [opset, shape, a_type, b_type](bool generate_expected_outputs) -> IoData {
             Tensor a = MakeBenchmarkTensor(a_type, {shape.m, shape.k}, 433);
             Tensor b = MakeBenchmarkTensor(b_type, {shape.k, shape.n}, 434);
             if (!generate_expected_outputs) {
               return IoData{{std::move(a), std::move(b)}, {}, {}, false};
             }
             const onnx_light_cpu::MatMulIntegerKernel kernel{KernelContext{opset}};
             Tensor y = kernel(a, b);
             return IoData{{std::move(a), std::move(b)}, {std::move(y)}};
           },
           "backend-test", bt_ns::TestCaseTag::NONE,
           {bt_ns::TensorTypeSpec(static_cast<std::int32_t>(DataType::INT32), {shape.m, shape.n})});
    return;
  }
  Expect(registry, MakeNode("MatMulInteger", {"A", "B"}, {"Y"}), name, {opset}, {a_count, b_count},
         {y_count}, [opset, shape, a_type, b_type]() -> IoData {
           Tensor a = MakeBenchmarkTensor(a_type, {shape.m, shape.k}, 433);
           Tensor b = MakeBenchmarkTensor(b_type, {shape.k, shape.n}, 434);
           Tensor y = onnx_light_cpu::MatMulIntegerKernel{KernelContext{opset}}(a, b);
           return IoData{{std::move(a), std::move(b)}, {std::move(y)}};
         });
}

void RegisterZeroPointCase(std::vector<TestCase> &registry, const OpsetId &opset,
                           const std::string &suffix, DataType a_type, DataType b_type,
                           const Shape &a_shape, const Shape &b_shape, const Shape &az_shape,
                           const Shape &bz_shape) {
  const std::string name = "test_cpu_matmulinteger_zero_points_" + suffix + "_" +
                           DataTypeSuffix(a_type) + "x" + DataTypeSuffix(b_type);
  auto build = [=]() -> IoData {
    Tensor a = MakeBenchmarkTensor(a_type, a_shape, 433);
    Tensor b = MakeBenchmarkTensor(b_type, b_shape, 434);
    Tensor az = MakeBenchmarkTensor(a_type, az_shape, 435);
    Tensor bz = MakeBenchmarkTensor(b_type, bz_shape, 436);
    // Use independent onnx-light kernels, not the MatMulInteger under test.
    namespace reference = ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel;
    const KernelContext ctx{opset};
    const reference::Cast cast{ctx};
    const reference::Sub sub{ctx};
    const reference::MatMul matmul{ctx};
    Tensor y = matmul(sub(cast(a, DataType::INT32), cast(az, DataType::INT32)),
                      sub(cast(b, DataType::INT32), cast(bz, DataType::INT32)));
    return IoData{{std::move(a), std::move(b), std::move(az), std::move(bz)}, {std::move(y)}};
  };
  Expect(registry, MakeNode("MatMulInteger", {"A", "B", "az", "bz"}, {"Y"}), name, {opset},
         std::move(build));
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
          RegisterMatMulIntegerCase(registry, opset, shape, a_type, b_type, true);
        }
      }
    }
    return;
  }
  for (DataType a_type : data_types) {
    for (DataType b_type : data_types) {
      RegisterMatMulIntegerCase(registry, opset, {"small", 2, 3, 4}, a_type, b_type, false);
      RegisterZeroPointCase(registry, opset, "matrix", a_type, b_type, {3, 4}, {4, 2}, {3, 1},
                            {1, 2});
      RegisterZeroPointCase(registry, opset, "batched_a", a_type, b_type, {2, 3, 4}, {4, 2},
                            {2, 3, 1}, {});
      RegisterZeroPointCase(registry, opset, "batched_b", a_type, b_type, {3, 4}, {2, 4, 2}, {},
                            {2, 1, 2});
      RegisterZeroPointCase(registry, opset, "multiple_batches", a_type, b_type, {2, 3, 3, 4},
                            {2, 3, 4, 2}, {2, 3, 3, 1}, {2, 3, 1, 2});
      RegisterZeroPointCase(registry, opset, "broadcast_batches", a_type, b_type, {2, 1, 3, 4},
                            {3, 4, 2}, {2, 1, 3, 1}, {3, 1, 2});
      RegisterZeroPointCase(registry, opset, "broadcast_a", a_type, b_type, {3, 4}, {2, 3, 4, 2},
                            {3, 1}, {2, 3, 1, 2});
      RegisterZeroPointCase(registry, opset, "single_batch", a_type, b_type, {1, 1, 3, 4}, {4, 2},
                            {1, 1, 3, 1}, {2});
      RegisterZeroPointCase(registry, opset, "single_axis", a_type, b_type, {2, 1, 1, 4}, {3, 4, 1},
                            {2, 1, 1, 1}, {3, 1, 1});
      RegisterZeroPointCase(registry, opset, "vector_b", a_type, b_type, {2, 3, 3, 4}, {4},
                            {2, 3, 3, 1}, {});
      RegisterZeroPointCase(registry, opset, "vector_a", a_type, b_type, {4}, {2, 3, 4, 2}, {},
                            {2, 3, 1, 2});
    }
  }
}

} // namespace onnx_light_cpu::backend_test
