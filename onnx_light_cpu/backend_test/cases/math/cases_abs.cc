// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"
#include "onnx_light_cpu/backend_test/cases/math/include_math_cases.h"

#include "onnx_light_cpu/kernels/math/abs_kernel.h"

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
using bt_ns::TestCase;
using bt_ns::TestMode;
using rt_ns::DataType;
using rt_ns::DefaultOpset;
using rt_ns::KernelContext;
using rt_ns::OpsetId;
using rt_ns::Tensor;

} // namespace

// Abs — covers every element type ``AbsKernel`` implements:
// float32, float64, int8, int16, int32, int64, float16, bfloat16.
void RegisterCpuAbsCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);
  const std::vector<int64_t> shape = {2, 3};
  const std::vector<float> f = {-1.0f, 0.0f, 1.5f, -2.25f, 3.5f, -4.75f};

  if (mode == TestMode::BENCHMARK) {
    for (const int64_t size : {1024, 32768, 65535, 65536, 131072, 1048576, 4194304}) {
      for (const DataType data_type :
           {DataType::FLOAT, DataType::DOUBLE, DataType::INT8, DataType::INT16, DataType::INT32,
            DataType::INT64, DataType::FLOAT16, DataType::BFLOAT16}) {
        RegisterUnaryBenchmark(
            registry, "Abs", [opset] { return onnx_light_cpu::AbsKernel{KernelContext{opset}}; },
            opset, data_type, size);
      }
    }
    return;
  }

  Expect(registry, MakeNode("Abs", {"x"}, {"y"}), "test_cpu_abs_float32", {opset}, [=]() -> IoData {
    Tensor x = MakeTensor(DataType::FLOAT, shape, f);
    const onnx_light_cpu::AbsKernel kernel{KernelContext{opset}};
    return IoData{{x}, {kernel(x)}};
  });
  Expect(registry, MakeNode("Abs", {"x"}, {"y"}), "test_cpu_abs_float64", {opset}, [=]() -> IoData {
    Tensor x = MakeTensor(DataType::DOUBLE, shape, f);
    const onnx_light_cpu::AbsKernel kernel{KernelContext{opset}};
    return IoData{{x}, {kernel(x)}};
  });
  Expect(registry, MakeNode("Abs", {"x"}, {"y"}), "test_cpu_abs_int8", {opset}, [=]() -> IoData {
    Tensor x = Tensor::FromInt8("", shape, {-1, 0, 2, -127, 3, -5});
    const onnx_light_cpu::AbsKernel kernel{KernelContext{opset}};
    return IoData{{x}, {kernel(x)}};
  });
  Expect(registry, MakeNode("Abs", {"x"}, {"y"}), "test_cpu_abs_int16", {opset}, [=]() -> IoData {
    Tensor x = Tensor::FromInt16("", shape, {-1, 0, 2, -1000, 3, -5});
    const onnx_light_cpu::AbsKernel kernel{KernelContext{opset}};
    return IoData{{x}, {kernel(x)}};
  });
  Expect(registry, MakeNode("Abs", {"x"}, {"y"}), "test_cpu_abs_int32", {opset}, [=]() -> IoData {
    Tensor x = Tensor::FromInt32("", shape, {-1, 0, 2, -100000, 3, -5});
    const onnx_light_cpu::AbsKernel kernel{KernelContext{opset}};
    return IoData{{x}, {kernel(x)}};
  });
  Expect(registry, MakeNode("Abs", {"x"}, {"y"}), "test_cpu_abs_int64", {opset}, [=]() -> IoData {
    Tensor x = Tensor::FromInt64("", shape, {-1, 0, 2, -1000000000000LL, 3, -5});
    const onnx_light_cpu::AbsKernel kernel{KernelContext{opset}};
    return IoData{{x}, {kernel(x)}};
  });
  Expect(registry, MakeNode("Abs", {"x"}, {"y"}), "test_cpu_abs_float16", {opset}, [=]() -> IoData {
    Tensor x = MakeTensor(DataType::FLOAT16, shape, f);
    const onnx_light_cpu::AbsKernel kernel{KernelContext{opset}};
    return IoData{{x}, {kernel(x)}};
  });
  Expect(registry, MakeNode("Abs", {"x"}, {"y"}), "test_cpu_abs_bfloat16", {opset},
         [=]() -> IoData {
           Tensor x = MakeTensor(DataType::BFLOAT16, shape, f);
           const onnx_light_cpu::AbsKernel kernel{KernelContext{opset}};
           return IoData{{x}, {kernel(x)}};
         });
}

} // namespace onnx_light_cpu::backend_test
