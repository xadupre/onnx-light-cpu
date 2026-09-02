// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"
#include "onnx_light_cpu/backend_test/cases/math/include_math_cases.h"

#include "onnx_light_cpu/kernels/math/exp_log_kernel.h"

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

// Log — covers float32, float64, float16, bfloat16.
void RegisterCpuLogCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);
  const std::vector<int64_t> shape = {2, 3};
  const std::vector<float> f = {0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f};

  if (mode == TestMode::BENCHMARK) {
    // Positive inputs exercise the finite logarithm path rather than measuring
    // a data set dominated by invalid negative values and NaN outputs.
    for (const int64_t size : {1024, 65536, 131071, 131072, 262144, 1048576, 4194304}) {
      for (const DataType data_type :
           {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16}) {
        RegisterUnaryBenchmark(
            registry, "Log", [opset] { return onnx_light_cpu::LogKernel{KernelContext{opset}}; },
            opset, data_type, size);
      }
    }
    return;
  }

  Expect(registry, MakeNode("Log", {"x"}, {"y"}), "test_cpu_log_float32", {opset}, [=]() -> IoData {
    Tensor x = MakeTensor(DataType::FLOAT, shape, f);
    const onnx_light_cpu::LogKernel kernel{KernelContext{opset}};
    return IoData{{x}, {kernel(x)}};
  });
  Expect(registry, MakeNode("Log", {"x"}, {"y"}), "test_cpu_log_float64", {opset}, [=]() -> IoData {
    Tensor x = MakeTensor(DataType::DOUBLE, shape, f);
    const onnx_light_cpu::LogKernel kernel{KernelContext{opset}};
    return IoData{{x}, {kernel(x)}};
  });
  Expect(registry, MakeNode("Log", {"x"}, {"y"}), "test_cpu_log_float16", {opset}, [=]() -> IoData {
    Tensor x = MakeTensor(DataType::FLOAT16, shape, f);
    const onnx_light_cpu::LogKernel kernel{KernelContext{opset}};
    return IoData{{x}, {kernel(x)}};
  });
  Expect(registry, MakeNode("Log", {"x"}, {"y"}), "test_cpu_log_bfloat16", {opset},
         [=]() -> IoData {
           Tensor x = MakeTensor(DataType::BFLOAT16, shape, f);
           const onnx_light_cpu::LogKernel kernel{KernelContext{opset}};
           return IoData{{x}, {kernel(x)}};
         });
}

} // namespace onnx_light_cpu::backend_test
