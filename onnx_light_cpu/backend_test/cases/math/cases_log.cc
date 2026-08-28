// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"
#include "onnx_light_cpu/backend_test/cases/math/include_math_cases.h"

#include "onnx_light_cpu/kernels/math/exp_log_kernel.h"

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
using bt_ns::TestCase;
using bt_ns::TestMode;
using ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::DataType;
using rt_ns::DefaultOpset;
using rt_ns::KernelContext;
using rt_ns::OpsetId;
using rt_ns::Tensor;

// Builds a single-input / single-output unary ``Log`` NodeProto.
NodeProto MakeLogNode() {
  NodeProto node;
  node.set_op_type("Log");
  node.add_input("x");
  node.add_output("y");
  return node;
}

// Encodes ``values`` as a BFLOAT16 ``Tensor`` (raw 16-bit bit patterns),
// mirroring how onnx-light's own backend test cases build bfloat16 inputs.
Tensor MakeBfloat16Tensor(const std::vector<int64_t> &shape, const std::vector<float> &values) {
  std::vector<std::uint8_t> raw(values.size() * sizeof(std::uint16_t));
  auto *dst = reinterpret_cast<std::uint16_t *>(raw.data());
  for (std::size_t i = 0; i < values.size(); ++i) {
    dst[i] = rt_ns::FloatToBfloat16Bits(values[i]);
  }
  return Tensor("", static_cast<std::int32_t>(DataType::BFLOAT16), shape, std::move(raw));
}

} // namespace

// Log — covers float32, float64, float16, bfloat16.
void RegisterCpuLogCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);
  const onnx_light_cpu::LogKernel log_kernel{KernelContext{opset}};
  const std::vector<int64_t> shape = {2, 3};
  const std::vector<float> f = {0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f};

  if (mode == TestMode::BENCHMARK) {
    // Positive inputs exercise the finite logarithm path rather than measuring
    // a data set dominated by invalid negative values and NaN outputs.
    for (const int64_t size : {1024, 65536, 131071, 131072, 262144, 1048576, 4194304}) {
      for (const DataType data_type :
           {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16}) {
        RegisterUnaryBenchmark(registry, "Log", log_kernel, opset, data_type, size);
      }
    }
    return;
  }

  Expect(registry, MakeLogNode(), "test_cpu_log_float32", {opset}, [=]() -> IoData {
    Tensor x = Tensor::FromFloat("", shape, f);
    return IoData{{x}, {log_kernel(x)}};
  });
  Expect(registry, MakeLogNode(), "test_cpu_log_float64", {opset}, [=]() -> IoData {
    Tensor x = Tensor::FromDouble("", shape, {0.5, 1.0, 1.5, 2.0, 3.0, 4.0});
    return IoData{{x}, {log_kernel(x)}};
  });
  Expect(registry, MakeLogNode(), "test_cpu_log_float16", {opset}, [=]() -> IoData {
    Tensor x = rt_ns::MakeFloat16Tensor("", shape, f);
    return IoData{{x}, {log_kernel(x)}};
  });
  Expect(registry, MakeLogNode(), "test_cpu_log_bfloat16", {opset}, [=]() -> IoData {
    Tensor x = MakeBfloat16Tensor(shape, f);
    return IoData{{x}, {log_kernel(x)}};
  });
}

} // namespace onnx_light_cpu::backend_test
