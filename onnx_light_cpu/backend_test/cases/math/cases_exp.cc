// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

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
using bt_ns::ExpectBenchmarkUnaryFloat;
using bt_ns::IoData;
using bt_ns::TestCase;
using bt_ns::TestMode;
using ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::DataType;
using rt_ns::DefaultOpset;
using rt_ns::KernelContext;
using rt_ns::OpsetId;
using rt_ns::Tensor;

// Builds a single-input / single-output unary ``Exp`` NodeProto.
NodeProto MakeExpNode() {
  NodeProto node;
  node.set_op_type("Exp");
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

// Exp — covers float32, float64, float16, bfloat16.
void RegisterCpuExpCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);
  const onnx_light_cpu::ExpKernel exp_kernel{KernelContext{opset}};
  const std::vector<int64_t> shape = {2, 3};
  const std::vector<float> f = {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f};

  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkUnaryFloat("Exp", exp_kernel, "test_cpu_exp_benchmark", opset, registry);
    return;
  }

  Expect(registry, MakeExpNode(), "test_cpu_exp_float32", {opset}, [=]() -> IoData {
    Tensor x = Tensor::FromFloat("", shape, f);
    return IoData{{x}, {exp_kernel(x)}};
  });
  Expect(registry, MakeExpNode(), "test_cpu_exp_float64", {opset}, [=]() -> IoData {
    Tensor x = Tensor::FromDouble("", shape, {-2.0, -1.0, 0.0, 0.5, 1.0, 2.0});
    return IoData{{x}, {exp_kernel(x)}};
  });
  Expect(registry, MakeExpNode(), "test_cpu_exp_float16", {opset}, [=]() -> IoData {
    Tensor x = rt_ns::MakeFloat16Tensor("", shape, f);
    return IoData{{x}, {exp_kernel(x)}};
  });
  Expect(registry, MakeExpNode(), "test_cpu_exp_bfloat16", {opset}, [=]() -> IoData {
    Tensor x = MakeBfloat16Tensor(shape, f);
    return IoData{{x}, {exp_kernel(x)}};
  });
}

} // namespace onnx_light_cpu::backend_test
