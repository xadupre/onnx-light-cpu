// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/kernel_backend_test.h"

#include "onnx_light_cpu/kernels/logical/not_kernel.h"
#include "onnx_light_cpu/kernels/math/abs_kernel.h"
#include "onnx_light_cpu/kernels/math/exp_log_kernel.h"
#include "onnx_light_cpu/kernels/math/gemm_kernel.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/backend_test/io_data.h"
#include "onnx_core/backend_test/test_case_registry.h"
#include "onnx_core/runtime/cast_helper.h"
#include "onnx_core/runtime/kernel_context.h"
#include "onnx_core/runtime/simple_tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace onnx_light_cpu::backend_test {

namespace {

namespace bt_ns = ONNX_LIGHT_NAMESPACE::core::backend_test;
namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

using bt_ns::DispatchRegisterByOpType;
using bt_ns::Expect;
using bt_ns::IoData;
using bt_ns::OpRegisterModeMap;
using bt_ns::TestCase;
using bt_ns::TestMode;
using ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::DataType;
using rt_ns::DefaultOpset;
using rt_ns::KernelContext;
using rt_ns::OpsetId;
using rt_ns::Tensor;

// Builds a single-input / single-output unary NodeProto.
NodeProto MakeUnaryNode(const std::string &op_type) {
  NodeProto node;
  node.set_op_type(op_type);
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

// ---------------------------------------------------------------------------
// Abs — covers every element type ``AbsKernel`` implements:
// float32, float64, int8, int16, int32, int64, float16, bfloat16.
// ---------------------------------------------------------------------------
void RegisterAbsCases(std::vector<TestCase> &registry, TestMode mode) {
  if (mode != TestMode::TEST) {
    return;
  }
  const OpsetId opset = DefaultOpset(13);
  const onnx_light_cpu::AbsKernel abs_kernel{KernelContext{opset}};
  const std::vector<int64_t> shape = {2, 3};
  const std::vector<float> f = {-1.0f, 0.0f, 1.5f, -2.25f, 3.5f, -4.75f};

  Expect(registry, MakeUnaryNode("Abs"), "test_cpu_abs_float32", {opset}, [=]() -> IoData {
    Tensor x = Tensor::FromFloat("", shape, f);
    return IoData{{x}, {abs_kernel(x)}};
  });
  Expect(registry, MakeUnaryNode("Abs"), "test_cpu_abs_float64", {opset}, [=]() -> IoData {
    Tensor x = Tensor::FromDouble("", shape, {-1.0, 0.0, 1.5, -2.25, 3.5, -4.75});
    return IoData{{x}, {abs_kernel(x)}};
  });
  Expect(registry, MakeUnaryNode("Abs"), "test_cpu_abs_int8", {opset}, [=]() -> IoData {
    Tensor x = Tensor::FromInt8("", shape, {-1, 0, 2, -127, 3, -5});
    return IoData{{x}, {abs_kernel(x)}};
  });
  Expect(registry, MakeUnaryNode("Abs"), "test_cpu_abs_int16", {opset}, [=]() -> IoData {
    Tensor x = Tensor::FromInt16("", shape, {-1, 0, 2, -1000, 3, -5});
    return IoData{{x}, {abs_kernel(x)}};
  });
  Expect(registry, MakeUnaryNode("Abs"), "test_cpu_abs_int32", {opset}, [=]() -> IoData {
    Tensor x = Tensor::FromInt32("", shape, {-1, 0, 2, -100000, 3, -5});
    return IoData{{x}, {abs_kernel(x)}};
  });
  Expect(registry, MakeUnaryNode("Abs"), "test_cpu_abs_int64", {opset}, [=]() -> IoData {
    Tensor x = Tensor::FromInt64("", shape, {-1, 0, 2, -1000000000000LL, 3, -5});
    return IoData{{x}, {abs_kernel(x)}};
  });
  Expect(registry, MakeUnaryNode("Abs"), "test_cpu_abs_float16", {opset}, [=]() -> IoData {
    Tensor x = rt_ns::MakeFloat16Tensor("", shape, f);
    return IoData{{x}, {abs_kernel(x)}};
  });
  Expect(registry, MakeUnaryNode("Abs"), "test_cpu_abs_bfloat16", {opset}, [=]() -> IoData {
    Tensor x = MakeBfloat16Tensor(shape, f);
    return IoData{{x}, {abs_kernel(x)}};
  });
}

// ---------------------------------------------------------------------------
// Exp — covers float32, float64, float16, bfloat16.
// ---------------------------------------------------------------------------
void RegisterExpCases(std::vector<TestCase> &registry, TestMode mode) {
  if (mode != TestMode::TEST) {
    return;
  }
  const OpsetId opset = DefaultOpset(13);
  const onnx_light_cpu::ExpKernel exp_kernel{KernelContext{opset}};
  const std::vector<int64_t> shape = {2, 3};
  const std::vector<float> f = {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f};

  Expect(registry, MakeUnaryNode("Exp"), "test_cpu_exp_float32", {opset}, [=]() -> IoData {
    Tensor x = Tensor::FromFloat("", shape, f);
    return IoData{{x}, {exp_kernel(x)}};
  });
  Expect(registry, MakeUnaryNode("Exp"), "test_cpu_exp_float64", {opset}, [=]() -> IoData {
    Tensor x = Tensor::FromDouble("", shape, {-2.0, -1.0, 0.0, 0.5, 1.0, 2.0});
    return IoData{{x}, {exp_kernel(x)}};
  });
  Expect(registry, MakeUnaryNode("Exp"), "test_cpu_exp_float16", {opset}, [=]() -> IoData {
    Tensor x = rt_ns::MakeFloat16Tensor("", shape, f);
    return IoData{{x}, {exp_kernel(x)}};
  });
  Expect(registry, MakeUnaryNode("Exp"), "test_cpu_exp_bfloat16", {opset}, [=]() -> IoData {
    Tensor x = MakeBfloat16Tensor(shape, f);
    return IoData{{x}, {exp_kernel(x)}};
  });
}

// ---------------------------------------------------------------------------
// Log — covers float32, float64, float16, bfloat16.
// ---------------------------------------------------------------------------
void RegisterLogCases(std::vector<TestCase> &registry, TestMode mode) {
  if (mode != TestMode::TEST) {
    return;
  }
  const OpsetId opset = DefaultOpset(13);
  const onnx_light_cpu::LogKernel log_kernel{KernelContext{opset}};
  const std::vector<int64_t> shape = {2, 3};
  const std::vector<float> f = {0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f};

  Expect(registry, MakeUnaryNode("Log"), "test_cpu_log_float32", {opset}, [=]() -> IoData {
    Tensor x = Tensor::FromFloat("", shape, f);
    return IoData{{x}, {log_kernel(x)}};
  });
  Expect(registry, MakeUnaryNode("Log"), "test_cpu_log_float64", {opset}, [=]() -> IoData {
    Tensor x = Tensor::FromDouble("", shape, {0.5, 1.0, 1.5, 2.0, 3.0, 4.0});
    return IoData{{x}, {log_kernel(x)}};
  });
  Expect(registry, MakeUnaryNode("Log"), "test_cpu_log_float16", {opset}, [=]() -> IoData {
    Tensor x = rt_ns::MakeFloat16Tensor("", shape, f);
    return IoData{{x}, {log_kernel(x)}};
  });
  Expect(registry, MakeUnaryNode("Log"), "test_cpu_log_bfloat16", {opset}, [=]() -> IoData {
    Tensor x = MakeBfloat16Tensor(shape, f);
    return IoData{{x}, {log_kernel(x)}};
  });
}

// ---------------------------------------------------------------------------
// Not — the only element type ``NotKernel`` implements: bool.
// ---------------------------------------------------------------------------
void RegisterNotCases(std::vector<TestCase> &registry, TestMode mode) {
  if (mode != TestMode::TEST) {
    return;
  }
  const OpsetId opset = DefaultOpset(1);
  const onnx_light_cpu::NotKernel not_kernel{KernelContext{opset}};

  Expect(registry, MakeUnaryNode("Not"), "test_cpu_not_bool", {opset}, [=]() -> IoData {
    Tensor x("", DataType::BOOL, {2, 2}, {1, 0, 1, 0});
    return IoData{{x}, {not_kernel(x)}};
  });
}

// Builds a plain (no-attribute) two-input Gemm NodeProto.
NodeProto MakeGemmNode() {
  NodeProto node;
  node.set_op_type("Gemm");
  node.add_input("A");
  node.add_input("B");
  node.add_output("Y");
  return node;
}

// ---------------------------------------------------------------------------
// Gemm — Y = A @ B (default attributes), covering float32, float64, float16,
// bfloat16 (every element type ``GemmKernel`` implements).
// ---------------------------------------------------------------------------
void RegisterGemmCases(std::vector<TestCase> &registry, TestMode mode) {
  if (mode != TestMode::TEST) {
    return;
  }
  const OpsetId opset = DefaultOpset(13);
  const onnx_light_cpu::GemmKernel gemm_kernel{KernelContext{opset}};
  const std::vector<int64_t> a_shape = {2, 3};
  const std::vector<int64_t> b_shape = {3, 4};
  const std::vector<float> a = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  const std::vector<float> b = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f,  5.0f,
                                6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};

  Expect(registry, MakeGemmNode(), "test_cpu_gemm_float32", {opset}, [=]() -> IoData {
    Tensor ta = Tensor::FromFloat("", a_shape, a);
    Tensor tb = Tensor::FromFloat("", b_shape, b);
    return IoData{{ta, tb}, {gemm_kernel(ta, tb, 1.0f, false, false)}};
  });
  Expect(registry, MakeGemmNode(), "test_cpu_gemm_float64", {opset}, [=]() -> IoData {
    Tensor ta = Tensor::FromDouble("", a_shape, {0.0, 1.0, 2.0, 3.0, 4.0, 5.0});
    Tensor tb = Tensor::FromDouble("", b_shape,
                                   {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0});
    return IoData{{ta, tb}, {gemm_kernel(ta, tb, 1.0f, false, false)}};
  });
  Expect(registry, MakeGemmNode(), "test_cpu_gemm_float16", {opset}, [=]() -> IoData {
    Tensor ta = rt_ns::MakeFloat16Tensor("", a_shape, a);
    Tensor tb = rt_ns::MakeFloat16Tensor("", b_shape, b);
    return IoData{{ta, tb}, {gemm_kernel(ta, tb, 1.0f, false, false)}};
  });
  Expect(registry, MakeGemmNode(), "test_cpu_gemm_bfloat16", {opset}, [=]() -> IoData {
    Tensor ta = MakeBfloat16Tensor(a_shape, a);
    Tensor tb = MakeBfloat16Tensor(b_shape, b);
    return IoData{{ta, tb}, {gemm_kernel(ta, tb, 1.0f, false, false)}};
  });
}

} // namespace

void CollectCpuKernelTestCases(std::vector<TestCase> &registry, const std::string &op_type,
                               TestMode mode) {
  static const OpRegisterModeMap kEntries = {
      {"Abs", &RegisterAbsCases}, {"Exp", &RegisterExpCases},   {"Log", &RegisterLogCases},
      {"Not", &RegisterNotCases}, {"Gemm", &RegisterGemmCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries, mode);
}

void RegisterCpuKernelBackendTestCases() {
  // Registering the collector into onnx-light's global registry exactly once,
  // regardless of how many times this function is called, via a function-local
  // static initialized on first use.
  static const int kRegistered = bt_ns::RegisterTestCasesCollector(
      [](std::vector<TestCase> &registry, const std::string &op_type, bool /*include_big*/,
         TestMode mode) { CollectCpuKernelTestCases(registry, op_type, mode); });
  (void)kRegistered;
}

} // namespace onnx_light_cpu::backend_test
