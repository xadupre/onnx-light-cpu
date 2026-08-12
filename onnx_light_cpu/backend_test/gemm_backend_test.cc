// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/gemm_backend_test.h"

#include "onnx_light_cpu/kernels/math/gemm_kernel.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/cast_helper.h"
#include "onnx_core/runtime/kernel_context.h"
#include "onnx_core/runtime/random.h"
#include "onnx_core/runtime/simple_tensor.h"

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
using rt_ns::Randn;
using rt_ns::Tensor;

// Builds a plain (no-attribute) two-input ``Gemm`` NodeProto: Y = A @ B.
NodeProto MakeGemmNode() {
  NodeProto node;
  node.set_op_type("Gemm");
  node.add_input("A");
  node.add_input("B");
  node.add_output("Y");
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

// Gemm — Y = A @ B (default attributes), covering float32, float64, float16,
// bfloat16 (every element type ``GemmKernel`` implements).
void RegisterCpuGemmCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);
  const onnx_light_cpu::GemmKernel gemm_kernel{KernelContext{opset}};

  if (mode == TestMode::BENCHMARK) {
    const std::vector<int64_t> shape = {512, 512};
    const int64_t count = 512 * 512;
    Expect(registry, MakeGemmNode(), "test_cpu_gemm_benchmark", {opset}, {count, count}, {count},
           [gemm_kernel, shape]() -> IoData {
             Tensor a = Tensor::FromFloat("", shape, Randn<float>(shape, 433));
             Tensor b = Tensor::FromFloat("", shape, Randn<float>(shape, 434));
             Tensor y = gemm_kernel(a, b, 1.0f, false, false);
             return IoData{{std::move(a), std::move(b)}, {std::move(y)}};
           });
    return;
  }

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

} // namespace onnx_light_cpu::backend_test
