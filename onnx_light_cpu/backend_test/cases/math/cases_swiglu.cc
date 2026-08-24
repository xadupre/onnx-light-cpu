// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/math/include_math_cases.h"

#include "onnx_light_cpu/kernels/math/swiglu_kernel.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/kernel_context.h"

namespace onnx_light_cpu::backend_test {

void RegisterCpuSwiGLUCases(std::vector<TestCase> &registry, TestMode mode) {
  if (mode == TestMode::BENCHMARK) {
    return;
  }
  namespace bt_ns = ONNX_LIGHT_NAMESPACE::core::backend_test;
  namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
  const rt_ns::OpsetId opset(std::string(), 28);
  const SwiGLUKernel kernel(rt_ns::KernelContext{opset});
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("SwiGLU");
  node.add_input("gate");
  node.add_input("value");
  node.add_output("output");
  auto *alpha = node.add_attribute();
  alpha->set_name("alpha");
  alpha->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::FLOAT);
  alpha->set_f(0.5f);
  bt_ns::Expect(registry, std::move(node), "test_cpu_swiglu_alpha_float32", {opset},
                [=]() -> bt_ns::IoData {
                  rt_ns::Tensor gate =
                      rt_ns::Tensor::FromFloat("", {2, 3}, {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 3.0f});
                  rt_ns::Tensor value =
                      rt_ns::Tensor::FromFloat("", {2, 3}, {0.5f, -2.0f, 3.0f, 1.0f, -1.0f, 2.0f});
                  return bt_ns::IoData{{gate, value}, {kernel(gate, value, 0.5f)}};
                });
}

} // namespace onnx_light_cpu::backend_test
