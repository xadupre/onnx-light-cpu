// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/math/include_math_cases.h"

#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"
#include "onnx_light_cpu/kernels/math/swiglu_kernel.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/kernel_context.h"

namespace onnx_light_cpu::backend_test {

void RegisterCpuSwiGLUCases(std::vector<TestCase> &registry, TestMode mode) {
  namespace bt_ns = ONNX_LIGHT_NAMESPACE::core::backend_test;
  namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
  const rt_ns::OpsetId opset(std::string(), 28);
  if (mode == TestMode::BENCHMARK) {
    for (const std::int64_t size : {1024, 32768, 65535, 65536, 131072, 1048576, 4194304}) {
      for (const rt_ns::DataType data_type :
           {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE, rt_ns::DataType::FLOAT16,
            rt_ns::DataType::BFLOAT16}) {
        ONNX_LIGHT_NAMESPACE::NodeProto node;
        node.set_op_type("SwiGLU");
        node.add_input("gate");
        node.add_input("value");
        node.add_output("output");
        auto *alpha = node.add_attribute();
        alpha->set_name("alpha");
        alpha->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::FLOAT);
        alpha->set_f(0.5f);
        const std::string name = "test_cpu_swiglu_n" + std::to_string(size) + "_" +
                                 DataTypeSuffix(data_type) + "_benchmark";
        bt_ns::Expect(registry, std::move(node), name, {opset}, {size, size}, {size},
                      [=](bool generate_expected_outputs) -> bt_ns::IoData {
                        rt_ns::Tensor gate = MakeBenchmarkTensor(data_type, {size}, 987654321ULL);
                        rt_ns::Tensor value = MakeBenchmarkTensor(data_type, {size}, 987654322ULL);
                        if (!generate_expected_outputs) {
                          return bt_ns::IoData{{std::move(gate), std::move(value)}, {}, false};
                        }
                        const SwiGLUKernel kernel(rt_ns::KernelContext{opset});
                        rt_ns::Tensor output = kernel(gate, value, 0.5f);
                        return bt_ns::IoData{{std::move(gate), std::move(value)},
                                             {std::move(output)}};
                      });
      }
    }
    return;
  }
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
