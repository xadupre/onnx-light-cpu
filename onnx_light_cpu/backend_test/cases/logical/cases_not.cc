// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/logical/include_logical_cases.h"

#include "onnx_light_cpu/kernels/logical/not_kernel.h"

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
using ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::DataType;
using rt_ns::DefaultOpset;
using rt_ns::KernelContext;
using rt_ns::OpsetId;
using rt_ns::Tensor;

// Builds a single-input / single-output unary ``Not`` NodeProto.
NodeProto MakeNotNode() {
  NodeProto node;
  node.set_op_type("Not");
  node.add_input("x");
  node.add_output("y");
  return node;
}

} // namespace

// Not — the only element type ``NotKernel`` implements: bool.
void RegisterCpuNotCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(1);
  const onnx_light_cpu::NotKernel not_kernel{KernelContext{opset}};

  if (mode == TestMode::BENCHMARK) {
    for (const std::int64_t size : {1024, 32768, 65535, 65536, 131072, 1048576, 4194304}) {
      Expect(registry, MakeNotNode(), "test_cpu_not_n" + std::to_string(size) + "_benchmark",
             {opset}, {size}, {size}, [not_kernel, size]() -> IoData {
               std::vector<std::uint8_t> raw(static_cast<std::size_t>(size));
               for (std::size_t i = 0; i < raw.size(); ++i) {
                 raw[i] = static_cast<std::uint8_t>(i & 1U);
               }
               Tensor x("", static_cast<std::int32_t>(DataType::BOOL), {size}, std::move(raw));
               Tensor y = not_kernel(x);
               return IoData{{std::move(x)}, {std::move(y)}};
             });
    }
    return;
  }

  Expect(registry, MakeNotNode(), "test_cpu_not_bool", {opset}, [=]() -> IoData {
    Tensor x("", DataType::BOOL, {2, 2}, {1, 0, 1, 0});
    return IoData{{x}, {not_kernel(x)}};
  });
}

} // namespace onnx_light_cpu::backend_test
