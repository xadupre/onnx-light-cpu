// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"
#include "onnx_light_cpu/backend_test/cases/com_microsoft/include_com_microsoft_cases.h"
#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include <cstdint>
#include <string>
#include <vector>

namespace onnx_light_cpu::backend_test {

void RegisterCpuMicrosoftGeluCases(std::vector<TestCase> &registry, TestMode mode) {
  namespace bt_ns = ONNX_LIGHT_NAMESPACE::core::backend_test;
  namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
  using rt_ns::DataType;
  const auto shapes = mode == TestMode::BENCHMARK
                          ? std::vector<std::vector<int64_t>>{{4096, 128}}
                          : std::vector<std::vector<int64_t>>{{},   {0, 32}, {7},  {8},    {9},
                                                              {15}, {16},    {17}, {2, 32}};
  for (const auto &shape : shapes) {
    int64_t count = 1;
    for (const auto dim : shape) {
      count *= dim;
    }
    for (const auto type :
         {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16}) {
      ONNX_LIGHT_NAMESPACE::NodeProto node;
      node.set_op_type("Gelu");
      node.set_domain(kMicrosoftDomain);
      node.add_input("X");
      node.add_output("Y");
      std::string name = "test_cpu_microsoft_gelu";
      for (const auto dim : shape) {
        name += "_" + std::to_string(dim);
      }
      name += std::string("_") + DataTypeSuffix(type);
      if (mode == TestMode::BENCHMARK) {
        name += "_benchmark";
      }
      bt_ns::Expect(registry, node, name,
                    {rt_ns::DefaultOpset(20), rt_ns::OpsetId(kMicrosoftDomain, 1)}, {count},
                    {count},
                    [=](bool generate_expected_outputs) -> bt_ns::IoData {
                      auto x = MakeBenchmarkTensor(type, shape, 972);
                      if (!generate_expected_outputs) {
                        return bt_ns::IoData{{std::move(x)}, {}, {}, false};
                      }
                      const rt_ns::KernelContext kernel_context{rt_ns::DefaultOpset(20)};
                      const ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::Gelu oracle{kernel_context};
                      auto y = oracle(x);
                      return bt_ns::IoData{{std::move(x)}, {std::move(y)}};
                    },
                    "backend-test", bt_ns::TestCaseTag::AI_RT,
                    {bt_ns::TensorTypeSpec(static_cast<std::int32_t>(type), shape)});
    }
  }
}

} // namespace onnx_light_cpu::backend_test
