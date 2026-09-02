// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/memory/simple_tensor.h"

#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

namespace onnx_light_cpu::backend_test {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

ONNX_LIGHT_NAMESPACE::NodeProto MakeNode(std::string_view op_type,
                                         std::initializer_list<const char *> inputs,
                                         std::initializer_list<const char *> outputs);

const char *DataTypeSuffix(rt_ns::DataType data_type);

rt_ns::Tensor MakeTensor(rt_ns::DataType data_type, const rt_ns::Shape &shape,
                         const std::vector<float> &values);

rt_ns::Tensor MakeBenchmarkTensor(rt_ns::DataType data_type, const rt_ns::Shape &shape,
                                  std::uint64_t seed, bool positive = false);

template <typename KernelFactory>
void RegisterUnaryBenchmark(
    std::vector<ONNX_LIGHT_NAMESPACE::core::backend_test::TestCase> &registry,
    const std::string &op_type, KernelFactory kernel_factory, const rt_ns::OpsetId &opset,
    rt_ns::DataType data_type, std::int64_t size) {
  // Single-element timings measure dispatch overhead rather than unary throughput.
  if (size == 1) {
    return;
  }
  namespace bt_ns = ONNX_LIGHT_NAMESPACE::core::backend_test;
  ONNX_LIGHT_NAMESPACE::NodeProto node = MakeNode(op_type, {"x"}, {"y"});
  std::string op_tag = op_type;
  for (char &character : op_tag) {
    if (character >= 'A' && character <= 'Z') {
      character = static_cast<char>(character - 'A' + 'a');
    }
  }
  const std::string name = "test_cpu_" + op_tag + "_n" + std::to_string(size) + "_" +
                           DataTypeSuffix(data_type) + "_benchmark";
  bt_ns::Expect(
      registry, std::move(node), name, {opset}, {size}, {size},
      [kernel_factory, op_type, data_type, size](bool generate_expected_outputs) -> bt_ns::IoData {
        rt_ns::Tensor x = MakeBenchmarkTensor(data_type, {size}, 987654321ULL, op_type == "Log");
        if (!generate_expected_outputs) {
          return bt_ns::IoData{{std::move(x)}, {}, {}, false};
        }
        rt_ns::Tensor y = kernel_factory()(x);
        return bt_ns::IoData{{std::move(x)}, {std::move(y)}};
      },
      "backend-test", bt_ns::TestCaseTag::NONE,
      {bt_ns::TensorTypeSpec(static_cast<std::int32_t>(data_type), {size})});
}

} // namespace onnx_light_cpu::backend_test
