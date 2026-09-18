// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"
#include "onnx_light_cpu/backend_test/cases/math/include_math_cases.h"
#include "onnx_light_cpu/kernels/math/tanh_kernel.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace onnx_light_cpu::backend_test {

namespace {

namespace bt = ONNX_LIGHT_NAMESPACE::core::backend_test;
namespace rt = ONNX_LIGHT_NAMESPACE::core::runtime;

void RegisterLogitsBenchmark(std::vector<TestCase> &registry, rt::DataType type,
                             std::int64_t tokens) {
  const auto opset = rt::DefaultOpset(13);
  const rt::Shape shape{1, tokens, 202048};
  const std::int64_t count = tokens * shape.back();
  const std::string name = "test_cpu_tanh_logits_1x" + std::to_string(tokens) + "x202048_" +
                           DataTypeSuffix(type) + "_benchmark";
  auto build = [=](bool generate_expected_outputs) -> bt::IoData {
    auto x = MakeBenchmarkTensor(type, shape, 987654321ULL);
    if (!generate_expected_outputs) {
      return bt::IoData{{std::move(x)}, {}, {}, false};
    }
    const rt::KernelContext context{opset};
    const TanhKernel kernel{context};
    auto y = kernel(x);
    return bt::IoData{{std::move(x)}, {std::move(y)}};
  };
  bt::Expect(registry, MakeNode("Tanh", {"x"}, {"y"}), name, {opset}, {count}, {count},
             std::move(build), "backend-test", bt::TestCaseTag::NONE,
             {bt::TensorTypeSpec(static_cast<std::int32_t>(type), shape)});
}

} // namespace

void RegisterCpuTanhCases(std::vector<TestCase> &registry, TestMode mode) {
  const auto opset = rt::DefaultOpset(13);
  for (const auto type : {rt::DataType::FLOAT, rt::DataType::FLOAT16, rt::DataType::BFLOAT16}) {
    if (mode == TestMode::BENCHMARK) {
      for (const std::int64_t size : {8, 1024, 65535, 65536}) {
        RegisterUnaryBenchmark(
            registry, "Tanh", [opset] { return TanhKernel{rt::KernelContext{opset}}; }, opset, type,
            size);
      }
      for (const std::int64_t tokens : {1, 16, 128}) {
        RegisterLogitsBenchmark(registry, type, tokens);
      }
      continue;
    }
    const std::vector<float> input{-20, -4, -1, -0.25f, -0.0f, 0, 0.25f, 1, 4, 20};
    std::vector<float> expected;
    for (float value : input) {
      expected.push_back(std::tanh(value));
    }
    bt::Expect(registry, MakeNode("Tanh", {"x"}, {"y"}),
               "test_cpu_tanh_" + std::string(DataTypeSuffix(type)), {opset}, [=]() -> bt::IoData {
                 return bt::IoData{{MakeTensor(type, {2, 5}, input)},
                                   {MakeTensor(type, {2, 5}, expected)}};
               });
    registry.back().rtol = type == rt::DataType::FLOAT ? 2e-6 : 8e-3;
    registry.back().atol = 1e-7;
  }
}

} // namespace onnx_light_cpu::backend_test
