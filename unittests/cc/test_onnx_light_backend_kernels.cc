// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// C++ unit test for the onnx-light-cpu kernels driven through onnx-light's
// regular backend-test API.
//
// The onnx-light-cpu backend test *cases* live in a dedicated library
// (``lib_onnx_light_cpu_backend_test``) which merely *registers* them into
// onnx-light's shared backend test case registry. This unit test:
//
//   1. registers those new backend test cases
//      (``RegisterCpuKernelBackendTestCases``),
//   2. registers the accelerated kernels
//      (``onnx_light_cpu::RegisterAllKernels``), and
//   3. drives each registered case through onnx-light's regular runtime API
//      (``CollectTestCases`` + ``RuntimeContext`` / ``SubgraphSession``),
//      checking the model output — produced by the onnx-light-cpu kernel the
//      runtime dispatched to — matches the reference output shipped with the
//      case, via onnx-light's public ``CompareTensors`` helper (see
//      ``CompareTensor``).
//
// The runtime resolves each node by ``(domain, op_type)`` to whatever kernel is
// registered, so registering the onnx-light-cpu kernels means these cases
// exercise the accelerated kernels end to end, exactly like a model run through
// onnx-light itself.

#include "onnx_light_cpu/backend_test/collect_test_cases.h"
#include "onnx_light_cpu/kernels/register_kernels.h"

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_core/runtime/kernels/tensor_compare.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ONNX_LIGHT_NAMESPACE;
using core::backend_test::CollectTestCases;
using core::backend_test::DataSet;
using core::backend_test::TestCase;
using core::runtime::CompareTensors;
using core::runtime::DefaultOpset;
using core::runtime::KernelContext;
using core::runtime::RegisterModelFunctions;
using core::runtime::RuntimeContext;
using core::runtime::SubgraphSession;
using core::runtime::Tensor;
using core::runtime::TensorComparison;
using core::runtime::Tensors;

// onnx-light resolves kernels by (domain, op_type) only — never by opset
// version — so the concrete default opset used to build the runtime context
// does not affect which kernel a node dispatches to. A fixed recent version is
// therefore sufficient to drive every registered case.
constexpr int64_t kRuntimeDefaultOpsetVersion = 18;

// Compares a runtime output tensor against the reference output of a backend
// test case using onnx-light's public ``CompareTensors`` helper
// (``onnx_core/runtime/kernels/tensor_compare.h``) with the case's ``rtol``/``atol``.
void CompareTensor(const std::string &case_name, const Tensor &actual, const Tensor &expected,
                   double rtol, double atol, std::vector<std::string> &failures) {
  const TensorComparison result = CompareTensors(actual, expected, rtol, atol);
  if (!result.close) {
    failures.push_back(case_name + ": " + result.message);
  }
}

// Runs a single-node backend test case model through onnx-light's runtime and
// compares the produced output with the reference output. The runtime resolves
// the node to the registered onnx-light-cpu kernel. When ``compare`` is false
// (benchmark cases) only the output count is checked -- benchmark inputs are
// randomly drawn (e.g. negative values for ``Log`` produce NaN), so they are
// executed for timing coverage rather than numeric comparison.
void RunCaseThroughRuntime(const TestCase &tc, bool compare, std::vector<std::string> &failures) {
  const ModelProto &model = tc.model();
  const GraphProto &graph = model.ref_graph();
  for (const DataSet &ds : tc.data_sets()) {
    RuntimeContext rt(KernelContext(DefaultOpset(kRuntimeDefaultOpsetVersion)));
    RegisterModelFunctions(model, rt);

    std::vector<std::pair<std::string, Tensor>> bindings;
    bindings.reserve(ds.inputs.size());
    for (const Tensor &t : ds.inputs) {
      bindings.emplace_back(t.name, t);
    }

    SubgraphSession session(rt, graph);
    const Tensors outputs = session.Run(std::move(bindings), rt);

    if (outputs.size() != ds.outputs.size()) {
      failures.push_back(tc.name + ": output count mismatch");
      continue;
    }
    if (!compare) {
      continue;
    }
    for (size_t i = 0; i < ds.outputs.size(); ++i) {
      CompareTensor(tc.name, outputs[i], ds.outputs[i], tc.rtol, tc.atol, failures);
    }
  }
}

// Registers the onnx-light-cpu backend test cases and kernels, then runs every
// ``test_cpu_*`` case for ``op_type`` (in the given ``mode``) through the
// runtime, collecting failures. Correctness (``TEST``) cases have their outputs
// compared; benchmark (``BENCHMARK``) cases are only executed.
std::vector<std::string> RunCpuBackendCases(const std::string &op_type,
                                            core::backend_test::TestMode mode) {
  onnx_light_cpu::backend_test::RegisterCpuKernelBackendTestCases();
  onnx_light_cpu::RegisterAllKernels();

  const bool compare = mode == core::backend_test::TestMode::TEST;
  std::vector<std::string> failures;
  std::vector<TestCase> cases = CollectTestCases(op_type, /*include_big=*/false, mode);
  size_t cpu_cases = 0;
  for (const TestCase &tc : cases) {
    if (tc.name.rfind("test_cpu_", 0) != 0) {
      continue;
    }
    ++cpu_cases;
    RunCaseThroughRuntime(tc, compare, failures);
  }
  if (cpu_cases == 0) {
    failures.push_back("no onnx-light-cpu backend test cases registered for " + op_type);
  }
  return failures;
}

std::string Describe(const std::vector<std::string> &failures) {
  std::string message;
  for (const std::string &failure : failures) {
    message += failure;
    message += '\n';
  }
  return message;
}

TEST(OnnxLightBackendKernels, AbsRunsThroughRuntime) {
  const std::vector<std::string> failures =
      RunCpuBackendCases("Abs", core::backend_test::TestMode::TEST);
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, ExpRunsThroughRuntime) {
  const std::vector<std::string> failures =
      RunCpuBackendCases("Exp", core::backend_test::TestMode::TEST);
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, LogRunsThroughRuntime) {
  const std::vector<std::string> failures =
      RunCpuBackendCases("Log", core::backend_test::TestMode::TEST);
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, NotRunsThroughRuntime) {
  const std::vector<std::string> failures =
      RunCpuBackendCases("Not", core::backend_test::TestMode::TEST);
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, GemmRunsThroughRuntime) {
  const std::vector<std::string> failures =
      RunCpuBackendCases("Gemm", core::backend_test::TestMode::TEST);
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

// Benchmark-mode cases: registered per kernel alongside the correctness cases.
// The unit test exercises them (executes the large model through the runtime)
// to keep the benchmark registration covered; timings are collected by a
// dedicated benchmark harness, so only successful execution is asserted here.
TEST(OnnxLightBackendKernels, AbsBenchmarkRunsThroughRuntime) {
  const std::vector<std::string> failures =
      RunCpuBackendCases("Abs", core::backend_test::TestMode::BENCHMARK);
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, ExpBenchmarkRunsThroughRuntime) {
  const std::vector<std::string> failures =
      RunCpuBackendCases("Exp", core::backend_test::TestMode::BENCHMARK);
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, LogBenchmarkRunsThroughRuntime) {
  const std::vector<std::string> failures =
      RunCpuBackendCases("Log", core::backend_test::TestMode::BENCHMARK);
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, NotBenchmarkRunsThroughRuntime) {
  const std::vector<std::string> failures =
      RunCpuBackendCases("Not", core::backend_test::TestMode::BENCHMARK);
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, GemmBenchmarkRunsThroughRuntime) {
  const std::vector<std::string> failures =
      RunCpuBackendCases("Gemm", core::backend_test::TestMode::BENCHMARK);
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, GemmBenchmarkCorpusIsLazyAndCoversPriorityShapes) {
  onnx_light_cpu::backend_test::RegisterCpuKernelBackendTestCases();
  std::vector<TestCase> cases =
      CollectTestCases("Gemm", /*include_big=*/false, core::backend_test::TestMode::BENCHMARK);

  size_t cpu_cases = 0;
  bool has_constant_b = false;
  bool has_direct = false;
  bool has_skinny_m = false;
  bool has_skinny_n = false;
  bool has_large_k = false;
  bool has_split_k = false;
  bool has_transpose = false;
  bool has_scalar_bias = false;
  bool has_row_bias = false;
  bool has_column_bias = false;
  bool has_matrix_bias = false;
  bool has_float32 = false;
  bool has_float16 = false;
  bool has_bfloat16 = false;
  for (const TestCase &test_case : cases) {
    if (test_case.name.rfind("test_cpu_gemm_", 0) != 0) {
      continue;
    }
    ++cpu_cases;
    EXPECT_TRUE(test_case.is_lazy());
    EXPECT_FALSE(test_case.materialized());
    EXPECT_TRUE(test_case.name.ends_with("_benchmark"));
    if (test_case.name.find("constant_b") != std::string::npos) {
      has_constant_b = true;
      EXPECT_EQ(test_case.declared_input_element_counts.size(), 1u);
    }
    has_direct |= test_case.name.find("_direct_") != std::string::npos;
    has_skinny_m |= test_case.name.find("skinny_m") != std::string::npos;
    has_skinny_n |= test_case.name.find("skinny_n") != std::string::npos;
    has_large_k |= test_case.name.find("large_k") != std::string::npos;
    has_split_k |= test_case.name.find("split_k") != std::string::npos;
    has_transpose |= test_case.name.find("trans_") != std::string::npos;
    has_scalar_bias |= test_case.name.find("scalar_bias") != std::string::npos;
    has_row_bias |= test_case.name.find("row_bias") != std::string::npos;
    has_column_bias |= test_case.name.find("column_bias") != std::string::npos;
    has_matrix_bias |= test_case.name.find("matrix_bias") != std::string::npos;
    has_float32 |= test_case.name.find("_float32_") != std::string::npos;
    has_float16 |= test_case.name.find("_float16_") != std::string::npos;
    has_bfloat16 |= test_case.name.find("_bfloat16_") != std::string::npos;
  }
  // 16 prepared shapes, each registered for float32, float16 and bfloat16.
  EXPECT_EQ(cpu_cases, 48u);
  EXPECT_TRUE(has_constant_b);
  EXPECT_TRUE(has_direct);
  EXPECT_TRUE(has_skinny_m);
  EXPECT_TRUE(has_skinny_n);
  EXPECT_TRUE(has_large_k);
  EXPECT_TRUE(has_split_k);
  EXPECT_TRUE(has_transpose);
  EXPECT_TRUE(has_scalar_bias);
  EXPECT_TRUE(has_row_bias);
  EXPECT_TRUE(has_column_bias);
  EXPECT_TRUE(has_matrix_bias);
  EXPECT_TRUE(has_float32);
  EXPECT_TRUE(has_float16);
  EXPECT_TRUE(has_bfloat16);
}

} // namespace
