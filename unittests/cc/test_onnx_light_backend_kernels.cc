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
#include <set>
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
                                            core::backend_test::TestMode mode,
                                            const std::string &case_name = {}) {
  onnx_light_cpu::backend_test::RegisterCpuKernelBackendTestCases();
  onnx_light_cpu::RegisterAllKernels();

  const bool compare = mode == core::backend_test::TestMode::TEST;
  std::vector<std::string> failures;
  std::vector<TestCase> cases = CollectTestCases(op_type, /*include_big=*/false, mode);
  size_t cpu_cases = 0;
  for (const TestCase &tc : cases) {
    if (tc.name.rfind("test_cpu_", 0) != 0 || (!case_name.empty() && tc.name != case_name)) {
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

size_t CountCpuCasesAtMlOpset5(const std::string &op_type) {
  onnx_light_cpu::backend_test::RegisterCpuKernelBackendTestCases();
  const std::vector<TestCase> cases =
      CollectTestCases(op_type, /*include_big=*/false, core::backend_test::TestMode::TEST);
  size_t count = 0;
  for (const TestCase &test_case : cases) {
    if (test_case.name.rfind("test_cpu_", 0) != 0) {
      continue;
    }
    ++count;
    bool found_ml = false;
    for (const OperatorSetIdProto &opset : test_case.model().opset_import()) {
      if (opset.domain() == "ai.onnx.ml") {
        EXPECT_EQ(opset.version(), 5) << test_case.name;
        found_ml = true;
      }
    }
    EXPECT_TRUE(found_ml) << test_case.name;
  }
  return count;
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

TEST(OnnxLightBackendKernels, TreeEnsembleCorpusRegistersOnlyMlOpset5) {
  EXPECT_GE(CountCpuCasesAtMlOpset5("TreeEnsemble"), 36U);
  EXPECT_EQ(CountCpuCasesAtMlOpset5("TreeEnsembleRegressor"), 4U);
  EXPECT_EQ(CountCpuCasesAtMlOpset5("TreeEnsembleClassifier"), 2U);
}

TEST(OnnxLightBackendKernels, TreeEnsembleBenchmarkCoversPriorityDimensions) {
  onnx_light_cpu::backend_test::RegisterCpuKernelBackendTestCases();
  const std::vector<TestCase> cases = CollectTestCases("TreeEnsemble", /*include_big=*/false,
                                                       core::backend_test::TestMode::BENCHMARK);
  std::set<std::string> names;
  for (const TestCase &test_case : cases) {
    if (test_case.name.rfind("test_cpu_treeensemble_", 0) == 0) {
      names.insert(test_case.name);
    }
  }
  EXPECT_EQ(names.size(), 5U);
  EXPECT_TRUE(names.contains("test_cpu_treeensemble_t10_f4_b1_benchmark"));
  EXPECT_TRUE(names.contains("test_cpu_treeensemble_t100_f64_b8_benchmark"));
  EXPECT_TRUE(names.contains("test_cpu_treeensemble_t1000_f1024_b32_benchmark"));
  EXPECT_TRUE(names.contains("test_cpu_treeensemble_t10000_f4096_b1_benchmark"));
  EXPECT_TRUE(names.contains("test_cpu_treeensemble_t10000_f4096_b128_benchmark"));
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

TEST(OnnxLightBackendKernels, UnaryBenchmarkCorporaCoverParallelThresholds) {
  onnx_light_cpu::backend_test::RegisterCpuKernelBackendTestCases();
  std::set<int64_t> exp_sizes;
  std::set<int64_t> not_sizes;
  for (const std::string &op : {"Abs", "Exp", "Log", "Not"}) {
    std::vector<TestCase> cases =
        CollectTestCases(op, /*include_big=*/false, core::backend_test::TestMode::BENCHMARK);
    std::set<int64_t> sizes;
    for (const TestCase &test_case : cases) {
      if (test_case.name.rfind("test_cpu_", 0) == 0) {
        ASSERT_EQ(test_case.declared_input_element_counts.size(), 1u);
        sizes.insert(test_case.declared_input_element_counts[0]);
      }
    }
    EXPECT_GE(sizes.size(), 7u) << op;
    EXPECT_TRUE(sizes.contains(1048576)) << op;
    EXPECT_TRUE(sizes.contains(4194304)) << op;
    const int64_t threshold = op == "Log" ? 131072 : 65536;
    EXPECT_TRUE(sizes.contains(threshold - 1)) << op;
    EXPECT_TRUE(sizes.contains(threshold)) << op;
    if (op == "Exp") {
      exp_sizes = sizes;
    } else if (op == "Not") {
      not_sizes = sizes;
    }
  }
  EXPECT_EQ(not_sizes, exp_sizes);
}

TEST(OnnxLightBackendKernels, GemmBenchmarkRunsThroughRuntime) {
  // Execute one bounded representative case through the real multithreaded
  // runtime. The metadata test below validates the complete 72-case benchmark
  // corpus without materializing and executing every large timing workload as
  // part of the unit-test suite.
  const std::vector<std::string> failures = RunCpuBackendCases(
      "Gemm", core::backend_test::TestMode::BENCHMARK, "test_cpu_gemm_direct_float32_benchmark");
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
  bool has_direct_k32 = false;
  bool has_square_128 = false;
  bool has_square_1024 = false;
  bool has_skinny_m_small = false;
  bool has_skinny_n_small = false;
  bool has_large_k_1024 = false;
  bool has_split_k_16384 = false;
  bool has_transformer_decode = false;
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
    has_direct_k32 |= test_case.name.find("direct_k32") != std::string::npos;
    has_square_128 |= test_case.name.find("square_128") != std::string::npos;
    has_square_1024 |= test_case.name.find("square_1024") != std::string::npos;
    has_skinny_m_small |= test_case.name.find("skinny_m_small") != std::string::npos;
    has_skinny_n_small |= test_case.name.find("skinny_n_small") != std::string::npos;
    has_large_k_1024 |= test_case.name.find("large_k_1024") != std::string::npos;
    has_split_k_16384 |= test_case.name.find("split_k_16384") != std::string::npos;
    has_transformer_decode |=
        test_case.name.find("transformer_projection_decode") != std::string::npos;
  }
  // 24 prepared shapes, each registered for float32, float16 and bfloat16.
  EXPECT_EQ(cpu_cases, 72u);
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
  EXPECT_TRUE(has_direct_k32);
  EXPECT_TRUE(has_square_128);
  EXPECT_TRUE(has_square_1024);
  EXPECT_TRUE(has_skinny_m_small);
  EXPECT_TRUE(has_skinny_n_small);
  EXPECT_TRUE(has_large_k_1024);
  EXPECT_TRUE(has_split_k_16384);
  EXPECT_TRUE(has_transformer_decode);
}

} // namespace
