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

#include "onnx_light_cpu/backend_test/backend_correctness_runner.h"
#include "onnx_light_cpu/backend_test/collect_test_cases.h"
#include "onnx_light_cpu/impl/math/binary/binary_manifest.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"
#include "onnx_light_cpu/kernels/math/gemm_kernel.h"
#include "onnx_light_cpu/kernels/register_kernels.h"

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_core/runtime/kernels/tensor_compare.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
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

class TestCaseUnloadGuard {
public:
  explicit TestCaseUnloadGuard(TestCase &test_case) : test_case_(test_case) {}

  ~TestCaseUnloadGuard() {
    if (test_case_.materialized()) {
      test_case_.unload();
    }
  }

  TestCaseUnloadGuard(const TestCaseUnloadGuard &) = delete;
  TestCaseUnloadGuard &operator=(const TestCaseUnloadGuard &) = delete;

private:
  TestCase &test_case_;
};

const std::array<std::string_view, 7> kBinaryShapeTags = {
    "contiguous",   "left_scalar",     "right_scalar", "repeated_block",
    "inner_vector", "outer_broadcast", "general",
};

// TreeEnsemble's pre-existing v5 corpus names use the legacy "float" and
// "double" type tags.
const std::array<std::string_view, 21> kDataTypeNameTags = {
    "bool",   "float",  "float16", "bfloat16", "float32", "float64", "double",
    "int8",   "uint8",  "int16",   "uint16",   "int32",   "uint32",  "int64",
    "uint64", "string", "strings", "complex",  "float8",  "int4",    "uint4",
};

// Case names use 'x' to delimit paired input types, such as "int8xuint8".
bool IsDataTypeNameTag(std::string_view name, std::string_view tag) {
  if (tag.empty() || tag.size() > name.size()) {
    return false;
  }
  size_t position = name.find(tag);
  while (position != std::string_view::npos) {
    const size_t end = position + tag.size();
    const bool starts_tag = position == 0 || name[position - 1] == '_' || name[position - 1] == 'x';
    const bool ends_tag = end == name.size() || name[end] == '_' || name[end] == 'x';
    if (starts_tag && ends_tag) {
      return true;
    }
    position = name.find(tag, position + 1);
  }
  return false;
}

bool ContainsDataTypeName(std::string_view name) {
  return std::any_of(kDataTypeNameTags.begin(), kDataTypeNameTags.end(),
                     [name](std::string_view tag) { return IsDataTypeNameTag(name, tag); });
}

std::string Lowercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::string BinaryTypeSuffix(onnx_light_cpu::DataType type) {
  switch (type) {
  case onnx_light_cpu::DataType::BOOL:
    return "bool";
  case onnx_light_cpu::DataType::FLOAT:
    return "float32";
  case onnx_light_cpu::DataType::DOUBLE:
    return "float64";
  case onnx_light_cpu::DataType::FLOAT16:
    return "float16";
  case onnx_light_cpu::DataType::BFLOAT16:
    return "bfloat16";
  case onnx_light_cpu::DataType::INT8:
    return "int8";
  case onnx_light_cpu::DataType::INT16:
    return "int16";
  case onnx_light_cpu::DataType::INT32:
    return "int32";
  case onnx_light_cpu::DataType::INT64:
    return "int64";
  case onnx_light_cpu::DataType::UINT8:
    return "uint8";
  case onnx_light_cpu::DataType::UINT16:
    return "uint16";
  case onnx_light_cpu::DataType::UINT32:
    return "uint32";
  case onnx_light_cpu::DataType::UINT64:
    return "uint64";
  case onnx_light_cpu::DataType::STRING:
    return "string";
  default:
    throw std::invalid_argument("unsupported binary type");
  }
}

bool IsBinaryNonCommutative(onnx_light_cpu::BinaryOperator op) {
  using onnx_light_cpu::BinaryOperator;
  switch (op) {
  case BinaryOperator::kSub:
  case BinaryOperator::kDiv:
  case BinaryOperator::kPow:
  case BinaryOperator::kGreater:
  case BinaryOperator::kGreaterOrEqual:
  case BinaryOperator::kLess:
  case BinaryOperator::kLessOrEqual:
  case BinaryOperator::kBitShift:
    return true;
  default:
    return false;
  }
}

std::vector<TestCase> CollectCpuCases(const std::string &op_type,
                                      core::backend_test::TestMode mode) {
  onnx_light_cpu::backend_test::RegisterCpuKernelBackendTestCases();
  std::vector<TestCase> filtered;
  std::vector<TestCase> cases = CollectTestCases(op_type, /*include_big=*/false, mode);
  for (TestCase &tc : cases) {
    if (tc.name.rfind("test_cpu_", 0) == 0) {
      filtered.emplace_back(std::move(tc));
    }
  }
  return filtered;
}

std::int64_t DefaultModelOpsetVersion(const ModelProto &model) {
  for (const OperatorSetIdProto &opset : model.opset_import()) {
    if (opset.domain().empty() || opset.domain() == "ai.onnx") {
      return opset.version();
    }
  }
  throw std::invalid_argument("backend test model has no default-domain opset import");
}

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
    RuntimeContext rt(KernelContext(DefaultOpset(DefaultModelOpsetVersion(model))));
    RegisterModelFunctions(model, rt);

    std::vector<std::pair<std::string, Tensor>> bindings;
    bindings.reserve(ds.inputs.size());
    for (const Tensor &t : ds.inputs) {
      bindings.emplace_back(t.name, t);
    }

    SubgraphSession session(rt, graph);
    const Tensors outputs = session.Run(std::move(bindings), rt);

    if (!ds.expected_outputs_generated) {
      if (compare) {
        failures.push_back(tc.name + ": TEST case omitted expected outputs");
      }
      continue;
    }
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
std::vector<std::string>
RunCpuBackendCases(const std::string &op_type, core::backend_test::TestMode mode,
                   const std::string &case_name = {},
                   onnx_light_cpu::MicrosoftKernelImplementation microsoft_implementation =
                       onnx_light_cpu::MicrosoftKernelImplementation::OPTIMIZED) {
  onnx_light_cpu::backend_test::RegisterCpuKernelBackendTestCases();
  onnx_light_cpu::RegisterAllKernels(microsoft_implementation);

  const bool compare = mode == core::backend_test::TestMode::TEST;
  std::vector<std::string> failures;
  std::vector<TestCase> cases = CollectTestCases(op_type, /*include_big=*/false, mode);
  size_t cpu_cases = 0;
  for (TestCase &tc : cases) {
    if (tc.name.rfind("test_cpu_", 0) != 0 || (!case_name.empty() && tc.name != case_name)) {
      continue;
    }
    TestCaseUnloadGuard unload_guard(tc);
    ++cpu_cases;
    try {
      RunCaseThroughRuntime(tc, compare, failures);
    } catch (const std::exception &e) {
      failures.push_back(tc.name + ": " + e.what());
    }
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
  std::vector<TestCase> cases =
      CollectTestCases(op_type, /*include_big=*/false, core::backend_test::TestMode::TEST);
  size_t count = 0;
  for (TestCase &test_case : cases) {
    if (test_case.name.rfind("test_cpu_", 0) != 0) {
      continue;
    }
    TestCaseUnloadGuard unload_guard(test_case);
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

TEST(OnnxLightBackendKernels, AllRegisteredKernelsPassRegularBackendCorrectnessCorpus) {
  const onnx_light_cpu::backend_test::BackendCorrectnessReport report =
      onnx_light_cpu::backend_test::RunBackendCorrectnessTests();
  EXPECT_EQ(report.executed, report.passed) << report.Describe();
  EXPECT_TRUE(report.failed.empty()) << report.Describe();
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

TEST(OnnxLightBackendKernels, SwiGLURunsThroughRuntime) {
  const std::vector<std::string> failures =
      RunCpuBackendCases("SwiGLU", core::backend_test::TestMode::TEST);
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, BiasGeluRunsThroughRuntime) {
  const std::vector<std::string> failures =
      RunCpuBackendCases("BiasGelu", core::backend_test::TestMode::TEST);
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, CDistRunsThroughRuntime) {
  const std::vector<std::string> failures =
      RunCpuBackendCases("CDist", core::backend_test::TestMode::TEST);
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, GroupQueryAttentionRunsThroughRuntime) {
  const std::vector<std::string> failures =
      RunCpuBackendCases("GroupQueryAttention", core::backend_test::TestMode::TEST);
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, MicrosoftCorrectnessCasesRunWithNaiveRegistration) {
  for (const std::string &op_type : {"BiasGelu", "CDist", "GroupQueryAttention"}) {
    const auto failures = RunCpuBackendCases(op_type, core::backend_test::TestMode::TEST, {},
                                             onnx_light_cpu::MicrosoftKernelImplementation::NAIVE);
    EXPECT_TRUE(failures.empty()) << Describe(failures);
  }
  onnx_light_cpu::RegisterAllKernels();
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

TEST(OnnxLightBackendKernels, GemmCaseCollectionRetainsNoKernels) {
  const std::int64_t baseline = onnx_light_cpu::GemmKernel::ActiveInstanceCountForTesting();
  onnx_light_cpu::SetKernelUsageRecording(true);

  for (const core::backend_test::TestMode mode :
       {core::backend_test::TestMode::TEST, core::backend_test::TestMode::BENCHMARK}) {
    onnx_light_cpu::ClearUsedKernelNames();
    std::vector<TestCase> cases = CollectCpuCases("Gemm", mode);
    ASSERT_FALSE(cases.empty());
    EXPECT_EQ(onnx_light_cpu::GemmKernel::ActiveInstanceCountForTesting(), baseline);
    EXPECT_TRUE(onnx_light_cpu::UsedKernelNames().empty());
    for (const TestCase &test_case : cases) {
      EXPECT_FALSE(test_case.materialized()) << test_case.name;
    }
  }

  std::vector<TestCase> cases = CollectCpuCases("Gemm", core::backend_test::TestMode::TEST);
  const auto found = std::find_if(cases.begin(), cases.end(), [](const TestCase &test_case) {
    return test_case.name == "test_cpu_gemm_float32";
  });
  ASSERT_NE(found, cases.end());
  found->Materialize();
  EXPECT_EQ(onnx_light_cpu::GemmKernel::ActiveInstanceCountForTesting(), baseline);
  found->unload();
  EXPECT_EQ(onnx_light_cpu::GemmKernel::ActiveInstanceCountForTesting(), baseline);
}

TEST(OnnxLightBackendKernels, MatMulRunsThroughRuntime) {
  const std::vector<std::string> failures =
      RunCpuBackendCases("MatMul", core::backend_test::TestMode::TEST);
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, MatMulIntegerRunsThroughRuntime) {
  const std::vector<std::string> failures =
      RunCpuBackendCases("MatMulInteger", core::backend_test::TestMode::TEST);
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, RmsNormalizationRunsThroughRuntime) {
  const std::vector<std::string> failures =
      RunCpuBackendCases("RMSNormalization", core::backend_test::TestMode::TEST);
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, StandardNormalizationKernelsRunThroughRuntime) {
  std::vector<std::string> failures;
  for (const char *op_type :
       {"BatchNormalization", "GroupNormalization", "InstanceNormalization", "LayerNormalization",
        "LpNormalization", "MeanVarianceNormalization"}) {
    const std::vector<std::string> op_failures =
        RunCpuBackendCases(op_type, core::backend_test::TestMode::TEST);
    failures.insert(failures.end(), op_failures.begin(), op_failures.end());
  }
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, AttentionRunsThroughRuntime) {
  const std::vector<std::string> failures =
      RunCpuBackendCases("Attention", core::backend_test::TestMode::TEST);
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, AttentionPriorityBenchmarksRunThroughRuntime) {
  std::vector<std::string> failures;
  for (const std::string &name :
       {"test_cpu_attention_opset23_rank4_mha_q1_kv128_hd64_none_stateless_float16_benchmark",
        "test_cpu_attention_opset23_rank4_gqa_q128_kv128_hd64_causal_stateless_bfloat16_benchmark",
        "test_cpu_attention_opset23_rank3_mha_q128_kv128_hd64_none_stateless_float32_benchmark",
        "test_cpu_attention_opset23_rank4_mha_q1_kv1024_hd64_none_internal_cache_bfloat16_"
        "benchmark",
        "test_cpu_attention_opset24_rank4_mha_q8_kv1024_hd64_causal_nonpad_float32_benchmark",
        "test_cpu_attention_llm_qwen3_8b_opset23_rank3_gqa_q1_kv128_hd128_qh32_kvh8_causal_"
        "internal_cache_float16_benchmark",
        "test_cpu_attention_llm_qwen3_6_27b_opset23_rank3_gqa_q1_kv128_hd256_qh24_kvh4_causal_"
        "internal_cache_float16_benchmark",
        "test_cpu_attention_llm_qwen3_6_35b_a3b_opset23_rank3_gqa_q128_kv128_hd256_qh16_kvh2_"
        "causal_stateless_float16_benchmark"}) {
    const std::vector<std::string> case_failures =
        RunCpuBackendCases("Attention", core::backend_test::TestMode::BENCHMARK, name);
    failures.insert(failures.end(), case_failures.begin(), case_failures.end());
  }
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, BinaryElementwiseRunsThroughRuntime) {
  std::vector<std::string> failures;
  for (const auto &entry : onnx_light_cpu::GetBinaryManifest()) {
    const std::vector<std::string> op_failures =
        RunCpuBackendCases(std::string(entry.op_type), core::backend_test::TestMode::TEST);
    failures.insert(failures.end(), op_failures.begin(), op_failures.end());
  }
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, BinaryElementwiseCorpusCoversTypesShapesAndSwapCases) {
  for (const auto &entry : onnx_light_cpu::GetBinaryManifest()) {
    const std::vector<TestCase> cases =
        CollectCpuCases(std::string(entry.op_type), core::backend_test::TestMode::TEST);
    ASSERT_FALSE(cases.empty()) << entry.op_type;
    const std::string op_prefix = "test_cpu_" + Lowercase(std::string(entry.op_type)) + "_";
    for (std::string_view tag : kBinaryShapeTags) {
      EXPECT_TRUE(std::any_of(cases.begin(), cases.end(),
                              [&](const TestCase &test_case) {
                                return test_case.name.rfind(op_prefix, 0) == 0 &&
                                       test_case.name.find("_" + std::string(tag) + "_") !=
                                           std::string::npos;
                              }))
          << entry.op_type << " missing tag " << tag;
    }
    for (const auto &signature : entry.signatures) {
      const std::string type_tag = BinaryTypeSuffix(signature.left) + "x" +
                                   BinaryTypeSuffix(signature.right) + "_to_" +
                                   BinaryTypeSuffix(signature.output);
      EXPECT_TRUE(std::any_of(cases.begin(), cases.end(),
                              [&](const TestCase &test_case) {
                                return test_case.name.rfind(op_prefix, 0) == 0 &&
                                       test_case.name.find(type_tag) != std::string::npos;
                              }))
          << entry.op_type << " missing signature " << type_tag;
      if (IsBinaryNonCommutative(entry.op)) {
        EXPECT_TRUE(std::any_of(cases.begin(), cases.end(),
                                [&](const TestCase &test_case) {
                                  return test_case.name.rfind(op_prefix, 0) == 0 &&
                                         test_case.name.find(type_tag) != std::string::npos &&
                                         test_case.name.ends_with("_swapped");
                                }))
            << entry.op_type << " missing swapped signature " << type_tag;
      }
    }
    if (entry.op == onnx_light_cpu::BinaryOperator::kBitShift) {
      EXPECT_TRUE(std::any_of(cases.begin(), cases.end(), [](const TestCase &test_case) {
        return test_case.name.find("_left") != std::string::npos;
      }));
      EXPECT_TRUE(std::any_of(cases.begin(), cases.end(), [](const TestCase &test_case) {
        return test_case.name.find("_right") != std::string::npos;
      }));
    }
  }
}

TEST(OnnxLightBackendKernels, TreeEnsembleCorpusRegistersOnlyMlOpset5) {
  EXPECT_GE(CountCpuCasesAtMlOpset5("TreeEnsemble"), 36U);
  EXPECT_EQ(CountCpuCasesAtMlOpset5("TreeEnsembleRegressor"), 4U);
  EXPECT_EQ(CountCpuCasesAtMlOpset5("TreeEnsembleClassifier"), 2U);
}

TEST(OnnxLightBackendKernels, TreeEnsembleRunsThroughRuntime) {
  const std::vector<std::string> failures =
      RunCpuBackendCases("TreeEnsemble", core::backend_test::TestMode::TEST);
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, TreeEnsembleBenchmarkRunsThroughRuntime) {
  const std::vector<std::string> failures =
      RunCpuBackendCases("TreeEnsemble", core::backend_test::TestMode::BENCHMARK,
                         "test_cpu_treeensemble_t10_f4_b1_float32_benchmark");
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, TreeEnsembleBenchmarkCoversPriorityDimensions) {
  onnx_light_cpu::backend_test::RegisterCpuKernelBackendTestCases();
  const std::vector<TestCase> cases = CollectTestCases("TreeEnsemble", /*include_big=*/false,
                                                       core::backend_test::TestMode::BENCHMARK);
  std::set<std::string> names;
  for (const TestCase &test_case : cases) {
    if (test_case.name.rfind("test_cpu_treeensemble_", 0) == 0) {
      EXPECT_TRUE(names.insert(test_case.name).second) << test_case.name;
    }
  }
  EXPECT_EQ(names.size(), 5U);
  EXPECT_TRUE(names.contains("test_cpu_treeensemble_t10_f4_b1_float32_benchmark"));
  EXPECT_TRUE(names.contains("test_cpu_treeensemble_t100_f64_b8_float32_benchmark"));
  EXPECT_TRUE(names.contains("test_cpu_treeensemble_t1000_f1024_b32_float32_benchmark"));
  EXPECT_TRUE(names.contains("test_cpu_treeensemble_t10000_f4096_b1_float32_benchmark"));
  EXPECT_TRUE(names.contains("test_cpu_treeensemble_t10000_f4096_b128_float32_benchmark"));
}

TEST(OnnxLightBackendKernels, AttentionBenchmarkCoversPriorityCorpus) {
  onnx_light_cpu::backend_test::RegisterCpuKernelBackendTestCases();
  const std::vector<TestCase> cases =
      CollectTestCases("Attention", /*include_big=*/false, core::backend_test::TestMode::BENCHMARK);
  std::set<std::string> names;
  for (const TestCase &test_case : cases) {
    if (test_case.name.rfind("test_cpu_attention_", 0) == 0) {
      EXPECT_TRUE(names.insert(test_case.name).second) << test_case.name;
    }
  }
  EXPECT_EQ(names.size(), 69U);
  for (std::string_view name :
       {"test_cpu_attention_opset23_rank4_mha_q1_kv128_hd64_none_stateless_float16_benchmark",
        "test_cpu_attention_opset23_rank4_gqa_q128_kv128_hd64_causal_stateless_bfloat16_benchmark",
        "test_cpu_attention_opset23_rank3_mha_q128_kv128_hd64_none_stateless_float32_benchmark",
        "test_cpu_attention_llm_qwen3_8b_opset23_rank3_gqa_q128_kv128_hd128_qh32_kvh8_causal_"
        "stateless_float16_benchmark",
        "test_cpu_attention_llm_qwen3_8b_opset23_rank3_gqa_q1_kv128_hd128_qh32_kvh8_causal_"
        "internal_cache_float16_benchmark",
        "test_cpu_attention_llm_qwen3_8b_opset23_rank3_gqa_q1_kv1024_hd128_qh32_kvh8_causal_"
        "internal_cache_float16_benchmark",
        "test_cpu_attention_llm_qwen3_8b_opset23_rank3_gqa_q1_kv4096_hd128_qh32_kvh8_causal_"
        "internal_cache_float16_benchmark",
        "test_cpu_attention_llm_qwen3_6_27b_opset23_rank3_gqa_q128_kv128_hd256_qh24_kvh4_causal_"
        "stateless_float16_benchmark",
        "test_cpu_attention_llm_qwen3_6_27b_opset23_rank3_gqa_q1_kv4096_hd256_qh24_kvh4_causal_"
        "internal_cache_float16_benchmark",
        "test_cpu_attention_llm_qwen3_6_35b_a3b_opset23_rank3_gqa_q128_kv128_hd256_qh16_kvh2_"
        "causal_stateless_float16_benchmark",
        "test_cpu_attention_llm_qwen3_6_35b_a3b_opset23_rank3_gqa_q1_kv4096_hd256_qh16_kvh2_"
        "causal_internal_cache_float16_benchmark"}) {
    EXPECT_TRUE(names.contains(std::string(name))) << name;
  }
  for (std::string_view tag :
       {"_float32_benchmark", "_float16_benchmark", "_bfloat16_benchmark", "_rank3_", "_rank4_",
        "_mha_", "_gqa_", "_mqa_", "_hd64_", "_hd128_", "_none_", "_causal_", "_bool_",
        "_additive_", "_internal_cache_", "_nonpad_", "_hd256_", "_llm_qwen3_6_27b_",
        "_llm_qwen3_6_35b_a3b_"}) {
    EXPECT_TRUE(std::any_of(names.begin(), names.end(), [tag](const std::string &name) {
      return name.find(tag) != std::string::npos;
    })) << tag;
  }
  for (std::string_view tag : {"_q1_", "_q2_", "_q8_", "_q16_", "_q128_", "_q512_", "_kv1_",
                               "_kv128_", "_kv1024_", "_kv4096_", "_kv8192_"}) {
    EXPECT_TRUE(std::any_of(names.begin(), names.end(), [tag](const std::string &name) {
      return name.find(tag) != std::string::npos;
    })) << tag;
  }
}

// Guards the invariant fixed for GEMM benchmark cases (#369/#371): every
// TreeEnsemble/TreeEnsembleRegressor/TreeEnsembleClassifier backend test case
// name must be unique within its operator and must encode the attribute(s)
// that distinguish it from its siblings (value type / branch mode / aggregate
// / post-transform for the TreeEnsemble v5 corpus, the input element type for
// TreeEnsembleRegressor, and the label element type for
// TreeEnsembleClassifier).
TEST(OnnxLightBackendKernels, TreeEnsembleCaseNamesAreUniqueAndReflectAttributes) {
  onnx_light_cpu::backend_test::RegisterCpuKernelBackendTestCases();

  {
    const std::vector<TestCase> cases =
        CollectTestCases("TreeEnsemble", /*include_big=*/false, core::backend_test::TestMode::TEST);
    std::set<std::string> names;
    for (const TestCase &test_case : cases) {
      if (test_case.name.rfind("test_cpu_treeensemble_v5_", 0) != 0) {
        continue;
      }
      EXPECT_TRUE(names.insert(test_case.name).second) << test_case.name;
    }
    EXPECT_EQ(names.size(), 36U);
  }

  {
    const std::vector<TestCase> cases = CollectTestCases(
        "TreeEnsembleRegressor", /*include_big=*/false, core::backend_test::TestMode::TEST);
    std::set<std::string> names;
    for (const TestCase &test_case : cases) {
      if (test_case.name.rfind("test_cpu_treeensembleregressor_v5_", 0) != 0) {
        continue;
      }
      EXPECT_TRUE(names.insert(test_case.name).second) << test_case.name;
      const int dtype_tags = static_cast<int>(test_case.name.ends_with("_float")) +
                             static_cast<int>(test_case.name.ends_with("_double")) +
                             static_cast<int>(test_case.name.ends_with("_int32")) +
                             static_cast<int>(test_case.name.ends_with("_int64"));
      EXPECT_EQ(dtype_tags, 1) << test_case.name;
    }
    EXPECT_EQ(names.size(), 4U);
  }

  {
    const std::vector<TestCase> cases = CollectTestCases(
        "TreeEnsembleClassifier", /*include_big=*/false, core::backend_test::TestMode::TEST);
    std::set<std::string> names;
    for (const TestCase &test_case : cases) {
      if (test_case.name.rfind("test_cpu_treeensembleclassifier_v5_", 0) != 0) {
        continue;
      }
      EXPECT_TRUE(names.insert(test_case.name).second) << test_case.name;
      const int label_tags = static_cast<int>(test_case.name.ends_with("_strings")) +
                             static_cast<int>(test_case.name.ends_with("_int64"));
      EXPECT_EQ(label_tags, 1) << test_case.name;
    }
    EXPECT_EQ(names.size(), 2U);
  }
}

// Every onnx-light-cpu backend test case name (across every registered
// operator, not just TreeEnsemble) must be unique within its ``TestMode``:
// onnx-light's registry keys cases by name alone, so a collision between two
// operators would silently shadow one of the cases.
TEST(OnnxLightBackendKernels, AllCpuBackendCaseNamesAreGloballyUnique) {
  onnx_light_cpu::backend_test::RegisterCpuKernelBackendTestCases();
  for (core::backend_test::TestMode mode :
       {core::backend_test::TestMode::TEST, core::backend_test::TestMode::BENCHMARK}) {
    const std::vector<TestCase> cases =
        CollectTestCases(/*op_type=*/"", /*include_big=*/false, mode);
    std::vector<std::string> cpu_case_names;
    for (const TestCase &test_case : cases) {
      if (test_case.name.rfind("test_cpu_", 0) != 0) {
        continue;
      }
      cpu_case_names.push_back(test_case.name);
    }
    EXPECT_GT(cpu_case_names.size(), 0U);

    const std::set<std::string> names(cpu_case_names.begin(), cpu_case_names.end());
    if (names.size() != cpu_case_names.size()) {
      // Only pay the per-name insert cost to pinpoint the offending
      // duplicate(s) when a collision was actually detected above.
      std::set<std::string> seen;
      for (const std::string &name : cpu_case_names) {
        EXPECT_TRUE(seen.insert(name).second) << name;
      }
    }
  }
}

TEST(OnnxLightBackendKernels, AllCpuBackendCaseNamesContainDataTypeName) {
  onnx_light_cpu::backend_test::RegisterCpuKernelBackendTestCases();
  for (core::backend_test::TestMode mode :
       {core::backend_test::TestMode::TEST, core::backend_test::TestMode::BENCHMARK}) {
    const std::vector<TestCase> cases =
        CollectTestCases(/*op_type=*/"", /*include_big=*/false, mode);
    for (const TestCase &test_case : cases) {
      if (test_case.name.starts_with("test_cpu_")) {
        EXPECT_TRUE(ContainsDataTypeName(test_case.name)) << test_case.name;
      }
    }
  }
}

// Benchmark-mode cases are registered per kernel alongside the correctness
// cases. The runtime tests execute one bounded representative per corpus; the
// metadata tests below cover every large timing workload without materializing
// all of them in the unit-test suite.
TEST(OnnxLightBackendKernels, AbsBenchmarkRunsThroughRuntime) {
  const std::vector<std::string> failures = RunCpuBackendCases(
      "Abs", core::backend_test::TestMode::BENCHMARK, "test_cpu_abs_n1024_float32_benchmark");
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, ExpBenchmarkRunsThroughRuntime) {
  const std::vector<std::string> failures = RunCpuBackendCases(
      "Exp", core::backend_test::TestMode::BENCHMARK, "test_cpu_exp_n1024_float32_benchmark");
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, LogBenchmarkRunsThroughRuntime) {
  const std::vector<std::string> failures = RunCpuBackendCases(
      "Log", core::backend_test::TestMode::BENCHMARK, "test_cpu_log_n1024_float32_benchmark");
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, SwiGLUBenchmarkRunsThroughRuntime) {
  const std::vector<std::string> failures = RunCpuBackendCases(
      "SwiGLU", core::backend_test::TestMode::BENCHMARK, "test_cpu_swiglu_n1024_float32_benchmark");
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, BiasGeluBenchmarkRunsThroughRuntime) {
  const std::vector<std::string> failures =
      RunCpuBackendCases("BiasGelu", core::backend_test::TestMode::BENCHMARK,
                         "test_cpu_biasgelu_o4096_i256_float32_benchmark");
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, CDistBenchmarkRunsThroughRuntime) {
  const std::vector<std::string> failures =
      RunCpuBackendCases("CDist", core::backend_test::TestMode::BENCHMARK,
                         "test_cpu_cdist_m64_k64_n64_sqeuclidean_float32_benchmark");
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, NotBenchmarkRunsThroughRuntime) {
  const std::vector<std::string> failures = RunCpuBackendCases(
      "Not", core::backend_test::TestMode::BENCHMARK, "test_cpu_not_n1024_bool_benchmark");
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, BinaryBenchmarkCorporaAreUnmaterializedAndRunThroughRuntime) {
  std::vector<std::string> failures;
  for (const auto &entry : onnx_light_cpu::GetBinaryManifest()) {
    const std::vector<TestCase> cases =
        CollectCpuCases(std::string(entry.op_type), core::backend_test::TestMode::BENCHMARK);
    ASSERT_FALSE(cases.empty()) << entry.op_type;
    for (const TestCase &test_case : cases) {
      EXPECT_FALSE(test_case.materialized()) << test_case.name;
      EXPECT_TRUE(test_case.name.ends_with("_benchmark")) << test_case.name;
    }
    const auto representative =
        std::find_if(cases.begin(), cases.end(), [](const TestCase &test_case) {
          return test_case.name.find("_row_") != std::string::npos;
        });
    ASSERT_NE(representative, cases.end()) << entry.op_type;
    const std::vector<std::string> op_failures = RunCpuBackendCases(
        std::string(entry.op_type), core::backend_test::TestMode::BENCHMARK, representative->name);
    failures.insert(failures.end(), op_failures.begin(), op_failures.end());
  }
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, BinaryBenchmarkCorporaCoverEverySignatureAndPriorityShape) {
  const std::set<std::string> expected_shape_tags = {
      "contiguous", "left_scalar", "right_scalar", "row", "per_channel", "outer", "general",
  };
  const std::set<int64_t> expected_output_sizes = {
      4096,
      65536,
      1048576,
      4194304,
  };
  for (const auto &entry : onnx_light_cpu::GetBinaryManifest()) {
    const std::vector<TestCase> cases =
        CollectCpuCases(std::string(entry.op_type), core::backend_test::TestMode::BENCHMARK);
    std::set<std::string> shape_tags;
    std::set<int64_t> output_sizes;
    for (const TestCase &test_case : cases) {
      if (test_case.name.find("_llm_") != std::string::npos) {
        continue;
      }
      for (const std::string &shape_tag : expected_shape_tags) {
        if (test_case.name.find("_" + shape_tag + "_") != std::string::npos) {
          shape_tags.insert(shape_tag);
        }
      }
      ASSERT_EQ(test_case.declared_output_element_counts.size(), 1u) << test_case.name;
      output_sizes.insert(test_case.declared_output_element_counts[0]);
    }
    EXPECT_EQ(shape_tags, expected_shape_tags) << entry.op_type;
    EXPECT_EQ(output_sizes, expected_output_sizes) << entry.op_type;
    for (const auto &signature : entry.signatures) {
      if (signature.left == onnx_light_cpu::DataType::STRING) {
        continue;
      }
      const std::string signature_tag = "_" + BinaryTypeSuffix(signature.left) + "x" +
                                        BinaryTypeSuffix(signature.right) + "_to_" +
                                        BinaryTypeSuffix(signature.output);
      for (const std::string &shape_tag : expected_shape_tags) {
        EXPECT_TRUE(std::any_of(cases.begin(), cases.end(),
                                [&](const TestCase &test_case) {
                                  return test_case.name.find("_" + shape_tag + signature_tag) !=
                                         std::string::npos;
                                }))
            << entry.op_type << ": " << signature_tag << ", " << shape_tag;
      }
    }
  }
}

TEST(OnnxLightBackendKernels, BinaryArithmeticBenchmarksRunThroughRuntime) {
  const std::vector<std::pair<std::string, std::string>> cases = {
      {"Div", "test_cpu_div_v14_contiguous_float32xfloat32_to_float32_n4096_benchmark"},
      {"Sub", "test_cpu_sub_v14_contiguous_float32xfloat32_to_float32_n4096_benchmark"},
      {"Mod", "test_cpu_mod_v13_contiguous_float32xfloat32_to_float32_fmod1_n4096_benchmark"},
  };
  for (const auto &[op_type, case_name] : cases) {
    const std::vector<std::string> failures =
        RunCpuBackendCases(op_type, core::backend_test::TestMode::BENCHMARK, case_name);
    EXPECT_TRUE(failures.empty()) << case_name << ": " << Describe(failures);
  }
}

TEST(OnnxLightBackendKernels, BinaryBenchmarkCorporaCoverQwen3LlmWorkloads) {
  const std::vector<std::string> mul_names = {
      "test_cpu_mul_llm_qwen3_8b_v14_contiguous_s1_h12288_float16xfloat16_to_float16_n12288_"
      "benchmark",
      "test_cpu_mul_llm_qwen3_8b_v14_contiguous_s128_h12288_float16xfloat16_to_float16_n1572864_"
      "benchmark",
      "test_cpu_mul_llm_qwen3_8b_v14_contiguous_s512_h12288_float16xfloat16_to_float16_n6291456_"
      "benchmark",
  };
  const std::vector<std::string> sub_names = {
      "test_cpu_sub_llm_qwen3_8b_v14_right_scalar_batch1_int64xint64_to_int64_n1_benchmark",
  };
  for (const auto &[op_type, expected_names] :
       {std::pair<std::string_view, const std::vector<std::string> &>{"Mul", mul_names},
        {"Sub", sub_names}}) {
    const std::vector<TestCase> cases =
        CollectCpuCases(std::string(op_type), core::backend_test::TestMode::BENCHMARK);
    for (const std::string &name : expected_names) {
      const auto found =
          std::find_if(cases.begin(), cases.end(),
                       [&name](const TestCase &test_case) { return test_case.name == name; });
      ASSERT_NE(found, cases.end()) << name;
      EXPECT_FALSE(found->materialized()) << name;
    }
    const std::vector<std::string> failures = RunCpuBackendCases(
        std::string(op_type), core::backend_test::TestMode::BENCHMARK, expected_names.front());
    EXPECT_TRUE(failures.empty()) << Describe(failures);
  }
}

TEST(OnnxLightBackendKernels, UnaryBenchmarkCorporaCoverParallelThresholds) {
  onnx_light_cpu::backend_test::RegisterCpuKernelBackendTestCases();
  std::set<int64_t> exp_sizes;
  for (const std::string &op : {"Abs", "Exp", "Log"}) {
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
    }
  }
}

TEST(OnnxLightBackendKernels, UnaryBenchmarkCorporaCoverEverySupportedType) {
  onnx_light_cpu::backend_test::RegisterCpuKernelBackendTestCases();
  const std::vector<std::pair<std::string, std::set<std::string>>> expected = {
      {"Abs", {"float32", "float64", "int8", "int16", "int32", "int64", "float16", "bfloat16"}},
      {"Exp", {"float32", "float64", "float16", "bfloat16"}},
      {"Log", {"float32", "float64", "float16", "bfloat16"}},
      {"Not", {"bool"}},
  };
  for (const auto &[op, suffixes] : expected) {
    const std::vector<TestCase> cases =
        CollectTestCases(op, /*include_big=*/false, core::backend_test::TestMode::BENCHMARK);
    const std::size_t cpu_case_count =
        static_cast<std::size_t>(std::count_if(cases.begin(), cases.end(), [](const TestCase &tc) {
          return tc.name.rfind("test_cpu_", 0) == 0;
        }));
    EXPECT_EQ(cpu_case_count, suffixes.size() * 7) << op;
    for (const std::string &suffix : suffixes) {
      EXPECT_TRUE(std::any_of(cases.begin(), cases.end(),
                              [&suffix](const TestCase &test_case) {
                                return test_case.name.rfind("test_cpu_", 0) == 0 &&
                                       test_case.name.find("_" + suffix + "_benchmark") !=
                                           std::string::npos;
                              }))
          << op << ": " << suffix;
    }
  }
}

TEST(OnnxLightBackendKernels, MatMulBenchmarkCorporaCoverTypesAndPriorityShapes) {
  onnx_light_cpu::backend_test::RegisterCpuKernelBackendTestCases();
  const std::vector<TestCase> matmul =
      CollectTestCases("MatMul", /*include_big=*/false, core::backend_test::TestMode::BENCHMARK);
  EXPECT_EQ(
      std::count_if(matmul.begin(), matmul.end(),
                    [](const TestCase &tc) { return tc.name.rfind("test_cpu_matmul_", 0) == 0; }),
      33);
  for (const std::string &type : {"float32", "float64", "float16", "bfloat16"}) {
    for (const std::string &shape :
         {"square_64", "square_1024", "skinny_m", "skinny_n", "large_k"}) {
      const std::string expected = "test_cpu_matmul_" + shape + "_" + type + "_benchmark";
      EXPECT_TRUE(std::any_of(matmul.begin(), matmul.end(), [&expected](const TestCase &test_case) {
        return test_case.name == expected;
      })) << expected;
    }
  }
  for (const std::string &name :
       {"test_cpu_matmul_llm_qwen3_8b_qkv_m1_k4096_n6144_float16_benchmark",
        "test_cpu_matmul_llm_qwen3_8b_o_proj_m1_k4096_n4096_float16_benchmark",
        "test_cpu_matmul_llm_qwen3_8b_gate_up_m1_k4096_n12288_float16_benchmark",
        "test_cpu_matmul_llm_qwen3_8b_down_m1_k12288_n4096_float16_benchmark",
        "test_cpu_matmul_llm_qwen3_8b_lm_head_m1_k4096_n151936_float16_benchmark"}) {
    const auto found =
        std::find_if(matmul.begin(), matmul.end(),
                     [&name](const TestCase &test_case) { return test_case.name == name; });
    ASSERT_NE(found, matmul.end()) << name;
    EXPECT_FALSE(found->materialized()) << name;
  }

  const std::vector<TestCase> integer = CollectTestCases("MatMulInteger", /*include_big=*/false,
                                                         core::backend_test::TestMode::BENCHMARK);
  EXPECT_EQ(std::count_if(integer.begin(), integer.end(),
                          [](const TestCase &tc) {
                            return tc.name.rfind("test_cpu_matmulinteger_", 0) == 0;
                          }),
            20);
  for (const std::string &types : {"int8xint8", "int8xuint8", "uint8xint8", "uint8xuint8"}) {
    for (const std::string &shape : {"square_64", "square_512", "skinny_m", "large_k"}) {
      const std::string expected = "test_cpu_matmulinteger_" + shape + "_" + types + "_benchmark";
      EXPECT_TRUE(std::any_of(
          integer.begin(), integer.end(),
          [&expected](const TestCase &test_case) { return test_case.name == expected; }))
          << expected;
    }
  }
}

TEST(OnnxLightBackendKernels, RmsNormalizationBenchmarksCoverQwen3LlmWorkloads) {
  onnx_light_cpu::backend_test::RegisterCpuKernelBackendTestCases();
  const std::vector<TestCase> cases = CollectTestCases("RMSNormalization", /*include_big=*/false,
                                                       core::backend_test::TestMode::BENCHMARK);
  const std::vector<std::string> expected_names = {
      "test_cpu_rms_normalization_llm_qwen3_8b_hidden4096_s1_float16_benchmark",
      "test_cpu_rms_normalization_llm_qwen3_8b_hidden4096_s128_float16_benchmark",
      "test_cpu_rms_normalization_llm_qwen3_8b_q_norm_hd128_s1_qh32_float16_benchmark",
      "test_cpu_rms_normalization_llm_qwen3_8b_q_norm_hd128_s128_qh32_float16_benchmark",
      "test_cpu_rms_normalization_llm_qwen3_8b_k_norm_hd128_s1_kvh8_float16_benchmark",
      "test_cpu_rms_normalization_llm_qwen3_8b_k_norm_hd128_s128_kvh8_float16_benchmark",
  };
  for (const std::string &name : expected_names) {
    const auto found = std::find_if(cases.begin(), cases.end(), [&name](const TestCase &test_case) {
      return test_case.name == name;
    });
    ASSERT_NE(found, cases.end()) << name;
    EXPECT_FALSE(found->materialized()) << name;
  }
  const std::vector<std::string> failures = RunCpuBackendCases(
      "RMSNormalization", core::backend_test::TestMode::BENCHMARK, expected_names.front());
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, NormalizationBenchmarksAreUnmaterializedAndRunThroughRuntime) {
  onnx_light_cpu::backend_test::RegisterCpuKernelBackendTestCases();
  struct ExpectedCorpus {
    const char *op_type;
    std::vector<std::string> names;
  };
  const std::vector<ExpectedCorpus> expected = {
      {"BatchNormalization",
       {"test_cpu_batchnormalization_n32_c128_rank2_float32_benchmark",
        "test_cpu_batchnormalization_n8_c16_l64_rank3_float32_benchmark",
        "test_cpu_batchnormalization_n4_c32_h16_w16_rank4_float32_benchmark",
        "test_cpu_batchnormalization_n8_c64_h8_w8_rank4_float32_benchmark",
        "test_cpu_batchnormalization_n2_c16_d4_h8_w8_rank5_float32_benchmark",
        "test_cpu_batchnormalization_training_n4_c32_h8_w8_rank4_float32_benchmark"}},
      {"GroupNormalization",
       {"test_cpu_groupnormalization_n2_c16_g4_l64_rank3_float32_benchmark",
        "test_cpu_groupnormalization_n4_c32_g8_h16_w16_rank4_float32_benchmark",
        "test_cpu_groupnormalization_n1_c64_g32_h8_w8_rank4_float32_benchmark",
        "test_cpu_groupnormalization_n2_c24_g6_d4_h8_w8_rank5_float32_benchmark"}},
      {"InstanceNormalization",
       {"test_cpu_instancenormalization_n8_c16_l128_rank3_float32_benchmark",
        "test_cpu_instancenormalization_n4_c32_h16_w16_rank4_float32_benchmark",
        "test_cpu_instancenormalization_n2_c8_h64_w8_rank4_float32_benchmark",
        "test_cpu_instancenormalization_n2_c16_d4_h8_w8_rank5_float32_benchmark"}},
      {"LayerNormalization",
       {"test_cpu_layernormalization_small_r256_w128_axis1_float32_benchmark",
        "test_cpu_layernormalization_wide_r1_w4096_axis1_float32_benchmark",
        "test_cpu_layernormalization_llm_r128_w4096_axis1_float32_benchmark",
        "test_cpu_layernormalization_tall_r4096_w128_axis1_float32_benchmark",
        "test_cpu_layernormalization_rank3_n8_s16_w512_axis2_float32_benchmark",
        "test_cpu_layernormalization_rank3_n4_d8_w64_axis1_float32_benchmark"}},
      {"LpNormalization",
       {"test_cpu_lpnormalization_r256_w128_axis1_p2_float32_benchmark",
        "test_cpu_lpnormalization_b16_r32_w128_axis2_p2_float32_benchmark",
        "test_cpu_lpnormalization_b8_r64_w32_axis1_p2_float32_benchmark",
        "test_cpu_lpnormalization_n4_c8_h16_w16_axis1_p1_float32_benchmark",
        "test_cpu_lpnormalization_wide_r16_w4096_axis1_p1_float32_benchmark"}},
      {"MeanVarianceNormalization",
       {"test_cpu_meanvariancenormalization_r256_w128_axes1_rank2_float32_benchmark",
        "test_cpu_meanvariancenormalization_r64_w32_axes0_1_rank2_float32_benchmark",
        "test_cpu_meanvariancenormalization_n8_c32_l128_axes0_2_rank3_float32_benchmark",
        "test_cpu_meanvariancenormalization_n8_c32_h16_w16_default_rank4_float32_benchmark",
        "test_cpu_meanvariancenormalization_n4_c16_h32_w8_axes2_3_rank4_float32_benchmark",
        "test_cpu_meanvariancenormalization_n2_c8_d4_h8_w8_axes0_2_3_4_rank5_float32_benchmark"}},
      {"RMSNormalization",
       {"test_cpu_rms_normalization_small_r8_w32_axis1_float32_benchmark",
        "test_cpu_rms_normalization_r64_w512_float32_benchmark",
        "test_cpu_rms_normalization_wide_r1_w4096_axis1_float32_benchmark",
        "test_cpu_rms_normalization_llm_r128_w4096_axis1_float32_benchmark",
        "test_cpu_rms_normalization_tall_r4096_w128_axis1_float32_benchmark",
        "test_cpu_rms_normalization_rank3_n4_r16_w512_axis2_float32_benchmark",
        "test_cpu_rms_normalization_rank3_n2_h8_w16_axis1_float32_benchmark",
        "test_cpu_rms_normalization_rank4_n2_c4_h8_w16_axis2_float32_benchmark"}},
  };
  std::vector<std::string> failures;
  for (const ExpectedCorpus &corpus : expected) {
    std::vector<TestCase> cases = CollectTestCases(corpus.op_type, /*include_big=*/false,
                                                   core::backend_test::TestMode::BENCHMARK);
    std::set<std::string> actual_names;
    for (TestCase &test_case : cases) {
      if (test_case.name.rfind("test_cpu_", 0) != 0 ||
          !test_case.name.ends_with("_float32_benchmark")) {
        continue;
      }
      TestCaseUnloadGuard unload_guard(test_case);
      actual_names.insert(test_case.name);
      EXPECT_FALSE(test_case.materialized()) << test_case.name;
      const GraphProto &graph = test_case.model().ref_graph();
      EXPECT_EQ(graph.initializer_size(), 0) << test_case.name;
      ASSERT_EQ(test_case.data_sets().size(), 1u) << test_case.name;
      EXPECT_EQ(graph.input_size(), test_case.data_sets()[0].inputs.size()) << test_case.name;
    }
    EXPECT_EQ(actual_names, std::set<std::string>(corpus.names.begin(), corpus.names.end()))
        << corpus.op_type;
    const std::vector<std::string> op_failures = RunCpuBackendCases(
        corpus.op_type, core::backend_test::TestMode::BENCHMARK, corpus.names.front());
    failures.insert(failures.end(), op_failures.begin(), op_failures.end());
  }
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, NormalizationCasesCoverRequiredFloatingPointTypes) {
  onnx_light_cpu::backend_test::RegisterCpuKernelBackendTestCases();
  struct ExpectedTypes {
    const char *op_type;
    const char *test_prefix;
    const char *float16_benchmark;
    const char *bfloat16_benchmark;
  };
  const std::vector<ExpectedTypes> expected = {
      {"BatchNormalization", "test_cpu_batchnormalization_small_",
       "test_cpu_batchnormalization_n4_c16_h8_w8_rank4_float16_benchmark",
       "test_cpu_batchnormalization_n4_c16_h8_w8_rank4_bfloat16_benchmark"},
      {"GroupNormalization", "test_cpu_groupnormalization_small_",
       "test_cpu_groupnormalization_n2_c16_g4_h8_w8_rank4_float16_benchmark",
       "test_cpu_groupnormalization_n2_c16_g4_h8_w8_rank4_bfloat16_benchmark"},
      {"InstanceNormalization", "test_cpu_instancenormalization_small_",
       "test_cpu_instancenormalization_n2_c16_h8_w8_rank4_float16_benchmark",
       "test_cpu_instancenormalization_n2_c16_h8_w8_rank4_bfloat16_benchmark"},
      {"LayerNormalization", "test_cpu_layernormalization_axis1_",
       "test_cpu_layernormalization_r64_w512_axis1_float16_benchmark",
       "test_cpu_layernormalization_r64_w512_axis1_bfloat16_benchmark"},
      {"LpNormalization", "test_cpu_lpnormalization_axis0_p1_",
       "test_cpu_lpnormalization_b8_r16_w128_axis2_p2_float16_benchmark",
       "test_cpu_lpnormalization_b8_r16_w128_axis2_p2_bfloat16_benchmark"},
      {"MeanVarianceNormalization", "test_cpu_meanvariancenormalization_axis1_",
       "test_cpu_meanvariancenormalization_n4_c16_h8_w8_default_rank4_float16_benchmark",
       "test_cpu_meanvariancenormalization_n4_c16_h8_w8_default_rank4_bfloat16_benchmark"},
      {"RMSNormalization", "test_cpu_rms_normalization_small_",
       "test_cpu_rms_normalization_llm_qwen3_8b_hidden4096_s1_float16_benchmark",
       "test_cpu_rms_normalization_r64_w512_bfloat16_benchmark"},
  };
  for (const ExpectedTypes &entry : expected) {
    const std::vector<TestCase> tests =
        CollectTestCases(entry.op_type, /*include_big=*/false, core::backend_test::TestMode::TEST);
    const std::set<std::string> test_names = [&]() {
      std::set<std::string> names;
      for (const TestCase &test_case : tests) {
        names.insert(test_case.name);
      }
      return names;
    }();
    for (const char *suffix : {"float32", "float16", "bfloat16"}) {
      EXPECT_TRUE(test_names.contains(std::string(entry.test_prefix) + suffix))
          << entry.op_type << " " << suffix;
    }
    if (std::string_view(entry.op_type) == "RMSNormalization") {
      EXPECT_TRUE(test_names.contains("test_cpu_rms_normalization_mixed_x_float16_scale_float32"));
      EXPECT_TRUE(test_names.contains("test_cpu_rms_normalization_mixed_x_float32_scale_float16"));
    } else if (std::string_view(entry.op_type) == "GroupNormalization") {
      EXPECT_TRUE(test_names.contains("test_cpu_groupnormalization_opset18_group_affine_float32"));
    }

    const std::vector<TestCase> benchmarks = CollectTestCases(
        entry.op_type, /*include_big=*/false, core::backend_test::TestMode::BENCHMARK);
    const auto has_benchmark = [&](const char *name) {
      return std::any_of(benchmarks.begin(), benchmarks.end(),
                         [&](const TestCase &test_case) { return test_case.name == name; });
    };
    EXPECT_TRUE(has_benchmark(entry.float16_benchmark)) << entry.float16_benchmark;
    EXPECT_TRUE(has_benchmark(entry.bfloat16_benchmark)) << entry.bfloat16_benchmark;
  }
}

TEST(OnnxLightBackendKernels, MatMulBenchmarksRunThroughRuntime) {
  std::vector<std::string> failures =
      RunCpuBackendCases("MatMul", core::backend_test::TestMode::BENCHMARK,
                         "test_cpu_matmul_square_64_float32_benchmark");
  const std::vector<std::string> llm_failures =
      RunCpuBackendCases("MatMul", core::backend_test::TestMode::BENCHMARK,
                         "test_cpu_matmul_llm_qwen3_8b_qkv_m1_k4096_n6144_float16_benchmark");
  failures.insert(failures.end(), llm_failures.begin(), llm_failures.end());
  const std::vector<std::string> integer_failures =
      RunCpuBackendCases("MatMulInteger", core::backend_test::TestMode::BENCHMARK,
                         "test_cpu_matmulinteger_square_64_int8xuint8_benchmark");
  failures.insert(failures.end(), integer_failures.begin(), integer_failures.end());
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, GemmBenchmarkRunsThroughRuntime) {
  // Execute one bounded representative case through the real multithreaded
  // runtime. The metadata test below validates the complete 106-case benchmark
  // corpus without materializing and executing every large timing workload as
  // part of the unit-test suite.
  const std::vector<std::string> failures =
      RunCpuBackendCases("Gemm", core::backend_test::TestMode::BENCHMARK,
                         "test_cpu_gemm_direct_float32_transA_0_transB_0_bias_none_benchmark");
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, GemmBenchmarkCorpusIsUnmaterializedAndCoversPriorityShapes) {
  onnx_light_cpu::backend_test::RegisterCpuKernelBackendTestCases();
  std::vector<TestCase> cases =
      CollectTestCases("Gemm", /*include_big=*/false, core::backend_test::TestMode::BENCHMARK);

  size_t cpu_cases = 0;
  bool checked_runtime_inputs = false;
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
  bool has_float64 = false;
  bool has_float16 = false;
  bool has_bfloat16 = false;
  bool has_direct_k32 = false;
  bool has_square_128 = false;
  bool has_square_1024 = false;
  bool has_square_2048 = false;
  bool has_square_4096 = false;
  bool has_skinny_m_small = false;
  bool has_skinny_n_small = false;
  bool has_large_k_1024 = false;
  bool has_large_k_16384 = false;
  bool has_split_k_16384 = false;
  bool has_transformer_decode = false;
  size_t chained_cases = 0;
  std::set<std::string> names;
  for (TestCase &test_case : cases) {
    if (test_case.name.rfind("test_cpu_gemm_", 0) != 0) {
      continue;
    }
    TestCaseUnloadGuard unload_guard(test_case);
    if (test_case.name.find("_gemm_chain_") != std::string::npos) {
      ++chained_cases;
      EXPECT_NE(test_case.name.find("_float32"), std::string::npos) << test_case.name;
      EXPECT_FALSE(test_case.materialized()) << test_case.name;
      EXPECT_EQ(test_case.declared_input_element_counts.size(), 4u);
      EXPECT_EQ(test_case.declared_output_element_counts.size(), 1u);
      continue;
    }
    ++cpu_cases;
    EXPECT_TRUE(names.insert(test_case.name).second) << test_case.name;
    EXPECT_NE(test_case.name,
              "test_cpu_gemm_square_2048_float16_transA_0_transB_0_bias_none_benchmark");
    EXPECT_NE(test_case.name,
              "test_cpu_gemm_square_4096_float16_transA_0_transB_0_bias_none_benchmark");
    EXPECT_FALSE(test_case.materialized());
    EXPECT_TRUE(test_case.name.ends_with("_benchmark"));
    if (test_case.name.find("tiny_dynamic_float32") != std::string::npos) {
      checked_runtime_inputs = true;
      const GraphProto &graph = test_case.model().ref_graph();
      EXPECT_EQ(graph.initializer_size(), 0);
      EXPECT_EQ(graph.input_size(), 2);
      ASSERT_EQ(test_case.data_sets().size(), 1u);
      EXPECT_EQ(test_case.data_sets()[0].inputs.size(), 2u);
    }
    if (test_case.name.find("_bias_none_") != std::string::npos) {
      EXPECT_EQ(test_case.declared_input_element_counts.size(), 2u);
    }
    has_direct |= test_case.name.find("_direct_") != std::string::npos;
    has_skinny_m |= test_case.name.find("skinny_m") != std::string::npos;
    has_skinny_n |= test_case.name.find("skinny_n") != std::string::npos;
    has_large_k |= test_case.name.find("large_k") != std::string::npos;
    has_split_k |= test_case.name.find("split_k") != std::string::npos;
    has_transpose |= test_case.name.find("_transA_1_") != std::string::npos ||
                     test_case.name.find("_transB_1_") != std::string::npos;
    has_scalar_bias |= test_case.name.find("_bias_scalar_") != std::string::npos;
    has_row_bias |= test_case.name.find("_bias_row_") != std::string::npos;
    has_column_bias |= test_case.name.find("_bias_column_") != std::string::npos;
    has_matrix_bias |= test_case.name.find("_bias_matrix_") != std::string::npos;
    const int bias_tags =
        static_cast<int>(test_case.name.find("_bias_none_") != std::string::npos) +
        static_cast<int>(test_case.name.find("_bias_scalar_") != std::string::npos) +
        static_cast<int>(test_case.name.find("_bias_row_") != std::string::npos) +
        static_cast<int>(test_case.name.find("_bias_column_") != std::string::npos) +
        static_cast<int>(test_case.name.find("_bias_matrix_") != std::string::npos);
    EXPECT_EQ(bias_tags, 1) << test_case.name;
    const int trans_a_tags =
        static_cast<int>(test_case.name.find("_transA_0_") != std::string::npos) +
        static_cast<int>(test_case.name.find("_transA_1_") != std::string::npos);
    const int trans_b_tags =
        static_cast<int>(test_case.name.find("_transB_0_") != std::string::npos) +
        static_cast<int>(test_case.name.find("_transB_1_") != std::string::npos);
    EXPECT_EQ(trans_a_tags, 1) << test_case.name;
    EXPECT_EQ(trans_b_tags, 1) << test_case.name;
    has_float32 |= test_case.name.find("_float32_") != std::string::npos;
    has_float64 |= test_case.name.find("_float64_") != std::string::npos;
    has_float16 |= test_case.name.find("_float16_") != std::string::npos;
    has_bfloat16 |= test_case.name.find("_bfloat16_") != std::string::npos;
    has_direct_k32 |= test_case.name.find("direct_k32") != std::string::npos;
    has_square_128 |= test_case.name.find("square_128") != std::string::npos;
    has_square_1024 |= test_case.name.find("square_1024") != std::string::npos;
    has_square_2048 |= test_case.name.find("square_2048") != std::string::npos;
    has_square_4096 |= test_case.name.find("square_4096") != std::string::npos;
    has_skinny_m_small |= test_case.name.find("skinny_m_small") != std::string::npos;
    has_skinny_n_small |= test_case.name.find("skinny_n_small") != std::string::npos;
    has_large_k_1024 |= test_case.name.find("large_k_1024") != std::string::npos;
    has_large_k_16384 |= test_case.name.find("large_k_16384") != std::string::npos;
    has_split_k_16384 |= test_case.name.find("split_k_16384") != std::string::npos;
    has_transformer_decode |=
        test_case.name.find("transformer_projection_decode") != std::string::npos;
  }
  // 26 prepared shapes registered for four dtypes, excluding the two largest
  // square float16 cases.
  EXPECT_EQ(cpu_cases, 102u);
  EXPECT_EQ(chained_cases, 3u);
  EXPECT_TRUE(checked_runtime_inputs);
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
  EXPECT_TRUE(has_float64);
  EXPECT_TRUE(has_float16);
  EXPECT_TRUE(has_bfloat16);
  EXPECT_TRUE(has_direct_k32);
  EXPECT_TRUE(has_square_128);
  EXPECT_TRUE(has_square_1024);
  EXPECT_TRUE(has_square_2048);
  EXPECT_TRUE(has_square_4096);
  EXPECT_TRUE(has_skinny_m_small);
  EXPECT_TRUE(has_skinny_n_small);
  EXPECT_TRUE(has_large_k_1024);
  EXPECT_TRUE(has_large_k_16384);
  EXPECT_TRUE(has_split_k_16384);
  EXPECT_TRUE(has_transformer_decode);
}

TEST(OnnxLightBackendKernels, GemmBigBenchmarkCorpusCovers8192And16384OnEveryAxis) {
  onnx_light_cpu::backend_test::RegisterCpuKernelBackendTestCases();
  std::vector<TestCase> cases =
      CollectTestCases("Gemm", /*include_big=*/true, core::backend_test::TestMode::BENCHMARK);

  size_t big_cases = 0;
  bool has_large_m = false;
  bool has_large_n = false;
  bool has_large_k = false;
  bool has_larger_m = false;
  bool has_larger_n = false;
  bool has_larger_k = false;
  bool has_float32 = false;
  bool has_float64 = false;
  bool has_float16 = false;
  bool has_bfloat16 = false;
  for (const TestCase &test_case : cases) {
    if (test_case.name.rfind("test_cpu_gemm_big_", 0) != 0) {
      continue;
    }
    ++big_cases;
    EXPECT_FALSE(test_case.materialized());
    has_large_m |= test_case.name.find("_m8192_n128_k128_") != std::string::npos;
    has_large_n |= test_case.name.find("_m128_n8192_k128_") != std::string::npos;
    has_large_k |= test_case.name.find("_m128_n128_k8192_") != std::string::npos;
    has_larger_m |= test_case.name.find("_m16384_n128_k128_") != std::string::npos;
    has_larger_n |= test_case.name.find("_m128_n16384_k128_") != std::string::npos;
    has_larger_k |= test_case.name.find("_m128_n128_k16384_") != std::string::npos;
    has_float32 |= test_case.name.find("_float32_") != std::string::npos;
    has_float64 |= test_case.name.find("_float64_") != std::string::npos;
    has_float16 |= test_case.name.find("_float16_") != std::string::npos;
    has_bfloat16 |= test_case.name.find("_bfloat16_") != std::string::npos;
  }
  EXPECT_EQ(big_cases, 24U);
  EXPECT_TRUE(has_large_m);
  EXPECT_TRUE(has_large_n);
  EXPECT_TRUE(has_large_k);
  EXPECT_TRUE(has_larger_m);
  EXPECT_TRUE(has_larger_n);
  EXPECT_TRUE(has_larger_k);
  EXPECT_TRUE(has_float32);
  EXPECT_TRUE(has_float64);
  EXPECT_TRUE(has_float16);
  EXPECT_TRUE(has_bfloat16);
}

TEST(OnnxLightBackendKernels, BenchmarksOmitExpectedOutputsByDefault) {
  onnx_light_cpu::backend_test::RegisterCpuKernelBackendTestCases();
  const std::vector<TestCase> cases =
      CollectTestCases("", /*include_big=*/false, core::backend_test::TestMode::BENCHMARK);
  ASSERT_FALSE(cases.empty());
  for (const TestCase &test_case : cases) {
    if (test_case.name.rfind("test_cpu_", 0) != 0) {
      continue;
    }
    EXPECT_FALSE(test_case.has_expected_outputs()) << test_case.name;
  }
}

TEST(OnnxLightBackendKernels, BenchmarksGenerateExpectedOutputsWhenRequested) {
  onnx_light_cpu::backend_test::RegisterCpuKernelBackendTestCases();
  std::vector<TestCase> cases =
      CollectTestCases("Not", /*include_big=*/false, core::backend_test::TestMode::BENCHMARK,
                       /*generate_benchmark_expected_outputs=*/true);
  ASSERT_FALSE(cases.empty());
  for (TestCase &test_case : cases) {
    if (test_case.name.rfind("test_cpu_", 0) == 0) {
      TestCaseUnloadGuard unload_guard(test_case);
      ASSERT_EQ(test_case.data_sets().size(), 1u) << test_case.name;
      EXPECT_TRUE(test_case.data_sets()[0].expected_outputs_generated) << test_case.name;
      EXPECT_FALSE(test_case.data_sets()[0].outputs.empty()) << test_case.name;
    }
  }
}

TEST(OnnxLightBackendKernels, BenchmarkOutputCallbackRunsOnlyWhenRequested) {
  int input_only_output_calls = 0;
  TestCase input_only("input_only");
  input_only.build = [&input_only_output_calls](bool generate_expected_outputs) {
    core::backend_test::BuiltCase built;
    if (generate_expected_outputs) {
      ++input_only_output_calls;
    }
    built.data_sets.push_back(
        DataSet{{Tensor::FromBool("x", {1}, {1})}, {}, generate_expected_outputs});
    return built;
  };
  input_only.set_expected_outputs_generated(false);
  EXPECT_FALSE(input_only.data_sets()[0].expected_outputs_generated);
  EXPECT_TRUE(input_only.data_sets()[0].outputs.empty());
  EXPECT_EQ(input_only_output_calls, 0);
  EXPECT_EQ(input_only.data_sets().size(), 1u);
  EXPECT_EQ(input_only_output_calls, 0);

  int requested_output_calls = 0;
  TestCase requested("requested");
  requested.build = [&requested_output_calls](bool generate_expected_outputs) {
    core::backend_test::BuiltCase built;
    std::vector<Tensor> outputs;
    if (generate_expected_outputs) {
      ++requested_output_calls;
      outputs.emplace_back(Tensor::FromBool("y", {1}, {1}));
    }
    built.data_sets.push_back(
        DataSet{{Tensor::FromBool("x", {1}, {1})}, std::move(outputs), generate_expected_outputs});
    return built;
  };
  EXPECT_TRUE(requested.data_sets()[0].expected_outputs_generated);
  EXPECT_EQ(requested.data_sets()[0].outputs.size(), 1u);
  EXPECT_EQ(requested_output_calls, 1);
  EXPECT_EQ(requested.data_sets().size(), 1u);
  EXPECT_EQ(requested_output_calls, 1);
}

} // namespace
