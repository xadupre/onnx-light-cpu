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
//      runtime dispatched to — matches, byte for byte, the reference output
//      shipped with the case (see ``CompareTensor``).
//
// The runtime resolves each node by ``(domain, op_type)`` to whatever kernel is
// registered, so registering the onnx-light-cpu kernels means these cases
// exercise the accelerated kernels end to end, exactly like a model run through
// onnx-light itself.

#include "onnx_light_cpu/backend_test/kernel_backend_test.h"
#include "onnx_light_cpu/kernels/register_kernels.h"

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/runtime/kernel_context.h"
#include "onnx_core/runtime/run_nodes.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/simple_tensor.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ONNX_LIGHT_NAMESPACE;
using core::backend_test::CollectTestCases;
using core::backend_test::DataSet;
using core::backend_test::TestCase;
using core::runtime::DefaultOpset;
using core::runtime::KernelContext;
using core::runtime::RegisterModelFunctions;
using core::runtime::RuntimeContext;
using core::runtime::SubgraphSession;
using core::runtime::Tensor;
using core::runtime::Tensors;

// onnx-light resolves kernels by (domain, op_type) only — never by opset
// version — so the concrete default opset used to build the runtime context
// does not affect which kernel a node dispatches to. A fixed recent version is
// therefore sufficient to drive every registered case.
constexpr int64_t kRuntimeDefaultOpsetVersion = 18;

// Compares a runtime output tensor against the reference output of a backend
// test case for bit-exact equality, mirroring onnx-light's own backend
// run-model test (``ExpectTensorBitEqual`` in ``test_backend_run_model.cc``).
//
// A bit-exact byte comparison — rather than an ``rtol``/``atol`` tolerance — is
// the correct check here because every ``test_cpu_*`` case's expected output is
// produced by the same onnx-light-cpu kernel the runtime dispatches to (see
// ``kernel_backend_test.cc``). The runtime therefore has to reproduce those
// exact bytes, so the comparison stays dtype-agnostic and needs no per-type
// (float / float16 / bfloat16) decoding.
void CompareTensor(const std::string &case_name, const Tensor &actual, const Tensor &expected,
                   std::vector<std::string> &failures) {
  if (actual.data_type != expected.data_type) {
    failures.push_back(case_name + ": output data type " + std::to_string(actual.data_type) +
                       " != expected " + std::to_string(expected.data_type));
    return;
  }
  if (actual.shape != expected.shape) {
    failures.push_back(case_name + ": output shape mismatch");
    return;
  }
  if (actual.size_bytes() != expected.size_bytes() ||
      std::memcmp(actual.bytes(), expected.bytes(), actual.size_bytes()) != 0) {
    failures.push_back(case_name + ": output does not match reference");
  }
}

// Runs a single-node backend test case model through onnx-light's runtime and
// compares the produced output with the reference output. The runtime resolves
// the node to the registered onnx-light-cpu kernel.
void RunCaseThroughRuntime(const TestCase &tc, std::vector<std::string> &failures) {
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
    for (size_t i = 0; i < ds.outputs.size(); ++i) {
      CompareTensor(tc.name, outputs[i], ds.outputs[i], failures);
    }
  }
}

// Registers the onnx-light-cpu backend test cases and kernels, then runs every
// ``test_cpu_*`` case for ``op_type`` through the runtime, collecting failures.
std::vector<std::string> RunCpuBackendCases(const std::string &op_type) {
  onnx_light_cpu::backend_test::RegisterCpuKernelBackendTestCases();
  onnx_light_cpu::RegisterAllKernels();

  std::vector<std::string> failures;
  std::vector<TestCase> cases = CollectTestCases(op_type);
  size_t cpu_cases = 0;
  for (const TestCase &tc : cases) {
    if (tc.name.rfind("test_cpu_", 0) != 0) {
      continue;
    }
    ++cpu_cases;
    RunCaseThroughRuntime(tc, failures);
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
  const std::vector<std::string> failures = RunCpuBackendCases("Abs");
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, ExpRunsThroughRuntime) {
  const std::vector<std::string> failures = RunCpuBackendCases("Exp");
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, LogRunsThroughRuntime) {
  const std::vector<std::string> failures = RunCpuBackendCases("Log");
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, NotRunsThroughRuntime) {
  const std::vector<std::string> failures = RunCpuBackendCases("Not");
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, GemmRunsThroughRuntime) {
  const std::vector<std::string> failures = RunCpuBackendCases("Gemm");
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

} // namespace
