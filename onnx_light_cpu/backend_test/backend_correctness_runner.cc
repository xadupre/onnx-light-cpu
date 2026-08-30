// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/backend_correctness_runner.h"

#include "onnx_light_cpu/backend_test/collect_test_cases.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/backend_test/test_case_registry.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_core/runtime/kernels/tensor_compare.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_proto/onnx_helper.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace onnx_light_cpu::backend_test {

namespace {

using namespace ONNX_LIGHT_NAMESPACE;

namespace bt_ns = ONNX_LIGHT_NAMESPACE::core::backend_test;
namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

using bt_ns::DataSet;
using bt_ns::TestCase;
using rt_ns::Tensor;

class TestCaseUnloadGuard {
public:
  explicit TestCaseUnloadGuard(TestCase &test_case) : test_case_(test_case) {}
  ~TestCaseUnloadGuard() {
    if (test_case_.materialized()) {
      test_case_.unload();
    }
  }

private:
  TestCase &test_case_;
};

std::int64_t ModelOpsetVersion(const ModelProto &model, const std::string &domain) {
  for (const OperatorSetIdProto &opset : model.opset_import()) {
    if (ONNX_LIGHT_NAMESPACE::NormaliseDomain(opset.domain()) == domain) {
      return opset.version();
    }
  }
  return -1;
}

bool IsApplicable(const TestCase &test_case, const KernelRegistration &kernel,
                  std::string &reason) {
  const ModelProto &model = test_case.model();
  const GraphProto &graph = model.ref_graph();
  const bool has_node =
      std::any_of(graph.node().begin(), graph.node().end(), [&kernel](const NodeProto &node) {
        return ONNX_LIGHT_NAMESPACE::NormaliseDomain(node.domain()) == kernel.domain &&
               node.op_type() == kernel.op_type;
      });
  if (!has_node) {
    reason = "model has no matching node";
    return false;
  }
  const std::int64_t opset = ModelOpsetVersion(model, kernel.domain);
  if (opset < 0) {
    reason = "model has no matching opset";
    return false;
  }
  if ((kernel.since_version && opset < *kernel.since_version) ||
      (kernel.until_version && opset > *kernel.until_version)) {
    reason = "model opset is outside the kernel's supported range";
    return false;
  }
  const bool has_supported_input = std::any_of(
      graph.input().begin(), graph.input().end(), [&kernel](const ValueInfoProto &input) {
        return input.has_type() && input.type().has_tensor_type() &&
               std::find(kernel.types.begin(), kernel.types.end(),
                         static_cast<rt_ns::DataType>(input.type().tensor_type().elem_type())) !=
                   kernel.types.end();
      });
  if (!has_supported_input) {
    reason = "model input types are unsupported";
    return false;
  }
  return true;
}

void CompareOutputs(const TestCase &test_case, const DataSet &data_set,
                    const rt_ns::Tensors &actual) {
  if (!data_set.expected_outputs_generated) {
    throw std::runtime_error("TEST case omitted expected outputs");
  }
  if (actual.size() != data_set.outputs.size()) {
    throw std::runtime_error("output count mismatch");
  }
  for (size_t index = 0; index < actual.size(); ++index) {
    const rt_ns::TensorComparison comparison = rt_ns::CompareTensors(
        actual[index], data_set.outputs[index], test_case.rtol, test_case.atol);
    if (!comparison.close) {
      throw std::runtime_error(comparison.message);
    }
  }
}

void RunCase(const TestCase &test_case, const KernelRegistration &kernel) {
  const ModelProto &model = test_case.model();
  const GraphProto &graph = model.ref_graph();
  for (const DataSet &data_set : test_case.data_sets()) {
    rt_ns::RuntimeContext runtime(
        rt_ns::KernelContext(rt_ns::DefaultOpset(ModelOpsetVersion(model, "ai.onnx"))));
    rt_ns::RegisterModelFunctions(model, runtime);
    std::vector<std::pair<std::string, Tensor>> bindings;
    bindings.reserve(data_set.inputs.size());
    for (const Tensor &input : data_set.inputs) {
      bindings.emplace_back(input.name, input);
    }
    rt_ns::SubgraphSession session(runtime, graph);
    ClearUsedKernelNames();
    CompareOutputs(test_case, data_set, session.Run(std::move(bindings), runtime));
    const std::vector<std::string> used = UsedKernelNames();
    if (std::find(used.begin(), used.end(), kernel.kernel_name) == used.end()) {
      throw std::runtime_error("expected kernel '" + kernel.kernel_name + "' did not run");
    }
  }
}

} // namespace

std::string BackendCorrectnessReport::Describe() const {
  std::ostringstream stream;
  stream << "executed=" << executed << ", passed=" << passed << ", skipped=" << skipped.size()
         << ", failed=" << failed.size();
  for (const BackendCaseResult &result : skipped) {
    stream << "\nskipped " << result.op_type << " " << result.case_name << ": " << result.reason;
  }
  for (const BackendCaseResult &result : failed) {
    stream << "\nfailed " << result.op_type << " " << result.case_name << ": " << result.reason;
  }
  return stream.str();
}

BackendCorrectnessReport RunBackendCorrectnessTests(MicrosoftKernelImplementation implementation) {
  RegisterCpuKernelBackendTestCases();
  RegisterAllKernels(implementation);
  BackendCorrectnessReport report;
  std::set<std::pair<std::string, std::string>> seen_cases;
  const std::vector<KernelRegistration> kernels = CollectRegisteredKernels(implementation);
  std::vector<bool> covered_kernels(kernels.size());
  for (size_t kernel_index = 0; kernel_index < kernels.size(); ++kernel_index) {
    const KernelRegistration &kernel = kernels[kernel_index];
    for (TestCase &test_case :
         bt_ns::CollectTestCases(kernel.op_type, /*include_big=*/false, bt_ns::TestMode::TEST)) {
      const auto case_key = std::make_pair(kernel.domain, test_case.name);
      if (seen_cases.contains(case_key)) {
        continue;
      }
      TestCaseUnloadGuard unload_guard(test_case);
      std::string reason;
      if (!IsApplicable(test_case, kernel, reason)) {
        report.skipped.push_back({kernel.op_type, test_case.name, std::move(reason)});
        continue;
      }
      seen_cases.insert(case_key);
      ++report.executed;
      try {
        RunCase(test_case, kernel);
        ++report.passed;
        covered_kernels[kernel_index] = true;
      } catch (const std::exception &error) {
        report.failed.push_back({kernel.op_type, test_case.name, error.what()});
      }
    }
  }
  for (size_t kernel_index = 0; kernel_index < kernels.size(); ++kernel_index) {
    const KernelRegistration &kernel = kernels[kernel_index];
    if (!covered_kernels[kernel_index]) {
      report.failed.push_back(
          {kernel.op_type, "",
           "no applicable TEST backend correctness case for " + kernel.kernel_name});
    }
  }
  return report;
}

} // namespace onnx_light_cpu::backend_test
