// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/collect_test_cases.h"

#include "onnx_light_cpu/backend_test/cases/attention/include_attention_cases.h"
#include "onnx_light_cpu/backend_test/cases/elementwise/include_elementwise_cases.h"
#include "onnx_light_cpu/backend_test/cases/logical/include_logical_cases.h"
#include "onnx_light_cpu/backend_test/cases/math/include_math_cases.h"
#include "onnx_light_cpu/backend_test/cases/traditionalml/include_traditionalml_cases.h"

#include "onnx_core/backend_test/test_case_registry.h"

#include <string>
#include <vector>

namespace onnx_light_cpu::backend_test {

namespace {

namespace bt_ns = ONNX_LIGHT_NAMESPACE::core::backend_test;

using bt_ns::TestCase;
using bt_ns::TestMode;

} // namespace

void CollectCpuKernelTestCases(std::vector<TestCase> &registry, const std::string &op_type,
                               TestMode mode) {
  CollectCpuMathTestCases(registry, op_type, mode);
  CollectCpuElementwiseTestCases(registry, op_type, mode);
  CollectCpuLogicalTestCases(registry, op_type, mode);
  CollectCpuTraditionalMlTestCases(registry, op_type, mode);
  CollectCpuAttentionTestCases(registry, op_type, mode);
}

void RegisterCpuKernelBackendTestCases() {
  // Registering the collector into onnx-light's global registry exactly once,
  // regardless of how many times this function is called, via a function-local
  // static initialized on first use.
  static const int kRegistered = bt_ns::RegisterTestCasesCollector(
      [](std::vector<TestCase> &registry, const std::string &op_type, bool /*include_big*/,
         TestMode mode) { CollectCpuKernelTestCases(registry, op_type, mode); });
  (void)kRegistered;
}

} // namespace onnx_light_cpu::backend_test
