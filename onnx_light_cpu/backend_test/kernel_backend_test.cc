// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/kernel_backend_test.h"

#include "onnx_light_cpu/backend_test/abs_backend_test.h"
#include "onnx_light_cpu/backend_test/exp_backend_test.h"
#include "onnx_light_cpu/backend_test/gemm_backend_test.h"
#include "onnx_light_cpu/backend_test/log_backend_test.h"
#include "onnx_light_cpu/backend_test/not_backend_test.h"

#include "onnx_core/backend_test/test_case_registry.h"

#include <string>
#include <vector>

namespace onnx_light_cpu::backend_test {

namespace {

namespace bt_ns = ONNX_LIGHT_NAMESPACE::core::backend_test;

using bt_ns::DispatchRegisterByOpType;
using bt_ns::OpRegisterModeMap;
using bt_ns::TestCase;
using bt_ns::TestMode;

} // namespace

void CollectCpuKernelTestCases(std::vector<TestCase> &registry, const std::string &op_type,
                               TestMode mode) {
  // Mirrors onnx-light's ``collect_<category>_cases.cc``: dispatch by op_type to
  // the per-kernel registration helper (one header + one ``.cc`` per kernel).
  static const OpRegisterModeMap kEntries = {
      {"Abs", &RegisterCpuAbsCases}, {"Exp", &RegisterCpuExpCases},   {"Log", &RegisterCpuLogCases},
      {"Not", &RegisterCpuNotCases}, {"Gemm", &RegisterCpuGemmCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries, mode);
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
