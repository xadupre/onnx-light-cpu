// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/math/include_math_cases.h"

namespace onnx_light_cpu::backend_test {

void CollectCpuMathTestCases(std::vector<TestCase> &registry, const std::string &op_type,
                             TestMode mode) {
  using ONNX_LIGHT_NAMESPACE::core::backend_test::DispatchRegisterByOpType;
  using ONNX_LIGHT_NAMESPACE::core::backend_test::OpRegisterModeMap;

  static const OpRegisterModeMap kEntries = {
      {"Abs", &RegisterCpuAbsCases},       {"Exp", &RegisterCpuExpCases},
      {"Gemm", &RegisterCpuGemmCases},     {"Log", &RegisterCpuLogCases},
      {"MatMul", &RegisterCpuMatMulCases}, {"MatMulInteger", &RegisterCpuMatMulIntegerCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries, mode);
}

} // namespace onnx_light_cpu::backend_test
