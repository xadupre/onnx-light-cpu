// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/elementwise/include_elementwise_cases.h"

namespace onnx_light_cpu::backend_test {

namespace {

#define ONNX_LIGHT_CPU_BINARY_CASE_WRAPPER(op_name)                                                \
  void RegisterCpu##op_name##Cases(std::vector<TestCase> &registry, TestMode mode) {               \
    RegisterCpuBinaryCases(registry, #op_name, mode);                                              \
  }

ONNX_LIGHT_CPU_BINARY_CASE_WRAPPER(Add)
ONNX_LIGHT_CPU_BINARY_CASE_WRAPPER(Sub)
ONNX_LIGHT_CPU_BINARY_CASE_WRAPPER(Mul)
ONNX_LIGHT_CPU_BINARY_CASE_WRAPPER(Div)
ONNX_LIGHT_CPU_BINARY_CASE_WRAPPER(Mod)
ONNX_LIGHT_CPU_BINARY_CASE_WRAPPER(Pow)
ONNX_LIGHT_CPU_BINARY_CASE_WRAPPER(Equal)
ONNX_LIGHT_CPU_BINARY_CASE_WRAPPER(Greater)
ONNX_LIGHT_CPU_BINARY_CASE_WRAPPER(GreaterOrEqual)
ONNX_LIGHT_CPU_BINARY_CASE_WRAPPER(Less)
ONNX_LIGHT_CPU_BINARY_CASE_WRAPPER(LessOrEqual)
ONNX_LIGHT_CPU_BINARY_CASE_WRAPPER(And)
ONNX_LIGHT_CPU_BINARY_CASE_WRAPPER(Or)
ONNX_LIGHT_CPU_BINARY_CASE_WRAPPER(Xor)
ONNX_LIGHT_CPU_BINARY_CASE_WRAPPER(BitwiseAnd)
ONNX_LIGHT_CPU_BINARY_CASE_WRAPPER(BitwiseOr)
ONNX_LIGHT_CPU_BINARY_CASE_WRAPPER(BitwiseXor)
ONNX_LIGHT_CPU_BINARY_CASE_WRAPPER(BitShift)
ONNX_LIGHT_CPU_BINARY_CASE_WRAPPER(PRelu)

#undef ONNX_LIGHT_CPU_BINARY_CASE_WRAPPER

} // namespace

void CollectCpuElementwiseTestCases(std::vector<TestCase> &registry, const std::string &op_type,
                                    TestMode mode) {
  using ONNX_LIGHT_NAMESPACE::core::backend_test::DispatchRegisterByOpType;
  using ONNX_LIGHT_NAMESPACE::core::backend_test::OpRegisterModeMap;

  static const OpRegisterModeMap kEntries = {
      {"Add", &RegisterCpuAddCases},
      {"Sub", &RegisterCpuSubCases},
      {"Mul", &RegisterCpuMulCases},
      {"Div", &RegisterCpuDivCases},
      {"Mod", &RegisterCpuModCases},
      {"Pow", &RegisterCpuPowCases},
      {"Equal", &RegisterCpuEqualCases},
      {"Greater", &RegisterCpuGreaterCases},
      {"GreaterOrEqual", &RegisterCpuGreaterOrEqualCases},
      {"Less", &RegisterCpuLessCases},
      {"LessOrEqual", &RegisterCpuLessOrEqualCases},
      {"And", &RegisterCpuAndCases},
      {"Or", &RegisterCpuOrCases},
      {"Xor", &RegisterCpuXorCases},
      {"BitwiseAnd", &RegisterCpuBitwiseAndCases},
      {"BitwiseOr", &RegisterCpuBitwiseOrCases},
      {"BitwiseXor", &RegisterCpuBitwiseXorCases},
      {"BitShift", &RegisterCpuBitShiftCases},
      {"PRelu", &RegisterCpuPReluCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries, mode);
}

} // namespace onnx_light_cpu::backend_test
