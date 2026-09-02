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
      {"Abs", &RegisterCpuAbsCases},
      {"BatchNormalization", &RegisterCpuBatchNormalizationCases},
      {"Exp", &RegisterCpuExpCases},
      {"Gemm", &RegisterCpuGemmCases},
      {"GroupNormalization", &RegisterCpuGroupNormalizationCases},
      {"InstanceNormalization", &RegisterCpuInstanceNormalizationCases},
      {"LayerNormalization", &RegisterCpuLayerNormalizationCases},
      {"Log", &RegisterCpuLogCases},
      {"LpNormalization", &RegisterCpuLpNormalizationCases},
      {"MatMul", &RegisterCpuMatMulCases},
      {"MatMulInteger", &RegisterCpuMatMulIntegerCases},
      {"MeanVarianceNormalization", &RegisterCpuMeanVarianceNormalizationCases},
      {"RMSNormalization", &RegisterCpuRmsNormalizationCases},
      {"Sigmoid", &RegisterCpuSigmoidCases},
      {"Softmax", &RegisterCpuSoftmaxCases},
      {"SwiGLU", &RegisterCpuSwiGLUCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries, mode);
}

} // namespace onnx_light_cpu::backend_test
