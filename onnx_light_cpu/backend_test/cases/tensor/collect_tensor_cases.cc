// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/tensor/include_tensor_cases.h"

namespace onnx_light_cpu::backend_test {

void CollectCpuTensorTestCases(std::vector<TestCase> &registry, const std::string &op_type,
                               TestMode mode) {
  using ONNX_LIGHT_NAMESPACE::core::backend_test::DispatchRegisterByOpType;
  using ONNX_LIGHT_NAMESPACE::core::backend_test::OpRegisterModeMap;

  static const OpRegisterModeMap kEntries = {
      {"Gather", &RegisterCpuGatherCases}, {"Slice", &RegisterCpuSliceCases},
      {"Concat", &RegisterCpuConcatCases}, {"Split", &RegisterCpuSplitCases},
      {"Cast", &RegisterCpuCastCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries, mode);
}

} // namespace onnx_light_cpu::backend_test
