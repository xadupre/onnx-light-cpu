// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/com_microsoft/include_com_microsoft_cases.h"

namespace onnx_light_cpu::backend_test {

void CollectCpuMicrosoftTestCases(std::vector<TestCase> &registry, const std::string &op_type,
                                  TestMode mode) {
  using ONNX_LIGHT_NAMESPACE::core::backend_test::DispatchRegisterByOpType;
  using ONNX_LIGHT_NAMESPACE::core::backend_test::OpRegisterModeMap;

  static const OpRegisterModeMap kEntries = {
      {"BiasGelu", &RegisterCpuBiasGeluCases},
      {"CDist", &RegisterCpuCDistCases},
      {"GroupQueryAttention", &RegisterCpuGroupQueryAttentionCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries, mode);
}

} // namespace onnx_light_cpu::backend_test
