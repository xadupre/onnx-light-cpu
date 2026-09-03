// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/attention/linear_attention_cases.h"
#include "onnx_light_cpu/backend_test/cases/com_microsoft/include_com_microsoft_cases.h"

namespace onnx_light_cpu::backend_test {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

void RegisterCpuMicrosoftLinearAttentionCases(std::vector<TestCase> &registry, TestMode mode) {
  if (mode == TestMode::BENCHMARK) {
    RegisterLinearAttentionCase(registry,
                                {"decode_h16_d128", 1, 1, 16, 16, 16, 128, 128, "gated_delta", true,
                                 false, false, rt_ns::DataType::FLOAT, 4101},
                                LinearAttentionCaseContract::kMicrosoft, true);
    return;
  }
  for (const LinearAttentionCase &test_case :
       {LinearAttentionCase{"linear", 1, 3, 2, 2, 2, 4, 3, "linear", false, false, false,
                            rt_ns::DataType::FLOAT, 4101},
        LinearAttentionCase{"gated", 1, 3, 2, 2, 2, 4, 3, "gated", false, false, false,
                            rt_ns::DataType::FLOAT, 4111},
        LinearAttentionCase{"delta", 1, 3, 2, 2, 2, 4, 3, "delta", false, false, false,
                            rt_ns::DataType::FLOAT, 4121},
        LinearAttentionCase{"gated_delta_past", 1, 3, 2, 2, 2, 4, 3, "gated_delta", true, false,
                            false, rt_ns::DataType::FLOAT, 4131},
        LinearAttentionCase{"inverse_grouping", 1, 2, 1, 2, 2, 4, 3, "linear", false, false, false,
                            rt_ns::DataType::FLOAT, 4141},
        LinearAttentionCase{"shared_key_head", 1, 2, 2, 1, 2, 4, 3, "linear", false, false, false,
                            rt_ns::DataType::FLOAT, 4151},
        LinearAttentionCase{"empty_sequence", 1, 0, 2, 2, 2, 4, 3, "linear", false, false, false,
                            rt_ns::DataType::FLOAT, 4161}}) {
    RegisterLinearAttentionCase(registry, test_case, LinearAttentionCaseContract::kMicrosoft,
                                false);
  }
}

} // namespace onnx_light_cpu::backend_test
