// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/attention/include_attention_cases.h"
#include "onnx_light_cpu/backend_test/cases/attention/linear_attention_cases.h"

namespace onnx_light_cpu::backend_test {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

void RegisterCpuLinearAttentionCases(std::vector<TestCase> &registry, TestMode mode) {
  if (mode == TestMode::BENCHMARK) {
    for (const LinearAttentionCase &test_case :
         {LinearAttentionCase{"qwen3_5_decode_t1_h16_d128_gated_delta_past", 1, 1, 16, 16, 16, 128,
                              128, "gated_delta", true, false, false, rt_ns::DataType::FLOAT, 2701},
          LinearAttentionCase{"qwen3_5_prefill_t128_h16_d128_gated_delta", 1, 128, 16, 16, 16, 128,
                              128, "gated_delta", false, false, false, rt_ns::DataType::FLOAT,
                              2711},
          LinearAttentionCase{"qwen3_5_prefill_t512_h16_d128_gated_delta", 1, 512, 16, 16, 16, 128,
                              128, "gated_delta", false, false, false, rt_ns::DataType::FLOAT,
                              2721},
          LinearAttentionCase{"qwen3_5_decode_t1_h16_d128_gated_delta_past", 1, 1, 16, 16, 16, 128,
                              128, "gated_delta", true, false, false, rt_ns::DataType::BFLOAT16,
                              2741},
          LinearAttentionCase{"qwen3_5_prefill_t128_h16_d128_gated_delta", 1, 128, 16, 16, 16, 128,
                              128, "gated_delta", false, false, false, rt_ns::DataType::BFLOAT16,
                              2751}}) {
      RegisterLinearAttentionCase(registry, test_case, LinearAttentionCaseContract::kOnnx, true);
    }
    return;
  }

  for (const LinearAttentionCase &test_case :
       {LinearAttentionCase{"linear_mha", 1, 3, 2, 2, 2, 4, 3, "linear", false, false, false,
                            rt_ns::DataType::FLOAT, 2601},
        LinearAttentionCase{"gated_per_dimension", 1, 3, 2, 2, 2, 4, 3, "gated", false, true, false,
                            rt_ns::DataType::FLOAT, 2611},
        LinearAttentionCase{"delta_shared_beta", 2, 2, 2, 2, 2, 4, 3, "delta", false, false, true,
                            rt_ns::DataType::FLOAT, 2621},
        LinearAttentionCase{"gqa_with_past", 1, 3, 4, 2, 2, 4, 3, "gated_delta", true, false, false,
                            rt_ns::DataType::FLOAT, 2631},
        LinearAttentionCase{"gated_delta", 1, 3, 2, 2, 2, 4, 3, "gated_delta", true, false, false,
                            rt_ns::DataType::FLOAT16, 2651},
        LinearAttentionCase{"gated_delta", 1, 3, 2, 2, 2, 4, 3, "gated_delta", true, false, false,
                            rt_ns::DataType::BFLOAT16, 2661}}) {
    RegisterLinearAttentionCase(registry, test_case, LinearAttentionCaseContract::kOnnx, false);
  }
}

} // namespace onnx_light_cpu::backend_test
