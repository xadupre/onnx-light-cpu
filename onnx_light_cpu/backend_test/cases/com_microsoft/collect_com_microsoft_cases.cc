// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/com_microsoft/include_com_microsoft_cases.h"

namespace onnx_light_cpu::backend_test {
namespace {

void RegisterMoEAndQuantizedReferenceCases(std::vector<TestCase> &registry, TestMode mode) {
  RegisterCpuMoECases(registry, mode);
  RegisterCpuQMoEReferenceCases(registry, mode);
}

} // namespace

void CollectCpuMicrosoftTestCases(std::vector<TestCase> &registry, const std::string &op_type,
                                  TestMode mode) {
  using ONNX_LIGHT_NAMESPACE::core::backend_test::DispatchRegisterByOpType;
  using ONNX_LIGHT_NAMESPACE::core::backend_test::OpRegisterModeMap;

  static const OpRegisterModeMap kEntries = {
      {"BiasGelu", &RegisterCpuBiasGeluCases},
      {"CDist", &RegisterCpuCDistCases},
      {"CausalConvWithState", &RegisterCpuCausalConvWithStateCases},
      {"GroupQueryAttention", &RegisterCpuGroupQueryAttentionCases},
      {"MultiHeadAttention", &RegisterCpuMultiHeadAttentionCases},
      {"LinearAttention", &RegisterCpuMicrosoftLinearAttentionCases},
      {"MatMulNBits", &RegisterCpuMatMulNBitsCases},
      {"MoE", &RegisterMoEAndQuantizedReferenceCases},
      {"QMoE", &RegisterCpuQMoECases},
      {"RotaryEmbedding", &RegisterCpuMicrosoftRotaryEmbeddingCases},
      {"SkipLayerNormalization", &RegisterCpuSkipLayerNormalizationCases},
      {"SkipSimplifiedLayerNormalization", &RegisterCpuSkipSimplifiedLayerNormalizationCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries, mode);
}

} // namespace onnx_light_cpu::backend_test
