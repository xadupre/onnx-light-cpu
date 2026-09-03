// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu/kernels/attention/linear_attention_shared.h"

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

#include <cstdint>
#include <string>

#ifndef ONNX_LIGHT_NAMESPACE
#define ONNX_LIGHT_NAMESPACE onnx_light
#endif

namespace onnx_light_cpu {

class LinearAttentionKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  using ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase::KernelBase;
  static constexpr const char *kName = "onnx_light_cpu::LinearAttention";

  struct Attributes {
    std::string update_rule = "gated_delta";
    std::int64_t query_heads = 0;
    std::int64_t key_value_heads = 0;
    float scale = 0.0f;
  };

  using Result = LinearAttentionResult;

  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;

  Result operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &query,
                    const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &key,
                    const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &value,
                    const Attributes &attributes,
                    const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *past_state = nullptr,
                    const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *decay = nullptr,
                    const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *beta = nullptr,
                    ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr,
                    bool has_state_output = true) const;
};

void RegisterLinearAttentionKernel();

} // namespace onnx_light_cpu
