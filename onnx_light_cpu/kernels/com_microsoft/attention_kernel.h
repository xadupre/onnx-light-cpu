// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/runtime_context.h"

namespace onnx_light_cpu {

class MicrosoftAttentionKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  MicrosoftAttentionKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                           const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx);
  static constexpr const char *kName = "onnx_light_cpu::MicrosoftAttention";
  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;
};

void RegisterMicrosoftAttentionKernel();
} // namespace onnx_light_cpu
