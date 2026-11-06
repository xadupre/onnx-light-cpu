// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/runtime_context.h"

namespace onnx_light_cpu {

/// ONNX Runtime contrib Gelu, opset 1 (no approximation attribute).
class MicrosoftGeluKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  static constexpr const char *kName = "onnx_light_cpu::MicrosoftGelu";
  MicrosoftGeluKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                      const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx);
  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;
};

void RegisterMicrosoftGeluKernel();

} // namespace onnx_light_cpu
