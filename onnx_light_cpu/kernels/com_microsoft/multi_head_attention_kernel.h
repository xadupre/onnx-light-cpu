// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

#ifndef ONNX_LIGHT_NAMESPACE
#define ONNX_LIGHT_NAMESPACE onnx_light
#endif

namespace onnx_light_cpu {

class MultiHeadAttentionKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  MultiHeadAttentionKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                           const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx);
  static constexpr const char *kName = "onnx_light_cpu::MultiHeadAttention";
  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;
};

void RegisterMultiHeadAttentionKernel();

} // namespace onnx_light_cpu
