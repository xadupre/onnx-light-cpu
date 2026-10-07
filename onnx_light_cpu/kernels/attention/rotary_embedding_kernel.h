// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

namespace onnx_light_cpu {

class RotaryEmbeddingKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  using ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase::KernelBase;
  static constexpr const char *kName = "onnx_light_cpu::RotaryEmbedding";

  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &input,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &cos,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &sin,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *positions, bool interleaved = false,
             std::int64_t rotary_dim = 0, std::int64_t num_heads = 0, bool microsoft = false,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;
  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;
};

void RegisterRotaryEmbeddingKernels();

} // namespace onnx_light_cpu
