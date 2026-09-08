// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

#include <cstdint>

namespace onnx_light_cpu {

struct SimplifiedLayerNormalizationResult {
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor y;
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor inv_std_var;
};

/// CPU implementation of the experimental ai.onnx operator used by Qwen exports.
class SimplifiedLayerNormalizationKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  using ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase::KernelBase;
  static constexpr const char *kName = "onnx_light_cpu::SimplifiedLayerNormalization";

  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;
  SimplifiedLayerNormalizationResult
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &x,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &scale, std::int64_t axis = -1,
             float epsilon = 1.0e-5F, std::int64_t stash_type = 1, bool output_inv_std_var = false,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;
};

void RegisterSimplifiedLayerNormalizationKernel();

} // namespace onnx_light_cpu
