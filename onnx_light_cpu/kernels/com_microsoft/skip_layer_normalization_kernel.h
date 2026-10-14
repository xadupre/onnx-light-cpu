// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

namespace onnx_light_cpu {

struct SkipLayerNormalizationResult {
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor output;
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor mean;
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor inv_std_var;
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor input_skip_bias_sum;
};

/// Computes fused residual addition and last-axis layer normalization.
class SkipLayerNormalizationKernel final : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  using ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase::KernelBase;
  static constexpr const char *kName = "onnx_light_cpu::SkipLayerNormalization";

  SkipLayerNormalizationKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                               const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx);

  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;

  SkipLayerNormalizationResult
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &input,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &skip,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &gamma,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *beta = nullptr,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *bias = nullptr,
             float epsilon = 1.0e-12F, bool output_sum = false, bool output_mean = false,
             bool output_inv_std_var = false,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;
};

void RegisterSkipLayerNormalizationKernel();

} // namespace onnx_light_cpu
