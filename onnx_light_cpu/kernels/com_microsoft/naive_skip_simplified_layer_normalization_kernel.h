// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu/kernels/com_microsoft/skip_simplified_layer_normalization_kernel.h"

namespace onnx_light_cpu {

/// Provides an independent scalar implementation of the fused residual RMS normalization.
class NaiveSkipSimplifiedLayerNormalizationKernel
    : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  using ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase::KernelBase;
  static constexpr const char *kName = "onnx_light_cpu::NaiveSkipSimplifiedLayerNormalization";

  NaiveSkipSimplifiedLayerNormalizationKernel(
      const ONNX_LIGHT_NAMESPACE::NodeProto &node,
      const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx);

  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;

  SkipSimplifiedLayerNormalizationResult
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &input,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &skip,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &gamma,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *bias = nullptr,
             float epsilon = 1.0e-12F, bool output_sum = false, bool output_mean = false,
             bool output_inv_std_var = false,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;
};

void RegisterNaiveSkipSimplifiedLayerNormalizationKernel();

} // namespace onnx_light_cpu
