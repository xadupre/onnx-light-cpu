// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

namespace onnx_light_cpu {

struct SkipSimplifiedLayerNormalizationResult {
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor output;
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor mean;
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor inv_std_var;
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor input_skip_bias_sum;
};

/// Computes fused residual addition and RMS normalization over the last dimension.
/// Accepts FLOAT, FLOAT16, and BFLOAT16 tensors with rank 2 or 3. Arithmetic uses FLOAT
/// until the final output conversion, including residual addition for the narrow types.
/// Optional outputs retain their ONNX slots: mean (1, zero), inverse RMS (2), and sum (3).
class SkipSimplifiedLayerNormalizationKernel
    : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  using ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase::KernelBase;
  static constexpr const char *kName = "onnx_light_cpu::SkipSimplifiedLayerNormalization";

  SkipSimplifiedLayerNormalizationKernel(
      const ONNX_LIGHT_NAMESPACE::NodeProto &node,
      const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx);

  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;

  /// Returns the normalized output and any requested FLOAT statistics or typed residual sum.
  SkipSimplifiedLayerNormalizationResult
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &input,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &skip,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &gamma,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *bias = nullptr,
             float epsilon = 1.0e-12F, bool output_sum = false, bool output_mean = false,
             bool output_inv_std_var = false,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;
};

void RegisterSkipSimplifiedLayerNormalizationKernel();

} // namespace onnx_light_cpu
