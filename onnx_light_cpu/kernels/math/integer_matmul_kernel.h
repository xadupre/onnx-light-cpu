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

class MatMulIntegerKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  explicit MatMulIntegerKernel(const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx)
      : KernelBase(ctx) {}

  static constexpr const char *kName = "onnx_light_cpu::MatMulInteger";

  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;

  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &a,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &b,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *a_zero_point = nullptr,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *b_zero_point = nullptr,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;
};

class QLinearMatMulKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  explicit QLinearMatMulKernel(const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx)
      : KernelBase(ctx) {}

  static constexpr const char *kName = "onnx_light_cpu::QLinearMatMul";

  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;

  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &a,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &a_scale,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &a_zero_point,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &b,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &b_scale,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &b_zero_point,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &y_scale,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &y_zero_point,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;
};

void RegisterIntegerMatMulKernels();

} // namespace onnx_light_cpu
