// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_light_cpu/impl/com_microsoft/qmoe.h"

namespace onnx_light_cpu {

class QMoEKernel final : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  QMoEKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
             const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx);

  static constexpr const char *kName = "onnx_light_cpu::QMoE";

  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &input,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &router,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &fc1,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &scales1,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *bias1,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &fc2,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &scales2,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *bias2,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *fc3 = nullptr,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *scales3 = nullptr,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *bias3 = nullptr,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;

  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;

private:
  QMoEAttributes attributes_;
};

void RegisterQMoEKernel();

} // namespace onnx_light_cpu
