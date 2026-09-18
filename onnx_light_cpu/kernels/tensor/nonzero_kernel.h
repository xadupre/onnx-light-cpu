// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

namespace onnx_light_cpu {

/// BOOL NonZero with a count-sized INT64 output in row-major index order.
class NonZeroKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  explicit NonZeroKernel(const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx);
  NonZeroKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx);

  static constexpr const char *kName = "onnx_light_cpu::NonZero";
  static constexpr bool CanRunInPlace() noexcept { return false; }

  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &x,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;
};

void RegisterNonZeroKernel();

} // namespace onnx_light_cpu
