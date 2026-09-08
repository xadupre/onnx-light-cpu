// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

namespace onnx_light_cpu {

/// ONNX Slice with modern tensor inputs and legacy opset 1-9 node attributes.
class SliceKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  using ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase::KernelBase;
  SliceKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
              const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx);

  static constexpr const char *kName = "onnx_light_cpu::Slice";
  static constexpr bool CanRunInPlace() noexcept { return false; }
  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;

  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &data,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &starts,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &ends,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *axes = nullptr,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *steps = nullptr,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;

  /// Output must have the expected dtype, shape, byte size and not overlap any input.
  void operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &data,
                  const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &starts,
                  const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &ends,
                  const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *axes,
                  const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *steps,
                  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &output) const;
};

void RegisterSliceKernel();

} // namespace onnx_light_cpu
