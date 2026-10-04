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

/// CPU ReduceSum implementation for INT64, preserving onnx-light's other types.
class ReduceSumKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  using KernelBase::KernelBase;

  ReduceSumKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                  const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx);

  static constexpr const char *kName = "onnx_light_cpu::ReduceSum";

  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;

private:
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  Compute(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &data,
          const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *axes, bool keepdims,
          bool noop_with_empty_axes, ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) const;
};

void RegisterReduceSumKernel();

} // namespace onnx_light_cpu
