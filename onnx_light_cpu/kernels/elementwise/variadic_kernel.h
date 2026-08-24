// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu/impl/math/binary/variadic_elementwise_plan.h"

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

#ifndef ONNX_LIGHT_NAMESPACE
#define ONNX_LIGHT_NAMESPACE onnx_light
#endif

namespace onnx_light_cpu {

class VariadicElementwiseKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  VariadicElementwiseKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                            const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx);

  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;

  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensors &inputs,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;

  void operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensors &inputs,
                  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &output) const;

private:
  VariadicOperator op_;
};

void RegisterVariadicElementwiseKernels();

} // namespace onnx_light_cpu
