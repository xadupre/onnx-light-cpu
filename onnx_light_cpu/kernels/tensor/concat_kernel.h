// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

#include <span>

namespace onnx_light_cpu {

/// Standard variadic ONNX Concat for every fixed-byte dtype supported by onnx-light.
class ConcatKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  using ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase::KernelBase;
  ConcatKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
               const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx);

  static constexpr const char *kName = "onnx_light_cpu::Concat";
  static constexpr bool CanRunInPlace() noexcept { return false; }
  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;

  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(std::span<const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor> inputs, int64_t axis,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;

  /// Output must match the inferred dtype, shape and byte size, and not overlap any input.
  void operator()(std::span<const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor> inputs, int64_t axis,
                  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &output) const;
};

void RegisterConcatKernel();

} // namespace onnx_light_cpu
