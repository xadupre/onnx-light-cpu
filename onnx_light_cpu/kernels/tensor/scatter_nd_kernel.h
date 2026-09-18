// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

#include <string>

namespace onnx_light_cpu {

/// ScatterND replacement (opset 11+), with INT32 indices as an extension.
/// Duplicate destinations are updated in tuple order (last tuple wins).
class ScatterNDKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  explicit ScatterNDKernel(const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx,
                           const std::string &reduction = "none");
  ScatterNDKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                  const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx);

  static constexpr const char *kName = "onnx_light_cpu::ScatterND";
  static constexpr bool CanRunInPlace() noexcept { return false; }

  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &data,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &indices,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &updates,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;

  /// Output must not overlap any input. Validation precedes all writes.
  void operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &data,
                  const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &indices,
                  const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &updates,
                  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &output) const;
};

void RegisterScatterNDKernel();

} // namespace onnx_light_cpu
