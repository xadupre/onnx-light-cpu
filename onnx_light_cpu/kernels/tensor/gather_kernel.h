// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

namespace onnx_light_cpu {

/// Standard ONNX Gather for every fixed-byte type supported by onnx-light.
class GatherKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  explicit GatherKernel(const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx,
                        int64_t axis = 0);
  GatherKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
               const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx);

  static constexpr const char *kName = "onnx_light_cpu::Gather";
  static constexpr bool CanRunInPlace() noexcept { return false; }

  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &data,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &indices,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;

  /// Validates output dtype, shape, size and non-overlap before writing.
  void operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &data,
                  const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &indices,
                  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &output) const;

private:
  int64_t axis_;
};

void RegisterGatherKernel();

} // namespace onnx_light_cpu
