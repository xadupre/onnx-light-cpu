// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

namespace onnx_light_cpu {

/// Shape-preserving numeric Cast, with concrete builtin compatibility for
/// STRING, float8 and packed sub-byte formats (no global dispatch recursion).
class CastKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  explicit CastKernel(const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx);
  CastKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
             const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx);

  static constexpr const char *kName = "onnx_light_cpu::Cast";
  static constexpr bool CanRunInPlace() noexcept { return false; }

  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &data, int32_t to,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &data, int32_t to, bool saturate,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;

  /// Validates dtype, shape, storage and non-overlap before any output writes.
  void operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &data, int32_t to,
                  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &output) const;
  void operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &data, int32_t to,
                  bool saturate, ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &output) const;

private:
  int32_t to_ = -1;
  bool saturate_ = true;
};

void RegisterCastKernel();

} // namespace onnx_light_cpu
