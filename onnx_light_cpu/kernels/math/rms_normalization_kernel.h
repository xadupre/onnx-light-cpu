// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

#include <cstdint>

#ifndef ONNX_LIGHT_NAMESPACE
#define ONNX_LIGHT_NAMESPACE onnx_light
#endif

namespace onnx_light_cpu {

class RmsNormalizationKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  using ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase::KernelBase;
  static constexpr const char *kName = "onnx_light_cpu::RMSNormalization";

  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &x,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &scale, std::int64_t axis = -1,
             float epsilon = 1.0e-5f, std::int64_t stash_type = 1,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;
};

void RegisterRmsNormalizationKernel();

} // namespace onnx_light_cpu
