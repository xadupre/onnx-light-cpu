// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

#include <cstdint>
#include <string>
#include <utility>

#ifndef ONNX_LIGHT_NAMESPACE
#define ONNX_LIGHT_NAMESPACE onnx_light
#endif

namespace onnx_light_cpu {

class NaiveCausalConvWithStateKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  using ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase::KernelBase;
  static constexpr const char *kName = "onnx_light_cpu::NaiveCausalConvWithState";

  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;

  std::pair<ONNX_LIGHT_NAMESPACE::core::runtime::Tensor,
            ONNX_LIGHT_NAMESPACE::core::runtime::Tensor>
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &input,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &weight,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *bias = nullptr,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *past_state = nullptr,
             const std::string &activation = "none", bool channels_last = false,
             std::int64_t dilation = 1,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;
};

void RegisterNaiveCausalConvWithStateKernel();

} // namespace onnx_light_cpu
