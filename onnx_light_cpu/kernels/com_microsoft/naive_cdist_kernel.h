// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

#include <string>

#ifndef ONNX_LIGHT_NAMESPACE
#define ONNX_LIGHT_NAMESPACE onnx_light
#endif

namespace onnx_light_cpu {

/// Readable scalar correctness oracle for ``com.microsoft::CDist``.
class NaiveCDistKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  using ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase::KernelBase;
  static constexpr const char *kName = "onnx_light_cpu::NaiveCDist";

  NaiveCDistKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                   const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx);

  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;

  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &a,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &b,
             const std::string &metric = "sqeuclidean",
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;

  void operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &a,
                  const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &b, const std::string &metric,
                  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &output) const;
};

void RegisterNaiveCDistKernel();

} // namespace onnx_light_cpu
