// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

#include <memory>

namespace onnx_light_cpu {

/// SIMD-accelerated kernel for the ONNX ``MatMul`` operator.
class MatMulKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  explicit MatMulKernel(const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx);
  MatMulKernel(const MatMulKernel &) = delete;
  MatMulKernel &operator=(const MatMulKernel &) = delete;
  ~MatMulKernel() override;

  static constexpr const char *kName = "onnx_light_cpu::MatMul";

  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;

  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &a,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &b,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;

private:
  struct MatMulPlanCache;
  std::unique_ptr<MatMulPlanCache> plan_cache_;

  static ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  Compute(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &a,
          const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &b,
          ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt, MatMulPlanCache *cache);
};

void RegisterMatMulKernel();

} // namespace onnx_light_cpu
