// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

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
  bool HasPreparations(const std::unordered_set<std::string> &immutable_inputs) const override;
  void DeclarePreparations(
      const std::unordered_set<std::string> &immutable_inputs,
      std::vector<ONNX_LIGHT_NAMESPACE::core::runtime::KernelPreparation> &preparations) override;

  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &a,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &b,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;

private:
  struct MatMulPlanCache;
  struct PreparedWeights;
  std::unique_ptr<MatMulPlanCache> plan_cache_;
  std::unique_ptr<PreparedWeights> prepared_weights_;

  static ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  Compute(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &a,
          const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &b,
          ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt, MatMulPlanCache *cache,
          const PreparedWeights *prepared);
};

void RegisterMatMulKernel();

} // namespace onnx_light_cpu
