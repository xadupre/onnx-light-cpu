// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu/impl/com_microsoft/matmul_nbits.h"

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

#include <cstdint>

namespace onnx_light_cpu {

struct MatMulNBitsAttributes {
  std::int64_t k = 0;
  std::int64_t n = 0;
  std::int64_t bits = 4;
  std::int64_t block_size = 0;
  std::int64_t accuracy_level = 0;
  std::int64_t weight_prepacked = 0;
};

class MatMulNBitsKernel final : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  MatMulNBitsKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                    const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx);

  static constexpr const char *kName = "onnx_light_cpu::MatMulNBits";
  static constexpr std::uint32_t kTuningAbi = 1;

  static void RegisterTuningSchemas();
  ONNX_LIGHT_NAMESPACE::core::runtime::KernelTuningKey
  TuningKey(std::int32_t element_type) const override;
  void Configure(
      const ONNX_LIGHT_NAMESPACE::core::runtime::KernelTuningParameters &parameters) override;

  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &a,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &b,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &scales,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *bias = nullptr,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;

  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;

private:
  MatMulNBitsAttributes attributes_;
  MatMulNBitsExecutionTuning tuning_ = kDefaultMatMulNBitsExecutionTuning;
};

void RegisterMatMulNBitsKernel();

} // namespace onnx_light_cpu
