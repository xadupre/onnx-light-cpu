// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

#include <cstdint>

namespace onnx_light_cpu {

struct GatherBlockQuantizedAttributes {
  int64_t gather_axis = 0;
  int64_t quantize_axis = 1;
  int64_t block_size = 128;
  int64_t bits = 4;
};

class GatherBlockQuantizedKernel final : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  explicit GatherBlockQuantizedKernel(const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx,
                                      GatherBlockQuantizedAttributes attributes = {});
  GatherBlockQuantizedKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                             const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx);

  static constexpr const char *kName = "onnx_light_cpu::GatherBlockQuantized";
  static constexpr bool CanRunInPlace() noexcept { return false; }

  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &data,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &indices,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &scales,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *zero_points = nullptr,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;
  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;

private:
  GatherBlockQuantizedAttributes attributes_;
};

void RegisterGatherBlockQuantizedKernel();

} // namespace onnx_light_cpu
