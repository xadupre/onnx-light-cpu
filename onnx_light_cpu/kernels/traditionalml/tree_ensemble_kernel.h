// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu/impl/traditionalml/tree_ensemble.h"

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/runtime_context.h"

#include <cstdint>
#include <memory>
#include <mutex>

namespace onnx_light_cpu {

/// Runtime kernel adapter for ``ai.onnx.ml::TreeEnsemble`` version 5.
class TreeEnsembleKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  explicit TreeEnsembleKernel(const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx);

  static constexpr const char *kName = "onnx_light_cpu::TreeEnsemble";

  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;

private:
  std::once_flag initialize_once_;
  std::unique_ptr<TreeEnsemblePlan> plan_;
  std::int32_t input_data_type_ = 0;
  std::int64_t feature_count_ = 0;
};

void RegisterTreeEnsembleKernel();

} // namespace onnx_light_cpu
