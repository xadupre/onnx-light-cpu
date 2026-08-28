// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#ifndef ONNX_LIGHT_NAMESPACE
#define ONNX_LIGHT_NAMESPACE onnx_light
#endif

namespace onnx_light_cpu {

struct BatchNormalizationResult {
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor y;
  std::optional<ONNX_LIGHT_NAMESPACE::core::runtime::Tensor> running_mean;
  std::optional<ONNX_LIGHT_NAMESPACE::core::runtime::Tensor> running_variance;
};

class BatchNormalizationKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  using ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase::KernelBase;
  static constexpr const char *kName = "onnx_light_cpu::BatchNormalization";

  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &x,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &scale,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &bias,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &mean,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &variance, float epsilon = 1.0e-5f,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;
  BatchNormalizationResult
  Compute(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &x,
          const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &scale,
          const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &bias,
          const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &mean,
          const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &variance, bool training_mode,
          float epsilon = 1.0e-5f, float momentum = 0.9f,
          ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr,
          bool output_running_mean = true, bool output_running_variance = true) const;
};

class GroupNormalizationKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  using ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase::KernelBase;
  static constexpr const char *kName = "onnx_light_cpu::GroupNormalization";

  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &x,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &scale,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &bias, std::int64_t num_groups,
             float epsilon = 1.0e-5f, std::int64_t stash_type = 1,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;
};

class InstanceNormalizationKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  using ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase::KernelBase;
  static constexpr const char *kName = "onnx_light_cpu::InstanceNormalization";

  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &x,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &scale,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &bias, float epsilon = 1.0e-5f,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;
};

struct LayerNormalizationResult {
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor y;
  std::optional<ONNX_LIGHT_NAMESPACE::core::runtime::Tensor> mean;
  std::optional<ONNX_LIGHT_NAMESPACE::core::runtime::Tensor> inv_std_dev;
};

class LayerNormalizationKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  using ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase::KernelBase;
  static constexpr const char *kName = "onnx_light_cpu::LayerNormalization";

  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;
  LayerNormalizationResult
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &x,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &scale,
             const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *bias = nullptr,
             std::int64_t axis = -1, float epsilon = 1.0e-5f, std::int64_t stash_type = 1,
             bool output_mean = false, bool output_inv_std_dev = false,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;
};

class LpNormalizationKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  using ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase::KernelBase;
  static constexpr const char *kName = "onnx_light_cpu::LpNormalization";

  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &x, std::int64_t axis = -1,
             std::int64_t p = 2,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;
};

class MeanVarianceNormalizationKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  using ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase::KernelBase;
  static constexpr const char *kName = "onnx_light_cpu::MeanVarianceNormalization";

  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &x,
             const std::vector<std::int64_t> &axes = {0, 2, 3},
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;
};

void RegisterNormalizationKernels();

} // namespace onnx_light_cpu
