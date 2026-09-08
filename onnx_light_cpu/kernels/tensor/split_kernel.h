// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

#include <initializer_list>
#include <span>
#include <vector>

namespace onnx_light_cpu {

/// Standard ONNX Split for all supported fixed-byte tensor types.
class SplitKernel : public ONNX_LIGHT_NAMESPACE::core::runtime::KernelBase {
public:
  explicit SplitKernel(const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx)
      : KernelBase(ctx) {}
  SplitKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
              const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx);

  static constexpr const char *kName = "onnx_light_cpu::Split";
  static constexpr bool CanRunInPlace() noexcept { return false; }

  void Run(ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext &rt) override;

  /// Exactly one of nonempty split sizes or positive num_outputs is required.
  /// num_outputs uses Split-18 ceil partitioning with a smaller final chunk.
  /// Outputs always own storage and preserve the input's bytes verbatim.
  std::vector<ONNX_LIGHT_NAMESPACE::core::runtime::Tensor>
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &data, int64_t axis,
             std::span<const int64_t> split, int64_t num_outputs,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const;

  std::vector<ONNX_LIGHT_NAMESPACE::core::runtime::Tensor>
  operator()(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &data, int64_t axis,
             std::initializer_list<int64_t> split, int64_t num_outputs,
             ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt = nullptr) const {
    return (*this)(data, axis, std::span<const int64_t>(split.begin(), split.size()), num_outputs,
                   rt);
  }

private:
  std::vector<ONNX_LIGHT_NAMESPACE::core::runtime::Tensor>
  Compute(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &data, int64_t axis,
          std::span<const int64_t> split, int64_t num_outputs,
          ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt,
          const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *split_input) const;
};

void RegisterSplitKernel();

} // namespace onnx_light_cpu
