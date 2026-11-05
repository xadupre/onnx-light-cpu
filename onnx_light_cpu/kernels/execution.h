// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ml_cpu_compute/execution.h"
#include "onnx_core/runtime/runtime_context.h"

#include <optional>

namespace onnx_light_cpu {

// Borrows the current executor and, when supplied, runtime scratch allocator.
// Explicit compute executor and allocator scopes are preserved independently.
class RuntimeExecutionScope {
public:
  explicit RuntimeExecutionScope(
      ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *runtime = nullptr) noexcept;

private:
  ml_cpu_compute::ExecutionExecutorView view_;
  ml_cpu_compute::ExecutionExecutorScope scope_;
  ml_cpu_compute::ExecutionAllocatorView allocator_view_;
  std::optional<ml_cpu_compute::ExecutionAllocatorScope> allocator_scope_;
};

} // namespace onnx_light_cpu
