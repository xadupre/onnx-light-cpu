// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cpu_compute/execution.h"

namespace onnx_light_cpu {

// Borrows the current onnx-light executor without replacing an explicit compute executor.
class RuntimeExecutionScope {
public:
  RuntimeExecutionScope() noexcept;

private:
  cpu_compute::ExecutionExecutorView view_;
  cpu_compute::ExecutionExecutorScope scope_;
};

} // namespace onnx_light_cpu
