// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu_compute/execution.h"

namespace onnx_light_cpu {

// Borrows the current onnx-light executor without replacing an explicit compute executor.
class RuntimeExecutionScope {
public:
  RuntimeExecutionScope() noexcept;

private:
  ExecutionExecutorView view_;
  ExecutionExecutorScope scope_;
};

} // namespace onnx_light_cpu
