// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/execution.h"

namespace onnx_light_cpu {
namespace {

const ExecutionExecutorView *&CurrentExecutorSlot() noexcept {
  thread_local const ExecutionExecutorView *executor = nullptr;
  return executor;
}

int &ExecutionDepthSlot() noexcept {
  thread_local int depth = 0;
  return depth;
}

} // namespace

const ExecutionExecutorView *CurrentExecutionExecutor() noexcept { return CurrentExecutorSlot(); }

ExecutionExecutorScope::ExecutionExecutorScope(const ExecutionExecutorView *executor) noexcept
    : previous_(CurrentExecutorSlot()) {
  CurrentExecutorSlot() = executor;
}

ExecutionExecutorScope::~ExecutionExecutorScope() { CurrentExecutorSlot() = previous_; }

namespace detail {

ExecutionRegionScope::ExecutionRegionScope() noexcept { ++ExecutionDepthSlot(); }

ExecutionRegionScope::~ExecutionRegionScope() { --ExecutionDepthSlot(); }

int ExecutionRegionDepth() noexcept { return ExecutionDepthSlot(); }

} // namespace detail
} // namespace onnx_light_cpu
