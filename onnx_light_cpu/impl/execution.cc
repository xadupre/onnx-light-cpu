// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/execution.h"

#ifdef ONNX_LIGHT_CPU_WITH_ONNX_LIGHT
#include "onnx_core/runtime/tuning/cpu_executor.h"
#endif

namespace onnx_light_cpu {
namespace {

const ExecutionExecutorView *&CurrentExecutorSlot() noexcept {
  thread_local const ExecutionExecutorView *executor = nullptr;
  return executor;
}

#ifdef ONNX_LIGHT_CPU_WITH_ONNX_LIGHT

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

struct CpuExecutorDispatch {
  void *task_context;
  ExecutionBlockFn task;
};

void RunCpuExecutorBlocks(void *context, int64_t num_blocks, void *task_context,
                          ExecutionBlockFn task) {
  auto *executor = static_cast<rt_ns::CpuExecutor *>(context);
  CpuExecutorDispatch dispatch{task_context, task};
  executor->ParallelFor(
      num_blocks, 1, &dispatch,
      [](void *dispatch_context, int64_t begin, int64_t end) {
        auto &current = *static_cast<CpuExecutorDispatch *>(dispatch_context);
        for (int64_t block = begin; block < end; ++block) {
          current.task(current.task_context, block);
        }
      },
      static_cast<uint32_t>(num_blocks));
}

const ExecutionExecutorView *CurrentRuntimeExecutor() noexcept {
  if (rt_ns::CpuExecutor *executor = rt_ns::CurrentCpuExecutor(); executor != nullptr) {
    thread_local ExecutionExecutorView view;
    view.context = executor;
    view.effective_threads = static_cast<int64_t>(executor->effective_threads());
    view.run_blocks = &RunCpuExecutorBlocks;
    return &view;
  }
  return nullptr;
}

#else

const ExecutionExecutorView *CurrentRuntimeExecutor() noexcept { return nullptr; }

#endif

int &ExecutionDepthSlot() noexcept {
  thread_local int depth = 0;
  return depth;
}

} // namespace

const ExecutionExecutorView *CurrentExecutionExecutor() noexcept {
  if (const ExecutionExecutorView *executor = CurrentExecutorSlot(); executor != nullptr) {
    return executor;
  }
  return CurrentRuntimeExecutor();
}

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
