// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu/impl/parallel_for.h"

#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/tuning/cpu_executor.h"

#include <cstdint>
#include <memory>
#include <stdexcept>

namespace onnx_light_cpu {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
using NodeProto = ONNX_LIGHT_NAMESPACE::NodeProto;

/**
 * Installs an onnx-light session executor for onnx-light-cpu ParallelFor calls.
 */
class SessionExecutorAdapter {
public:
  explicit SessionExecutorAdapter(rt_ns::RuntimeContext &runtime)
      : executor_(runtime.cpu_executor()),
        view_{this, executor_ == nullptr ? 1 : static_cast<int64_t>(executor_->effective_threads()),
              &SessionExecutorAdapter::RunBlocks},
        scope_(&view_) {
    if (executor_ == nullptr) {
      throw std::invalid_argument(
          "Registered onnx-light-cpu kernels require a session CpuExecutor.");
    }
  }

private:
  struct Dispatch {
    const ParallelForExecutorView *view;
    void *task_context;
    ParallelForBlockFn task;
  };

  static void RunBlocks(void *context, int64_t num_blocks, void *task_context,
                        ParallelForBlockFn task) {
    auto &self = *static_cast<SessionExecutorAdapter *>(context);
    Dispatch dispatch{&self.view_, task_context, task};
    self.executor_->ParallelFor(
        num_blocks, 1, &dispatch,
        [](void *dispatch_context, int64_t begin, int64_t end) {
          auto &current = *static_cast<Dispatch *>(dispatch_context);
          ParallelForExecutorScope executor_scope(current.view);
          for (int64_t block = begin; block < end; ++block) {
            current.task(current.task_context, block);
          }
        },
        static_cast<uint32_t>(num_blocks));
  }

  rt_ns::CpuExecutor *executor_;
  ParallelForExecutorView view_;
  ParallelForExecutorScope scope_;
};

/**
 * Wraps a registered kernel so its internal parallel regions use the session
 * executor while direct standalone instances retain the CPU library pool.
 */
template <typename Kernel> class SessionKernel final : public Kernel {
public:
  SessionKernel(const NodeProto &node, rt_ns::RuntimeContext &runtime)
      : Kernel(runtime.kernel_ctx()) {
    if (runtime.cpu_executor() == nullptr) {
      throw std::invalid_argument(
          "Registered onnx-light-cpu kernels require a session CpuExecutor.");
    }
    this->set_node(node);
  }

  void Run(rt_ns::RuntimeContext &runtime) override {
    SessionExecutorAdapter executor(runtime);
    Kernel::Run(runtime);
  }
};

template <typename Kernel>
std::unique_ptr<rt_ns::KernelBase> MakeSessionKernel(const NodeProto &node,
                                                     rt_ns::RuntimeContext &runtime) {
  return std::make_unique<SessionKernel<Kernel>>(node, runtime);
}

} // namespace onnx_light_cpu
