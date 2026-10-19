// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/execution.h"

#include "onnx_core/runtime/tuning/cpu_executor.h"

namespace onnx_light_cpu {
namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

struct CpuExecutorDispatch {
  void *task_context;
  ml_cpu_compute::ExecutionBlockFn task;
};

void RunCpuExecutorBlocks(void *context, int64_t num_blocks, void *task_context,
                          ml_cpu_compute::ExecutionBlockFn task) {
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

ml_cpu_compute::ExecutionParallelPlan PlanCpuExecutor(void *context, int64_t total,
                                                      const ml_cpu_compute::ExecutionWorkCost &cost,
                                                      int64_t maximum_participants,
                                                      int64_t preferred_participants) {
  auto *executor = static_cast<rt_ns::CpuExecutor *>(context);
  const rt_ns::CpuParallelPlan plan = executor->PlanParallelFor(
      total, rt_ns::CpuLoopCost{cost.bytes_read, cost.bytes_written, cost.compute_cycles},
      rt_ns::CpuParallelConstraints{
          static_cast<uint32_t>(std::max<int64_t>(maximum_participants, 1)),
          static_cast<uint32_t>(std::max<int64_t>(preferred_participants, 0))});
  return {plan.grain_size, static_cast<int64_t>(plan.participants)};
}

ml_cpu_compute::ExecutionExecutorView RuntimeExecutorView() noexcept {
  auto *executor = rt_ns::CurrentCpuExecutor();
  return executor == nullptr ? ml_cpu_compute::ExecutionExecutorView{}
                             : ml_cpu_compute::ExecutionExecutorView{
                                   executor, static_cast<int64_t>(executor->effective_threads()),
                                   &RunCpuExecutorBlocks, &PlanCpuExecutor};
}

} // namespace

RuntimeExecutionScope::RuntimeExecutionScope() noexcept
    : view_(RuntimeExecutorView()), scope_(ml_cpu_compute::HasExecutionExecutorScope()
                                               ? ml_cpu_compute::CurrentExecutionExecutor()
                                           : view_.context != nullptr ? &view_
                                                                      : nullptr) {}

} // namespace onnx_light_cpu
