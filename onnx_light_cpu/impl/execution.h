// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>

namespace onnx_light_cpu {

// CpuExecutor bridge contract for kernel implementations
//
// Ownership and lifetime:
// - onnx-light owns the CpuExecutor and its workers; the session keeps an executor
//   lease alive and installs it for the duration of a run. This bridge only borrows
//   it and never creates a pool or extends the executor lifetime.
// - CurrentExecutionExecutor() returns a non-owning, thread-local view, or nullptr
//   without an installed executor. The runtime adapter reuses that view on later
//   lookups; its address is not a stable executor identity.
// - An explicit ExecutionExecutorScope borrows its view and context on the calling
//   thread and restores the previous binding on destruction, including unwinding.
//   The installer must keep both alive through the scope and all submitted work.
//   A nullptr scope removes the explicit override, allowing runtime lookup again.
//
// Dispatch and failures:
// - run_blocks must be valid whenever parallel dispatch is possible. It invokes
//   task(task_context, i) exactly once for each i in [0, num_blocks) on success.
//   Dispatch is synchronous: no callback may remain active after run_blocks exits,
//   including on failure. It must not retain task_context or the callable.
// - Range callbacks may run concurrently, in any order, on the caller or workers.
//   Captures and buffers must stay alive until the helper finishes; writes must be
//   disjoint or synchronized. total <= 0 invokes no callback.
// - Range callbacks MUST NOT throw, matching CpuExecutor::ParallelFor. Validate
//   before dispatch, or record errors safely and report them after completion.
//   There is no worker-exception capture/rethrow guarantee; throwing on a worker
//   can terminate the process. Inline callbacks currently propagate exceptions,
//   but kernels must not depend on whether a particular call executes inline.
// - Exceptions from planning or dispatch itself propagate to the kernel caller;
//   the bridge does not catch them, translate them to status codes, or roll back
//   output writes. plan_parallel is optional; its absence uses the fallback schedule.
//
// Parallelism and thread counts:
// - Submit work only through ExecuteRanges / ExecuteCostedRanges, not a private
//   pool, OpenMP region, or a retained executor. Every range callback, even an
//   inline one, enters a thread-local region; nested helpers execute serially
//   without another executor submission.
// - ExecutionThreadCount() is the current executor's effective participant budget
//   (including the caller), clamped to at least one; without an executor it is one.
//   It is not the number of active workers and does not itself become one in a
//   nested region. Work size, alignment, schedule limits, and cost planning may
//   select fewer participants. ExecutionSchedule::max_participants is clamped to
//   at least one (zero does not mean unlimited); preferred_participants == 0
//   leaves the choice to the cost model.
//
// Retention:
// - Kernels must not cache executor/view/context pointers, callbacks, task contexts,
//   or executor-dependent scheduling state across invocations, nor transfer the
//   thread-local binding to another thread. Reacquire the executor and recompute
//   its scheduling decisions for each invocation. Executor-independent immutable
//   kernel data and tuning constants may be retained.

using ExecutionBlockFn = void (*)(void *, int64_t);

struct ExecutionWorkCost {
  double bytes_read = 0.0;
  double bytes_written = 0.0;
  double compute_cycles = 0.0;
};

struct ExecutionParallelPlan {
  int64_t grain_size = 1;
  int64_t participants = 1;
};

struct ExecutionExecutorView {
  void *context = nullptr;
  int64_t effective_threads = 1;
  void (*run_blocks)(void *context, int64_t num_blocks, void *task_context,
                     ExecutionBlockFn task) = nullptr;
  ExecutionParallelPlan (*plan_parallel)(void *context, int64_t total,
                                         const ExecutionWorkCost &cost,
                                         int64_t maximum_participants,
                                         int64_t preferred_participants) = nullptr;
};

struct ExecutionSchedule {
  // Non-positive values are normalized to one by ExecutionBlockCount.
  int64_t min_parallel_size = 1;
  int64_t min_block_size = 1;
  int64_t max_participants = 1;
  // Zero leaves the participant count to the executor cost model.
  int64_t preferred_participants = 0;
};

const ExecutionExecutorView *CurrentExecutionExecutor() noexcept;

class ExecutionExecutorScope {
public:
  explicit ExecutionExecutorScope(const ExecutionExecutorView *executor) noexcept;
  ExecutionExecutorScope(const ExecutionExecutorScope &) = delete;
  ExecutionExecutorScope &operator=(const ExecutionExecutorScope &) = delete;
  ~ExecutionExecutorScope();

private:
  const ExecutionExecutorView *previous_;
};

namespace detail {

class ExecutionRegionScope {
public:
  ExecutionRegionScope() noexcept;
  ExecutionRegionScope(const ExecutionRegionScope &) = delete;
  ExecutionRegionScope &operator=(const ExecutionRegionScope &) = delete;
  ~ExecutionRegionScope();
};

int ExecutionRegionDepth() noexcept;

} // namespace detail

inline constexpr int64_t kExecutionGrainSize = 1 << 15;
inline constexpr int64_t kExecutionSimdWidthBytes = 64;

inline int64_t ExecutionThreadCount() noexcept {
  const ExecutionExecutorView *executor = CurrentExecutionExecutor();
  return executor == nullptr ? 1 : std::max<int64_t>(executor->effective_threads, 1);
}

inline bool ExecutionInParallelRegion() noexcept { return detail::ExecutionRegionDepth() != 0; }

template <typename T> inline constexpr int64_t ExecutionSimdLanes() noexcept {
  static_assert(sizeof(T) > 0, "element type must be a complete type");
  constexpr int64_t lanes = kExecutionSimdWidthBytes / static_cast<int64_t>(sizeof(T));
  return lanes > 0 ? lanes : 1;
}

inline int64_t ExecutionBlockCount(int64_t total, double cost_per_element = 1.0) {
  if (total <= 0) {
    return 0;
  }
  if (ExecutionInParallelRegion()) {
    return 1;
  }
  const int64_t max_threads = ExecutionThreadCount();
  if (max_threads <= 1) {
    return 1;
  }
  if (!(cost_per_element > 0.0)) {
    cost_per_element = 1.0;
  }
  const double total_work = static_cast<double>(total) * cost_per_element;
  if (total_work < static_cast<double>(kExecutionGrainSize)) {
    return 1;
  }
  const int64_t max_useful_blocks =
      static_cast<int64_t>(total_work / static_cast<double>(kExecutionGrainSize));
  return std::min<int64_t>(total,
                           std::min<int64_t>(max_threads, std::max<int64_t>(1, max_useful_blocks)));
}

inline int64_t ExecutionBlockCount(int64_t total, const ExecutionSchedule &schedule) {
  if (total <= 0) {
    return 0;
  }
  if (ExecutionInParallelRegion() || total < std::max<int64_t>(schedule.min_parallel_size, 1)) {
    return 1;
  }
  const int64_t max_threads =
      std::min(ExecutionThreadCount(), std::max<int64_t>(schedule.max_participants, 1));
  const int64_t useful_blocks = total / std::max<int64_t>(schedule.min_block_size, 1);
  return std::min(max_threads, std::max<int64_t>(useful_blocks, 1));
}

template <typename Fn>
void ExecuteRanges(int64_t total, const ExecutionSchedule &schedule, int64_t block_multiple,
                   Fn fn) {
  if (total <= 0) {
    return;
  }
  const ExecutionExecutorView *executor = CurrentExecutionExecutor();
  int64_t num_blocks = ExecutionBlockCount(total, schedule);
  if (executor == nullptr || num_blocks <= 1) {
    detail::ExecutionRegionScope region;
    fn(0, total);
    return;
  }

  block_multiple = std::max<int64_t>(block_multiple, 1);
  int64_t block = (total + num_blocks - 1) / num_blocks;
  block = ((block + block_multiple - 1) / block_multiple) * block_multiple;
  num_blocks = (total + block - 1) / block;

  auto run_block = [&fn, block, total](int64_t index) {
    const int64_t begin = index * block;
    if (begin >= total) {
      return;
    }
    detail::ExecutionRegionScope region;
    fn(begin, std::min(begin + block, total));
  };
  using Callable = decltype(run_block);
  executor->run_blocks(
      executor->context, num_blocks, static_cast<void *>(&run_block),
      [](void *context, int64_t block_index) { (*static_cast<Callable *>(context))(block_index); });
}

template <typename Fn> void ExecuteRanges(int64_t total, const ExecutionSchedule &schedule, Fn fn) {
  ExecuteRanges(total, schedule, int64_t{1}, std::move(fn));
}

template <typename Fn>
void ExecuteCostedRanges(int64_t total, const ExecutionWorkCost &cost,
                         const ExecutionSchedule &fallback, int64_t block_multiple, Fn fn) {
  if (total <= 0) {
    return;
  }
  const ExecutionExecutorView *executor = CurrentExecutionExecutor();
  if (executor == nullptr || executor->run_blocks == nullptr ||
      executor->plan_parallel == nullptr || ExecutionInParallelRegion()) {
    ExecuteRanges(total, fallback, block_multiple, std::move(fn));
    return;
  }
  const int64_t participant_limit = std::min(std::max<int64_t>(fallback.max_participants, 1),
                                             std::max<int64_t>(executor->effective_threads, 1));
  const ExecutionParallelPlan plan =
      executor->plan_parallel(executor->context, total, cost, participant_limit,
                              std::max<int64_t>(fallback.preferred_participants, 0));
  const int64_t grain = std::max<int64_t>(plan.grain_size, 1);
  const int64_t participants = std::min(participant_limit, std::max<int64_t>(plan.participants, 1));
  const int64_t num_blocks = std::min(participants, total / grain);
  if (num_blocks <= 1) {
    detail::ExecutionRegionScope region;
    fn(0, total);
    return;
  }

  block_multiple = std::max<int64_t>(block_multiple, 1);
  int64_t block = total / num_blocks + (total % num_blocks != 0);
  block += (block_multiple - block % block_multiple) % block_multiple;
  const int64_t aligned_blocks = total / block + (total % block != 0);
  auto run_block = [&fn, block, total](int64_t index) {
    const int64_t begin = index * block;
    if (begin >= total) {
      return;
    }
    detail::ExecutionRegionScope region;
    fn(begin, std::min(begin + block, total));
  };
  using Callable = decltype(run_block);
  executor->run_blocks(
      executor->context, aligned_blocks, static_cast<void *>(&run_block),
      [](void *context, int64_t block_index) { (*static_cast<Callable *>(context))(block_index); });
}

template <typename Fn>
void ExecuteRanges(int64_t total, double cost_per_element, int64_t block_multiple, Fn fn) {
  if (total <= 0) {
    return;
  }
  const ExecutionExecutorView *executor = CurrentExecutionExecutor();
  int64_t num_blocks = ExecutionBlockCount(total, cost_per_element);
  if (executor == nullptr || num_blocks <= 1) {
    detail::ExecutionRegionScope region;
    fn(0, total);
    return;
  }

  block_multiple = std::max<int64_t>(block_multiple, 1);
  int64_t block = (total + num_blocks - 1) / num_blocks;
  block = ((block + block_multiple - 1) / block_multiple) * block_multiple;
  num_blocks = (total + block - 1) / block;

  auto run_block = [&fn, block, total](int64_t index) {
    const int64_t begin = index * block;
    if (begin >= total) {
      return;
    }
    detail::ExecutionRegionScope region;
    fn(begin, std::min(begin + block, total));
  };
  using Callable = decltype(run_block);
  executor->run_blocks(
      executor->context, num_blocks, static_cast<void *>(&run_block),
      [](void *context, int64_t block_index) { (*static_cast<Callable *>(context))(block_index); });
}

template <typename Fn> void ExecuteRanges(int64_t total, double cost_per_element, Fn fn) {
  ExecuteRanges(total, cost_per_element, int64_t{1}, std::move(fn));
}

template <typename Fn> void ExecuteRanges(int64_t total, Fn fn) {
  ExecuteRanges(total, 1.0, int64_t{1}, std::move(fn));
}

} // namespace onnx_light_cpu
