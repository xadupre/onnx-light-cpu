// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>

namespace onnx_light_cpu {

using ExecutionBlockFn = void (*)(void *, int64_t);

struct ExecutionExecutorView {
  void *context = nullptr;
  int64_t effective_threads = 1;
  void (*run_blocks)(void *context, int64_t num_blocks, void *task_context,
                     ExecutionBlockFn task) = nullptr;
};

struct ExecutionSchedule {
  int64_t min_parallel_size = 1;
  int64_t min_block_size = 1;
  int64_t max_participants = 1;
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
