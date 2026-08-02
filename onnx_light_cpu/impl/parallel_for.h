// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

/**
 * @file parallel_for.h
 * @brief Persistent thread pool and cost-aware block-parallel iteration helper
 *        for element-wise kernels.
 *
 * The public entry point is :cpp:func:`ParallelFor`. Before dispatching any
 * worker threads it consults a small cost model
 * (:cpp:func:`ParallelForBlockCount`) that decides *whether* parallelising is
 * worthwhile: waking worker threads has a fixed latency, so tiny ranges (or
 * cheap per-element work) stay on the calling thread where they are faster.
 */

namespace onnx_light_cpu {

/// Approximate amount of per-element work, in abstract "work units", below
/// which running a block on its own thread is slower than running it inline.
///
/// One work unit is roughly the cost of a single trivial element-wise
/// operation (a load, a cheap arithmetic op and a store). A block must carry at
/// least this many work units to justify the thread-wakeup latency; the cost
/// model in :cpp:func:`ParallelForBlockCount` never creates a block smaller
/// than this. The value mirrors the grain size used by common element-wise
/// runtimes (32768 trivial iterations).
inline constexpr int64_t kParallelForGrainSize = 1 << 15; // 32768 work units

/// Returns the number of participating threads :cpp:func:`ParallelFor` may use.
///
/// Resolves to ``std::thread::hardware_concurrency()``, falling back to ``1``
/// when the hardware count is not available. The result is always ``>= 1`` and
/// counts the calling thread, which always participates in the work.
inline int64_t ParallelForThreadCount() noexcept {
  const unsigned int cores = std::thread::hardware_concurrency();
  return cores == 0 ? 1 : static_cast<int64_t>(cores);
}

/**
 * Cost model deciding how many blocks :cpp:func:`ParallelFor` should use.
 *
 * The decision combines the *processor* (how many hardware threads are
 * available, via :cpp:func:`ParallelForThreadCount`) with an *estimate of the
 * loop cost* (``total`` iterations, each costing about ``cost_per_element``
 * work units, see :cpp:var:`kParallelForGrainSize`). Parallelising only pays
 * off once the total estimated work exceeds one grain; otherwise the whole
 * range runs inline on the calling thread.
 *
 * @param total            Number of iterations. Values ``<= 0`` yield ``0``.
 * @param cost_per_element Relative cost of a single iteration in work units
 *                         (``1.0`` for a trivial element-wise op). Non-positive
 *                         values are treated as ``1.0``. Heavier kernels (e.g.
 *                         ``exp``/``log``) should pass a value ``> 1`` so that
 *                         smaller ranges still parallelise.
 * @return The number of contiguous blocks to split ``[0, total)`` into. A
 *         return value of ``1`` means "run inline, do not parallelize".
 */
inline int64_t ParallelForBlockCount(int64_t total, double cost_per_element = 1.0) noexcept {
  if (total <= 0) {
    return 0;
  }
  const int64_t max_threads = ParallelForThreadCount();
  if (max_threads <= 1) {
    return 1;
  }
  if (!(cost_per_element > 0.0)) { // also guards NaN
    cost_per_element = 1.0;
  }
  // Estimated total work expressed in grain-size units.
  const double total_work = static_cast<double>(total) * cost_per_element;
  if (total_work < static_cast<double>(kParallelForGrainSize)) {
    // Too little work: thread-wakeup latency would dominate.
    return 1;
  }
  // Use as many blocks as participants, but never so many that a block would
  // hold less than one grain of work.
  const int64_t max_useful_blocks =
      static_cast<int64_t>(total_work / static_cast<double>(kParallelForGrainSize));
  int64_t num_blocks =
      std::min<int64_t>(max_threads, std::max<int64_t>(int64_t{1}, max_useful_blocks));
  // Never create more blocks than there are iterations.
  num_blocks = std::min<int64_t>(num_blocks, total);
  return num_blocks < 1 ? 1 : num_blocks;
}

/**
 * A persistent pool of worker threads that stay alive between parallel regions.
 *
 * Unlike spawning fresh ``std::thread`` objects per call, the workers are
 * created once and parked on a condition variable, so dispatching a region only
 * costs a notify plus a wait instead of thread creation/teardown. This keeps the
 * per-call overhead low for kernels invoked many times.
 *
 * The pool exposes a single primitive, :cpp:func:`Run`, that executes a set of
 * indexed blocks with a static assignment: block ``0`` runs on the calling
 * thread and block ``j`` runs on worker ``j - 1``. It deliberately offers no
 * reduction/combine step: callers write disjoint output ranges, so results are
 * independent of how blocks map to threads (bit-exact). The pool handles several
 * scenarios:
 *   - no workers available (single core): every block runs inline on the caller;
 *   - a single block: runs inline without touching the workers;
 *   - nested calls from inside a running block: run inline to avoid deadlock;
 *   - concurrent calls from unrelated threads: serialized so one region runs at
 *     a time, each still internally parallel.
 */
class ThreadPool {
public:
  /// Type-erased block callable: ``fn(context, block_index)``.
  using TaskFn = void (*)(void *, int64_t);

  /// Creates a pool with ``num_workers`` parked worker threads.
  ///
  /// @param num_workers Number of worker threads to spawn. Values ``<= 0``
  ///                    create a pool with no workers, in which case
  ///                    :cpp:func:`Run` executes every block on the caller.
  explicit ThreadPool(int64_t num_workers) {
    if (num_workers < 0) {
      num_workers = 0;
    }
    workers_.reserve(static_cast<size_t>(num_workers));
    for (int64_t i = 0; i < num_workers; ++i) {
      workers_.emplace_back([this, i]() { WorkerLoop(i); });
    }
  }

  ThreadPool(const ThreadPool &) = delete;
  ThreadPool &operator=(const ThreadPool &) = delete;

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
      ++generation_;
    }
    cv_work_.notify_all();
    for (std::thread &worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  /// Returns the number of worker threads (the calling thread is not counted).
  int64_t worker_count() const noexcept { return static_cast<int64_t>(workers_.size()); }

  /**
   * Runs ``fn(block)`` for every ``block`` in ``[0, num_blocks)``, then blocks
   * until all blocks finish.
   *
   * Block ``0`` runs on the calling thread and block ``j`` runs on worker
   * ``j - 1`` (static assignment). ``fn`` is invoked concurrently and must only
   * touch data disjoint per block; it must not throw. ``num_blocks`` must not
   * exceed ``worker_count() + 1`` when workers are used; :cpp:func:`ParallelFor`
   * enforces this.
   *
   * @param num_blocks Number of blocks to run. Values ``<= 0`` are a no-op.
   * @param fn         Callable invoked as ``fn(int64_t block)``.
   */
  template <typename Fn> void Run(int64_t num_blocks, Fn &&fn) {
    if (num_blocks <= 0) {
      return;
    }
    if (workers_.empty() || num_blocks == 1 || InPool()) {
      // No workers, a single block, or a nested call from within a running
      // block: run inline serially to stay correct and deadlock-free.
      for (int64_t b = 0; b < num_blocks; ++b) {
        fn(b);
      }
      return;
    }

    using Callable = std::remove_reference_t<Fn>;
    Callable &callable = fn;
    TaskFn invoker = [](void *ctx, int64_t b) { (*static_cast<Callable *>(ctx))(b); };

    // Serialize regions so a single set of shared fields describes one active
    // region at a time, even when unrelated threads call Run concurrently.
    std::lock_guard<std::mutex> region(region_mu_);
    {
      std::lock_guard<std::mutex> lock(mu_);
      task_ctx_ = static_cast<void *>(std::addressof(callable));
      task_fn_ = invoker;
      num_blocks_ = num_blocks;
      remaining_.store(num_blocks - 1, std::memory_order_relaxed);
      ++generation_;
    }
    cv_work_.notify_all();

    // The calling thread runs block 0. Mark it in-pool so a nested ParallelFor
    // launched from fn falls back to the serial path.
    bool &in_pool = InPoolFlag();
    const bool was_in_pool = in_pool;
    in_pool = true;
    fn(static_cast<int64_t>(0));
    in_pool = was_in_pool;

    std::unique_lock<std::mutex> lock(mu_);
    cv_done_.wait(lock, [this]() { return remaining_.load(std::memory_order_acquire) == 0; });
  }

private:
  static bool &InPoolFlag() noexcept {
    thread_local bool in_pool = false;
    return in_pool;
  }
  static bool InPool() noexcept { return InPoolFlag(); }

  void WorkerLoop(int64_t worker_index) {
    InPoolFlag() = true;
    const int64_t my_block = worker_index + 1;
    uint64_t last_generation = 0;
    for (;;) {
      void *ctx = nullptr;
      TaskFn fn = nullptr;
      int64_t num_blocks = 0;
      {
        // Snapshot the whole region under the lock so each wake processes
        // exactly one generation and never mixes fields across regions.
        std::unique_lock<std::mutex> lock(mu_);
        cv_work_.wait(
            lock, [this, last_generation]() { return stop_ || generation_ != last_generation; });
        if (stop_) {
          return;
        }
        last_generation = generation_;
        ctx = task_ctx_;
        fn = task_fn_;
        num_blocks = num_blocks_;
      }
      if (my_block < num_blocks) {
        fn(ctx, my_block);
        if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
          // Last worker block of the region finished: wake the waiting caller.
          std::lock_guard<std::mutex> lock(mu_);
          cv_done_.notify_one();
        }
      }
    }
  }

  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::mutex region_mu_;
  std::condition_variable cv_work_;
  std::condition_variable cv_done_;
  void *task_ctx_ = nullptr;
  TaskFn task_fn_ = nullptr;
  int64_t num_blocks_ = 0;
  std::atomic<int64_t> remaining_{0};
  uint64_t generation_ = 0;
  bool stop_ = false;
};

/// Returns the process-wide :cpp:class:`ThreadPool` used by :cpp:func:`ParallelFor`.
///
/// The pool is constructed on first use with ``ParallelForThreadCount() - 1``
/// worker threads (the calling thread makes up the last participant) and lives
/// for the remainder of the process. Threads are therefore created once and
/// reused across every ``ParallelFor`` call.
inline ThreadPool &GlobalThreadPool() {
  static ThreadPool pool(ParallelForThreadCount() - 1);
  return pool;
}

/**
 * Splits the half-open range ``[0, total)`` into contiguous blocks and invokes
 * ``fn(begin, end)`` once per block, using the cost model to decide whether
 * parallelising is worthwhile.
 *
 * When :cpp:func:`ParallelForBlockCount` decides the work is too small for the
 * current processor (few iterations and/or cheap ``cost_per_element``), or only
 * one thread is available, the whole range is processed inline on the calling
 * thread, so ``fn`` must be safe to call once with the full range. Every block
 * is disjoint and covers the range exactly once, so the observable result is
 * independent of the number of threads: kernels that only map input elements to
 * output elements (no cross-element accumulation) stay bit-exact.
 *
 * ``fn`` is invoked concurrently from several threads and must therefore only
 * touch data disjoint per block (typically writing ``output[begin, end)`` from
 * ``input[begin, end)``). It must not throw.
 *
 * @param total            Number of iterations. Values ``<= 0`` are a no-op.
 * @param cost_per_element Relative cost of a single iteration in work units
 *                         (see :cpp:func:`ParallelForBlockCount`).
 * @param fn               Callable invoked as ``fn(int64_t begin, int64_t end)``
 *                         for each block, covering the sub-range
 *                         ``[begin, end)``.
 */
template <typename Fn> void ParallelFor(int64_t total, double cost_per_element, Fn fn) {
  if (total <= 0) {
    return;
  }
  const int64_t num_blocks = ParallelForBlockCount(total, cost_per_element);
  if (num_blocks <= 1) {
    fn(static_cast<int64_t>(0), total);
    return;
  }

  const int64_t block = (total + num_blocks - 1) / num_blocks;
  GlobalThreadPool().Run(num_blocks, [&fn, block, total](int64_t b) {
    const int64_t begin = b * block;
    if (begin >= total) {
      return;
    }
    const int64_t end = std::min(begin + block, total);
    fn(begin, end);
  });
}

/// Convenience overload assuming trivial per-element cost (``1`` work unit).
///
/// @param total Number of iterations. Values ``<= 0`` are a no-op.
/// @param fn    Callable invoked as ``fn(int64_t begin, int64_t end)``.
template <typename Fn> void ParallelFor(int64_t total, Fn fn) {
  ParallelFor(total, 1.0, std::move(fn));
}

} // namespace onnx_light_cpu
