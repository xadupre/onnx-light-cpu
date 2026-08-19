// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu/impl/thread_topology.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

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

/// Width, in bytes, of the widest SIMD register the kernels may use.
///
/// The element-wise kernels vectorise their inner loop, processing several
/// values per instruction (e.g. AVX handles 8 ``float`` at a time, AVX-512
/// handles 16). The widest register targeted is AVX-512 at 64 bytes. Sizing a
/// :cpp:func:`ParallelFor` block to a multiple of the corresponding lane count
/// keeps every block a whole number of SIMD vectors, so the vectorised loop
/// never falls back to a scalar remainder in the middle of the range.
inline constexpr int64_t kParallelForSimdWidthBytes = 64;

#ifndef ONNX_LIGHT_CPU_MAX_THREADS
#define ONNX_LIGHT_CPU_MAX_THREADS 0
#endif

static_assert(ONNX_LIGHT_CPU_MAX_THREADS >= 0,
              "ONNX_LIGHT_CPU_MAX_THREADS must be a non-negative integer");

/// Maximum number of threads allowed to participate in a parallel region.
///
/// A value of zero leaves the limit to runtime topology detection. Positive
/// values impose the build-time ceiling requested through the CMake
/// ``ONNX_LIGHT_CPU_MAX_THREADS`` setting.
inline constexpr int64_t kParallelForMaxThreads = ONNX_LIGHT_CPU_MAX_THREADS == 0
                                                      ? std::numeric_limits<int64_t>::max()
                                                      : ONNX_LIGHT_CPU_MAX_THREADS;

namespace detail {

inline int64_t ResolveParallelForThreadCount(int64_t logical_threads, int64_t physical_cores,
                                             const char *configured) noexcept {
  logical_threads = std::clamp<int64_t>(logical_threads, 1, kParallelForMaxThreads);
  physical_cores = std::clamp<int64_t>(physical_cores, 1, logical_threads);
  if (configured == nullptr || *configured == '\0') {
    return physical_cores;
  }
  const std::string_view text(configured);
  int64_t requested = 0;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), requested);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || requested <= 0) {
    return physical_cores;
  }
  return std::min(logical_threads, requested);
}

inline int64_t ResolveParallelForThreadCount(int64_t available, const char *configured) noexcept {
  return ResolveParallelForThreadCount(available, available, configured);
}

inline int64_t ResolveParallelForSpinCount(const char *configured) noexcept {
  constexpr int64_t default_spin_count = 2000;
  constexpr int64_t maximum_spin_count = 1000000;
  if (configured == nullptr || *configured == '\0') {
    return default_spin_count;
  }
  const std::string_view text(configured);
  int64_t requested = 0;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), requested);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || requested < 0) {
    return default_spin_count;
  }
  return std::min(requested, maximum_spin_count);
}

inline void ParallelForCpuRelax() noexcept {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
  _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
  __asm__ volatile("yield");
#else
  std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

inline int &ParallelRegionDepth() noexcept {
  thread_local int depth = 0;
  return depth;
}

} // namespace detail

/// Returns the number of participating threads :cpp:func:`ParallelFor` may use.
///
/// Uses one available logical thread per detected physical core by default,
/// caps it at :cpp:var:`kParallelForMaxThreads`, and applies the optional
/// ``ONNX_LIGHT_CPU_NUM_THREADS`` process environment limit. An explicit limit
/// may consume SMT siblings after physical cores. The result is always ``>= 1``
/// and counts the calling thread, which always participates in the work.
inline int64_t ParallelForThreadCount() {
  static const int64_t thread_count = []() {
    const CpuTopology &topology = GetCpuTopology();
    return detail::ResolveParallelForThreadCount(
        static_cast<int64_t>(topology.logical_thread_count),
        static_cast<int64_t>(topology.physical_core_count),
        std::getenv("ONNX_LIGHT_CPU_NUM_THREADS"));
  }();
  return thread_count;
}

inline int64_t ParallelForSpinCount() noexcept {
  static const int64_t spin_count =
      detail::ResolveParallelForSpinCount(std::getenv("ONNX_LIGHT_CPU_SPIN_COUNT"));
  return spin_count;
}

/// Marks work dispatched by a caller-owned thread pool.
///
/// Construct this guard inside each caller worker before invoking a kernel.
/// Any nested :cpp:func:`ParallelFor` then runs inline on that worker instead
/// of waking the process-wide pool, avoiding nested workers and
/// oversubscription.
class ParallelForExternalRegion {
public:
  ParallelForExternalRegion() noexcept { ++detail::ParallelRegionDepth(); }
  ParallelForExternalRegion(const ParallelForExternalRegion &) = delete;
  ParallelForExternalRegion &operator=(const ParallelForExternalRegion &) = delete;
  ~ParallelForExternalRegion() { --detail::ParallelRegionDepth(); }
};

/// Number of ``T`` elements the widest SIMD register processes at once.
///
/// Used as the ``block_multiple`` argument of :cpp:func:`ParallelFor` so block
/// boundaries land on SIMD-vector boundaries for element type ``T`` (e.g. 16 for
/// ``float``, 8 for ``double``, 32 for a 2-byte half, 64 for ``int8_t``). The
/// result is always ``>= 1``.
template <typename T> inline constexpr int64_t ParallelForSimdLanes() noexcept {
  static_assert(sizeof(T) > 0, "element type must be a complete type");
  constexpr int64_t lanes = kParallelForSimdWidthBytes / static_cast<int64_t>(sizeof(T));
  return lanes > 0 ? lanes : 1;
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
inline int64_t ParallelForBlockCount(int64_t total, double cost_per_element = 1.0) {
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
 * created once and use a configurable, bounded spin before parking on a
 * condition variable, so dispatching nearby regions often avoids a scheduler
 * wakeup without busy-waiting indefinitely.
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

  /// Creates a pool with ``num_workers`` persistent worker threads.
  ///
  /// @param num_workers Number of worker threads to spawn. Values ``<= 0``
  ///                    create a pool with no workers, in which case
  ///                    :cpp:func:`Run` executes every block on the caller.
  /// @param affinities  Optional processor targets, one per worker.
  explicit ThreadPool(int64_t num_workers, std::vector<CpuAffinity> affinities = {})
      : affinities_(std::move(affinities)) {
    if (num_workers < 0) {
      num_workers = 0;
    }
    if (affinities_.size() > static_cast<std::size_t>(num_workers)) {
      affinities_.resize(static_cast<std::size_t>(num_workers));
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
      stop_.store(true, std::memory_order_release);
      generation_.fetch_add(1, std::memory_order_release);
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

  /// Returns true while the calling thread executes a pool region.
  static bool IsInParallelRegion() noexcept { return detail::ParallelRegionDepth() != 0; }

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
    if (workers_.empty() || num_blocks == 1 || IsInParallelRegion()) {
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
      generation_.fetch_add(1, std::memory_order_release);
    }
    cv_work_.notify_all();

    // The calling thread runs block 0. Mark it in-pool so a nested ParallelFor
    // launched from fn falls back to the serial path.
    int &depth = detail::ParallelRegionDepth();
    ++depth;
    fn(static_cast<int64_t>(0));
    --depth;

    for (int64_t spin = 0; spin < ParallelForSpinCount(); ++spin) {
      if (remaining_.load(std::memory_order_acquire) == 0) {
        return;
      }
      detail::ParallelForCpuRelax();
    }
    std::unique_lock<std::mutex> lock(mu_);
    cv_done_.wait(lock, [this]() { return remaining_.load(std::memory_order_acquire) == 0; });
  }

private:
  void WorkerLoop(int64_t worker_index) {
    if (static_cast<std::size_t>(worker_index) < affinities_.size()) {
      SetCurrentThreadAffinity(affinities_[static_cast<std::size_t>(worker_index)]);
    }
    detail::ParallelRegionDepth() = 1;
    const int64_t my_block = worker_index + 1;
    uint64_t last_generation = 0;
    for (;;) {
      for (int64_t spin = 0; spin < ParallelForSpinCount(); ++spin) {
        if (stop_.load(std::memory_order_acquire) ||
            generation_.load(std::memory_order_acquire) != last_generation) {
          break;
        }
        detail::ParallelForCpuRelax();
      }
      void *ctx = nullptr;
      TaskFn fn = nullptr;
      int64_t num_blocks = 0;
      {
        // Snapshot the whole region under the lock so each wake processes
        // exactly one generation and never mixes fields across regions.
        std::unique_lock<std::mutex> lock(mu_);
        cv_work_.wait(lock, [this, last_generation]() {
          return stop_.load(std::memory_order_acquire) ||
                 generation_.load(std::memory_order_acquire) != last_generation;
        });
        if (stop_.load(std::memory_order_acquire)) {
          return;
        }
        last_generation = generation_.load(std::memory_order_acquire);
        ctx = task_ctx_;
        fn = task_fn_;
        num_blocks = num_blocks_;
      }
      if (my_block < num_blocks) {
        fn(ctx, my_block);
        if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
          // Last worker block of the region finished: wake the waiting caller.
          cv_done_.notify_one();
        }
      }
    }
  }

  std::vector<std::thread> workers_;
  std::vector<CpuAffinity> affinities_;
  std::mutex mu_;
  std::mutex region_mu_;
  std::condition_variable cv_work_;
  std::condition_variable cv_done_;
  void *task_ctx_ = nullptr;
  TaskFn task_fn_ = nullptr;
  int64_t num_blocks_ = 0;
  std::atomic<int64_t> remaining_{0};
  std::atomic<uint64_t> generation_{0};
  std::atomic<bool> stop_{false};
};

/// Returns the process-wide :cpp:class:`ThreadPool` used by :cpp:func:`ParallelFor`.
///
/// The pool is constructed on first use with ``ParallelForThreadCount() - 1``
/// worker threads (the calling thread makes up the last participant) and lives
/// for the remainder of the process. Threads are therefore created once and
/// reused across every ``ParallelFor`` call.
inline ThreadPool &GlobalThreadPool() {
  static const std::vector<CpuAffinity> worker_affinities = []() {
    const CpuTopology &topology = GetCpuTopology();
    std::vector<CpuAffinity> affinities =
        SelectCpuAffinities(topology, static_cast<std::size_t>(ParallelForThreadCount()));
    CpuAffinity current;
    if (GetCurrentThreadAffinity(current)) {
      const auto current_thread = std::find_if(
          topology.threads.begin(), topology.threads.end(), [current](const CpuThread &thread) {
            return thread.affinity.group == current.group && thread.affinity.index == current.index;
          });
      if (current_thread != topology.threads.end()) {
        const auto same_core =
            std::find_if(affinities.begin(), affinities.end(),
                         [&topology, current_thread](const CpuAffinity &affinity) {
                           const auto thread =
                               std::find_if(topology.threads.begin(), topology.threads.end(),
                                            [affinity](const CpuThread &candidate) {
                                              return candidate.affinity.group == affinity.group &&
                                                     candidate.affinity.index == affinity.index;
                                            });
                           return thread != topology.threads.end() &&
                                  thread->core_index == current_thread->core_index;
                         });
        if (same_core != affinities.end()) {
          affinities.erase(same_core);
        }
      }
    }
    const std::size_t worker_count =
        static_cast<std::size_t>(std::max<int64_t>(0, ParallelForThreadCount() - 1));
    if (affinities.size() > worker_count) {
      affinities.resize(worker_count);
    }
    return affinities;
  }();
  static ThreadPool pool(ParallelForThreadCount() - 1, worker_affinities);
  return pool;
}

inline bool ParallelForInParallelRegion() noexcept { return ThreadPool::IsInParallelRegion(); }

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
 * ``block_multiple`` sizes each block to a whole number of SIMD vectors so a
 * vectorised kernel processes full registers per block instead of a scalar
 * remainder at every block boundary.
 *
 * @param total            Number of iterations. Values ``<= 0`` are a no-op.
 * @param cost_per_element Relative cost of a single iteration in work units
 *                         (see :cpp:func:`ParallelForBlockCount`).
 * @param block_multiple   Round every block size up to a multiple of this many
 *                         iterations (except the final block, which stops at
 *                         ``total``). Pass the SIMD lane count for the kernel's
 *                         element type (see :cpp:func:`ParallelForSimdLanes`) so
 *                         each block is a whole number of SIMD vectors. Values
 *                         ``<= 1`` disable alignment.
 * @param fn               Callable invoked as ``fn(int64_t begin, int64_t end)``
 *                         for each block, covering the sub-range
 *                         ``[begin, end)``.
 */
template <typename Fn>
void ParallelFor(int64_t total, double cost_per_element, int64_t block_multiple, Fn fn) {
  if (total <= 0) {
    return;
  }
  int64_t num_blocks = ParallelForBlockCount(total, cost_per_element);
  if (num_blocks <= 1) {
    fn(static_cast<int64_t>(0), total);
    return;
  }

  if (block_multiple < 1) {
    block_multiple = 1;
  }
  // Base block size, rounded up so every block boundary (except the final one)
  // lands on a multiple of block_multiple. This keeps each block a whole number
  // of SIMD vectors, so the vectorised inner loop of a kernel never has to
  // process a scalar remainder in the middle of the range.
  int64_t block = (total + num_blocks - 1) / num_blocks;
  block = ((block + block_multiple - 1) / block_multiple) * block_multiple;
  // Rounding the block size up can leave later blocks empty; recompute the block
  // count so we never wake a worker that would receive an empty range.
  num_blocks = (total + block - 1) / block;

  GlobalThreadPool().Run(num_blocks, [&fn, block, total](int64_t b) {
    const int64_t begin = b * block;
    if (begin >= total) {
      return;
    }
    const int64_t end = std::min(begin + block, total);
    fn(begin, end);
  });
}

/// Convenience overload without SIMD-vector block alignment (``block_multiple``
/// of ``1``).
///
/// @param total            Number of iterations. Values ``<= 0`` are a no-op.
/// @param cost_per_element Relative cost of a single iteration in work units.
/// @param fn               Callable invoked as ``fn(int64_t begin, int64_t end)``.
template <typename Fn> void ParallelFor(int64_t total, double cost_per_element, Fn fn) {
  ParallelFor(total, cost_per_element, int64_t{1}, std::move(fn));
}

/// Convenience overload assuming trivial per-element cost (``1`` work unit) and
/// no SIMD-vector block alignment.
///
/// @param total Number of iterations. Values ``<= 0`` are a no-op.
/// @param fn    Callable invoked as ``fn(int64_t begin, int64_t end)``.
template <typename Fn> void ParallelFor(int64_t total, Fn fn) {
  ParallelFor(total, 1.0, int64_t{1}, std::move(fn));
}

} // namespace onnx_light_cpu
