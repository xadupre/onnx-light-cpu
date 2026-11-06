// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "ml_cpu_compute/execution.h"
#include "ml_cpu_compute/math/gemm/gemm_plan.h"
#include "ml_cpu_compute/math/half_conversion.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

// GEMM packing and split-K buffers use aligned new. Count their live payloads,
// not the output, thread-launch bookkeeping, or merely the largest allocation.
struct AllocationHeader {
  void *allocation;
  std::size_t bytes;
  std::uint64_t generation;
};

std::atomic<std::uint64_t> generation{0};
std::atomic<bool> recording{false};
std::atomic<std::size_t> live_bytes{0}, peak_bytes{0};
thread_local bool in_callback = false;
thread_local int callback_parent_depth = -1;
std::atomic<std::size_t> callback_allocations{0};

bool AllocatingInCallback() {
  return in_callback || (callback_parent_depth >= 0 &&
                         ml_cpu_compute::detail::ExecutionRegionDepth() > callback_parent_depth);
}

struct CallbackScope {
  CallbackScope() { in_callback = true; }
  ~CallbackScope() { in_callback = false; }
};

void RecordPeak(std::size_t bytes) {
  std::size_t previous = peak_bytes.load();
  while (previous < bytes && !peak_bytes.compare_exchange_weak(previous, bytes)) {
  }
}

template <typename Fn> std::size_t PackingPeak(Fn fn) {
  fn();
  ++generation;
  live_bytes = 0;
  peak_bytes = 0;
  struct RecordingScope {
    ~RecordingScope() { recording = false; }
  } scope;
  recording = true;
  fn();
  recording = false;
  EXPECT_EQ(live_bytes.load(), 0U);
  return peak_bytes.load();
}

struct ConcurrentExecutor {
  static void Run(void *, std::int64_t blocks, void *context,
                  ml_cpu_compute::ExecutionBlockFn task) {
    std::vector<std::thread> threads;
    for (std::int64_t block = 1; block < blocks; ++block) {
      threads.emplace_back([=] { task(context, block); });
    }
    task(context, 0);
    for (auto &thread : threads) {
      thread.join();
    }
  }
};

} // namespace

void *operator new(std::size_t bytes) {
  if (AllocatingInCallback()) {
    ++callback_allocations;
  }
  if (void *pointer = std::malloc(std::max<std::size_t>(bytes, 1))) {
    return pointer;
  }
  throw std::bad_alloc();
}

void operator delete(void *pointer) noexcept { std::free(pointer); }
void operator delete(void *pointer, std::size_t) noexcept { std::free(pointer); }
void *operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete[](void *pointer) noexcept { ::operator delete(pointer); }
void operator delete[](void *pointer, std::size_t) noexcept { ::operator delete(pointer); }

void *operator new(std::size_t bytes, std::align_val_t alignment) {
  if (AllocatingInCallback()) {
    ++callback_allocations;
  }
  const std::size_t align =
      std::max(static_cast<std::size_t>(alignment), alignof(AllocationHeader));
  const std::size_t overhead = align - 1 + sizeof(AllocationHeader);
  if (bytes > std::numeric_limits<std::size_t>::max() - overhead) {
    throw std::bad_alloc();
  }
  void *allocation = std::malloc(bytes + overhead);
  if (allocation == nullptr) {
    throw std::bad_alloc();
  }
  const auto address = reinterpret_cast<std::uintptr_t>(allocation) + sizeof(AllocationHeader);
  const auto aligned = (address + align - 1) & ~(align - 1);
  auto *header = reinterpret_cast<AllocationHeader *>(aligned) - 1;
  const bool tracked = recording.load();
  *header = {allocation, bytes, tracked ? generation.load() : 0};
  if (tracked) {
    RecordPeak(live_bytes.fetch_add(bytes) + bytes);
  }
  return reinterpret_cast<void *>(aligned);
}

void operator delete(void *pointer, std::align_val_t) noexcept {
  if (pointer == nullptr) {
    return;
  }
  const auto *header = static_cast<AllocationHeader *>(pointer) - 1;
  if (header->generation != 0 && header->generation == generation.load()) {
    live_bytes.fetch_sub(header->bytes);
  }
  std::free(header->allocation);
}

void *operator new[](std::size_t bytes, std::align_val_t alignment) {
  return ::operator new(bytes, alignment);
}
void operator delete[](void *pointer, std::align_val_t alignment) noexcept {
  ::operator delete(pointer, alignment);
}
void operator delete(void *pointer, std::size_t, std::align_val_t alignment) noexcept {
  ::operator delete(pointer, alignment);
}
void operator delete[](void *pointer, std::size_t, std::align_val_t alignment) noexcept {
  ::operator delete(pointer, alignment);
}

namespace {

struct AllocationCheckingExecutor {
  std::size_t submissions = 0;
  std::int64_t maximum_blocks = 0;

  static void Run(void *self, std::int64_t blocks, void *context,
                  ml_cpu_compute::ExecutionBlockFn task) {
    auto &executor = *static_cast<AllocationCheckingExecutor *>(self);
    ++executor.submissions;
    executor.maximum_blocks = std::max(executor.maximum_blocks, blocks);
    std::vector<std::thread> threads;
    for (std::int64_t block = blocks - 1; block > 0; --block) {
      threads.emplace_back([=] {
        CallbackScope callback;
        task(context, block);
      });
    }
    {
      CallbackScope callback;
      task(context, 0);
    }
    for (auto &thread : threads) {
      thread.join();
    }
  }
};

struct TrackingAllocator {
  struct Handle {
    std::size_t bytes, alignment;
  };
  std::size_t allocations = 0, releases = 0, live = 0, peak = 0;
  std::size_t fail_after = std::numeric_limits<std::size_t>::max();
  std::size_t callback_calls = 0;

  static ml_cpu_compute::ExecutionAllocation Allocate(void *context, std::size_t bytes,
                                                      std::size_t alignment) {
    auto &self = *static_cast<TrackingAllocator *>(context);
    if (AllocatingInCallback()) {
      ++self.callback_calls;
    }
    if (self.allocations == self.fail_after) {
      return {};
    }
    // Keep the size with the opaque handle so release accounting is independent
    // of the aligned payload address borrowed by GEMM.
    const std::size_t offset = (sizeof(Handle) + alignment - 1) / alignment * alignment;
    const std::size_t capacity = bytes + offset;
    auto *handle = static_cast<Handle *>(::operator new(capacity, std::align_val_t{alignment}));
    *handle = {bytes, alignment};
    void *data = reinterpret_cast<char *>(handle) + offset;
    ++self.allocations;
    self.live += bytes;
    self.peak = std::max(self.peak, self.live);
    return {data, handle};
  }

  static void Release(void *context, ml_cpu_compute::ExecutionAllocation allocation) noexcept {
    if (allocation.handle == nullptr) {
      return;
    }
    auto &self = *static_cast<TrackingAllocator *>(context);
    auto *handle = static_cast<Handle *>(allocation.handle);
    self.live -= handle->bytes;
    ++self.releases;
    ::operator delete(handle, std::align_val_t{handle->alignment});
  }

  ml_cpu_compute::ExecutionAllocatorView View() { return {this, &Allocate, &Release}; }
};

template <typename T>
void CheckGenericWorkerScratch(std::size_t m, std::size_t n, std::size_t k, std::int64_t threads,
                               bool trans_a, bool nested, int half = 0) {
  using namespace ml_cpu_compute;
  std::vector<T> a(m * k), b(n * k), y(m * n, T(-1000)), expected(m * n, T(0));
  for (std::size_t row = 0; row < m; ++row) {
    for (std::size_t inner = 0; inner < k; ++inner) {
      const T value = T(static_cast<int>((row + 3 * inner) % 7) - 3) / T(8);
      a[trans_a ? inner * m + row : row * k + inner] = value;
    }
  }
  for (std::size_t column = 0; column < n; ++column) {
    for (std::size_t inner = 0; inner < k; ++inner) {
      b[column * k + inner] = T(static_cast<int>((column * 5 + inner) % 11) - 5) / T(8);
    }
  }
  for (std::size_t row = 0; row < m; ++row) {
    for (std::size_t column = 0; column < n; ++column) {
      for (std::size_t inner = 0; inner < k; ++inner) {
        expected[row * n + column] +=
            a[trans_a ? inner * m + row : row * k + inner] * b[column * k + inner];
      }
    }
  }
  std::vector<std::uint16_t> ah, bh;
  if (half != 0) {
    const auto convert = [half](T value) {
      return half == 1 ? detail::FloatToFloat16Bits(static_cast<float>(value))
                       : detail::FloatToBFloat16Bits(static_cast<float>(value));
    };
    ah.resize(a.size());
    bh.resize(b.size());
    std::transform(a.begin(), a.end(), ah.begin(), convert);
    std::transform(b.begin(), b.end(), bh.begin(), convert);
  }

  AllocationCheckingExecutor tracker;
  ExecutionExecutorView executor{&tracker, threads, &AllocationCheckingExecutor::Run};
  ExecutionExecutorScope scope(threads == 1 ? nullptr : &executor);
  TrackingAllocator allocation_tracker;
  const auto allocator = allocation_tracker.View();
  ExecutionAllocatorScope allocation_scope(&allocator);
  const GemmPlan<T> plan({trans_a, true, m, n, k});
  const GemmHalfPlan half_plan({half == 2, trans_a, true, m, n, k});
  const auto &blocking = half != 0 ? half_plan.blocking() : plan.blocking();
  SCOPED_TRACE(::testing::Message()
               << "element_size=" << sizeof(T) << ", half=" << half << ", blocking=" << blocking.mc
               << "," << blocking.nc << "," << blocking.kc);
  const auto run = [&] {
    if constexpr (std::is_same_v<T, float>) {
      if (half != 0) {
        detail::GemmHalfPlanned<GemmAlgorithm::kGeneral>(half == 2, trans_a, true, m, n, k, 1.0f,
                                                         ah.data(), bh.data(), y.data(), &blocking,
                                                         &half_plan.compact_blocking());
      } else {
        detail::GemmFloat32Planned<GemmAlgorithm::kGeneral>(
            trans_a, true, m, n, k, T(1), a.data(), b.data(), T(0), nullptr, y.data(), &blocking);
      }
    } else {
      detail::GemmFloat64Planned<GemmAlgorithm::kGeneral>(
          trans_a, true, m, n, k, T(1), a.data(), b.data(), T(0), nullptr, y.data(), &blocking);
    }
  };
  callback_allocations = 0;
  callback_parent_depth = nested ? 1 : 0;
  if (nested) {
    detail::ExecutionRegionScope region;
    run();
  } else {
    run();
  }
  callback_parent_depth = -1;
  EXPECT_EQ(callback_allocations.load(), 0U);
  EXPECT_EQ(allocation_tracker.callback_calls, 0U);
  EXPECT_GT(allocation_tracker.allocations, 0U);
  EXPECT_EQ(allocation_tracker.allocations, allocation_tracker.releases);
  EXPECT_EQ(allocation_tracker.live, 0U);
  // Forced detail entry points do not apply the plan's shape-specific participant
  // limit, so estimate their actual executor budget rather than the plan budget.
  const std::size_t bound = detail::GemmProductScratchBytes(
      GemmAlgorithm::kGeneral, trans_a, true, m, n, k, blocking, sizeof(T),
      static_cast<std::size_t>(threads), half != 0,
      half != 0 ? &half_plan.compact_blocking() : nullptr, half == 2);
  EXPECT_LE(allocation_tracker.peak, bound);
  if (nested || threads == 1) {
    EXPECT_EQ(tracker.submissions, 0U);
  } else {
    EXPECT_LE(tracker.maximum_blocks, threads);
    if (m >= 257) {
      EXPECT_GT(tracker.maximum_blocks, 1);
    }
  }
  EXPECT_EQ(y, expected);
}

TEST(GemmScratchAllocations, AllocatorScopesRestoreBindingsAndExplicitHeapFallback) {
  using namespace ml_cpu_compute;
  TrackingAllocator first, second;
  const auto first_view = first.View();
  const auto second_view = second.View();
  const auto *previous = CurrentExecutionAllocator();
  const bool had_scope = HasExecutionAllocatorScope();
  {
    ExecutionAllocatorScope scope(&first_view);
    EXPECT_TRUE(HasExecutionAllocatorScope());
    EXPECT_EQ(CurrentExecutionAllocator(), &first_view);
    ExecutionBuffer empty(0);
    EXPECT_EQ(empty.data(), nullptr);
    EXPECT_EQ(first.allocations, 0U);
    {
      ExecutionAllocatorScope heap(nullptr);
      ExecutionBuffer buffer(129);
      EXPECT_EQ(reinterpret_cast<std::uintptr_t>(buffer.data()) % 64, 0U);
      EXPECT_EQ(CurrentExecutionAllocator(), nullptr);
      EXPECT_EQ(first.allocations, 0U);
    }
    EXPECT_EQ(CurrentExecutionAllocator(), &first_view);
    EXPECT_THROW(
        [&] {
          ExecutionAllocatorScope nested(&second_view);
          ExecutionBuffer buffer(129, 128);
          EXPECT_EQ(reinterpret_cast<std::uintptr_t>(buffer.data()) % 128, 0U);
          throw std::runtime_error("unwind");
        }(),
        std::runtime_error);
    EXPECT_EQ(CurrentExecutionAllocator(), &first_view);
    EXPECT_EQ(second.allocations, 1U);
    EXPECT_EQ(second.releases, 1U);
    EXPECT_EQ(second.live, 0U);
    ExecutionBuffer buffer(17);
    EXPECT_EQ(first.allocations, 1U);
  }
  EXPECT_EQ(first.live, 0U);
  EXPECT_EQ(first.releases, 1U);
  EXPECT_EQ(CurrentExecutionAllocator(), previous);
  EXPECT_EQ(HasExecutionAllocatorScope(), had_scope);
}

TEST(GemmScratchAllocations, AllocatorRejectsInvalidContractsAndReleasesMisalignedStorage) {
  using namespace ml_cpu_compute;
  EXPECT_THROW(ExecutionBuffer(1, 0), std::invalid_argument);
  EXPECT_THROW(ExecutionBuffer(1, 3), std::invalid_argument);
  TrackingAllocator tracker;
  auto allocator = tracker.View();
  {
    allocator.allocate = nullptr;
    ExecutionAllocatorScope scope(&allocator);
    EXPECT_THROW(ExecutionBuffer(1), std::invalid_argument);
  }
  {
    allocator = tracker.View();
    allocator.release = nullptr;
    ExecutionAllocatorScope scope(&allocator);
    EXPECT_THROW(ExecutionBuffer(1), std::invalid_argument);
  }
  allocator = tracker.View();
  allocator.allocate = [](void *context, std::size_t bytes, std::size_t alignment) {
    auto allocation = TrackingAllocator::Allocate(context, bytes, alignment);
    allocation.data = static_cast<char *>(allocation.data) + 1;
    return allocation;
  };
  ExecutionAllocatorScope scope(&allocator);
  EXPECT_THROW(ExecutionBuffer(1), std::bad_alloc);
  EXPECT_EQ(tracker.allocations, 1U);
  EXPECT_EQ(tracker.releases, 1U);
  EXPECT_EQ(tracker.live, 0U);
}

TEST(GemmScratchAllocations, GenericCallbacksDoNotAllocateAndOwnDisjointScratch) {
  for (const auto shape : {std::array<std::size_t, 3>{49, 1057, 259}, {257, 65, 259}}) {
    const auto [m, n, k] = shape;
    for (const std::int64_t threads : {1, 3, 7}) {
      for (const bool trans_a : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "M=" << m << ", N=" << n << ", threads=" << threads
                                          << ", trans_a=" << trans_a);
        CheckGenericWorkerScratch<float>(m, n, k, threads, trans_a, false);
        CheckGenericWorkerScratch<double>(m, n, k, threads, trans_a, false);
        CheckGenericWorkerScratch<float>(m, n, k, threads, trans_a, false, 1);
        CheckGenericWorkerScratch<float>(m, n, k, threads, trans_a, false, 2);
      }
    }
  }
}

TEST(GemmScratchAllocations, NestedGenericExecutionDoesNotResubmit) {
  CheckGenericWorkerScratch<float>(49, 1057, 259, 7, true, true);
  CheckGenericWorkerScratch<double>(49, 1057, 259, 7, true, true);
  CheckGenericWorkerScratch<float>(49, 1057, 259, 7, true, true, 1);
  CheckGenericWorkerScratch<float>(49, 1057, 259, 7, true, true, 2);
}

TEST(GemmScratchAllocations, AllocationFailurePrecedesDispatchAndOutputWrites) {
  using namespace ml_cpu_compute;
  constexpr std::size_t m = 129, n = 1057, k = 259;
  const GemmPlan<float> plan({true, true, m, n, k});
  const std::vector<float> a(m * k, 0.25f), b(n * k, 0.5f);
  std::vector<float> y(m * n, -1.0f);
  AllocationCheckingExecutor executor_tracker;
  ExecutionExecutorView executor{&executor_tracker, 7, &AllocationCheckingExecutor::Run};
  ExecutionExecutorScope executor_scope(&executor);
  TrackingAllocator tracker;
  tracker.fail_after = 0;
  const auto allocator = tracker.View();
  ExecutionAllocatorScope allocator_scope(&allocator);
  EXPECT_THROW(plan.Execute(a.data(), b.data(), nullptr, y.data()), std::bad_alloc);
  EXPECT_EQ(executor_tracker.submissions, 0U);
  EXPECT_EQ(tracker.live, 0U);
  EXPECT_TRUE(std::all_of(y.begin(), y.end(), [](float value) { return value == -1.0f; }));
}

TEST(GemmScratchAllocations, SharedAndSplitKScratchFailuresPrecedeEveryDispatch) {
  using namespace ml_cpu_compute;
  for (const auto shape : {std::array<std::size_t, 3>{257, 65, 259}, {32, 31, 8193}}) {
    const auto [m, n, k] = shape;
    const GemmPlan<float> plan({true, true, m, n, k});
    const std::vector<float> a(m * k, 0.25f), b(n * k, 0.5f);
    std::size_t allocation_count = 0;
    {
      std::vector<float> y(m * n);
      AllocationCheckingExecutor executor_tracker;
      ExecutionExecutorView executor{&executor_tracker, 7, &AllocationCheckingExecutor::Run};
      ExecutionExecutorScope executor_scope(&executor);
      TrackingAllocator tracker;
      const auto allocator = tracker.View();
      ExecutionAllocatorScope allocator_scope(&allocator);
      plan.Execute(a.data(), b.data(), nullptr, y.data());
      allocation_count = tracker.allocations;
    }
    ASSERT_GT(allocation_count, 0U);
    for (std::size_t fail_after = 0; fail_after < allocation_count; ++fail_after) {
      SCOPED_TRACE(::testing::Message()
                   << "M=" << m << ", N=" << n << ", fail_after=" << fail_after);
      std::vector<float> y(m * n, -1.0f);
      AllocationCheckingExecutor executor_tracker;
      ExecutionExecutorView executor{&executor_tracker, 7, &AllocationCheckingExecutor::Run};
      ExecutionExecutorScope executor_scope(&executor);
      TrackingAllocator tracker;
      tracker.fail_after = fail_after;
      const auto allocator = tracker.View();
      ExecutionAllocatorScope allocator_scope(&allocator);
      EXPECT_THROW(plan.Execute(a.data(), b.data(), nullptr, y.data()), std::bad_alloc);
      EXPECT_EQ(executor_tracker.submissions, 0U);
      EXPECT_EQ(tracker.live, 0U);
      EXPECT_EQ(tracker.allocations, tracker.releases);
      EXPECT_TRUE(std::all_of(y.begin(), y.end(), [](float value) { return value == -1.0f; }));
    }
  }
}

TEST(GemmScratchAllocations, PackingSlicePaddingAndByteOverflowPrecedeAllocation) {
  using namespace ml_cpu_compute;
  TrackingAllocator tracker;
  const auto allocator = tracker.View();
  ExecutionAllocatorScope allocator_scope(&allocator);
  ExecutionExecutorScope serial(nullptr);
  const float a = 1.0f, b = 1.0f;
  float output = -1.0f;
  for (const std::size_t rows : {std::numeric_limits<std::size_t>::max(),
                                 std::numeric_limits<std::size_t>::max() / sizeof(float) + 1}) {
    const GemmBlocking blocking{rows, 1, 1, 4, 16};
    EXPECT_THROW(detail::GemmFloat32Planned<GemmAlgorithm::kGeneral>(
                     true, true, rows, 1, 1, 1.0f, &a, &b, 0.0f, nullptr, &output, &blocking),
                 std::invalid_argument);
  }
  EXPECT_EQ(output, -1.0f);
  EXPECT_EQ(tracker.allocations, 0U);
  EXPECT_EQ(tracker.live, 0U);
}

TEST(GemmScratchAllocations, SplitKReservesNestedPackingBeforeDispatch) {
  using namespace ml_cpu_compute;
  constexpr std::size_t m = 32, n = 31, k = 8193;
  const GemmPlan<float> plan({true, true, m, n, k});
  const std::vector<float> a(m * k, 0.25f), b(n * k, 0.5f);
  std::vector<float> y(m * n);
  ASSERT_EQ(plan.algorithm(), GemmAlgorithm::kSplitK);
  AllocationCheckingExecutor executor_tracker;
  ExecutionExecutorView executor{&executor_tracker, 7, &AllocationCheckingExecutor::Run};
  ExecutionExecutorScope executor_scope(&executor);
  TrackingAllocator tracker;
  const auto allocator = tracker.View();
  ExecutionAllocatorScope allocator_scope(&allocator);
  callback_allocations = 0;
  callback_parent_depth = 0;
  plan.Execute(a.data(), b.data(), nullptr, y.data());
  callback_parent_depth = -1;
  EXPECT_EQ(callback_allocations.load(), 0U);
  EXPECT_EQ(tracker.callback_calls, 0U);
  EXPECT_GT(executor_tracker.maximum_blocks, 1);
  EXPECT_EQ(tracker.live, 0U);
  EXPECT_EQ(tracker.allocations, tracker.releases);
  EXPECT_LE(tracker.peak, plan.ScratchBytes(7));
  EXPECT_TRUE(
      std::all_of(y.begin(), y.end(), [](float value) { return value == float(k) * 0.125f; }));
}

template <typename T>
void CheckBorrowedBatchScratch(int half = 0, bool trans_b = true, bool borrowed = true,
                               std::size_t m = 32, std::size_t n = 65, std::size_t k = 1031) {
  using namespace ml_cpu_compute;
  constexpr std::size_t batches = 11;
  const MatMulPlan<T> plan(std::array<std::size_t, 3>{batches, m, k},
                           trans_b ? std::array<std::size_t, 2>{n, k}
                                   : std::array<std::size_t, 2>{k, n},
                           false, trans_b);
  std::vector<T> a(batches * m * k);
  const std::vector<T> b(n * k, T(0.5));
  for (std::size_t batch = 0; batch < batches; ++batch) {
    std::fill_n(a.data() + batch * m * k, m * k, T(batch + 1) / T(16));
  }
  std::vector<T> y(batches * m * n);
  AllocationCheckingExecutor executor_tracker;
  ExecutionExecutorView executor{&executor_tracker, 7, &AllocationCheckingExecutor::Run};
  ExecutionExecutorScope executor_scope(&executor);
  TrackingAllocator tracker;
  const auto allocator = tracker.View();
  ExecutionAllocatorScope allocator_scope(borrowed ? &allocator : nullptr);
  const auto check = [&](auto execute, std::size_t scratch_bytes) {
    const std::size_t previous_allocations = tracker.allocations;
    executor_tracker.submissions = 0;
    executor_tracker.maximum_blocks = 0;
    callback_allocations = 0;
    callback_parent_depth = 0;
    execute();
    callback_parent_depth = -1;
    EXPECT_EQ(callback_allocations.load(), 0U);
    EXPECT_EQ(tracker.callback_calls, 0U);
    EXPECT_EQ(tracker.allocations - previous_allocations, borrowed && scratch_bytes != 0 ? 1U : 0U);
    EXPECT_EQ(executor_tracker.submissions, 1U);
    EXPECT_GT(executor_tracker.maximum_blocks, 1);
    EXPECT_LT(executor_tracker.maximum_blocks, 7);
    EXPECT_EQ(tracker.allocations, tracker.releases);
    EXPECT_EQ(tracker.live, 0U);
    for (std::size_t batch = 0; batch < batches; ++batch) {
      const T expected = T(batch + 1) * T(k) / T(32);
      EXPECT_TRUE(std::all_of(y.begin() + batch * m * n, y.begin() + (batch + 1) * m * n,
                              [expected](T value) { return value == expected; }));
    }
  };
  if constexpr (std::is_same_v<T, float>) {
    if (half != 0) {
      const GemmHalfPlan half_plan({half == 2, false, trans_b, m, n, k});
      std::vector<std::uint16_t> ah(a.size());
      std::transform(a.begin(), a.end(), ah.begin(), [half](float value) {
        return half == 1 ? detail::FloatToFloat16Bits(value) : detail::FloatToBFloat16Bits(value);
      });
      const std::vector<std::uint16_t> bh(n * k, half == 1 ? 0x3800 : 0x3f00);
      std::vector<std::uint16_t> converted(y.size());
      GemmEpilogue<float> epilogue;
      epilogue.output_conversion =
          half == 1 ? GemmOutputConversion::kFloat16 : GemmOutputConversion::kBFloat16;
      epilogue.converted_output = converted.data();
      check([&] { plan.ExecuteHalf(half_plan, ah.data(), bh.data(), epilogue, y.data()); },
            half_plan.ScratchBytes(1));
      for (std::size_t index = 0; index < y.size(); ++index) {
        EXPECT_EQ(converted[index], half == 1 ? detail::FloatToFloat16Bits(y[index])
                                              : detail::FloatToBFloat16Bits(y[index]));
      }
      EXPECT_LE(tracker.peak + y.size() * sizeof(float), plan.HalfScratchBytes(half_plan, 7));
      return;
    }
  }
  check([&] { plan.Execute(a.data(), b.data(), y.data()); }, plan.gemm_plan().ScratchBytes(1));
  GemmPlanOptions<T> serial_options{false, trans_b, m, n, k};
  serial_options.maximum_participants = 1;
  const GemmPlan<T> serial_plan(serial_options);
  check(
      [&] {
        StridedBatchedGemm(serial_plan, batches, a.data(), m * k, b.data(), 0,
                           static_cast<const T *>(nullptr), 0, y.data(), m * n);
      },
      serial_plan.ScratchBytes(1));
  std::array<GroupedGemmProblem<T>, batches> problems;
  auto alternate_options = serial_options;
  alternate_options.trans_a = true;
  alternate_options.blocking = {1, 17, 129};
  const GemmPlan<T> alternate_plan(alternate_options);
  for (std::size_t batch = 0; batch < batches; ++batch) {
    problems[batch] = {batch % 2 == 0 ? &serial_plan : &alternate_plan, a.data() + batch * m * k,
                       b.data(), nullptr, y.data() + batch * m * n};
  }
  check([&] { GroupedGemm<T>(problems); },
        std::max(serial_plan.ScratchBytes(1), alternate_plan.ScratchBytes(1)));
  EXPECT_LE(tracker.peak, std::max(plan.ScratchBytes(7), 7 * detail::GemmBatchScratchBytes(
                                                                 alternate_plan.ScratchBytes(1))));
}

TEST(GemmScratchAllocations, BorrowedAllocatorOwnsBatchedAndGroupedPacking) {
  for (bool trans_b : {false, true}) {
    CheckBorrowedBatchScratch<float>(0, trans_b);
    CheckBorrowedBatchScratch<double>(0, trans_b);
    CheckBorrowedBatchScratch<float>(1, trans_b);
    CheckBorrowedBatchScratch<float>(2, trans_b);
  }
}

TEST(GemmScratchAllocations, HeapBackedBatchCallbacksAlsoBorrowPreallocatedScratch) {
  CheckBorrowedBatchScratch<float>(0, true, false);
  CheckBorrowedBatchScratch<float>(1, false, false);
}

TEST(GemmScratchAllocations, BatchScratchCoversDirectSkinnyAndSerialSplitKDispatch) {
  for (const auto shape : {std::array<std::size_t, 3>{128, 128, 256},
                           {1, 257, 16385},
                           {32, 1, 16385},
                           {32, 31, 8193}}) {
    const auto [m, n, k] = shape;
    SCOPED_TRACE(::testing::Message() << m << ":" << n << ":" << k);
    CheckBorrowedBatchScratch<float>(0, false, true, m, n, k);
    CheckBorrowedBatchScratch<double>(0, false, true, m, n, k);
    CheckBorrowedBatchScratch<float>(1, false, true, m, n, k);
    CheckBorrowedBatchScratch<float>(2, false, true, m, n, k);
  }
}

TEST(GemmScratchAllocations, BatchReservationFailurePrecedesDispatchAndOutputWrites) {
  using namespace ml_cpu_compute;
  constexpr std::size_t batches = 11, m = 32, n = 65, k = 1031;
  const MatMulPlan<float> plan(std::array<std::size_t, 3>{batches, m, k},
                               std::array<std::size_t, 2>{n, k}, false, true);
  GemmPlanOptions<float> options{false, true, m, n, k};
  options.maximum_participants = 1;
  const GemmPlan<float> serial(options);
  const std::vector<float> a(batches * m * k, 0.25f), b(n * k, 0.5f);
  std::vector<float> y(batches * m * n, -1.0f);
  std::array<GroupedGemmProblem<float>, batches> problems;
  for (std::size_t batch = 0; batch < batches; ++batch) {
    problems[batch] = {&serial, a.data() + batch * m * k, b.data(), nullptr,
                       y.data() + batch * m * n};
  }
  AllocationCheckingExecutor executor_tracker;
  ExecutionExecutorView executor{&executor_tracker, 7, &AllocationCheckingExecutor::Run};
  ExecutionExecutorScope executor_scope(&executor);
  TrackingAllocator tracker;
  tracker.fail_after = 0;
  const auto allocator = tracker.View();
  ExecutionAllocatorScope allocator_scope(&allocator);
  EXPECT_THROW(plan.Execute(a.data(), b.data(), y.data()), std::bad_alloc);
  EXPECT_THROW(StridedBatchedGemm(serial, batches, a.data(), m * k, b.data(), 0,
                                  static_cast<const float *>(nullptr), 0, y.data(), m * n),
               std::bad_alloc);
  EXPECT_THROW(GroupedGemm<float>(problems), std::bad_alloc);
  for (bool bf16 : {false, true}) {
    const GemmHalfPlan half({bf16, false, true, m, n, k});
    const std::vector<std::uint16_t> ah(a.size(), bf16 ? 0x3e80 : 0x3400);
    const std::vector<std::uint16_t> bh(b.size(), bf16 ? 0x3f00 : 0x3800);
    std::vector<std::uint16_t> converted(y.size(), 0xffff);
    GemmEpilogue<float> epilogue;
    epilogue.output_conversion =
        bf16 ? GemmOutputConversion::kBFloat16 : GemmOutputConversion::kFloat16;
    epilogue.converted_output = converted.data();
    EXPECT_THROW(plan.ExecuteHalf(half, ah.data(), bh.data(), epilogue, y.data()), std::bad_alloc);
    EXPECT_TRUE(std::all_of(converted.begin(), converted.end(),
                            [](std::uint16_t value) { return value == 0xffff; }));
  }
  EXPECT_EQ(executor_tracker.submissions, 0U);
  EXPECT_EQ(tracker.live, 0U);
  EXPECT_TRUE(std::all_of(y.begin(), y.end(), [](float value) { return value == -1.0f; }));
}

TEST(GemmScratchAllocations, ExplicitBatchScratchChecksCapacityAndRestoresReuseState) {
  using namespace ml_cpu_compute;
  constexpr std::size_t m = 32, n = 65, k = 1031;
  const GemmPlan<float> plan({true, true, m, n, k});
  const std::vector<float> a(m * k, 0.25f), b(n * k, 0.5f);
  std::vector<float> y(m * n, -1.0f);
  const std::size_t bytes = detail::GemmBatchScratchBytes(plan.ScratchBytes(1));
  ExecutionBuffer storage(bytes);
  detail::GemmScratch scratch{static_cast<std::byte *>(storage.data()), 1};
  EXPECT_THROW(plan.Execute(a.data(), b.data(), nullptr, y.data(), &scratch),
               std::invalid_argument);
  EXPECT_EQ(scratch.used, 0U);
  EXPECT_TRUE(std::all_of(y.begin(), y.end(), [](float value) { return value == -1.0f; }));
  scratch.capacity = bytes;
  for (int repeat = 0; repeat < 2; ++repeat) {
    plan.Execute(a.data(), b.data(), nullptr, y.data(), &scratch);
    EXPECT_EQ(scratch.used, 0U);
    EXPECT_TRUE(std::all_of(y.begin(), y.end(), [](float value) { return value == k * 0.125f; }));
  }
}

TEST(GemmScratchAllocations, BorrowedAllocatorPreservesAllocationFreeBatchParallelism) {
  using namespace ml_cpu_compute;
  constexpr std::size_t batches = 1024, m = 32, n = 64, k = 16;
  const MatMulPlan<float> plan(std::array<std::size_t, 3>{batches, m, k},
                               std::array<std::size_t, 2>{k, n});
  ASSERT_EQ(plan.ScratchBytes(4), 0U);
  const std::vector<float> a(batches * m * k, 0.25f), b(k * n, 0.5f);
  std::vector<float> y(batches * m * n);
  AllocationCheckingExecutor executor_tracker;
  ExecutionExecutorView executor{&executor_tracker, 4, &AllocationCheckingExecutor::Run};
  ExecutionExecutorScope executor_scope(&executor);
  TrackingAllocator tracker;
  const auto allocator = tracker.View();
  ExecutionAllocatorScope allocator_scope(&allocator);
  callback_allocations = 0;
  callback_parent_depth = 0;
  plan.Execute(a.data(), b.data(), y.data());
  callback_parent_depth = -1;
  EXPECT_GT(executor_tracker.maximum_blocks, 1);
  EXPECT_EQ(callback_allocations.load(), 0U);
  EXPECT_EQ(tracker.allocations, 0U);
  EXPECT_TRUE(std::all_of(y.begin(), y.end(), [](float value) { return value == 2.0f; }));
}

TEST(GemmScratchAllocations, ConcurrentPackingAndSplitKFitPlanBounds) {
  for (const auto shape : {std::array<std::size_t, 3>{128, 513, 257}, {32, 32, 16384}}) {
    const auto [m, n, k] = shape;
    const ml_cpu_compute::GemmPlan<float> f32({false, false, m, n, k});
    const ml_cpu_compute::GemmPlan<double> f64({false, false, m, n, k});
    const ml_cpu_compute::GemmHalfPlan half({false, false, false, m, n, k});
    const ml_cpu_compute::GemmHalfPlan bf16({true, false, false, m, n, k});
    const std::vector<float> a(m * k, 0.25f), b(k * n, 0.5f);
    const std::vector<double> ad(m * k, 0.25), bd(k * n, 0.5);
    const std::vector<std::uint16_t> ah(m * k, 0x3400), bh(k * n, 0x3800);
    const std::vector<std::uint16_t> ab(m * k, 0x3e80), bb(k * n, 0x3f00);
    std::vector<float> y(m * n);
    std::vector<double> yd(m * n);
    ml_cpu_compute::GemmEpilogue<float> epilogue;
    ml_cpu_compute::ExecutionExecutorView executor{nullptr, 4, &ConcurrentExecutor::Run};
    ml_cpu_compute::ExecutionExecutorScope scope(&executor);
    const std::size_t float_peak =
        PackingPeak([&] { f32.Execute(a.data(), b.data(), nullptr, y.data()); });
    EXPECT_GT(float_peak, 0U);
    EXPECT_LE(float_peak, f32.ScratchBytes(4));
    EXPECT_LE(PackingPeak([&] { f64.Execute(ad.data(), bd.data(), nullptr, yd.data()); }),
              f64.ScratchBytes(4));
    EXPECT_LE(PackingPeak([&] { half.Execute(ah.data(), bh.data(), epilogue, y.data()); }),
              half.ScratchBytes(4));
    EXPECT_LE(PackingPeak([&] { bf16.Execute(ab.data(), bb.data(), epilogue, y.data()); }),
              bf16.ScratchBytes(4));
  }
}

TEST(GemmScratchAllocations, ConcurrentBroadcastProductsFitPlanBounds) {
  constexpr std::size_t batches = 4, m = 128, n = 513, k = 257;
  const ml_cpu_compute::MatMulPlan<float> plan(std::array<std::size_t, 3>{batches, m, k},
                                               std::array<std::size_t, 2>{k, n});
  const ml_cpu_compute::GemmHalfPlan half({false, false, false, m, n, k});
  const std::vector<float> a(batches * m * k, 0.25f), b(k * n, 0.5f);
  const std::vector<std::uint16_t> ah(batches * m * k, 0x3400), bh(k * n, 0x3800);
  std::vector<float> workspace(batches * m * n);
  std::vector<std::uint16_t> y(batches * m * n);
  ml_cpu_compute::GemmEpilogue<float> epilogue;
  epilogue.output_conversion = ml_cpu_compute::GemmOutputConversion::kFloat16;
  epilogue.converted_output = y.data();
  ml_cpu_compute::ExecutionExecutorView executor{nullptr, 4, &ConcurrentExecutor::Run};
  ml_cpu_compute::ExecutionExecutorScope scope(&executor);
  EXPECT_LE(PackingPeak([&] { plan.Execute(a.data(), b.data(), workspace.data()); }),
            plan.ScratchBytes(4));
  const std::size_t packing = PackingPeak(
      [&] { plan.ExecuteHalf(half, ah.data(), bh.data(), epilogue, workspace.data()); });
  EXPECT_LE(packing + workspace.size() * sizeof(float), plan.HalfScratchBytes(false, 4));
}

TEST(GemmScratchAllocations, ChangingTransposedAShapeDoesNotRetainPackedCapacity) {
  constexpr std::size_t n = 65, k = 128;
  const ml_cpu_compute::GemmPlan<float> large(
      ml_cpu_compute::GemmPlanOptions<float>{true, false, 4096, n, k});
  const ml_cpu_compute::GemmPlan<float> small(
      ml_cpu_compute::GemmPlanOptions<float>{true, false, 64, n, k});
  const std::vector<float> large_a(4096 * k, 0.25f), small_a(64 * k, 0.25f);
  const std::vector<float> b(k * n, 0.5f);
  std::vector<float> large_y(4096 * n), small_y(64 * n);
  ml_cpu_compute::ExecutionExecutorScope serial(nullptr);

  large.Execute(large_a.data(), b.data(), nullptr, large_y.data());
  EXPECT_EQ(small.ScratchBytes(1), 64U * k * sizeof(float));
  EXPECT_EQ(PackingPeak([&] { small.Execute(small_a.data(), b.data(), nullptr, small_y.data()); }),
            small.ScratchBytes(1));
}

TEST(GemmScratchAllocations, CustomAvx512ColumnOverrideFitsPlanBound) {
  if (ml_cpu_compute::EffectiveGemmSimdLevel() < ml_cpu_compute::SimdLevel::kAVX512) {
    GTEST_SKIP() << "Requires AVX-512 to select the 16-column FP32 packing override.";
  }
  ml_cpu_compute::GemmPlanOptions<float> options;
  options.trans_a = true;
  options.m = 128;
  options.n = 4096;
  options.k = 256;
  options.blocking = {128, 1, 256};
  options.maximum_participants = 1;
  const ml_cpu_compute::GemmPlan<float> plan(options);
  const std::vector<float> a(options.m * options.k, 0.25f), b(options.k * options.n, 0.5f);
  std::vector<float> y(options.m * options.n);
  ml_cpu_compute::ExecutionExecutorScope serial(nullptr);
  EXPECT_EQ(plan.ScratchBytes(1), 147456U);
  EXPECT_EQ(PackingPeak([&] { plan.Execute(a.data(), b.data(), nullptr, y.data()); }), 147456U);
}

template <typename T> void CheckSubMrBlockingFitsPlanBound() {
  for (const bool trans_b : {false, true}) {
    ml_cpu_compute::GemmPlanOptions<T> options;
    options.trans_a = true;
    options.trans_b = trans_b;
    options.m = 24;
    options.n = 4096;
    options.k = 256;
    options.blocking = {1, 4096, 256};
    options.maximum_participants = 1;
    options.m = 2 * ml_cpu_compute::GemmPlan<T>(options).blocking().mr;
    const ml_cpu_compute::GemmPlan<T> plan(options);
    SCOPED_TRACE(::testing::Message()
                 << "element_size=" << sizeof(T) << ", M=" << options.m << ", trans_b=" << trans_b);
    ASSERT_LT(plan.blocking().mc, plan.blocking().mr);
    const std::vector<T> a(options.m * options.k, T(0.25)), b(options.k * options.n, T(0.5));
    std::vector<T> y(options.m * options.n);
    ml_cpu_compute::ExecutionExecutorScope serial(nullptr);
    const std::size_t peak =
        PackingPeak([&] { plan.Execute(a.data(), b.data(), nullptr, y.data()); });
    EXPECT_GE(peak, options.k * options.n * sizeof(T));
    EXPECT_LE(peak, plan.ScratchBytes(1));
  }
}

TEST(GemmScratchAllocations, CustomSubMrBlockingFitsPlanBound) {
  CheckSubMrBlockingFitsPlanBound<float>();
  CheckSubMrBlockingFitsPlanBound<double>();
}

TEST(GemmScratchAllocations, CustomHalfBlockingFitsMatMulPlanEstimate) {
  constexpr std::size_t m = 128, n = 4096, k = 512;
  const ml_cpu_compute::MatMulPlan<float> matmul(m, n, k, false, true);
  ml_cpu_compute::GemmHalfPlanOptions options{true, false, true, m, n, k};
  options.blocking = {128, 4096, 512};
  options.maximum_participants = 1;
  const ml_cpu_compute::GemmHalfPlan half(options);
  const std::vector<std::uint16_t> a(m * k, 0x3e80), b(n * k, 0x3f00);
  std::vector<float> workspace(m * n);
  std::vector<std::uint16_t> output(m * n);
  ml_cpu_compute::GemmEpilogue<float> epilogue;
  epilogue.output_conversion = ml_cpu_compute::GemmOutputConversion::kBFloat16;
  epilogue.converted_output = output.data();
  ml_cpu_compute::ExecutionExecutorScope serial(nullptr);
  const std::size_t workspace_bytes = workspace.size() * sizeof(float);
  ASSERT_GE(half.ScratchBytes(1), 8U * 1024 * 1024);
  const std::size_t packing = PackingPeak(
      [&] { matmul.ExecuteHalf(half, a.data(), b.data(), epilogue, workspace.data()); });
  EXPECT_LE(packing + workspace_bytes, matmul.HalfScratchBytes(half, 1));
  EXPECT_THROW(
      matmul.HalfScratchBytes(ml_cpu_compute::GemmHalfPlan({true, false, false, m, n, k}), 1),
      std::invalid_argument);
}

} // namespace
