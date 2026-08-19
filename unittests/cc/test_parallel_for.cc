// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/parallel_for.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <set>
#include <thread>
#include <utility>
#include <vector>

namespace {

using onnx_light_cpu::CpuAffinity;
using onnx_light_cpu::CpuCoreKind;
using onnx_light_cpu::CpuThread;
using onnx_light_cpu::CpuTopology;
using onnx_light_cpu::GetCpuTopology;
using onnx_light_cpu::GetCurrentThreadAffinity;
using onnx_light_cpu::kParallelForGrainSize;
using onnx_light_cpu::kParallelForMaxThreads;
using onnx_light_cpu::ParallelFor;
using onnx_light_cpu::ParallelForBlockCount;
using onnx_light_cpu::ParallelForBlockFn;
using onnx_light_cpu::ParallelForExecutorScope;
using onnx_light_cpu::ParallelForExecutorView;
using onnx_light_cpu::ParallelForExternalRegion;
using onnx_light_cpu::ParallelForSimdLanes;
using onnx_light_cpu::ParallelForSpinCount;
using onnx_light_cpu::ParallelForThreadCount;
using onnx_light_cpu::SelectCpuAffinities;
using onnx_light_cpu::StandaloneThreadPoolCreationCount;
using onnx_light_cpu::ThreadPool;
using onnx_light_cpu::detail::ResolveParallelForSpinCount;
using onnx_light_cpu::detail::ResolveParallelForThreadCount;

// ---------------------------------------------------------------------------
// Cost model: ParallelForBlockCount
// ---------------------------------------------------------------------------

TEST(ParallelForThreadCount, IsStableAndCapped) {
  const int64_t first = ParallelForThreadCount();
  const CpuTopology &topology = GetCpuTopology();
  EXPECT_EQ(ParallelForThreadCount(), first);
  EXPECT_EQ(first,
            ResolveParallelForThreadCount(static_cast<int64_t>(topology.logical_thread_count),
                                          static_cast<int64_t>(topology.physical_core_count),
                                          std::getenv("ONNX_LIGHT_CPU_NUM_THREADS")));
  EXPECT_GE(first, 1);
  EXPECT_LE(first, kParallelForMaxThreads);
}

TEST(ParallelForThreadCount, ResolvesRuntimeLimit) {
  const int64_t default_count = std::min<int64_t>(8, kParallelForMaxThreads);
  EXPECT_EQ(ResolveParallelForThreadCount(8, nullptr), default_count);
  EXPECT_EQ(ResolveParallelForThreadCount(8, ""), default_count);
  EXPECT_EQ(ResolveParallelForThreadCount(8, "1"), 1);
  EXPECT_EQ(ResolveParallelForThreadCount(8, "2"), 2);
  EXPECT_EQ(ResolveParallelForThreadCount(2, "8"), 2);
  EXPECT_EQ(ResolveParallelForThreadCount(8, "0"), default_count);
  EXPECT_EQ(ResolveParallelForThreadCount(8, "-1"), default_count);
  EXPECT_EQ(ResolveParallelForThreadCount(8, "invalid"), default_count);
}

TEST(ParallelForThreadCount, DefaultsToPhysicalCoresAndAllowsExplicitSmt) {
  EXPECT_EQ(ResolveParallelForThreadCount(16, 8, nullptr),
            std::min<int64_t>(8, kParallelForMaxThreads));
  EXPECT_EQ(ResolveParallelForThreadCount(16, 8, ""), std::min<int64_t>(8, kParallelForMaxThreads));
  EXPECT_EQ(ResolveParallelForThreadCount(16, 8, "12"),
            std::min<int64_t>(12, kParallelForMaxThreads));
  EXPECT_EQ(ResolveParallelForThreadCount(4, 2, "12"),
            std::min<int64_t>(4, kParallelForMaxThreads));
}

TEST(ParallelForSpinCount, ResolvesBoundedConfiguration) {
  EXPECT_EQ(ResolveParallelForSpinCount(nullptr), 2000);
  EXPECT_EQ(ResolveParallelForSpinCount(""), 2000);
  EXPECT_EQ(ResolveParallelForSpinCount("0"), 0);
  EXPECT_EQ(ResolveParallelForSpinCount("4096"), 4096);
  EXPECT_EQ(ResolveParallelForSpinCount("2000000"), 1000000);
  EXPECT_EQ(ResolveParallelForSpinCount("-1"), 2000);
  EXPECT_EQ(ResolveParallelForSpinCount("invalid"), 2000);
  EXPECT_GE(ParallelForSpinCount(), 0);
  EXPECT_LE(ParallelForSpinCount(), 1000000);
}

TEST(CpuTopology, DetectsUsableProcessorCounts) {
  const CpuTopology &topology = GetCpuTopology();
  EXPECT_GE(topology.logical_thread_count, 1u);
  EXPECT_GE(topology.physical_core_count, 1u);
  EXPECT_LE(topology.physical_core_count, topology.logical_thread_count);
  EXPECT_EQ(topology.logical_thread_count, topology.threads.size());
  EXPECT_LE(topology.performance_core_count + topology.efficiency_core_count,
            topology.physical_core_count);
}

TEST(CpuTopology, SelectsPhysicalPerformanceCoresBeforeEfficiencyAndSmt) {
  CpuTopology topology;
  topology.logical_thread_count = 5;
  topology.physical_core_count = 3;
  topology.performance_core_count = 1;
  topology.efficiency_core_count = 1;
  topology.threads = {
      CpuThread{{0, 4}, 2, CpuCoreKind::kEfficiency, true},
      CpuThread{{0, 1}, 0, CpuCoreKind::kPerformance, false},
      CpuThread{{0, 3}, 1, CpuCoreKind::kUnknown, false},
      CpuThread{{0, 0}, 0, CpuCoreKind::kPerformance, true},
      CpuThread{{0, 2}, 1, CpuCoreKind::kUnknown, true},
  };

  const std::vector<CpuAffinity> affinities = SelectCpuAffinities(topology, 5);

  ASSERT_EQ(affinities.size(), 5u);
  EXPECT_EQ(affinities[0].index, 0u);
  EXPECT_EQ(affinities[1].index, 2u);
  EXPECT_EQ(affinities[2].index, 4u);
  EXPECT_EQ(affinities[3].index, 1u);
  EXPECT_EQ(affinities[4].index, 3u);
}

TEST(ThreadPool, PinsWorkersToSelectedProcessors) {
#if defined(__linux__) || defined(_WIN32)
  std::vector<CpuAffinity> affinities = SelectCpuAffinities(GetCpuTopology(), 2);
  if (affinities.size() < 2) {
    GTEST_SKIP() << "Affinity test requires at least two available processors.";
  }
  affinities.erase(affinities.begin());
  ThreadPool pool(1, affinities);
  std::array<CpuAffinity, 1> observed = {};
  std::array<bool, 1> detected = {};

  pool.Run(2, [&](int64_t block) {
    if (block > 0) {
      detected[static_cast<std::size_t>(block - 1)] =
          GetCurrentThreadAffinity(observed[static_cast<std::size_t>(block - 1)]);
    }
  });

  for (std::size_t index = 0; index < affinities.size(); ++index) {
    ASSERT_TRUE(detected[index]);
    EXPECT_EQ(observed[index].group, affinities[index].group);
    EXPECT_EQ(observed[index].index, affinities[index].index);
  }
#else
  GTEST_SKIP() << "Thread affinity is only applied on Linux and Windows.";
#endif
}

TEST(ParallelForBlockCount, NonPositiveTotalIsZero) {
  EXPECT_EQ(ParallelForBlockCount(0), 0);
  EXPECT_EQ(ParallelForBlockCount(-10), 0);
}

TEST(ParallelForBlockCount, SmallRangeRunsInline) {
  // Well below one grain of trivial work: never worth parallelizing.
  EXPECT_EQ(ParallelForBlockCount(1), 1);
  EXPECT_EQ(ParallelForBlockCount(kParallelForGrainSize - 1), 1);
}

TEST(ParallelForBlockCount, NeverMoreBlocksThanThreads) {
  const int64_t threads = ParallelForThreadCount();
  // A huge range should still cap the block count at the thread count.
  const int64_t blocks = ParallelForBlockCount(kParallelForGrainSize * 100000);
  EXPECT_GE(blocks, 1);
  EXPECT_LE(blocks, threads);
}

TEST(ParallelForBlockCount, NeverMoreBlocksThanIterations) {
  // 5 very expensive iterations: work justifies more blocks, but there are
  // only 5 iterations so at most 5 blocks can be produced.
  const int64_t blocks = ParallelForBlockCount(5, /*cost_per_element=*/1e9);
  EXPECT_GE(blocks, 1);
  EXPECT_LE(blocks, 5);
}

TEST(ParallelForBlockCount, HigherCostParallelizesSmallerRanges) {
  // A range that is inline for trivial work...
  const int64_t small = kParallelForGrainSize / 8;
  EXPECT_EQ(ParallelForBlockCount(small, /*cost_per_element=*/1.0), 1);
  if (ParallelForThreadCount() > 1) {
    // ...becomes worth parallelizing once each element is much more expensive.
    EXPECT_GT(ParallelForBlockCount(small, /*cost_per_element=*/1024.0), 1);
  }
}

TEST(ParallelForBlockCount, NonPositiveCostTreatedAsOne) {
  const int64_t total = kParallelForGrainSize * 8;
  EXPECT_EQ(ParallelForBlockCount(total, 0.0), ParallelForBlockCount(total, 1.0));
  EXPECT_EQ(ParallelForBlockCount(total, -3.0), ParallelForBlockCount(total, 1.0));
}

// ---------------------------------------------------------------------------
// ParallelFor: correctness (coverage + disjointedness, thread-count independent)
// ---------------------------------------------------------------------------

struct InjectedExecutorObservation {
  int64_t dispatches = 0;
  int64_t maximum_blocks = 0;
};

void RunInjectedBlocks(void *context, int64_t num_blocks, void *task_context,
                       ParallelForBlockFn task) {
  auto &observation = *static_cast<InjectedExecutorObservation *>(context);
  ++observation.dispatches;
  observation.maximum_blocks = std::max(observation.maximum_blocks, num_blocks);
  for (int64_t block = 0; block < num_blocks; ++block) {
    task(task_context, block);
  }
}

TEST(ParallelFor, InjectedExecutorControlsParticipantsAndNestedDispatch) {
  InjectedExecutorObservation observation;
  ParallelForExecutorView executor{&observation, 3, &RunInjectedBlocks};
  const uint64_t private_pools_before = StandaloneThreadPoolCreationCount();
  {
    ParallelForExecutorScope scope(&executor);
    EXPECT_EQ(ParallelForThreadCount(), 3);
    ParallelFor(kParallelForGrainSize * 8, [&](int64_t begin, int64_t) {
      if (begin == 0) {
        ParallelFor(kParallelForGrainSize * 8, [](int64_t, int64_t) {});
      }
    });
  }

  EXPECT_EQ(observation.dispatches, 2);
  EXPECT_LE(observation.maximum_blocks, 3);
  EXPECT_EQ(StandaloneThreadPoolCreationCount(), private_pools_before);
}

TEST(ParallelFor, StandaloneExecutionLazilyConstructsPrivatePool) {
  const uint64_t private_pools_before = StandaloneThreadPoolCreationCount();

  ParallelFor(kParallelForGrainSize * 8, [](int64_t, int64_t) {});

  EXPECT_EQ(StandaloneThreadPoolCreationCount(), private_pools_before + 1);
}

TEST(ParallelFor, EmptyRangeIsNoOp) {
  int calls = 0;
  ParallelFor(0, [&](int64_t, int64_t) { ++calls; });
  EXPECT_EQ(calls, 0);
}

TEST(ParallelFor, CoversRangeExactlyOnce) {
  for (int64_t total : {int64_t{1}, int64_t{100}, kParallelForGrainSize,
                        kParallelForGrainSize * 4 + 7, kParallelForGrainSize * 37 + 3}) {
    std::vector<std::atomic<int>> hits(static_cast<std::size_t>(total));
    for (auto &h : hits) {
      h.store(0, std::memory_order_relaxed);
    }
    ParallelFor(total, [&](int64_t begin, int64_t end) {
      EXPECT_LE(begin, end);
      for (int64_t i = begin; i < end; ++i) {
        hits[static_cast<std::size_t>(i)].fetch_add(1, std::memory_order_relaxed);
      }
    });
    for (int64_t i = 0; i < total; ++i) {
      EXPECT_EQ(hits[static_cast<std::size_t>(i)].load(std::memory_order_relaxed), 1)
          << "at index " << i << " total=" << total;
    }
  }
}

TEST(ParallelFor, ElementwiseResultMatchesSerial) {
  // A large enough range to exercise the parallel path when multiple threads
  // are available, while still being correct on a single core.
  const int64_t total = kParallelForGrainSize * 8 + 123;
  std::vector<int> in(static_cast<std::size_t>(total));
  std::vector<int> out(static_cast<std::size_t>(total), -1);
  for (int64_t i = 0; i < total; ++i) {
    in[static_cast<std::size_t>(i)] = static_cast<int>(i % 97) - 48;
  }
  ParallelFor(total, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const int v = in[static_cast<std::size_t>(i)];
      out[static_cast<std::size_t>(i)] = v < 0 ? -v : v;
    }
  });
  for (int64_t i = 0; i < total; ++i) {
    const int v = in[static_cast<std::size_t>(i)];
    EXPECT_EQ(out[static_cast<std::size_t>(i)], v < 0 ? -v : v) << "at index " << i;
  }
}

TEST(ParallelFor, NestedCallStaysCorrect) {
  // A ParallelFor launched from within another must not deadlock and must still
  // cover its inner range exactly once (it falls back to the serial path).
  const int64_t outer = kParallelForGrainSize * 4;
  const int64_t inner = 256;
  std::atomic<int64_t> sum{0};
  ParallelFor(outer, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      (void)i;
    }
    ParallelFor(inner, [&](int64_t b, int64_t e) {
      for (int64_t j = b; j < e; ++j) {
        sum.fetch_add(1, std::memory_order_relaxed);
      }
    });
  });
  // The inner loop runs once per outer block; just ensure it made progress and
  // every inner iteration was counted a whole number of times.
  EXPECT_EQ(sum.load(std::memory_order_relaxed) % inner, 0);
  EXPECT_GT(sum.load(std::memory_order_relaxed), 0);
}

TEST(ParallelFor, ExternalRegionDoesNotWakeNestedWorkers) {
  std::set<std::thread::id> threads;
  std::mutex mutex;
  {
    ParallelForExternalRegion external_region;
    ParallelFor(kParallelForGrainSize * 8, /*cost_per_element=*/64.0, [&](int64_t, int64_t) {
      std::lock_guard<std::mutex> lock(mutex);
      threads.insert(std::this_thread::get_id());
    });
  }
  EXPECT_EQ(threads.size(), 1u);
}

// ---------------------------------------------------------------------------
// ParallelFor: SIMD-aligned block boundaries
// ---------------------------------------------------------------------------

TEST(ParallelForSimdLanes, MatchesWidestRegisterForType) {
  // 64-byte (AVX-512) register: lane count is 64 / sizeof(T), at least 1.
  EXPECT_EQ(ParallelForSimdLanes<float>(), 16);
  EXPECT_EQ(ParallelForSimdLanes<double>(), 8);
  EXPECT_EQ(ParallelForSimdLanes<std::uint16_t>(), 32);
  EXPECT_EQ(ParallelForSimdLanes<std::int8_t>(), 64);
  EXPECT_EQ(ParallelForSimdLanes<std::int32_t>(), 16);
  EXPECT_EQ(ParallelForSimdLanes<std::int64_t>(), 8);
}

TEST(ParallelFor, BlocksAreMultiplesOfSimdWidth) {
  const int64_t multiple = 16;
  // Costs high enough to force the parallel path even on modest core counts,
  // and totals that are deliberately not multiples of the SIMD width.
  for (int64_t total : {kParallelForGrainSize * 8 + 5, kParallelForGrainSize * 37 + 3,
                        kParallelForGrainSize * 3 + 1}) {
    std::mutex mu;
    std::vector<std::pair<int64_t, int64_t>> ranges;
    ParallelFor(total, /*cost_per_element=*/64.0, multiple, [&](int64_t begin, int64_t end) {
      std::lock_guard<std::mutex> lock(mu);
      ranges.emplace_back(begin, end);
    });
    std::sort(ranges.begin(), ranges.end());
    ASSERT_FALSE(ranges.empty());
    int64_t expected_begin = 0;
    for (std::size_t k = 0; k < ranges.size(); ++k) {
      const auto [begin, end] = ranges[k];
      // Contiguous, disjoint coverage of [0, total).
      EXPECT_EQ(begin, expected_begin) << "total=" << total;
      EXPECT_LT(begin, end);
      // Every block begins on a SIMD-vector boundary.
      EXPECT_EQ(begin % multiple, 0) << "total=" << total;
      // Every block ends on a SIMD-vector boundary, except the final one which
      // may stop early at total.
      if (k + 1 < ranges.size()) {
        EXPECT_EQ(end % multiple, 0) << "total=" << total;
      }
      expected_begin = end;
    }
    EXPECT_EQ(expected_begin, total) << "total=" << total;
  }
}

TEST(ParallelFor, SimdAlignedCoversRangeExactlyOnce) {
  const int64_t multiple = 16;
  const int64_t total = kParallelForGrainSize * 8 + 5;
  std::vector<std::atomic<int>> hits(static_cast<std::size_t>(total));
  for (auto &h : hits) {
    h.store(0, std::memory_order_relaxed);
  }
  ParallelFor(total, /*cost_per_element=*/64.0, multiple, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      hits[static_cast<std::size_t>(i)].fetch_add(1, std::memory_order_relaxed);
    }
  });
  for (int64_t i = 0; i < total; ++i) {
    EXPECT_EQ(hits[static_cast<std::size_t>(i)].load(std::memory_order_relaxed), 1)
        << "at index " << i;
  }
}

} // namespace
