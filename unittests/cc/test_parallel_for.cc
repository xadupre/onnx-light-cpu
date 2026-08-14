// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/parallel_for.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <utility>
#include <vector>

namespace {

using onnx_light_cpu::kParallelForGrainSize;
using onnx_light_cpu::kParallelForMaxThreads;
using onnx_light_cpu::ParallelFor;
using onnx_light_cpu::ParallelForBlockCount;
using onnx_light_cpu::ParallelForSimdLanes;
using onnx_light_cpu::ParallelForThreadCount;
using onnx_light_cpu::detail::ResolveParallelForThreadCount;

// ---------------------------------------------------------------------------
// Cost model: ParallelForBlockCount
// ---------------------------------------------------------------------------

TEST(ParallelForThreadCount, IsStableAndCapped) {
  const int64_t first = ParallelForThreadCount();
  EXPECT_EQ(ParallelForThreadCount(), first);
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
