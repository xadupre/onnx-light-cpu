// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/parallel_for.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

using onnx_light_cpu::kParallelForGrainSize;
using onnx_light_cpu::ParallelFor;
using onnx_light_cpu::ParallelForBlockCount;
using onnx_light_cpu::ParallelForThreadCount;

// ---------------------------------------------------------------------------
// Cost model: ParallelForBlockCount
// ---------------------------------------------------------------------------

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
// ParallelFor: correctness (coverage + disjointness, thread-count independent)
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

} // namespace
