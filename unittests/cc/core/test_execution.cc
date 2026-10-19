// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "cpu_compute/execution.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

using cpu_compute::CurrentExecutionExecutor;
using cpu_compute::ExecuteRanges;
using cpu_compute::ExecutionBlockCount;
using cpu_compute::ExecutionBlockFn;
using cpu_compute::ExecutionExecutorScope;
using cpu_compute::ExecutionExecutorView;
using cpu_compute::ExecutionInParallelRegion;
using cpu_compute::ExecutionParallelPlan;
using cpu_compute::ExecutionThreadCount;
using cpu_compute::ExecutionWorkCost;

struct InlineExecutor {
  int64_t dispatches = 0;
  int64_t blocks = 0;

  static void Run(void *context, int64_t num_blocks, void *task_context,
                  cpu_compute::ExecutionBlockFn task) {
    auto &self = *static_cast<InlineExecutor *>(context);
    ++self.dispatches;
    self.blocks = num_blocks;
    for (int64_t block = num_blocks; block > 0; --block) {
      task(task_context, block - 1);
    }
  }
};

struct CostExecutor : InlineExecutor {
  cpu_compute::ExecutionWorkCost observed_cost;
  int64_t observed_maximum = 0;
  int64_t observed_preferred = 0;

  static cpu_compute::ExecutionParallelPlan Plan(void *context, int64_t,
                                                 const cpu_compute::ExecutionWorkCost &cost,
                                                 int64_t maximum_participants,
                                                 int64_t preferred_participants) {
    auto &self = *static_cast<CostExecutor *>(context);
    self.observed_cost = cost;
    self.observed_maximum = maximum_participants;
    self.observed_preferred = preferred_participants;
    return {100, 3};
  }
};

TEST(Execution, StandaloneIsSerial) {
  EXPECT_EQ(cpu_compute::CurrentExecutionExecutor(), nullptr);
  EXPECT_EQ(cpu_compute::ExecutionThreadCount(), 1);
  EXPECT_EQ(cpu_compute::ExecutionBlockCount(1000000), 1);

  int calls = 0;
  cpu_compute::ExecuteRanges(1000000, [&](int64_t begin, int64_t end) {
    EXPECT_TRUE(cpu_compute::ExecutionInParallelRegion());
    EXPECT_EQ(begin, 0);
    EXPECT_EQ(end, 1000000);
    ++calls;
  });
  EXPECT_EQ(calls, 1);
}

TEST(Execution, InjectedExecutorOwnsParallelDispatch) {
  InlineExecutor executor;
  cpu_compute::ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run};
  std::vector<std::atomic<int>> hits(200000);
  {
    cpu_compute::ExecutionExecutorScope scope(&view);
    EXPECT_EQ(cpu_compute::ExecutionThreadCount(), 4);
    cpu_compute::ExecuteRanges(
        static_cast<int64_t>(hits.size()), 1.0, 16, [&hits](int64_t begin, int64_t end) {
          EXPECT_TRUE(cpu_compute::ExecutionInParallelRegion());
          for (int64_t i = begin; i < end; ++i) {
            hits[static_cast<std::size_t>(i)].fetch_add(1, std::memory_order_relaxed);
          }
        });
  }

  EXPECT_EQ(executor.dispatches, 1);
  EXPECT_GT(executor.blocks, 1);
  EXPECT_LE(executor.blocks, 4);
  for (const auto &hit : hits) {
    EXPECT_EQ(hit.load(std::memory_order_relaxed), 1);
  }
  EXPECT_EQ(cpu_compute::CurrentExecutionExecutor(), nullptr);
}

TEST(Execution, SmallInjectedWorkRemainsInline) {
  InlineExecutor executor;
  cpu_compute::ExecutionExecutorView view{&executor, 8, &InlineExecutor::Run};
  int calls = 0;
  {
    cpu_compute::ExecutionExecutorScope scope(&view);
    cpu_compute::ExecuteRanges(8, [&calls](int64_t begin, int64_t end) {
      EXPECT_EQ(begin, 0);
      EXPECT_EQ(end, 8);
      ++calls;
    });
  }
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(executor.dispatches, 0);
}

TEST(Execution, NestedScopesRestoreBorrowedViewsAndClampThreads) {
  InlineExecutor executor;
  cpu_compute::ExecutionExecutorView outer{&executor, 4, &InlineExecutor::Run};
  cpu_compute::ExecutionExecutorView inner{&executor, 0, &InlineExecutor::Run};
  {
    cpu_compute::ExecutionExecutorScope outer_scope(&outer);
    EXPECT_EQ(cpu_compute::CurrentExecutionExecutor(), &outer);
    {
      cpu_compute::ExecutionExecutorScope inner_scope(&inner);
      EXPECT_EQ(cpu_compute::CurrentExecutionExecutor(), &inner);
      EXPECT_EQ(cpu_compute::ExecutionThreadCount(), 1);
      EXPECT_EQ(cpu_compute::ExecutionBlockCount(1000000), 1);
    }
    EXPECT_EQ(cpu_compute::CurrentExecutionExecutor(), &outer);
    EXPECT_EQ(cpu_compute::ExecutionThreadCount(), 4);
  }
  EXPECT_EQ(cpu_compute::CurrentExecutionExecutor(), nullptr);
}

TEST(Execution, DispatchFailurePropagatesAndRestoresScope) {
  cpu_compute::ExecutionExecutorView view{
      nullptr, 4, [](void *, int64_t, void *, cpu_compute::ExecutionBlockFn) {
        throw std::runtime_error("dispatch failed");
      }};
  int calls = 0;
  const auto run = [&]() {
    cpu_compute::ExecutionExecutorScope scope(&view);
    cpu_compute::ExecuteRanges(1000000, [&](int64_t, int64_t) { ++calls; });
  };
  EXPECT_THROW(run(), std::runtime_error);
  EXPECT_EQ(calls, 0);
  EXPECT_EQ(cpu_compute::CurrentExecutionExecutor(), nullptr);
  EXPECT_FALSE(cpu_compute::ExecutionInParallelRegion());
}

TEST(Execution, NestedRangesDoNotDispatchAgain) {
  InlineExecutor executor;
  cpu_compute::ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run};
  {
    cpu_compute::ExecutionExecutorScope scope(&view);
    cpu_compute::ExecuteRanges(
        4, static_cast<double>(cpu_compute::kExecutionGrainSize), [](int64_t begin, int64_t end) {
          EXPECT_EQ(cpu_compute::ExecutionThreadCount(), 4);
          EXPECT_EQ(cpu_compute::ExecutionBlockCount(1000000), 1);
          for (int64_t outer = begin; outer < end; ++outer) {
            (void)outer;
            int calls = 0;
            cpu_compute::ExecuteRanges(1000000, [&calls](int64_t inner_begin, int64_t inner_end) {
              EXPECT_EQ(inner_begin, 0);
              EXPECT_EQ(inner_end, 1000000);
              ++calls;
            });
            EXPECT_EQ(calls, 1);
          }
        });
  }
  EXPECT_EQ(executor.dispatches, 1);
}

TEST(Execution, CostedRangesUseExecutorPlan) {
  CostExecutor executor;
  cpu_compute::ExecutionExecutorView view{&executor, 8, &CostExecutor::Run, &CostExecutor::Plan};
  std::vector<std::atomic<int>> hits(1000);
  {
    cpu_compute::ExecutionExecutorScope scope(&view);
    cpu_compute::ExecuteCostedRanges(
        static_cast<int64_t>(hits.size()), cpu_compute::ExecutionWorkCost{4.0, 4.0, 3.0},
        cpu_compute::ExecutionSchedule{1, 1, 5, 3}, 16, [&hits](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            hits[static_cast<std::size_t>(i)].fetch_add(1, std::memory_order_relaxed);
          }
        });
  }

  EXPECT_EQ(executor.observed_cost.bytes_read, 4.0);
  EXPECT_EQ(executor.observed_cost.bytes_written, 4.0);
  EXPECT_EQ(executor.observed_cost.compute_cycles, 3.0);
  EXPECT_EQ(executor.observed_maximum, 5);
  EXPECT_EQ(executor.observed_preferred, 3);
  EXPECT_EQ(executor.blocks, 3);
  for (const auto &hit : hits) {
    EXPECT_EQ(hit.load(std::memory_order_relaxed), 1);
  }
}

} // namespace
