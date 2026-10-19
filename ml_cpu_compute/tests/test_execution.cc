// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "ml_cpu_compute/execution.h"
#include "ml_cpu_compute/math/math.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

using ml_cpu_compute::CurrentExecutionExecutor;
using ml_cpu_compute::ExecuteRanges;
using ml_cpu_compute::ExecutionBlockCount;
using ml_cpu_compute::ExecutionBlockFn;
using ml_cpu_compute::ExecutionExecutorScope;
using ml_cpu_compute::ExecutionExecutorView;
using ml_cpu_compute::ExecutionInParallelRegion;
using ml_cpu_compute::ExecutionParallelPlan;
using ml_cpu_compute::ExecutionThreadCount;
using ml_cpu_compute::ExecutionWorkCost;

struct InlineExecutor {
  int64_t dispatches = 0;
  int64_t blocks = 0;

  static void Run(void *context, int64_t num_blocks, void *task_context,
                  ml_cpu_compute::ExecutionBlockFn task) {
    auto &self = *static_cast<InlineExecutor *>(context);
    ++self.dispatches;
    self.blocks = num_blocks;
    for (int64_t block = num_blocks; block > 0; --block) {
      task(task_context, block - 1);
    }
  }
};

struct CostExecutor : InlineExecutor {
  ml_cpu_compute::ExecutionWorkCost observed_cost;
  int64_t observed_maximum = 0;
  int64_t observed_preferred = 0;

  static ml_cpu_compute::ExecutionParallelPlan Plan(void *context, int64_t,
                                                    const ml_cpu_compute::ExecutionWorkCost &cost,
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
  EXPECT_EQ(ml_cpu_compute::CurrentExecutionExecutor(), nullptr);
  EXPECT_EQ(ml_cpu_compute::ExecutionThreadCount(), 1);
  EXPECT_EQ(ml_cpu_compute::ExecutionBlockCount(1000000), 1);

  int calls = 0;
  ml_cpu_compute::ExecuteRanges(1000000, [&](int64_t begin, int64_t end) {
    EXPECT_TRUE(ml_cpu_compute::ExecutionInParallelRegion());
    EXPECT_EQ(begin, 0);
    EXPECT_EQ(end, 1000000);
    ++calls;
  });
  EXPECT_EQ(calls, 1);
}

TEST(Execution, InjectedExecutorOwnsParallelDispatch) {
  InlineExecutor executor;
  ml_cpu_compute::ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run};
  std::vector<std::atomic<int>> hits(200000);
  {
    ml_cpu_compute::ExecutionExecutorScope scope(&view);
    EXPECT_EQ(ml_cpu_compute::ExecutionThreadCount(), 4);
    ml_cpu_compute::ExecuteRanges(
        static_cast<int64_t>(hits.size()), 1.0, 16, [&hits](int64_t begin, int64_t end) {
          EXPECT_TRUE(ml_cpu_compute::ExecutionInParallelRegion());
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
  EXPECT_EQ(ml_cpu_compute::CurrentExecutionExecutor(), nullptr);
}

TEST(Execution, SmallInjectedWorkRemainsInline) {
  InlineExecutor executor;
  ml_cpu_compute::ExecutionExecutorView view{&executor, 8, &InlineExecutor::Run};
  int calls = 0;
  {
    ml_cpu_compute::ExecutionExecutorScope scope(&view);
    ml_cpu_compute::ExecuteRanges(8, [&calls](int64_t begin, int64_t end) {
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
  ml_cpu_compute::ExecutionExecutorView outer{&executor, 4, &InlineExecutor::Run};
  ml_cpu_compute::ExecutionExecutorView inner{&executor, 0, &InlineExecutor::Run};
  {
    ml_cpu_compute::ExecutionExecutorScope outer_scope(&outer);
    EXPECT_EQ(ml_cpu_compute::CurrentExecutionExecutor(), &outer);
    {
      ml_cpu_compute::ExecutionExecutorScope inner_scope(&inner);
      EXPECT_EQ(ml_cpu_compute::CurrentExecutionExecutor(), &inner);
      EXPECT_EQ(ml_cpu_compute::ExecutionThreadCount(), 1);
      EXPECT_EQ(ml_cpu_compute::ExecutionBlockCount(1000000), 1);
    }
    EXPECT_EQ(ml_cpu_compute::CurrentExecutionExecutor(), &outer);
    EXPECT_EQ(ml_cpu_compute::ExecutionThreadCount(), 4);
  }
  EXPECT_EQ(ml_cpu_compute::CurrentExecutionExecutor(), nullptr);
}

TEST(Execution, DispatchFailurePropagatesAndRestoresScope) {
  ml_cpu_compute::ExecutionExecutorView view{
      nullptr, 4, [](void *, int64_t, void *, ml_cpu_compute::ExecutionBlockFn) {
        throw std::runtime_error("dispatch failed");
      }};
  int calls = 0;
  const auto run = [&]() {
    ml_cpu_compute::ExecutionExecutorScope scope(&view);
    ml_cpu_compute::ExecuteRanges(1000000, [&](int64_t, int64_t) { ++calls; });
  };
  EXPECT_THROW(run(), std::runtime_error);
  EXPECT_EQ(calls, 0);
  EXPECT_EQ(ml_cpu_compute::CurrentExecutionExecutor(), nullptr);
  EXPECT_FALSE(ml_cpu_compute::ExecutionInParallelRegion());
}

TEST(Execution, NestedRangesDoNotDispatchAgain) {
  InlineExecutor executor;
  ml_cpu_compute::ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run};
  {
    ml_cpu_compute::ExecutionExecutorScope scope(&view);
    ml_cpu_compute::ExecuteRanges(4, static_cast<double>(ml_cpu_compute::kExecutionGrainSize),
                                  [](int64_t begin, int64_t end) {
                                    EXPECT_EQ(ml_cpu_compute::ExecutionThreadCount(), 4);
                                    EXPECT_EQ(ml_cpu_compute::ExecutionBlockCount(1000000), 1);
                                    for (int64_t outer = begin; outer < end; ++outer) {
                                      (void)outer;
                                      int calls = 0;
                                      ml_cpu_compute::ExecuteRanges(
                                          1000000,
                                          [&calls](int64_t inner_begin, int64_t inner_end) {
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
  ml_cpu_compute::ExecutionExecutorView view{&executor, 8, &CostExecutor::Run, &CostExecutor::Plan};
  std::vector<std::atomic<int>> hits(1000);
  {
    ml_cpu_compute::ExecutionExecutorScope scope(&view);
    ml_cpu_compute::ExecuteCostedRanges(
        static_cast<int64_t>(hits.size()), ml_cpu_compute::ExecutionWorkCost{4.0, 4.0, 3.0},
        ml_cpu_compute::ExecutionSchedule{1, 1, 5, 3}, 16, [&hits](int64_t begin, int64_t end) {
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

TEST(Execution, NullScopeRestoresOuterExecutorAfterUnwinding) {
  InlineExecutor executor;
  ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run};
  ExecutionExecutorScope outer(&view);
  const auto run = [&]() {
    ExecutionExecutorScope serial(nullptr);
    EXPECT_EQ(CurrentExecutionExecutor(), nullptr);
    EXPECT_EQ(ExecutionThreadCount(), 1);
    ExecuteRanges(1000000, [](int64_t begin, int64_t end) {
      EXPECT_EQ(begin, 0);
      EXPECT_EQ(end, 1000000);
    });
    throw std::runtime_error("caller failed");
  };
  EXPECT_THROW(run(), std::runtime_error);
  EXPECT_EQ(CurrentExecutionExecutor(), &view);
  EXPECT_EQ(ExecutionThreadCount(), 4);
  EXPECT_EQ(executor.dispatches, 0);
  EXPECT_FALSE(ExecutionInParallelRegion());
}

TEST(Execution, BindingsAndNestedRegionsAreThreadLocal) {
  InlineExecutor executor;
  ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run};
  ExecutionExecutorScope outer(&view);
  ExecuteRanges(1, [&](int64_t, int64_t) {
    EXPECT_TRUE(ExecutionInParallelRegion());
    std::thread worker([] {
      EXPECT_EQ(CurrentExecutionExecutor(), nullptr);
      EXPECT_EQ(ExecutionThreadCount(), 1);
      EXPECT_FALSE(ExecutionInParallelRegion());
      InlineExecutor local_executor;
      ExecutionExecutorView local_view{&local_executor, 2, &InlineExecutor::Run};
      {
        ExecutionExecutorScope local(&local_view);
        ExecuteRanges(1000000, [](int64_t, int64_t) {
          EXPECT_EQ(ExecutionThreadCount(), 2);
          EXPECT_TRUE(ExecutionInParallelRegion());
        });
      }
      EXPECT_EQ(local_executor.dispatches, 1);
      EXPECT_EQ(CurrentExecutionExecutor(), nullptr);
      EXPECT_FALSE(ExecutionInParallelRegion());
    });
    worker.join();
    EXPECT_EQ(CurrentExecutionExecutor(), &view);
    EXPECT_TRUE(ExecutionInParallelRegion());
  });
  EXPECT_FALSE(ExecutionInParallelRegion());
  EXPECT_EQ(executor.dispatches, 0);
}

TEST(Execution, EmptyRangesNeitherPlanNorDispatch) {
  InlineExecutor executor;
  ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run,
                             [](void *, int64_t, const ExecutionWorkCost &, int64_t, int64_t) {
                               ADD_FAILURE() << "Empty work must not call the planner.";
                               return ExecutionParallelPlan{1, 4};
                             }};
  ExecutionExecutorScope scope(&view);
  const ml_cpu_compute::ExecutionSchedule schedule{1, 1, 4};
  const auto callback = [](int64_t, int64_t) { ADD_FAILURE() << "Unexpected empty callback."; };
  for (int64_t total : {int64_t{0}, int64_t{-1}}) {
    ExecuteRanges(total, callback);
    ExecuteRanges(total, schedule, 8, callback);
    ml_cpu_compute::ExecuteCostedRanges(total, {}, schedule, 8, callback);
  }
  EXPECT_EQ(executor.dispatches, 0);
  EXPECT_FALSE(ExecutionInParallelRegion());
}

TEST(Execution, CostedFallbackCoversAlignedBlocksAndPartialTail) {
  InlineExecutor executor;
  ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run};
  ExecutionExecutorScope scope(&view);
  const ml_cpu_compute::ExecutionSchedule schedule{1, 1, 4};
  std::vector<int> hits(37);
  ml_cpu_compute::ExecuteCostedRanges(
      static_cast<int64_t>(hits.size()), {}, schedule, 8, [&](int64_t begin, int64_t end) {
        EXPECT_EQ(begin % 8, 0);
        EXPECT_LE(end, static_cast<int64_t>(hits.size()));
        ml_cpu_compute::ExecuteCostedRanges(end - begin, {}, schedule, 8,
                                            [&](int64_t inner_begin, int64_t inner_end) {
                                              EXPECT_EQ(inner_begin, 0);
                                              EXPECT_EQ(inner_end, end - begin);
                                              for (int64_t i = begin; i < end; ++i) {
                                                ++hits[static_cast<std::size_t>(i)];
                                              }
                                            });
      });
  EXPECT_EQ(executor.dispatches, 1);
  EXPECT_GT(executor.blocks, 1);
  EXPECT_LE(executor.blocks, 4);
  for (int hit : hits) {
    EXPECT_EQ(hit, 1);
  }
  EXPECT_FALSE(ExecutionInParallelRegion());
}

TEST(Execution, PlannerFailurePropagatesBeforeDispatch) {
  InlineExecutor executor;
  ExecutionExecutorView view{
      &executor, 4, &InlineExecutor::Run,
      [](void *, int64_t, const ExecutionWorkCost &, int64_t, int64_t) -> ExecutionParallelPlan {
        throw std::runtime_error("planner failed");
      }};
  const auto run = [&]() {
    ExecutionExecutorScope scope(&view);
    ml_cpu_compute::ExecuteCostedRanges(
        1000, {}, ml_cpu_compute::ExecutionSchedule{1, 1, 4}, 8,
        [](int64_t, int64_t) { ADD_FAILURE() << "Planning failed before this callback."; });
  };
  EXPECT_THROW(run(), std::runtime_error);
  EXPECT_EQ(executor.dispatches, 0);
  EXPECT_EQ(CurrentExecutionExecutor(), nullptr);
  EXPECT_FALSE(ExecutionInParallelRegion());
}

TEST(Execution, NumericalRoutineReacquiresExecutorForEachCall) {
  InlineExecutor first;
  InlineExecutor second;
  ExecutionExecutorView first_view{&first, 4, &InlineExecutor::Run};
  ExecutionExecutorView second_view{&second, 2, &InlineExecutor::Run};
  constexpr std::size_t count = 200003;
  std::vector<float> values(count + 2, -3.0f);
  const auto run = [&]() {
    ml_cpu_compute::AbsFloat32WithTuning(values.data() + 1, values.data() + 1, count,
                                         ml_cpu_compute::UnaryExecutionTuning{1, 64, 4, false});
  };
  {
    ExecutionExecutorScope scope(&first_view);
    run();
  }
  run();
  EXPECT_EQ(first.dispatches, 1);
  {
    ExecutionExecutorScope scope(&second_view);
    run();
  }
  EXPECT_EQ(first.dispatches, 1);
  EXPECT_EQ(second.dispatches, 1);
  EXPECT_EQ(values.front(), -3.0f);
  EXPECT_EQ(values.back(), -3.0f);
  for (std::size_t i = 1; i <= count; ++i) {
    ASSERT_EQ(values[i], 3.0f) << i;
  }
  EXPECT_EQ(CurrentExecutionExecutor(), nullptr);
}

} // namespace
