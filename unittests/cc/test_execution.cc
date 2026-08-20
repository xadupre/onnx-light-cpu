// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/execution.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <vector>

namespace {

using onnx_light_cpu::CurrentExecutionExecutor;
using onnx_light_cpu::ExecuteRanges;
using onnx_light_cpu::ExecutionBlockCount;
using onnx_light_cpu::ExecutionBlockFn;
using onnx_light_cpu::ExecutionExecutorScope;
using onnx_light_cpu::ExecutionExecutorView;
using onnx_light_cpu::ExecutionInParallelRegion;
using onnx_light_cpu::ExecutionThreadCount;

struct InlineExecutor {
  int64_t dispatches = 0;
  int64_t blocks = 0;

  static void Run(void *context, int64_t num_blocks, void *task_context, ExecutionBlockFn task) {
    auto &self = *static_cast<InlineExecutor *>(context);
    ++self.dispatches;
    self.blocks = num_blocks;
    for (int64_t block = num_blocks; block > 0; --block) {
      task(task_context, block - 1);
    }
  }
};

TEST(Execution, StandaloneIsSerial) {
  EXPECT_EQ(CurrentExecutionExecutor(), nullptr);
  EXPECT_EQ(ExecutionThreadCount(), 1);
  EXPECT_EQ(ExecutionBlockCount(1000000), 1);

  int calls = 0;
  ExecuteRanges(1000000, [&](int64_t begin, int64_t end) {
    EXPECT_TRUE(ExecutionInParallelRegion());
    EXPECT_EQ(begin, 0);
    EXPECT_EQ(end, 1000000);
    ++calls;
  });
  EXPECT_EQ(calls, 1);
}

TEST(Execution, InjectedExecutorOwnsParallelDispatch) {
  InlineExecutor executor;
  ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run};
  std::vector<std::atomic<int>> hits(200000);
  {
    ExecutionExecutorScope scope(&view);
    EXPECT_EQ(ExecutionThreadCount(), 4);
    ExecuteRanges(static_cast<int64_t>(hits.size()), 1.0, 16, [&hits](int64_t begin, int64_t end) {
      EXPECT_TRUE(ExecutionInParallelRegion());
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
  EXPECT_EQ(CurrentExecutionExecutor(), nullptr);
}

TEST(Execution, SmallInjectedWorkRemainsInline) {
  InlineExecutor executor;
  ExecutionExecutorView view{&executor, 8, &InlineExecutor::Run};
  int calls = 0;
  {
    ExecutionExecutorScope scope(&view);
    ExecuteRanges(8, [&calls](int64_t begin, int64_t end) {
      EXPECT_EQ(begin, 0);
      EXPECT_EQ(end, 8);
      ++calls;
    });
  }
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(executor.dispatches, 0);
}

TEST(Execution, NestedRangesDoNotDispatchAgain) {
  InlineExecutor executor;
  ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run};
  {
    ExecutionExecutorScope scope(&view);
    ExecuteRanges(4, static_cast<double>(onnx_light_cpu::kExecutionGrainSize),
                  [](int64_t begin, int64_t end) {
                    for (int64_t outer = begin; outer < end; ++outer) {
                      (void)outer;
                      int calls = 0;
                      ExecuteRanges(1000000, [&calls](int64_t inner_begin, int64_t inner_end) {
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

} // namespace
