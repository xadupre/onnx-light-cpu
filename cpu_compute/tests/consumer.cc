// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "cpu_compute/execution.h"
#include "cpu_compute/math/math.h"

#include <stdexcept>
#include <vector>

namespace {

struct Executor {
  int dispatches = 0;

  static void Run(void *context, int64_t blocks, void *task_context,
                  cpu_compute::ExecutionBlockFn task) {
    ++static_cast<Executor *>(context)->dispatches;
    for (int64_t block = blocks; block > 0; --block) {
      task(task_context, block - 1);
    }
  }
};

} // namespace

int main() {
  using namespace cpu_compute;
  const float input[] = {-1.0f, 2.0f, -3.0f, 4.0f};
  float output[4];
  AbsFloat32(input, output, 4);
  for (int i = 0; i < 4; ++i) {
    if (output[i] != static_cast<float>(i + 1)) {
      throw std::runtime_error("Standalone Abs produced an incorrect result.");
    }
  }
  if (CurrentExecutionExecutor() != nullptr || ExecutionThreadCount() != 1) {
    throw std::runtime_error("Standalone execution must default to serial.");
  }
  Executor executor;
  ExecutionExecutorView view{&executor, 4, &Executor::Run};
  std::vector<int> hits(200000);
  {
    ExecutionExecutorScope scope(&view);
    ExecuteRanges(static_cast<int64_t>(hits.size()), 1.0, 4, [&](int64_t begin, int64_t end) {
      ExecuteRanges(end - begin, [&](int64_t nested_begin, int64_t nested_end) {
        for (int64_t i = begin + nested_begin; i < begin + nested_end; ++i) {
          ++hits[static_cast<std::size_t>(i)];
        }
      });
    });
  }
  if (executor.dispatches != 1 || CurrentExecutionExecutor() != nullptr) {
    throw std::runtime_error("Borrowed executor dispatch or scope restoration failed.");
  }
  for (int hit : hits) {
    if (hit != 1) {
      throw std::runtime_error("Standalone executor did not cover every element exactly once.");
    }
  }
}
