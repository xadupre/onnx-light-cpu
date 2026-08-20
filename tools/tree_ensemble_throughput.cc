// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/thread_topology.h"
#include "onnx_light_cpu/reference/tree_ensemble_reference.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using onnx_light_cpu::ExecutionBlockFn;
using onnx_light_cpu::ExecutionExecutorScope;
using onnx_light_cpu::ExecutionExecutorView;
using onnx_light_cpu::reference::TreeBranchMode;
using onnx_light_cpu::reference::TreeEnsembleAttributes;
using onnx_light_cpu::reference::TreeEnsembleExecutionStrategy;
using onnx_light_cpu::reference::TreeEnsemblePlan;

struct ThreadExecutor {
  explicit ThreadExecutor(std::size_t threads) {
    workers.reserve(threads);
    for (std::size_t index = 0; index < threads; ++index) {
      workers.emplace_back([this] { Worker(); });
    }
  }

  ~ThreadExecutor() {
    {
      std::lock_guard lock(mutex);
      stopping = true;
      ++generation;
    }
    start.notify_all();
    for (std::thread &worker : workers) {
      worker.join();
    }
  }

  static void Run(void *context, std::int64_t num_blocks, void *task_context,
                  ExecutionBlockFn task) {
    auto &self = *static_cast<ThreadExecutor *>(context);
    {
      std::lock_guard lock(self.mutex);
      self.next.store(0, std::memory_order_relaxed);
      self.pending = num_blocks;
      self.num_blocks = num_blocks;
      self.task_context = task_context;
      self.task = task;
      ++self.generation;
    }
    self.start.notify_all();
    std::unique_lock lock(self.mutex);
    self.done.wait(lock, [&] { return self.pending == 0; });
  }

  void Worker() {
    std::size_t observed_generation = 0;
    for (;;) {
      {
        std::unique_lock lock(mutex);
        start.wait(lock, [&] { return stopping || generation != observed_generation; });
        if (stopping) {
          return;
        }
        observed_generation = generation;
      }
      for (;;) {
        const std::int64_t block = next.fetch_add(1, std::memory_order_relaxed);
        if (block >= num_blocks) {
          break;
        }
        task(task_context, block);
        std::lock_guard lock(mutex);
        if (--pending == 0) {
          done.notify_one();
        }
      }
    }
  }

  std::vector<std::thread> workers;
  std::mutex mutex;
  std::condition_variable start;
  std::condition_variable done;
  std::atomic<std::int64_t> next{0};
  std::size_t generation = 0;
  std::int64_t pending = 0;
  std::int64_t num_blocks = 0;
  void *task_context = nullptr;
  ExecutionBlockFn task = nullptr;
  bool stopping = false;
};

TreeEnsembleAttributes MakeForest(std::size_t trees) {
  TreeEnsembleAttributes attributes;
  attributes.n_features = 1;
  attributes.n_targets = 1;
  attributes.base_values = {0.5};
  for (std::size_t tree = 0; tree < trees; ++tree) {
    const std::int64_t leaf = static_cast<std::int64_t>(2 * tree);
    attributes.tree_roots.push_back(static_cast<std::int64_t>(tree));
    attributes.nodes_featureids.push_back(0);
    attributes.nodes_splits.push_back(0.0);
    attributes.nodes_modes.push_back(TreeBranchMode::kLeq);
    attributes.nodes_truenodeids.push_back(leaf);
    attributes.nodes_falsenodeids.push_back(leaf + 1);
    attributes.nodes_trueleafs.push_back(1);
    attributes.nodes_falseleafs.push_back(1);
    attributes.leaf_targetids.push_back(0);
    attributes.leaf_targetids.push_back(0);
    attributes.leaf_weights.push_back(0.25);
    attributes.leaf_weights.push_back(-0.25);
  }
  return attributes;
}

std::string_view StrategyName(TreeEnsembleExecutionStrategy strategy) {
  switch (strategy) {
  case TreeEnsembleExecutionStrategy::kRowParallel:
    return "row_parallel";
  case TreeEnsembleExecutionStrategy::kTreeParallel:
    return "tree_parallel";
  case TreeEnsembleExecutionStrategy::kTreeMajorBatch:
    return "tree_major_batch";
  }
  return "unknown";
}

void Measure(std::string_view scenario, std::string_view configuration, std::size_t trees,
             std::size_t rows, std::size_t threads) {
  const TreeEnsemblePlan plan(MakeForest(trees));
  std::vector<double> input(rows);
  for (std::size_t row = 0; row < rows; ++row) {
    input[row] = (row % 2 == 0) ? -1.0 : 1.0;
  }
  ThreadExecutor executor(threads);
  ExecutionExecutorView view{&executor, static_cast<std::int64_t>(threads), &ThreadExecutor::Run};
  ExecutionExecutorScope scope(&view);
  const auto decision = plan.SelectExecution(rows);
  for (int iteration = 0; iteration < 3; ++iteration) {
    (void)plan.Evaluate(input, rows);
  }
  std::vector<double> samples;
  samples.reserve(11);
  double checksum = 0.0;
  for (int iteration = 0; iteration < 11; ++iteration) {
    const auto begin = std::chrono::steady_clock::now();
    const std::vector<double> output = plan.Evaluate(input, rows);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
    checksum += output[static_cast<std::size_t>(iteration) % output.size()];
  }
  std::sort(samples.begin(), samples.end());
  const double median_us = samples[samples.size() / 2];
  std::cout << scenario << "," << configuration << "," << trees << "," << rows << "," << threads
            << "," << StrategyName(decision.strategy) << "," << decision.workspace_bytes << ","
            << std::fixed << std::setprecision(2) << median_us << ","
            << static_cast<double>(rows) * 1.0e6 / median_us << "," << checksum << "\n";
}

} // namespace

int main() {
  const std::size_t physical_threads =
      std::max<std::size_t>(onnx_light_cpu::GetCpuTopology().physical_core_count, 1);
  const std::vector<std::pair<std::string_view, std::size_t>> configurations{
      {"1", 1}, {"2", 2}, {"4", 4}, {"physical", physical_threads}};
  std::cout << "scenario,configuration,trees,rows,threads,strategy,workspace_bytes,median_us,"
               "rows_per_s,checksum\n";
  for (const auto &[configuration, threads] : configurations) {
    Measure("single_row_large_forest", configuration, 1024, 1, threads);
    Measure("large_batch_large_forest", configuration, 81, 4096, threads);
    Measure("large_batch_small_forest", configuration, 3, 4096, threads);
  }
  return 0;
}
