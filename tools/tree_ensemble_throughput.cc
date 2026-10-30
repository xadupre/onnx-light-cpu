// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "ml_cpu_compute/execution.h"
#include "ml_cpu_compute/thread_topology.h"
#include "ml_cpu_compute/traditionalml/tree_ensemble.h"

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

using ml_cpu_compute::ExecutionBlockFn;
using ml_cpu_compute::ExecutionExecutorScope;
using ml_cpu_compute::ExecutionExecutorView;
using ml_cpu_compute::TreeBranchMode;
using ml_cpu_compute::TreeEnsembleAttributes;
using ml_cpu_compute::TreeEnsembleExecutionStrategy;
using ml_cpu_compute::TreeEnsemblePlan;

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
                  ml_cpu_compute::ExecutionBlockFn task) {
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
  ml_cpu_compute::ExecutionBlockFn task = nullptr;
  bool stopping = false;
};

ml_cpu_compute::TreeEnsembleAttributes MakeForest(std::size_t trees, std::size_t features = 1,
                                                  std::size_t depth = 1) {
  ml_cpu_compute::TreeEnsembleAttributes attributes;
  attributes.n_features = static_cast<std::int64_t>(features);
  attributes.n_targets = 1;
  attributes.value_type = ml_cpu_compute::DataType::FLOAT;
  attributes.base_values = {0.5};
  const std::size_t internal_nodes = (1U << depth) - 1;
  const std::size_t leaves = 1U << depth;
  for (std::size_t tree = 0; tree < trees; ++tree) {
    attributes.tree_roots.push_back(static_cast<std::int64_t>(tree * internal_nodes));
    for (std::size_t node = 0; node < internal_nodes; ++node) {
      attributes.nodes_featureids.push_back(static_cast<std::int64_t>((tree + node) % features));
      attributes.nodes_splits.push_back(static_cast<double>(static_cast<int>(node % 7) - 3) * 0.25);
      attributes.nodes_modes.push_back(ml_cpu_compute::TreeBranchMode::kLeq);
      for (bool true_branch : {true, false}) {
        const std::size_t child = 2 * node + (true_branch ? 1 : 2);
        const bool leaf = child >= internal_nodes;
        (true_branch ? attributes.nodes_truenodeids : attributes.nodes_falsenodeids)
            .push_back(static_cast<std::int64_t>(leaf ? tree * leaves + child - internal_nodes
                                                      : tree * internal_nodes + child));
        (true_branch ? attributes.nodes_trueleafs : attributes.nodes_falseleafs)
            .push_back(leaf ? 1 : 0);
      }
    }
    for (std::size_t leaf = 0; leaf < leaves; ++leaf) {
      attributes.leaf_targetids.push_back(0);
      attributes.leaf_weights.push_back(
          static_cast<double>(static_cast<int>((leaf + tree) % 11) - 5) /
          (static_cast<double>(trees) * 8.0));
    }
  }
  return attributes;
}

std::string_view StrategyName(ml_cpu_compute::TreeEnsembleExecutionStrategy strategy) {
  switch (strategy) {
  case ml_cpu_compute::TreeEnsembleExecutionStrategy::kRowParallel:
    return "row_parallel";
  case ml_cpu_compute::TreeEnsembleExecutionStrategy::kTreeParallel:
    return "tree_parallel";
  case ml_cpu_compute::TreeEnsembleExecutionStrategy::kTreeMajorBatch:
    return "tree_major_batch";
  case ml_cpu_compute::TreeEnsembleExecutionStrategy::kInterleavedRows:
    return "interleaved_rows";
  }
  return "unknown";
}

void Measure(std::string_view scenario, std::string_view configuration, std::size_t trees,
             std::size_t rows, std::size_t threads, std::size_t features = 1,
             std::size_t depth = 1) {
  const ml_cpu_compute::TreeEnsemblePlan plan(MakeForest(trees, features, depth));
  std::vector<float> input(rows * features);
  std::vector<float> output(rows);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<float>(static_cast<int>(index % 17) - 8) * 0.25F;
  }
  ThreadExecutor executor(threads);
  ml_cpu_compute::ExecutionExecutorView view{&executor, static_cast<std::int64_t>(threads),
                                             &ThreadExecutor::Run};
  ml_cpu_compute::ExecutionExecutorScope scope(&view);
  const auto decision = plan.SelectExecution(rows);
  for (int iteration = 0; iteration < 3; ++iteration) {
    plan.EvaluateInto(input.data(), input.size(), rows, output.data());
  }
  std::vector<double> samples;
  samples.reserve(11);
  double checksum = 0.0;
  for (int iteration = 0; iteration < 11; ++iteration) {
    const auto begin = std::chrono::steady_clock::now();
    plan.EvaluateInto(input.data(), input.size(), rows, output.data());
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
      std::max<std::size_t>(ml_cpu_compute::GetCpuTopology().physical_core_count, 1);
  const std::vector<std::pair<std::string_view, std::size_t>> configurations{
      {"1", 1}, {"2", 2}, {"4", 4}, {"physical", physical_threads}};
  std::cout << "scenario,configuration,trees,rows,threads,strategy,workspace_bytes,median_us,"
               "rows_per_s,checksum\n";
  for (const auto &[configuration, threads] : configurations) {
    Measure("single_row_large_forest", configuration, 1024, 1, threads);
    Measure("large_batch_large_forest", configuration, 81, 4096, threads);
    Measure("large_batch_small_forest", configuration, 3, 4096, threads);
    Measure("balanced_medium", configuration, 1000, 32, threads, 1024, 4);
    Measure("balanced_large_single_row", configuration, 10000, 1, threads, 4096, 4);
    Measure("balanced_large_batch", configuration, 10000, 128, threads, 4096, 4);
    Measure("balanced_large_batch_tail", configuration, 10000, 129, threads, 4097, 4);
  }
  return 0;
}
