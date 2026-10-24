// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "ml_cpu_compute/traditionalml/tree_ensemble.h"
#include "onnx_light_cpu/backend_test/cases/traditionalml/tree_ensemble_corpus.h"

#include "ml_cpu_compute/execution.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using ml_cpu_compute::ClassLabels;
using ml_cpu_compute::DataType;
using ml_cpu_compute::LegacyTreeAttributes;
using ml_cpu_compute::SelectTreeEnsembleCacheBlocking;
using ml_cpu_compute::TreeAggregate;
using ml_cpu_compute::TreeBranchMode;
using ml_cpu_compute::TreeEnsembleAttributes;
using ml_cpu_compute::TreeEnsembleCacheBlocking;
using ml_cpu_compute::TreeEnsembleClassifierAttributes;
using ml_cpu_compute::TreeEnsembleExecutionStrategy;
using ml_cpu_compute::TreeEnsembleNodeLayout;
using ml_cpu_compute::TreeEnsembleOracle;
using ml_cpu_compute::TreeEnsemblePlan;
using ml_cpu_compute::TreeEnsembleRegressorAttributes;
using ml_cpu_compute::TreeEnsembleTraversal;
using ml_cpu_compute::TreeEnsembleTuningPolicy;
using ml_cpu_compute::TreePostTransform;
using onnx_light_cpu::backend_test::GenerateTreeEnsembleV5Corpus;

ml_cpu_compute::TreeEnsembleAttributes Stump() {
  ml_cpu_compute::TreeEnsembleAttributes attributes;
  attributes.n_features = 1;
  attributes.n_targets = 1;
  attributes.tree_roots = {0};
  attributes.nodes_featureids = {0};
  attributes.nodes_splits = {0.0};
  attributes.nodes_modes = {ml_cpu_compute::TreeBranchMode::kLeq};
  attributes.nodes_truenodeids = {0};
  attributes.nodes_falsenodeids = {1};
  attributes.nodes_trueleafs = {1};
  attributes.nodes_falseleafs = {1};
  attributes.leaf_targetids = {0, 0};
  attributes.leaf_weights = {1.0, -1.0};
  return attributes;
}

ml_cpu_compute::TreeEnsembleAttributes StumpForest(std::size_t trees, std::size_t targets = 2) {
  ml_cpu_compute::TreeEnsembleAttributes attributes;
  attributes.n_features = 1;
  attributes.n_targets = static_cast<std::int64_t>(targets);
  attributes.value_type = DataType::FLOAT;
  attributes.base_values.resize(targets, 0.5);
  for (std::size_t tree = 0; tree < trees; ++tree) {
    const std::int64_t node = static_cast<std::int64_t>(tree);
    const std::int64_t leaf = static_cast<std::int64_t>(2 * tree);
    attributes.tree_roots.push_back(node);
    attributes.nodes_featureids.push_back(0);
    attributes.nodes_splits.push_back(0.0);
    attributes.nodes_modes.push_back(ml_cpu_compute::TreeBranchMode::kLeq);
    attributes.nodes_truenodeids.push_back(leaf);
    attributes.nodes_falsenodeids.push_back(leaf + 1);
    attributes.nodes_trueleafs.push_back(1);
    attributes.nodes_falseleafs.push_back(1);
    attributes.leaf_targetids.push_back(static_cast<std::int64_t>(tree % targets));
    attributes.leaf_targetids.push_back(static_cast<std::int64_t>(tree % targets));
    attributes.leaf_weights.push_back(0.25);
    attributes.leaf_weights.push_back(-0.25);
  }
  return attributes;
}

ml_cpu_compute::TreeEnsembleAttributes SymmetricTree() {
  ml_cpu_compute::TreeEnsembleAttributes attributes;
  attributes.n_features = 2;
  attributes.n_targets = 1;
  attributes.tree_roots = {0};
  attributes.nodes_featureids = {0, 1, 1};
  attributes.nodes_splits = {0.0, 0.0, 0.0};
  attributes.nodes_modes = {ml_cpu_compute::TreeBranchMode::kLeq,
                            ml_cpu_compute::TreeBranchMode::kLeq,
                            ml_cpu_compute::TreeBranchMode::kLeq};
  attributes.nodes_truenodeids = {1, 0, 2};
  attributes.nodes_falsenodeids = {2, 1, 3};
  attributes.nodes_trueleafs = {0, 1, 1};
  attributes.nodes_falseleafs = {0, 1, 1};
  attributes.leaf_targetids = {0, 0, 0, 0};
  attributes.leaf_weights = {1.0, 2.0, 3.0, 4.0};
  return attributes;
}

ml_cpu_compute::TreeEnsembleAttributes DeepTree(std::size_t depth) {
  ml_cpu_compute::TreeEnsembleAttributes attributes;
  attributes.n_features = 1;
  attributes.n_targets = 1;
  attributes.tree_roots = {0};
  attributes.nodes_featureids.resize(depth, 0);
  attributes.nodes_splits.resize(depth, 0.0);
  attributes.nodes_modes.resize(depth, ml_cpu_compute::TreeBranchMode::kLeq);
  attributes.nodes_truenodeids.resize(depth);
  attributes.nodes_falsenodeids.resize(depth, 1);
  attributes.nodes_trueleafs.resize(depth, 0);
  attributes.nodes_falseleafs.resize(depth, 1);
  attributes.nodes_hitrates.resize(depth, 1.0);
  for (std::size_t node = 0; node + 1 < depth; ++node) {
    attributes.nodes_truenodeids[node] = static_cast<std::int64_t>(node + 1);
  }
  attributes.nodes_truenodeids.back() = 0;
  attributes.nodes_trueleafs.back() = 1;
  attributes.leaf_targetids = {0, 0};
  attributes.leaf_weights = {1.0, -1.0};
  return attributes;
}

ml_cpu_compute::TreeEnsembleAttributes BalancedForest(std::size_t trees) {
  constexpr std::size_t kDepth = 4;
  constexpr std::size_t kInternalNodes = (1U << kDepth) - 1;
  constexpr std::size_t kLeaves = 1U << kDepth;
  ml_cpu_compute::TreeEnsembleAttributes attributes;
  attributes.n_features = 8;
  attributes.n_targets = 1;
  attributes.value_type = DataType::FLOAT;
  attributes.base_values = {0.1};
  for (std::size_t tree = 0; tree < trees; ++tree) {
    const std::size_t node_offset = tree * kInternalNodes;
    const std::size_t leaf_offset = tree * kLeaves;
    attributes.tree_roots.push_back(static_cast<std::int64_t>(node_offset));
    for (std::size_t node = 0; node < kInternalNodes; ++node) {
      attributes.nodes_featureids.push_back(static_cast<std::int64_t>((tree + node) % 8));
      attributes.nodes_splits.push_back(static_cast<double>(static_cast<int>(node % 3) - 1) * 0.5);
      attributes.nodes_modes.push_back(ml_cpu_compute::TreeBranchMode::kLeq);
      for (const bool true_branch : {true, false}) {
        const std::size_t child = 2 * node + (true_branch ? 1 : 2);
        const bool leaf = child >= kInternalNodes;
        const std::int64_t child_id = static_cast<std::int64_t>(
            leaf ? leaf_offset + child - kInternalNodes : node_offset + child);
        (true_branch ? attributes.nodes_truenodeids : attributes.nodes_falsenodeids)
            .push_back(child_id);
        (true_branch ? attributes.nodes_trueleafs : attributes.nodes_falseleafs)
            .push_back(leaf ? 1 : 0);
      }
    }
    for (std::size_t leaf = 0; leaf < kLeaves; ++leaf) {
      attributes.leaf_targetids.push_back(0);
      attributes.leaf_weights.push_back(
          static_cast<double>(static_cast<int>((leaf + tree) % 9) - 4) / 32.0);
    }
  }
  return attributes;
}

ml_cpu_compute::TreeEnsembleTuningPolicy
OneRegionPolicy(ml_cpu_compute::TreeEnsembleExecutionStrategy strategy, std::size_t threads,
                std::size_t targets, std::size_t batch_rows = 1) {
  ml_cpu_compute::TreeEnsembleTuningPolicy policy;
  policy.regions.push_back(
      {std::nullopt, strategy, batch_rows, threads, 1, 1,
       threads * batch_rows * targets * (sizeof(double) + sizeof(std::size_t))});
  return policy;
}

struct ThreadedExecutor {
  std::atomic<std::size_t> dispatches{0};
  std::atomic<std::size_t> dispatched_blocks{0};
  std::atomic<std::size_t> maximum_active{0};

  static void Run(void *context, int64_t num_blocks, void *task_context,
                  ml_cpu_compute::ExecutionBlockFn task) {
    auto &self = *static_cast<ThreadedExecutor *>(context);
    self.dispatches.fetch_add(1, std::memory_order_relaxed);
    self.dispatched_blocks.store(static_cast<std::size_t>(num_blocks), std::memory_order_relaxed);
    std::atomic<std::size_t> active{0};
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(num_blocks));
    for (int64_t block = num_blocks; block > 0; --block) {
      workers.emplace_back([&, index = block - 1] {
        const std::size_t now = active.fetch_add(1, std::memory_order_relaxed) + 1;
        std::size_t maximum = self.maximum_active.load(std::memory_order_relaxed);
        while (now > maximum && !self.maximum_active.compare_exchange_weak(
                                    maximum, now, std::memory_order_relaxed)) {
        }
        task(task_context, index);
        active.fetch_sub(1, std::memory_order_relaxed);
      });
    }
    for (std::thread &worker : workers) {
      worker.join();
    }
  }
};

ml_cpu_compute::LegacyTreeAttributes LegacyStump() {
  ml_cpu_compute::LegacyTreeAttributes tree;
  tree.n_features = 1;
  tree.nodes_treeids = {0, 0, 0};
  tree.nodes_nodeids = {0, 1, 2};
  tree.nodes_featureids = {0, 0, 0};
  tree.nodes_values = {0.0, 0.0, 0.0};
  tree.nodes_modes = {"BRANCH_LEQ", "LEAF", "LEAF"};
  tree.nodes_truenodeids = {1, 0, 0};
  tree.nodes_falsenodeids = {2, 0, 0};
  return tree;
}

TEST(TreeEnsembleOracle, CorpusCoversV5ContractAndIsDeterministic) {
  const auto cases = GenerateTreeEnsembleV5Corpus();
  ASSERT_GE(cases.size(), 36U);
  std::set<ml_cpu_compute::TreeBranchMode> modes;
  std::set<ml_cpu_compute::TreeAggregate> aggregates;
  std::set<ml_cpu_compute::TreePostTransform> transforms;
  std::set<DataType> types;
  bool has_empty = false;
  bool has_multi_target = false;
  bool has_large_membership = false;
  bool has_depth_extreme = false;
  for (const auto &test_case : cases) {
    EXPECT_EQ(test_case.domain, "ai.onnx.ml");
    EXPECT_EQ(test_case.op_type, "TreeEnsemble");
    EXPECT_EQ(test_case.opset, 5);
    EXPECT_EQ(ml_cpu_compute::TreeEnsembleOracle(test_case.attributes)
                  .Evaluate(test_case.input, test_case.rows),
              test_case.expected)
        << test_case.name;
    modes.insert(test_case.attributes.nodes_modes.begin(), test_case.attributes.nodes_modes.end());
    aggregates.insert(test_case.attributes.aggregate);
    transforms.insert(test_case.attributes.post_transform);
    types.insert(test_case.attributes.value_type);
    has_empty |= test_case.rows == 0;
    has_multi_target |= test_case.attributes.n_targets > 1;
    has_large_membership |= test_case.attributes.membership_values.size() > 1000;
    has_depth_extreme |=
        test_case.attributes.nodes_modes.size() >= 64 && test_case.attributes.n_features >= 4096;
  }
  EXPECT_EQ(modes.size(), 7U);
  EXPECT_EQ(aggregates.size(), 4U);
  EXPECT_EQ(transforms.size(), 5U);
  EXPECT_EQ(types.size(), 3U);
  EXPECT_TRUE(has_empty);
  EXPECT_TRUE(has_multi_target);
  EXPECT_TRUE(has_large_membership);
  EXPECT_TRUE(has_depth_extreme);
}

TEST(TreeEnsembleOracle, CanonicalPlanLowersAndEvaluatesDeterministically) {
  ml_cpu_compute::TreeEnsembleAttributes attributes = Stump();
  attributes.value_type = DataType::FLOAT;
  const ml_cpu_compute::TreeEnsemblePlan plan(attributes);
  EXPECT_EQ(plan.tree_roots().size(), 1U);
  EXPECT_EQ(plan.nodes().size(), 1U);
  EXPECT_EQ(plan.leaves().size(), 2U);
  EXPECT_GT(plan.workspace_bytes(), 0U);
  EXPECT_EQ(plan.Evaluate({-1.0, 3.0}, 2), (std::vector<double>{1.0, -1.0}));
}

TEST(TreeEnsembleOracle, RuntimeCompactionRetainsTypedEvaluationAndReleasesConstructionData) {
  ml_cpu_compute::TreeEnsemblePlan plan(StumpForest(1024, 1));
  const std::size_t prepared_bytes = plan.prepared_storage_bytes();
  const std::vector<float> input{-1.0F, 1.0F};
  std::vector<float> output(2);

  plan.CompactRuntimeStorage();
  plan.EvaluateInto(input.data(), input.size(), input.size(), output.data());

  EXPECT_EQ(output, (std::vector<float>{256.5F, -255.5F}));
  EXPECT_TRUE(plan.nodes().empty());
  EXPECT_LT(plan.prepared_storage_bytes(), prepared_bytes / 2);
}

TEST(TreeEnsembleOracle, CanonicalPlanAppliesBaseValues) {
  ml_cpu_compute::TreeEnsembleAttributes attributes = Stump();
  attributes.value_type = DataType::FLOAT;
  attributes.base_values = {0.5};
  const ml_cpu_compute::TreeEnsemblePlan plan(attributes);
  EXPECT_EQ(plan.base_values(), (std::vector<double>{0.5}));
  EXPECT_EQ(plan.Evaluate({-1.0, 3.0}, 2), (std::vector<double>{1.5, -0.5}));
  EXPECT_EQ(ml_cpu_compute::TreeEnsembleOracle(attributes).Evaluate({-1.0, 3.0}, 2),
            (std::vector<double>{1.5, -0.5}));
}

TEST(TreeEnsembleOracle, SchedulingDecisionUsesCacheCapacity) {
  const ml_cpu_compute::TreeEnsemblePlan small_forest(StumpForest(79));
  EXPECT_EQ(small_forest.SelectExecution(1, 4).strategy,
            ml_cpu_compute::TreeEnsembleExecutionStrategy::kTreeMajorBatch);

  const ml_cpu_compute::TreeEnsemblePlan large_forest(StumpForest(81));
  EXPECT_EQ(large_forest.cache_blocking().trees_per_l1_block, 81U);
  EXPECT_EQ(large_forest.SelectExecution(1, 4).strategy,
            ml_cpu_compute::TreeEnsembleExecutionStrategy::kTreeMajorBatch);
  EXPECT_EQ(large_forest.SelectExecution(50, 4).strategy,
            ml_cpu_compute::TreeEnsembleExecutionStrategy::kTreeMajorBatch);
  EXPECT_EQ(large_forest.SelectExecution(51, 4).strategy,
            ml_cpu_compute::TreeEnsembleExecutionStrategy::kRowParallel);

  const ml_cpu_compute::TreeEnsemblePlan very_large_forest(StumpForest(10000));
  EXPECT_EQ(very_large_forest.SelectExecution(51, 4).strategy,
            ml_cpu_compute::TreeEnsembleExecutionStrategy::kTreeParallel);
  EXPECT_EQ(very_large_forest.SelectExecution(127, 4).strategy,
            ml_cpu_compute::TreeEnsembleExecutionStrategy::kTreeParallel);
  EXPECT_EQ(very_large_forest.SelectExecution(128, 4).strategy,
            ml_cpu_compute::TreeEnsembleExecutionStrategy::kTreeParallel);
  EXPECT_EQ(very_large_forest.SelectExecution(129, 4).strategy,
            ml_cpu_compute::TreeEnsembleExecutionStrategy::kRowParallel);

  const ml_cpu_compute::TreeEnsemblePlan prepared_very_large_forest(
      StumpForest(10000), ml_cpu_compute::TreeEnsembleExecutionTuning{128, 50, 128, 4});
  EXPECT_EQ(prepared_very_large_forest.SelectExecution(127, 4).strategy,
            ml_cpu_compute::TreeEnsembleExecutionStrategy::kTreeParallel);
  EXPECT_EQ(prepared_very_large_forest.SelectExecution(128, 4).strategy,
            ml_cpu_compute::TreeEnsembleExecutionStrategy::kTreeParallel);
  EXPECT_EQ(prepared_very_large_forest.SelectExecution(129, 4).strategy,
            ml_cpu_compute::TreeEnsembleExecutionStrategy::kRowParallel);

  const ml_cpu_compute::TreeEnsemblePlan few_trees(StumpForest(3));
  EXPECT_EQ(few_trees.SelectExecution(51, 4).strategy,
            ml_cpu_compute::TreeEnsembleExecutionStrategy::kRowParallel);
  EXPECT_EQ(few_trees.SelectExecution(1000, 1).strategy,
            ml_cpu_compute::TreeEnsembleExecutionStrategy::kTreeMajorBatch);
  EXPECT_EQ(few_trees.SelectExecution(1000, 1).batch_rows, 128U);

  const ml_cpu_compute::TreeEnsemblePlan four_single_target_trees(StumpForest(4, 1));
  const ml_cpu_compute::TreeEnsemblePlan four_multi_target_trees(StumpForest(4, 2));
  EXPECT_EQ(four_single_target_trees.SelectExecution(51, 4).strategy,
            ml_cpu_compute::TreeEnsembleExecutionStrategy::kRowParallel);
  EXPECT_EQ(four_multi_target_trees.SelectExecution(51, 4).strategy,
            ml_cpu_compute::TreeEnsembleExecutionStrategy::kRowParallel);
}

TEST(TreeEnsembleOracle, CacheBlockingUsesL1SizedFourTreeGroups) {
  const ml_cpu_compute::TreeEnsembleCacheBlocking four_tree_block =
      ml_cpu_compute::SelectTreeEnsembleCacheBlocking(8, 8, 0, 700, 0, 4096);
  EXPECT_EQ(four_tree_block.trees_per_l1_block, 4U);
  EXPECT_EQ(four_tree_block.parallel_tree_threshold, 4U);

  const ml_cpu_compute::TreeEnsembleCacheBlocking three_tree_block =
      ml_cpu_compute::SelectTreeEnsembleCacheBlocking(8, 8, 0, 1000, 0, 4096);
  EXPECT_EQ(three_tree_block.trees_per_l1_block, 3U);
  EXPECT_EQ(three_tree_block.parallel_tree_threshold, 3U);

  const ml_cpu_compute::TreeEnsembleCacheBlocking oversized_tree =
      ml_cpu_compute::SelectTreeEnsembleCacheBlocking(8, 8, 0, 8192, 0, 4096);
  EXPECT_EQ(oversized_tree.trees_per_l1_block, 1U);
  EXPECT_EQ(oversized_tree.parallel_tree_threshold, 1U);

  const ml_cpu_compute::TreeEnsembleCacheBlocking missing_l1 =
      ml_cpu_compute::SelectTreeEnsembleCacheBlocking(8, 8, 0, 700, 0, 0);
  EXPECT_EQ(missing_l1.trees_per_l1_block, 8U);
  EXPECT_EQ(missing_l1.parallel_tree_threshold, 8U);
}

TEST(TreeEnsembleOracle, SchedulingWorkspaceIsBoundedByActiveBatch) {
  const ml_cpu_compute::TreeEnsemblePlan plan(StumpForest(81));
  const auto one_row = plan.SelectExecution(1, 4);
  const auto many_rows = plan.SelectExecution(1000000, 4);
  const std::size_t accumulator_bytes = sizeof(double) + sizeof(std::size_t);
  EXPECT_EQ(one_row.workspace_bytes, 2U * sizeof(double));
  EXPECT_EQ(many_rows.batch_rows, 1U);
  EXPECT_EQ(many_rows.workspace_bytes, 4U * 1U * 2U * accumulator_bytes);
}

TEST(TreeEnsembleOracle, ExecutionTuningControlsSchedulingThresholdsAndParticipants) {
  ThreadedExecutor executor;
  ml_cpu_compute::ExecutionExecutorView view{&executor, 8, &ThreadedExecutor::Run};
  ml_cpu_compute::ExecutionExecutorScope scope(&view);
  ml_cpu_compute::TreeEnsembleAttributes attributes = StumpForest(10000);
  attributes.value_type = DataType::DOUBLE;
  ml_cpu_compute::TreeEnsemblePlan plan(attributes);
  plan.ConfigureExecutionTuning({64, 10, 20, 2});
  EXPECT_EQ(plan.SelectExecution(10, 8).strategy,
            ml_cpu_compute::TreeEnsembleExecutionStrategy::kTreeMajorBatch);
  const auto tree_parallel = plan.SelectExecution(11, 8);
  EXPECT_EQ(tree_parallel.strategy, ml_cpu_compute::TreeEnsembleExecutionStrategy::kTreeParallel);
  EXPECT_EQ(tree_parallel.participants, 2U);
  EXPECT_EQ(tree_parallel.batch_rows, 11U);
  const auto row_parallel = plan.SelectExecution(21, 8);
  EXPECT_EQ(row_parallel.strategy, ml_cpu_compute::TreeEnsembleExecutionStrategy::kRowParallel);
  EXPECT_EQ(row_parallel.participants, 2U);

  const std::vector<double> input(129, -1.0);
  EXPECT_EQ(plan.Evaluate(input, input.size()),
            ml_cpu_compute::TreeEnsembleOracle(plan.attributes()).Evaluate(input, input.size()));
  EXPECT_GT(executor.dispatches.load(std::memory_order_relaxed), 0U);
  EXPECT_LE(executor.dispatched_blocks.load(std::memory_order_relaxed), 2U);

  const ml_cpu_compute::TreeEnsembleAttributes balanced_attributes = BalancedForest(1024);
  ml_cpu_compute::TreeEnsemblePlan balanced_plan(balanced_attributes);
  balanced_plan.ConfigureExecutionTuning({1, 10, 128, 2});
  constexpr std::size_t kBalancedRows = 65;
  const auto balanced_decision = balanced_plan.SelectExecution(kBalancedRows);
  ASSERT_EQ(balanced_decision.strategy,
            ml_cpu_compute::TreeEnsembleExecutionStrategy::kTreeParallel);
  ASSERT_EQ(balanced_decision.batch_rows, 1U);
  EXPECT_LE(balanced_decision.participants * balanced_decision.batch_rows * sizeof(float),
            balanced_decision.workspace_bytes);
  std::vector<float> balanced_input(kBalancedRows * 8, -1.0F);
  std::vector<float> balanced_output(kBalancedRows);
  executor.dispatches.store(0, std::memory_order_relaxed);
  executor.dispatched_blocks.store(0, std::memory_order_relaxed);
  balanced_plan.EvaluateInto(balanced_input.data(), balanced_input.size(), kBalancedRows,
                             balanced_output.data());
  const std::vector<double> balanced_reference_input(balanced_input.begin(), balanced_input.end());
  const auto balanced_expected = ml_cpu_compute::TreeEnsembleOracle(balanced_attributes)
                                     .Evaluate(balanced_reference_input, kBalancedRows);
  for (std::size_t row = 0; row < kBalancedRows; ++row) {
    EXPECT_NEAR(balanced_output[row], balanced_expected[row], 1e-5) << row;
  }
  EXPECT_GT(executor.dispatches.load(std::memory_order_relaxed), 0U);
  EXPECT_LE(executor.dispatched_blocks.load(std::memory_order_relaxed), 2U);
}

TEST(TreeEnsembleOracle, ExecutionTuningAcceptsUnitRowThreshold) {
  ThreadedExecutor executor;
  ml_cpu_compute::ExecutionExecutorView view{&executor, 4, &ThreadedExecutor::Run};
  ml_cpu_compute::ExecutionExecutorScope scope(&view);
  ml_cpu_compute::TreeEnsemblePlan plan(StumpForest(10000));
  EXPECT_NO_THROW(plan.ConfigureExecutionTuning({128, 1, 128, 4}));
  const auto &regions = plan.tuning_policy().regions;
  ASSERT_EQ(regions.size(), 3U);
  EXPECT_EQ(regions[0].maximum_rows, 1U);
  EXPECT_EQ(regions[1].maximum_rows, 128U);
  EXPECT_FALSE(regions[2].maximum_rows.has_value());
}

TEST(TreeEnsembleOracle, RejectedExecutionTuningPreservesPlanState) {
  ThreadedExecutor executor;
  ml_cpu_compute::ExecutionExecutorView view{&executor, 4, &ThreadedExecutor::Run};
  ml_cpu_compute::ExecutionExecutorScope scope(&view);
  const ml_cpu_compute::TreeEnsembleAttributes attributes = StumpForest(3);
  ml_cpu_compute::TreeEnsemblePlan plan(attributes);
  const ml_cpu_compute::TreeEnsembleExecutionTuning tuning = plan.execution_tuning();
  const ml_cpu_compute::TreeEnsembleTuningPolicy policy = plan.tuning_policy();
  EXPECT_THROW(plan.ConfigureExecutionTuning({0, 50, 128, 0}), std::invalid_argument);
  EXPECT_EQ(plan.execution_tuning(), tuning);
  EXPECT_EQ(plan.tuning_policy(), policy);
  const std::vector<double> input(51, -1.0);
  EXPECT_EQ(plan.Evaluate(input, input.size()),
            ml_cpu_compute::TreeEnsembleOracle(attributes).Evaluate(input, input.size()));
}

TEST(TreeEnsembleOracle, PreparedPolicyCoversEveryInclusiveRowCrossover) {
  ThreadedExecutor executor;
  ml_cpu_compute::ExecutionExecutorView view{&executor, 4, &ThreadedExecutor::Run};
  ml_cpu_compute::ExecutionExecutorScope scope(&view);
  const ml_cpu_compute::TreeEnsemblePlan plan(StumpForest(81));
  const auto &regions = plan.tuning_policy().regions;
  ASSERT_EQ(regions.size(), 2U);
  EXPECT_EQ(regions[0].maximum_rows, 50U);
  EXPECT_FALSE(regions[1].maximum_rows.has_value());
  EXPECT_LE(regions.size(), 4U);

  EXPECT_EQ(plan.SelectExecution(0, 4).strategy,
            ml_cpu_compute::TreeEnsembleExecutionStrategy::kTreeMajorBatch);
  EXPECT_EQ(plan.SelectExecution(1, 4).strategy,
            ml_cpu_compute::TreeEnsembleExecutionStrategy::kTreeMajorBatch);
  EXPECT_EQ(plan.SelectExecution(2, 4).strategy,
            ml_cpu_compute::TreeEnsembleExecutionStrategy::kTreeMajorBatch);
  EXPECT_EQ(plan.SelectExecution(50, 4).strategy,
            ml_cpu_compute::TreeEnsembleExecutionStrategy::kTreeMajorBatch);
  EXPECT_EQ(plan.SelectExecution(51, 4).strategy,
            ml_cpu_compute::TreeEnsembleExecutionStrategy::kRowParallel);

  const ml_cpu_compute::TreeEnsemblePlan few_trees(StumpForest(3));
  ASSERT_EQ(few_trees.tuning_policy().regions.size(), 2U);
  EXPECT_EQ(few_trees.SelectExecution(50, 4).strategy,
            ml_cpu_compute::TreeEnsembleExecutionStrategy::kTreeMajorBatch);
  EXPECT_EQ(few_trees.SelectExecution(51, 4).strategy,
            ml_cpu_compute::TreeEnsembleExecutionStrategy::kRowParallel);
}

TEST(TreeEnsembleOracle, AdvancedLayoutsAndInterleavingRetainPortableResults) {
  ml_cpu_compute::TreeEnsembleAttributes attributes = SymmetricTree();
  attributes.nodes_hitrates = {0.8, 0.2, 0.7};
  const std::vector<double> input{-2.0, -2.0, -2.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 0.0};
  const std::vector<double> expected =
      ml_cpu_compute::TreeEnsembleOracle(attributes).Evaluate(input, 5);
  const ml_cpu_compute::TreeEnsemblePlan key_plan(
      attributes, ml_cpu_compute::TreeEnsembleExecutionTuning{128, 50, 128, 2});
  EXPECT_TRUE(key_plan.all_trees_are_symmetric());
  for (const ml_cpu_compute::TreeEnsembleNodeLayout layout :
       {ml_cpu_compute::TreeEnsembleNodeLayout::kCompactAosIndex,
        ml_cpu_compute::TreeEnsembleNodeLayout::kSplitSoa,
        ml_cpu_compute::TreeEnsembleNodeLayout::kPreorderHot}) {
    ml_cpu_compute::TreeEnsembleTuningPolicy policy = key_plan.tuning_policy();
    policy.layout = layout;
    policy.traversal = ml_cpu_compute::TreeEnsembleTraversal::kSymmetric;
    const ml_cpu_compute::TreeEnsemblePlan plan(attributes, 2, policy);
    EXPECT_EQ(plan.Evaluate(input, 5), expected);
  }

  ml_cpu_compute::TreeEnsembleTuningPolicy interleaved = key_plan.tuning_policy();
  interleaved.regions[0].strategy = ml_cpu_compute::TreeEnsembleExecutionStrategy::kInterleavedRows;
  const ml_cpu_compute::TreeEnsemblePlan plan(attributes, 2, interleaved);
  EXPECT_EQ(plan.SelectExecution(5, 2).strategy,
            ml_cpu_compute::TreeEnsembleExecutionStrategy::kInterleavedRows);
  EXPECT_EQ(plan.Evaluate(input, 5), expected);
}

TEST(TreeEnsembleOracle, OptimizedFloat16MatchesCompleteV5Corpus) {
  for (const auto &test_case : GenerateTreeEnsembleV5Corpus()) {
    if (test_case.attributes.value_type != DataType::FLOAT16) {
      continue;
    }
    const ml_cpu_compute::TreeEnsemblePlan key_plan(
        test_case.attributes, ml_cpu_compute::TreeEnsembleExecutionTuning{128, 50, 128, 1});
    ml_cpu_compute::TreeEnsembleTuningPolicy policy = key_plan.tuning_policy();
    policy.optimized_float16 = true;
    const ml_cpu_compute::TreeEnsemblePlan plan(test_case.attributes, 1, policy);
    EXPECT_EQ(plan.Evaluate(test_case.input, test_case.rows), test_case.expected) << test_case.name;
  }
}

TEST(TreeEnsembleOracle, ExplicitAdvancedPolicyRetainsPortableResults) {
  const ml_cpu_compute::TreeEnsembleAttributes attributes = StumpForest(3, 64);
  const ml_cpu_compute::TreeEnsemblePlan baseline(
      attributes, ml_cpu_compute::TreeEnsembleExecutionTuning{128, 50, 128, 1});
  ml_cpu_compute::TreeEnsembleTuningPolicy policy = baseline.tuning_policy();
  policy.traversal = ml_cpu_compute::TreeEnsembleTraversal::kStump;
  policy.target_layout = ml_cpu_compute::TreeEnsembleTargetLayout::kSparse;
  policy.traversal_prefetch_distance = 1;
  const ml_cpu_compute::TreeEnsemblePlan plan(attributes, 1, policy);
  const std::vector<double> input{-1.0, 1.0, -1.0, 1.0, -1.0};
  EXPECT_EQ(plan.Evaluate(input, input.size()),
            ml_cpu_compute::TreeEnsembleOracle(attributes).Evaluate(input, input.size()));
  EXPECT_LT(plan.SelectExecution(input.size(), 1).workspace_bytes,
            baseline.SelectExecution(input.size(), 1).workspace_bytes);
}

TEST(TreeEnsembleOracle, EverySchedulingStrategyMatchesScalarAcrossThreadCounts) {
  const ml_cpu_compute::TreeEnsembleAttributes attributes = StumpForest(81);
  const ml_cpu_compute::TreeEnsembleOracle oracle(attributes);
  const ml_cpu_compute::TreeEnsemblePlan plan(attributes);
  for (const std::size_t rows : {1U, 49U, 50U, 51U, 129U}) {
    std::vector<double> input(rows);
    for (std::size_t row = 0; row < rows; ++row) {
      input[row] = (row % 2 == 0) ? -1.0 : 1.0;
    }
    const std::vector<double> expected = oracle.Evaluate(input, rows);
    for (const int64_t threads : {1, 2, 4}) {
      ThreadedExecutor executor;
      ml_cpu_compute::ExecutionExecutorView view{&executor, threads, &ThreadedExecutor::Run};
      ml_cpu_compute::ExecutionExecutorScope scope(&view);
      EXPECT_EQ(plan.Evaluate(input, rows), expected) << "rows=" << rows << ", threads=" << threads;
      EXPECT_LE(executor.maximum_active.load(std::memory_order_relaxed),
                static_cast<std::size_t>(threads));
    }
  }

  const ml_cpu_compute::TreeEnsembleAttributes row_attributes = StumpForest(3);
  const ml_cpu_compute::TreeEnsembleOracle row_oracle(row_attributes);
  const ml_cpu_compute::TreeEnsemblePlan row_plan(row_attributes);
  std::vector<double> input(51, -1.0);
  ThreadedExecutor executor;
  ml_cpu_compute::ExecutionExecutorView view{&executor, 4, &ThreadedExecutor::Run};
  ml_cpu_compute::ExecutionExecutorScope scope(&view);
  EXPECT_EQ(row_plan.SelectExecution(input.size()).strategy,
            ml_cpu_compute::TreeEnsembleExecutionStrategy::kRowParallel);
  EXPECT_EQ(row_plan.Evaluate(input, input.size()), row_oracle.Evaluate(input, input.size()));
}

TEST(TreeEnsembleOracle, BalancedFloatForestUsesExactFixedDepthTraversal) {
  const ml_cpu_compute::TreeEnsembleAttributes attributes = BalancedForest(1024);
  const ml_cpu_compute::TreeEnsemblePlan plan(attributes);
  EXPECT_TRUE(plan.all_trees_are_balanced());
  EXPECT_FALSE(plan.all_trees_are_symmetric());

  constexpr std::size_t kRows = 70;
  std::vector<float> input(kRows * 8);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<float>(static_cast<int>(index % 11) - 5) * 0.25F;
  }
  const std::vector<double> oracle_input(input.begin(), input.end());
  const std::vector<double> expected =
      ml_cpu_compute::TreeEnsembleOracle(attributes).Evaluate(oracle_input, kRows);
  std::vector<float> sequential(kRows);
  plan.EvaluateInto(input.data(), input.size(), kRows, sequential.data());
  for (std::size_t row = 0; row < kRows; ++row) {
    EXPECT_FLOAT_EQ(sequential[row], static_cast<float>(expected[row])) << "row=" << row;
  }

  std::vector<float> actual(kRows);
  ThreadedExecutor executor;
  ml_cpu_compute::ExecutionExecutorView view{&executor, 4, &ThreadedExecutor::Run};
  ml_cpu_compute::ExecutionExecutorScope scope(&view);
  plan.EvaluateInto(input.data(), input.size(), kRows, actual.data());

  EXPECT_GT(executor.dispatches.load(std::memory_order_relaxed), 0U);
  for (std::size_t row = 0; row < kRows; ++row) {
    EXPECT_NEAR(actual[row], expected[row], 1e-5) << "row=" << row;
  }
}

TEST(TreeEnsembleOracle, BalancedFloatForestUsesRowParallelFixedDepthTraversal) {
  const ml_cpu_compute::TreeEnsembleAttributes attributes = BalancedForest(1024);
  const ml_cpu_compute::TreeEnsemblePlan plan(attributes);
  constexpr std::size_t kRows = 131;
  std::vector<float> input(kRows * 8);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<float>(static_cast<int>(index % 11) - 5) * 0.25F;
  }
  const std::vector<double> oracle_input(input.begin(), input.end());
  const std::vector<double> expected =
      ml_cpu_compute::TreeEnsembleOracle(attributes).Evaluate(oracle_input, kRows);

  std::vector<float> actual(kRows);
  ThreadedExecutor executor;
  ml_cpu_compute::ExecutionExecutorView view{&executor, 4, &ThreadedExecutor::Run};
  ml_cpu_compute::ExecutionExecutorScope scope(&view);
  EXPECT_EQ(plan.SelectExecution(kRows).strategy,
            ml_cpu_compute::TreeEnsembleExecutionStrategy::kRowParallel);
  plan.EvaluateInto(input.data(), input.size(), kRows, actual.data());

  EXPECT_GT(executor.dispatches.load(std::memory_order_relaxed), 0U);
  EXPECT_EQ(executor.dispatched_blocks.load(std::memory_order_relaxed), 4U);
  for (std::size_t row = 0; row < kRows; ++row) {
    EXPECT_NEAR(actual[row], expected[row], 1e-5) << "row=" << row;
  }
}

TEST(TreeEnsembleOracle, BalancedFloatRowVectorsPreserveRootRoutingAndTailBounds) {
  constexpr std::size_t kFeatures = 4096;
  constexpr std::size_t kBatchRows = 16;
  constexpr float kSentinel = -42.0F;
  for (auto attributes : {StumpForest(17, 1), BalancedForest(17)}) {
    attributes.n_features = kFeatures;
    attributes.value_type = DataType::FLOAT;
    for (std::size_t node = 0; node < attributes.nodes_featureids.size(); ++node) {
      attributes.nodes_featureids[node] = node % 2 == 0 ? 0 : kFeatures - 1;
    }
    const TreeEnsembleOracle oracle(attributes);
    for (const auto strategy : {TreeEnsembleExecutionStrategy::kRowParallel,
                                TreeEnsembleExecutionStrategy::kTreeMajorBatch}) {
      const auto policy = OneRegionPolicy(strategy, 2, 1, kBatchRows);
      TreeEnsemblePlan plan(attributes, 2, policy);
      plan.CompactRuntimeStorage();
      for (std::size_t rows = 0; rows <= 33; ++rows) {
        SCOPED_TRACE(::testing::Message() << "rows=" << rows << " depth=" << plan.max_depth()
                                          << " strategy=" << static_cast<int>(strategy));
        std::vector<float> input(rows * kFeatures);
        const float values[] = {-std::numeric_limits<float>::infinity(),
                                -0.5F,
                                -0.0F,
                                0.0F,
                                0.5F,
                                std::numeric_limits<float>::infinity(),
                                std::numeric_limits<float>::quiet_NaN()};
        for (std::size_t row = 0; row < rows; ++row) {
          input[row * kFeatures] = values[row % 7];
          input[row * kFeatures + kFeatures - 1] = values[(row + 3) % 7];
        }
        const std::vector<double> reference_input(input.begin(), input.end());
        const auto expected = oracle.Evaluate(reference_input, rows);
        std::vector<float> actual(rows + 2, kSentinel);
        ThreadedExecutor executor;
        ml_cpu_compute::ExecutionExecutorView view{&executor, 2, &ThreadedExecutor::Run};
        ml_cpu_compute::ExecutionExecutorScope scope(&view);
        if (rows != 0) {
          const auto decision = plan.SelectExecution(rows);
          ASSERT_EQ(decision.strategy, strategy);
          ASSERT_EQ(decision.batch_rows, std::min(rows, kBatchRows));
        }
        plan.EvaluateInto(rows == 0 ? nullptr : input.data(), input.size(), rows,
                          actual.data() + 1);
        if (strategy == TreeEnsembleExecutionStrategy::kRowParallel && rows > kBatchRows) {
          EXPECT_GT(executor.dispatches.load(std::memory_order_relaxed), 0U);
        } else if (strategy == TreeEnsembleExecutionStrategy::kTreeMajorBatch) {
          EXPECT_EQ(executor.dispatches.load(std::memory_order_relaxed), 0U);
        }
        EXPECT_EQ(actual.front(), kSentinel);
        EXPECT_EQ(actual.back(), kSentinel);
        for (std::size_t row = 0; row < rows; ++row) {
          EXPECT_FLOAT_EQ(actual[row + 1], static_cast<float>(expected[row])) << row;
        }
      }
    }
  }
}

TEST(TreeEnsembleOracle, SingleRowBalancedFloatKeepsBaseInsideTreeReduction) {
  auto attributes = StumpForest(2, 1);
  attributes.base_values = {16777216.0};
  attributes.leaf_weights = {1.0, 1.0, -1.0, -1.0};
  const ml_cpu_compute::TreeEnsemblePlan plan(attributes);
  const std::vector<float> input = {0.0F};
  std::vector<float> actual = {-42.0F, 0.0F, -42.0F};
  ThreadedExecutor executor;
  ml_cpu_compute::ExecutionExecutorView view{&executor, 4, &ThreadedExecutor::Run};
  {
    ml_cpu_compute::ExecutionExecutorScope scope(&view);
    plan.EvaluateInto(input.data(), input.size(), 1, actual.data() + 1);
  }
  EXPECT_EQ(executor.dispatches.load(std::memory_order_relaxed), 0U);
  EXPECT_EQ(actual[0], -42.0F);
  EXPECT_EQ(actual[1], 16777215.0F);
  EXPECT_EQ(actual[2], -42.0F);
}

TEST(TreeEnsembleOracle, SingleRowBalancedFloatHandlesVectorTreeTails) {
  for (const std::size_t trees : {7U, 8U, 9U, 15U, 16U, 17U, 97U}) {
    const ml_cpu_compute::TreeEnsembleAttributes attributes = BalancedForest(trees);
    const ml_cpu_compute::TreeEnsembleOracle oracle(attributes);
    const ml_cpu_compute::TreeEnsembleTuningPolicy policy =
        OneRegionPolicy(ml_cpu_compute::TreeEnsembleExecutionStrategy::kTreeParallel, 4, 1);
    ml_cpu_compute::TreeEnsemblePlan plan(attributes, 4, policy);
    plan.CompactRuntimeStorage();
    std::vector<float> input(static_cast<std::size_t>(attributes.n_features));
    for (std::size_t index = 0; index < input.size(); ++index) {
      input[index] = static_cast<float>(static_cast<int>(index) - 4) * 0.25F;
    }
    const std::vector<double> reference_input(input.begin(), input.end());
    const std::vector<double> expected = oracle.Evaluate(reference_input, 1);
    std::vector<float> actual(1);
    ThreadedExecutor executor;
    ml_cpu_compute::ExecutionExecutorView view{&executor, 4, &ThreadedExecutor::Run};
    {
      ml_cpu_compute::ExecutionExecutorScope scope(&view);
      plan.EvaluateInto(input.data(), input.size(), 1, actual.data());
    }
    EXPECT_GT(executor.dispatches.load(std::memory_order_relaxed), 0U);
    EXPECT_FLOAT_EQ(actual[0], static_cast<float>(expected[0])) << "trees=" << trees;
  }
}

TEST(TreeEnsembleOracle, BalancedFloatPartitionsPreserveTailsBaseAndMissingValues) {
  for (const auto &attributes : {StumpForest(97, 1), BalancedForest(97)}) {
    const ml_cpu_compute::TreeEnsembleOracle oracle(attributes);
    for (const auto strategy : {ml_cpu_compute::TreeEnsembleExecutionStrategy::kTreeParallel,
                                ml_cpu_compute::TreeEnsembleExecutionStrategy::kRowParallel,
                                ml_cpu_compute::TreeEnsembleExecutionStrategy::kTreeMajorBatch}) {
      const ml_cpu_compute::TreeEnsembleTuningPolicy policy = OneRegionPolicy(strategy, 4, 1, 128);
      ml_cpu_compute::TreeEnsemblePlan plan(attributes, 4, policy);
      plan.CompactRuntimeStorage();
      std::vector<std::size_t> row_counts = {1, 8, 63, 64, 65, 127, 128};
      for (std::size_t rows = 15; rows <= 33; ++rows) {
        row_counts.push_back(rows);
      }
      for (const std::size_t rows : row_counts) {
        SCOPED_TRACE(::testing::Message()
                     << "rows=" << rows << " strategy=" << static_cast<int>(strategy)
                     << " features=" << attributes.n_features);
        std::vector<float> input(rows * static_cast<std::size_t>(attributes.n_features));
        for (std::size_t i = 0; i < input.size(); ++i) {
          input[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.25F;
        }
        input.front() = std::numeric_limits<float>::quiet_NaN();
        if (input.size() > 2) {
          input[1] = std::numeric_limits<float>::infinity();
          input[2] = -std::numeric_limits<float>::infinity();
        }
        const std::vector<double> reference_input(input.begin(), input.end());
        const auto expected = oracle.Evaluate(reference_input, rows);
        std::vector<float> actual(rows);
        ThreadedExecutor executor;
        ml_cpu_compute::ExecutionExecutorView view{&executor, 4, &ThreadedExecutor::Run};
        {
          ml_cpu_compute::ExecutionExecutorScope scope(&view);
          plan.EvaluateInto(input.data(), input.size(), rows, actual.data());
        }
        EXPECT_LE(executor.maximum_active.load(std::memory_order_relaxed), 4U);
        for (std::size_t row = 0; row < rows; ++row) {
          EXPECT_NEAR(actual[row], expected[row], 1e-5) << row;
        }
      }
    }
  }
}

TEST(TreeEnsembleOracle, BalancedFloatTreePartitionsRespectThreadLimitsAndNesting) {
  constexpr std::size_t rows = 65;
  const auto attributes = BalancedForest(97);
  const std::vector<float> input(rows * static_cast<std::size_t>(attributes.n_features), -1.0F);
  const std::vector<double> reference_input(input.begin(), input.end());
  const auto expected =
      ml_cpu_compute::TreeEnsembleOracle(attributes).Evaluate(reference_input, rows);
  for (const std::size_t policy_limit : {2U, 80U}) {
    const ml_cpu_compute::TreeEnsembleTuningPolicy policy = OneRegionPolicy(
        ml_cpu_compute::TreeEnsembleExecutionStrategy::kTreeParallel, policy_limit, 1, rows);
    const ml_cpu_compute::TreeEnsemblePlan plan(attributes, 80, policy);
    for (const int64_t threads : {1, 3, 80}) {
      for (const bool nested : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "policy_limit=" << policy_limit
                                          << " threads=" << threads << " nested=" << nested);
        ThreadedExecutor executor;
        ml_cpu_compute::ExecutionExecutorView view{&executor, threads, &ThreadedExecutor::Run};
        ml_cpu_compute::ExecutionExecutorScope scope(&view);
        const auto decision = plan.SelectExecution(rows);
        if (decision.strategy == ml_cpu_compute::TreeEnsembleExecutionStrategy::kTreeParallel) {
          const std::size_t padded_workspace =
              decision.participants * (rows + ml_cpu_compute::ExecutionSimdLanes<float>()) *
              sizeof(float);
          EXPECT_LE(padded_workspace, decision.workspace_bytes);
        }
        std::vector<float> actual(rows);
        if (nested) {
          ml_cpu_compute::detail::ExecutionRegionScope region;
          plan.EvaluateInto(input.data(), input.size(), rows, actual.data());
        } else {
          plan.EvaluateInto(input.data(), input.size(), rows, actual.data());
        }
        EXPECT_LE(executor.dispatched_blocks.load(std::memory_order_relaxed),
                  std::min({policy_limit, static_cast<std::size_t>(threads), std::size_t{64}}));
        if (nested || threads == 1) {
          EXPECT_EQ(executor.dispatches.load(std::memory_order_relaxed), 0U);
        } else {
          EXPECT_GT(executor.dispatches.load(std::memory_order_relaxed), 0U);
        }
        for (std::size_t row = 0; row < rows; ++row) {
          EXPECT_NEAR(actual[row], expected[row], 1e-5) << row;
        }
      }
    }
  }
}

TEST(TreeEnsembleOracle, BalancedFloatTreePartitionsCoverOffsetBatchesAndTail) {
  constexpr std::size_t rows = 65;
  const auto attributes = BalancedForest(97);
  const auto policy =
      OneRegionPolicy(ml_cpu_compute::TreeEnsembleExecutionStrategy::kTreeParallel, 2, 1, 32);
  ml_cpu_compute::TreeEnsemblePlan plan(attributes, 2, policy);
  plan.CompactRuntimeStorage();
  std::vector<float> input(rows * static_cast<std::size_t>(attributes.n_features));
  for (std::size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>(static_cast<int>((i * 7 + i / 8) % 23) - 11) * 0.25F;
  }
  const std::vector<double> reference_input(input.begin(), input.end());
  const auto expected =
      ml_cpu_compute::TreeEnsembleOracle(attributes).Evaluate(reference_input, rows);
  std::vector<float> actual(rows + 2, -42.0F);
  ThreadedExecutor executor;
  ml_cpu_compute::ExecutionExecutorView view{&executor, 2, &ThreadedExecutor::Run};
  {
    ml_cpu_compute::ExecutionExecutorScope scope(&view);
    const auto decision = plan.SelectExecution(rows);
    ASSERT_EQ(decision.strategy, ml_cpu_compute::TreeEnsembleExecutionStrategy::kTreeParallel);
    ASSERT_EQ(decision.batch_rows, 32U);
    ASSERT_EQ(decision.participants, 2U);
    plan.EvaluateInto(input.data(), input.size(), rows, actual.data() + 1);
  }
  EXPECT_EQ(executor.dispatches.load(std::memory_order_relaxed), 3U);
  EXPECT_EQ(executor.dispatched_blocks.load(std::memory_order_relaxed), 2U);
  EXPECT_EQ(actual.front(), -42.0F);
  EXPECT_EQ(actual.back(), -42.0F);
  for (std::size_t row = 0; row < rows; ++row) {
    EXPECT_NEAR(actual[row + 1], expected[row], 1e-5) << row;
  }
}

TEST(TreeEnsembleOracle, ThresholdsSignedZeroInfinityAndMissingRouting) {
  ml_cpu_compute::TreeEnsembleAttributes attributes = Stump();
  attributes.value_type = DataType::DOUBLE;
  attributes.nodes_splits = {-0.0};
  attributes.nodes_modes = {ml_cpu_compute::TreeBranchMode::kEq};
  attributes.nodes_missing_value_tracks_true = {1};
  const double infinity = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<double> actual = ml_cpu_compute::TreeEnsembleOracle(std::move(attributes))
                                         .Evaluate({-0.0, 0.0, -infinity, infinity, nan}, 5);
  EXPECT_EQ(actual, (std::vector<double>{1.0, 1.0, -1.0, -1.0, 1.0}));
}

TEST(TreeEnsembleOracle, Float16RoundsAtTieToEvenBoundary) {
  ml_cpu_compute::TreeEnsembleAttributes attributes = Stump();
  attributes.value_type = DataType::FLOAT16;
  attributes.nodes_splits = {1.00048828125};
  const std::vector<double> actual = ml_cpu_compute::TreeEnsembleOracle(std::move(attributes))
                                         .Evaluate({1.0, 1.00048828125, 1.0009765625}, 3);
  EXPECT_EQ(actual, (std::vector<double>{1.0, 1.0, -1.0}));
}

TEST(TreeEnsembleOracle, MembershipUsesNonEmptyNanDelimitedSets) {
  ml_cpu_compute::TreeEnsembleAttributes attributes = Stump();
  attributes.nodes_modes = {ml_cpu_compute::TreeBranchMode::kMember};
  attributes.membership_values = {-0.0, 2.0, 2.0, std::numeric_limits<double>::quiet_NaN()};
  const std::vector<double> actual =
      ml_cpu_compute::TreeEnsembleOracle(std::move(attributes)).Evaluate({0.0, 2.0, 3.0}, 3);
  EXPECT_EQ(actual, (std::vector<double>{1.0, 1.0, -1.0}));
}

TEST(TreeEnsembleOracle, AggregatesMultipleTargetsAndAppliesTransforms) {
  ml_cpu_compute::TreeEnsembleAttributes attributes;
  attributes.n_features = 1;
  attributes.n_targets = 2;
  attributes.tree_roots = {0, 1, 2, 3};
  attributes.nodes_featureids = {0, 0, 0, 0};
  attributes.nodes_splits = {0, 0, 0, 0};
  attributes.nodes_modes = {
      ml_cpu_compute::TreeBranchMode::kLeq, ml_cpu_compute::TreeBranchMode::kLeq,
      ml_cpu_compute::TreeBranchMode::kLeq, ml_cpu_compute::TreeBranchMode::kLeq};
  attributes.nodes_truenodeids = {0, 1, 2, 3};
  attributes.nodes_falsenodeids = {0, 1, 2, 3};
  attributes.nodes_trueleafs = {1, 1, 1, 1};
  attributes.nodes_falseleafs = {1, 1, 1, 1};
  attributes.leaf_targetids = {0, 0, 1, 1};
  attributes.leaf_weights = {2.0, 4.0, 1.0, 3.0};
  attributes.aggregate = ml_cpu_compute::TreeAggregate::kAverage;
  attributes.post_transform = ml_cpu_compute::TreePostTransform::kSoftmax;
  const std::vector<double> actual =
      ml_cpu_compute::TreeEnsembleOracle(std::move(attributes)).Evaluate({0.0}, 1);
  ASSERT_EQ(actual.size(), 2U);
  EXPECT_NEAR(actual[0], 0.6224593, 1e-6);
  EXPECT_NEAR(actual[1], 0.3775407, 1e-6);
}

class TreeEnsembleSoftmaxZero : public ::testing::TestWithParam<std::size_t> {};

TEST_P(TreeEnsembleSoftmaxZero, ZeroSumUsesUniformFallback) {
  const std::size_t targets = GetParam();
  std::vector<std::vector<double>> scores = {std::vector<double>(targets, 0.0),
                                             std::vector<double>(targets, -0.0)};
  if (targets > 1) {
    std::vector<double> cancelling(targets, 0.0);
    cancelling[0] = 5e-8;
    cancelling[1] = -5e-8;
    scores.push_back(std::move(cancelling));
  }
  for (DataType type : {DataType::FLOAT, DataType::DOUBLE}) {
    for (const auto &base_values : scores) {
      SCOPED_TRACE(::testing::PrintToString(base_values));
      ml_cpu_compute::TreeEnsembleAttributes attributes = StumpForest(targets, targets);
      attributes.value_type = type;
      attributes.post_transform = ml_cpu_compute::TreePostTransform::kSoftmaxZero;
      attributes.base_values = base_values;
      std::fill(attributes.leaf_weights.begin(), attributes.leaf_weights.end(), 0.0);
      for (const auto &actual :
           {ml_cpu_compute::TreeEnsembleOracle(attributes).Evaluate({-1.0, 1.0}, 2),
            ml_cpu_compute::TreeEnsemblePlan(attributes).Evaluate({-1.0, 1.0}, 2)}) {
        ASSERT_EQ(actual.size(), 2 * targets);
        for (std::size_t row = 0; row < 2; ++row) {
          double sum = 0.0;
          for (std::size_t target = 0; target < targets; ++target) {
            const double value = actual[row * targets + target];
            EXPECT_TRUE(std::isfinite(value));
            EXPECT_NEAR(value, 1.0 / static_cast<double>(targets), 1e-6);
            sum += value;
          }
          EXPECT_NEAR(sum, 1.0, 1e-6);
        }
      }
    }
  }
}

TEST_P(TreeEnsembleSoftmaxZero, NonzeroSumRetainsNormalization) {
  const std::size_t targets = GetParam();
  for (double score : {5e-8, 2.0, -2.0, 1000.0}) {
    for (bool one_active_target : {false, true}) {
      SCOPED_TRACE(score);
      SCOPED_TRACE(one_active_target);
      ml_cpu_compute::TreeEnsembleAttributes attributes = StumpForest(targets, targets);
      attributes.value_type = DataType::DOUBLE;
      attributes.post_transform = ml_cpu_compute::TreePostTransform::kSoftmaxZero;
      attributes.base_values.assign(targets, one_active_target ? 0.0 : score);
      attributes.base_values[0] = score;
      std::fill(attributes.leaf_weights.begin(), attributes.leaf_weights.end(), 0.0);
      for (const auto &actual : {ml_cpu_compute::TreeEnsembleOracle(attributes).Evaluate({0.0}, 1),
                                 ml_cpu_compute::TreeEnsemblePlan(attributes).Evaluate({0.0}, 1)}) {
        ASSERT_EQ(actual.size(), targets);
        double sum = 0.0;
        for (std::size_t target = 0; target < targets; ++target) {
          EXPECT_TRUE(std::isfinite(actual[target]));
          const double expected =
              one_active_target ? (target == 0 ? 1.0 : 0.0) : 1.0 / static_cast<double>(targets);
          EXPECT_NEAR(actual[target], expected, 1e-12);
          sum += actual[target];
        }
        EXPECT_NEAR(sum, 1.0, 1e-12);
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(TargetCounts, TreeEnsembleSoftmaxZero, ::testing::Values(1U, 2U, 3U, 5U));

TEST(TreeEnsembleOracle, AggregatesMinMaxWithoutZeroBiasAndRetainsAverageTreeDivisor) {
  ml_cpu_compute::TreeEnsembleAttributes average;
  average.n_features = 1;
  average.n_targets = 2;
  average.aggregate = ml_cpu_compute::TreeAggregate::kAverage;
  average.tree_roots = {0, 1};
  average.nodes_featureids = {0, 0};
  average.nodes_splits = {0.0, 0.0};
  average.nodes_modes = {ml_cpu_compute::TreeBranchMode::kLeq,
                         ml_cpu_compute::TreeBranchMode::kLeq};
  average.nodes_truenodeids = {0, 2};
  average.nodes_falsenodeids = {1, 3};
  average.nodes_trueleafs = {1, 1};
  average.nodes_falseleafs = {1, 1};
  average.leaf_targetids = {0, 0, 1, 1};
  average.leaf_weights = {6.0, 4.0, 2.0, 1.0};
  const std::vector<double> average_actual =
      ml_cpu_compute::TreeEnsembleOracle(average).Evaluate({0.0}, 1);
  EXPECT_EQ(average_actual, (std::vector<double>{3.0, 1.0}));

  ml_cpu_compute::TreeEnsembleAttributes minimum = average;
  minimum.aggregate = ml_cpu_compute::TreeAggregate::kMin;
  minimum.leaf_weights = {5.0, 10.0, 3.0, 7.0};
  const std::vector<double> minimum_actual =
      ml_cpu_compute::TreeEnsembleOracle(minimum).Evaluate({0.0}, 1);
  EXPECT_EQ(minimum_actual, (std::vector<double>{5.0, 3.0}));

  ml_cpu_compute::TreeEnsembleAttributes maximum = average;
  maximum.aggregate = ml_cpu_compute::TreeAggregate::kMax;
  maximum.leaf_weights = {-5.0, -10.0, -3.0, -7.0};
  const std::vector<double> maximum_actual =
      ml_cpu_compute::TreeEnsembleOracle(maximum).Evaluate({0.0}, 1);
  EXPECT_EQ(maximum_actual, (std::vector<double>{-5.0, -3.0}));

  ml_cpu_compute::TreeEnsembleAttributes biased_min = minimum;
  biased_min.base_values = {10.0, 2.0};
  biased_min.leaf_weights = {5.0, 12.0, 7.0, 9.0};
  EXPECT_EQ(ml_cpu_compute::TreeEnsembleOracle(biased_min).Evaluate({0.0}, 1),
            (std::vector<double>{15.0, 9.0}));

  const ml_cpu_compute::TreeEnsemblePlan minimum_plan(minimum);
  EXPECT_EQ(minimum_plan.Evaluate({0.0}, 1), (std::vector<double>{5.0, 3.0}));
  const ml_cpu_compute::TreeEnsemblePlan maximum_plan(maximum);
  EXPECT_EQ(maximum_plan.Evaluate({0.0}, 1), (std::vector<double>{-5.0, -3.0}));
}

TEST(TreeEnsembleOracle, RejectsMalformedAttributeLengthsAndIndices) {
  ml_cpu_compute::TreeEnsembleAttributes attributes = Stump();
  attributes.nodes_splits.clear();
  EXPECT_THROW(ml_cpu_compute::TreeEnsembleOracle(std::move(attributes)), std::invalid_argument);

  attributes = Stump();
  attributes.nodes_featureids[0] = 1;
  EXPECT_THROW(ml_cpu_compute::TreeEnsembleOracle(std::move(attributes)), std::invalid_argument);

  attributes = Stump();
  attributes.nodes_truenodeids[0] = 2;
  EXPECT_THROW(ml_cpu_compute::TreeEnsembleOracle(std::move(attributes)), std::invalid_argument);

  attributes = Stump();
  attributes.leaf_targetids[0] = 1;
  EXPECT_THROW(ml_cpu_compute::TreeEnsembleOracle(std::move(attributes)), std::invalid_argument);

  attributes = Stump();
  attributes.tree_roots[0] = 1;
  EXPECT_THROW(ml_cpu_compute::TreeEnsembleOracle(std::move(attributes)), std::invalid_argument);
}

TEST(TreeEnsembleOracle, RejectsCyclesSharedAndUnreachableNodes) {
  ml_cpu_compute::TreeEnsembleAttributes attributes = Stump();
  attributes.nodes_featureids = {0, 0};
  attributes.nodes_splits = {0, 0};
  attributes.nodes_modes = {ml_cpu_compute::TreeBranchMode::kLeq,
                            ml_cpu_compute::TreeBranchMode::kLeq};
  attributes.nodes_truenodeids = {1, 0};
  attributes.nodes_falsenodeids = {0, 1};
  attributes.nodes_trueleafs = {0, 0};
  attributes.nodes_falseleafs = {1, 1};
  EXPECT_THROW(ml_cpu_compute::TreeEnsembleOracle{attributes}, std::invalid_argument);

  attributes.nodes_truenodeids = {1, 0};
  attributes.nodes_falsenodeids = {1, 1};
  attributes.nodes_trueleafs = {0, 1};
  attributes.nodes_falseleafs[0] = 0;
  EXPECT_THROW(ml_cpu_compute::TreeEnsembleOracle{attributes}, std::invalid_argument);

  attributes = Stump();
  attributes.nodes_featureids.push_back(0);
  attributes.nodes_splits.push_back(0);
  attributes.nodes_modes.push_back(ml_cpu_compute::TreeBranchMode::kLeq);
  attributes.nodes_truenodeids.push_back(0);
  attributes.nodes_falsenodeids.push_back(1);
  attributes.nodes_trueleafs.push_back(1);
  attributes.nodes_falseleafs.push_back(1);
  EXPECT_THROW(ml_cpu_compute::TreeEnsembleOracle(std::move(attributes)), std::invalid_argument);
}

TEST(TreeEnsembleOracle, RejectsTopologyBeyondDepthBudget) {
  EXPECT_THROW(ml_cpu_compute::TreeEnsembleOracle(DeepTree((128U << 10) + 1)),
               std::invalid_argument);
}

TEST(TreeEnsembleOracle, DeepTopologyPreprocessingIsIterative) {
  constexpr std::size_t kDepth = 100'000;
  const ml_cpu_compute::TreeEnsemblePlan plan(DeepTree(kDepth));
  EXPECT_EQ(plan.max_depth(), kDepth);
  EXPECT_EQ(plan.average_depth(), kDepth);
  EXPECT_FALSE(plan.all_trees_are_balanced());
  EXPECT_FALSE(plan.all_trees_are_symmetric());
  const auto result = plan.Evaluate({-1.0}, 1);
  ASSERT_EQ(result.size(), 1U);
  EXPECT_EQ(result[0], 1.0);
}

TEST(TreeEnsembleOracle, RejectsMalformedMembershipDelimiters) {
  ml_cpu_compute::TreeEnsembleAttributes attributes = Stump();
  attributes.nodes_modes = {ml_cpu_compute::TreeBranchMode::kMember};
  EXPECT_THROW(ml_cpu_compute::TreeEnsembleOracle{attributes}, std::invalid_argument);

  attributes.membership_values = {std::numeric_limits<double>::quiet_NaN()};
  EXPECT_THROW(ml_cpu_compute::TreeEnsembleOracle{attributes}, std::invalid_argument);

  attributes.membership_values = {1.0};
  EXPECT_THROW(ml_cpu_compute::TreeEnsembleOracle{attributes}, std::invalid_argument);

  attributes.membership_values = {1.0, std::numeric_limits<double>::quiet_NaN(), 2.0};
  EXPECT_THROW(ml_cpu_compute::TreeEnsembleOracle(std::move(attributes)), std::invalid_argument);
}

TEST(TreeEnsembleOracle, DeprecatedRegressorV5AdapterHandlesIntegerInputs) {
  ml_cpu_compute::TreeEnsembleRegressorAttributes attributes;
  attributes.tree = LegacyStump();
  attributes.n_targets = 2;
  attributes.aggregate = ml_cpu_compute::TreeAggregate::kSum;
  attributes.post_transform = ml_cpu_compute::TreePostTransform::kLogistic;
  attributes.target_treeids = {0, 0, 0, 0};
  attributes.target_nodeids = {1, 1, 2, 2};
  attributes.target_ids = {0, 1, 0, 1};
  attributes.target_weights = {1.0, -1.0, 2.0, -2.0};
  attributes.base_values = {0.5, 0.5};
  const std::vector<float> actual =
      ml_cpu_compute::EvaluateTreeEnsembleRegressor(attributes, {-1, 1}, 2);
  ASSERT_EQ(actual.size(), 4U);
  EXPECT_NEAR(actual[0], 0.8175745f, 1e-6f);
  EXPECT_NEAR(actual[1], 0.3775407f, 1e-6f);
  EXPECT_NEAR(actual[2], 0.9241418f, 1e-6f);
  EXPECT_NEAR(actual[3], 0.1824255f, 1e-6f);
}

TEST(TreeEnsembleOracle, DeprecatedClassifierComposesLabelsAndUsesStableTies) {
  ml_cpu_compute::TreeEnsembleClassifierAttributes attributes;
  attributes.tree = LegacyStump();
  attributes.class_treeids = {0, 0, 0, 0};
  attributes.class_nodeids = {1, 1, 2, 2};
  attributes.class_ids = {0, 1, 0, 1};
  attributes.class_weights = {1.0, 1.0, 0.0, 2.0};
  attributes.labels = ClassLabels(std::vector<std::string>{"left", "right"});
  const auto actual = ml_cpu_compute::EvaluateTreeEnsembleClassifier(attributes, {-1, 1}, 2);
  EXPECT_TRUE(actual.integer_labels.empty());
  EXPECT_EQ(actual.string_labels, (std::vector<std::string>{"left", "right"}));
  EXPECT_EQ(actual.scores, (std::vector<float>{1.0f, 1.0f, 0.0f, 2.0f}));
}

TEST(TreeEnsembleOracle, DeprecatedAdaptersRejectLabelsAndTargetMetadata) {
  ml_cpu_compute::TreeEnsembleClassifierAttributes classifier;
  classifier.tree = LegacyStump();
  classifier.labels = ClassLabels(std::vector<std::int64_t>{});
  EXPECT_THROW(ml_cpu_compute::EvaluateTreeEnsembleClassifier(classifier, {0}, 1),
               std::invalid_argument);

  ml_cpu_compute::TreeEnsembleRegressorAttributes regressor;
  regressor.tree = LegacyStump();
  regressor.n_targets = 1;
  regressor.target_treeids = {0};
  regressor.target_nodeids = {1};
  regressor.target_ids = {1};
  regressor.target_weights = {1};
  EXPECT_THROW(ml_cpu_compute::EvaluateTreeEnsembleRegressor(regressor, {-1}, 1),
               std::invalid_argument);
}

} // namespace
