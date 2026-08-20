// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/reference/tree_ensemble_corpus.h"
#include "onnx_light_cpu/reference/tree_ensemble_reference.h"

#include "onnx_light_cpu/impl/execution.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using onnx_light_cpu::reference::ClassLabels;
using onnx_light_cpu::reference::GenerateTreeEnsembleV5Corpus;
using onnx_light_cpu::reference::LegacyTreeAttributes;
using onnx_light_cpu::reference::TreeAggregate;
using onnx_light_cpu::reference::TreeBranchMode;
using onnx_light_cpu::reference::TreeEnsembleAttributes;
using onnx_light_cpu::reference::TreeEnsembleCalibrationCandidate;
using onnx_light_cpu::reference::TreeEnsembleCalibrationMeasurement;
using onnx_light_cpu::reference::TreeEnsembleCalibrationOptions;
using onnx_light_cpu::reference::TreeEnsembleCalibrationStage;
using onnx_light_cpu::reference::TreeEnsembleClassifierAttributes;
using onnx_light_cpu::reference::TreeEnsembleExecutionStrategy;
using onnx_light_cpu::reference::TreeEnsembleNodeLayout;
using onnx_light_cpu::reference::TreeEnsemblePlan;
using onnx_light_cpu::reference::TreeEnsembleProfileSource;
using onnx_light_cpu::reference::TreeEnsembleReference;
using onnx_light_cpu::reference::TreeEnsembleRegressorAttributes;
using onnx_light_cpu::reference::TreeEnsembleTuningContext;
using onnx_light_cpu::reference::TreeEnsembleTuningPolicy;
using onnx_light_cpu::reference::TreeEnsembleTuningRegistry;
using onnx_light_cpu::reference::TreePostTransform;
using onnx_light_cpu::reference::TreeValueType;

TreeEnsembleAttributes Stump() {
  TreeEnsembleAttributes attributes;
  attributes.n_features = 1;
  attributes.n_targets = 1;
  attributes.tree_roots = {0};
  attributes.nodes_featureids = {0};
  attributes.nodes_splits = {0.0};
  attributes.nodes_modes = {TreeBranchMode::kLeq};
  attributes.nodes_truenodeids = {0};
  attributes.nodes_falsenodeids = {1};
  attributes.nodes_trueleafs = {1};
  attributes.nodes_falseleafs = {1};
  attributes.leaf_targetids = {0, 0};
  attributes.leaf_weights = {1.0, -1.0};
  return attributes;
}

TreeEnsembleAttributes StumpForest(std::size_t trees, std::size_t targets = 2) {
  TreeEnsembleAttributes attributes;
  attributes.n_features = 1;
  attributes.n_targets = static_cast<std::int64_t>(targets);
  attributes.value_type = TreeValueType::kFloat32;
  attributes.base_values.resize(targets, 0.5);
  for (std::size_t tree = 0; tree < trees; ++tree) {
    const std::int64_t node = static_cast<std::int64_t>(tree);
    const std::int64_t leaf = static_cast<std::int64_t>(2 * tree);
    attributes.tree_roots.push_back(node);
    attributes.nodes_featureids.push_back(0);
    attributes.nodes_splits.push_back(0.0);
    attributes.nodes_modes.push_back(TreeBranchMode::kLeq);
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

TreeEnsembleTuningPolicy OneRegionPolicy(TreeEnsembleExecutionStrategy strategy,
                                         std::size_t threads, std::size_t targets,
                                         std::size_t batch_rows = 1) {
  TreeEnsembleTuningPolicy policy;
  policy.regions.push_back(
      {std::nullopt, strategy, batch_rows, threads, 1, 1,
       threads * batch_rows * targets * (sizeof(double) + sizeof(std::size_t))});
  return policy;
}

struct ThreadedExecutor {
  std::atomic<std::size_t> dispatches{0};
  std::atomic<std::size_t> maximum_active{0};

  static void Run(void *context, int64_t num_blocks, void *task_context,
                  onnx_light_cpu::ExecutionBlockFn task) {
    auto &self = *static_cast<ThreadedExecutor *>(context);
    self.dispatches.fetch_add(1, std::memory_order_relaxed);
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

LegacyTreeAttributes LegacyStump() {
  LegacyTreeAttributes tree;
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

TEST(TreeEnsembleReference, CorpusCoversV5ContractAndIsDeterministic) {
  const auto cases = GenerateTreeEnsembleV5Corpus();
  ASSERT_GE(cases.size(), 36U);
  std::set<TreeBranchMode> modes;
  std::set<TreeAggregate> aggregates;
  std::set<TreePostTransform> transforms;
  std::set<TreeValueType> types;
  bool has_empty = false;
  bool has_multi_target = false;
  bool has_large_membership = false;
  bool has_depth_extreme = false;
  for (const auto &test_case : cases) {
    EXPECT_EQ(test_case.domain, "ai.onnx.ml");
    EXPECT_EQ(test_case.op_type, "TreeEnsemble");
    EXPECT_EQ(test_case.opset, 5);
    EXPECT_EQ(TreeEnsembleReference(test_case.attributes).Evaluate(test_case.input, test_case.rows),
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

TEST(TreeEnsembleReference, CanonicalPlanLowersAndEvaluatesDeterministically) {
  TreeEnsembleAttributes attributes = Stump();
  attributes.value_type = TreeValueType::kFloat32;
  const TreeEnsemblePlan plan(attributes);
  EXPECT_EQ(plan.model_signature().find("tree_ensemble_v5"), 0U);
  EXPECT_EQ(plan.tree_roots().size(), 1U);
  EXPECT_EQ(plan.nodes().size(), 1U);
  EXPECT_EQ(plan.leaves().size(), 2U);
  EXPECT_GT(plan.workspace_bytes(), 0U);
  EXPECT_EQ(plan.Evaluate({-1.0, 3.0}, 2), (std::vector<double>{1.0, -1.0}));
}

TEST(TreeEnsembleReference, CanonicalPlanAppliesBaseValues) {
  TreeEnsembleAttributes attributes = Stump();
  attributes.value_type = TreeValueType::kFloat32;
  attributes.base_values = {0.5};
  const TreeEnsemblePlan plan(attributes);
  EXPECT_EQ(plan.base_values(), (std::vector<double>{0.5}));
  EXPECT_EQ(plan.Evaluate({-1.0, 3.0}, 2), (std::vector<double>{1.5, -0.5}));
  EXPECT_EQ(TreeEnsembleReference(attributes).Evaluate({-1.0, 3.0}, 2),
            (std::vector<double>{1.5, -0.5}));
}

TEST(TreeEnsembleReference, SchedulingDecisionMatchesOrtCrossovers) {
  const TreeEnsemblePlan small_forest(StumpForest(79));
  EXPECT_EQ(small_forest.SelectExecution(1, 4).strategy,
            TreeEnsembleExecutionStrategy::kTreeMajorBatch);

  const TreeEnsemblePlan large_forest(StumpForest(81));
  EXPECT_EQ(large_forest.SelectExecution(1, 4).strategy,
            TreeEnsembleExecutionStrategy::kTreeParallel);
  EXPECT_EQ(large_forest.SelectExecution(50, 4).strategy,
            TreeEnsembleExecutionStrategy::kTreeMajorBatch);
  EXPECT_EQ(large_forest.SelectExecution(51, 4).strategy,
            TreeEnsembleExecutionStrategy::kTreeParallel);

  const TreeEnsemblePlan few_trees(StumpForest(3));
  EXPECT_EQ(few_trees.SelectExecution(51, 4).strategy, TreeEnsembleExecutionStrategy::kRowParallel);
  EXPECT_EQ(few_trees.SelectExecution(1000, 1).strategy,
            TreeEnsembleExecutionStrategy::kTreeMajorBatch);
  EXPECT_EQ(few_trees.SelectExecution(1000, 1).batch_rows, 128U);

  const TreeEnsemblePlan four_single_target_trees(StumpForest(4, 1));
  const TreeEnsemblePlan four_multi_target_trees(StumpForest(4, 2));
  EXPECT_EQ(four_single_target_trees.SelectExecution(51, 4).strategy,
            TreeEnsembleExecutionStrategy::kRowParallel);
  EXPECT_EQ(four_multi_target_trees.SelectExecution(51, 4).strategy,
            TreeEnsembleExecutionStrategy::kTreeParallel);
}

TEST(TreeEnsembleReference, SchedulingWorkspaceIsBoundedByActiveBatch) {
  const TreeEnsemblePlan plan(StumpForest(81));
  const auto one_row = plan.SelectExecution(1, 4);
  const auto many_rows = plan.SelectExecution(1000000, 4);
  const std::size_t accumulator_bytes = sizeof(double) + sizeof(std::size_t);
  EXPECT_EQ(one_row.workspace_bytes, 4U * 1U * 2U * accumulator_bytes);
  EXPECT_EQ(many_rows.batch_rows, 128U);
  EXPECT_EQ(many_rows.workspace_bytes, 4U * 128U * 2U * accumulator_bytes);
}

TEST(TreeEnsembleReference, PreparedPolicyCoversEveryInclusiveRowCrossover) {
  const TreeEnsemblePlan plan(StumpForest(81), TreeEnsembleTuningContext{"test-cpu", 4}, nullptr);
  const auto &regions = plan.tuning_policy().regions;
  ASSERT_EQ(regions.size(), 3U);
  EXPECT_EQ(regions[0].maximum_rows, 1U);
  EXPECT_EQ(regions[1].maximum_rows, 50U);
  EXPECT_FALSE(regions[2].maximum_rows.has_value());
  EXPECT_LE(regions.size(), 4U);

  EXPECT_EQ(plan.SelectExecution(0, 4).strategy, TreeEnsembleExecutionStrategy::kTreeMajorBatch);
  EXPECT_EQ(plan.SelectExecution(1, 4).strategy, TreeEnsembleExecutionStrategy::kTreeParallel);
  EXPECT_EQ(plan.SelectExecution(2, 4).strategy, TreeEnsembleExecutionStrategy::kTreeMajorBatch);
  EXPECT_EQ(plan.SelectExecution(50, 4).strategy, TreeEnsembleExecutionStrategy::kTreeMajorBatch);
  EXPECT_EQ(plan.SelectExecution(51, 4).strategy, TreeEnsembleExecutionStrategy::kTreeParallel);

  const TreeEnsemblePlan few_trees(StumpForest(3), TreeEnsembleTuningContext{"test-cpu", 4},
                                   nullptr);
  ASSERT_EQ(few_trees.tuning_policy().regions.size(), 2U);
  EXPECT_EQ(few_trees.SelectExecution(50, 4).strategy,
            TreeEnsembleExecutionStrategy::kTreeMajorBatch);
  EXPECT_EQ(few_trees.SelectExecution(51, 4).strategy, TreeEnsembleExecutionStrategy::kRowParallel);
}

TEST(TreeEnsembleReference, ModelKeyAndStructuralDigestAreStableAndExact) {
  TreeEnsembleAttributes first = StumpForest(4);
  TreeEnsembleAttributes value_only = first;
  value_only.nodes_splits[0] = 42.0;
  value_only.leaf_weights[0] = 99.0;
  TreeEnsembleAttributes structural_change = first;
  structural_change.nodes_missing_value_tracks_true.assign(4, 0);
  structural_change.nodes_missing_value_tracks_true[0] = 1;

  const TreeEnsemblePlan first_plan(first, TreeEnsembleTuningContext{"cpu+features", 3}, nullptr);
  const TreeEnsemblePlan value_plan(value_only, TreeEnsembleTuningContext{"cpu+features", 3},
                                    nullptr);
  const TreeEnsemblePlan changed_plan(structural_change,
                                      TreeEnsembleTuningContext{"cpu+features", 3}, nullptr);
  EXPECT_EQ(first_plan.model_signature(), value_plan.model_signature());
  EXPECT_NE(first_plan.model_signature(), changed_plan.model_signature());
  EXPECT_EQ(first_plan.model_signature().size(), std::string("tree_ensemble_v5:").size() + 16U);
  EXPECT_EQ(first_plan.model_key().processor, "cpu+features");
  EXPECT_EQ(first_plan.model_key().threads, 3U);
  EXPECT_EQ(first_plan.model_key().model_digest, first_plan.model_signature());
  EXPECT_EQ(first_plan.model_key().library, "onnx_light_cpu");
  EXPECT_EQ(first_plan.model_key().domain, "ai.onnx.ml");
  EXPECT_EQ(first_plan.model_key().opset, 5);
}

TEST(TreeEnsembleReference, ExactProfileOverridesPortableAndDoesNotLeakAcrossModels) {
  const TreeEnsembleTuningContext context{"test-cpu", 4};
  const TreeEnsemblePlan baseline(StumpForest(81), context, nullptr);
  TreeEnsembleTuningRegistry registry;
  registry.PutPortable(baseline.structural_buckets(),
                       OneRegionPolicy(TreeEnsembleExecutionStrategy::kRowParallel, 2, 2));
  registry.PutExact(baseline.model_key(),
                    OneRegionPolicy(TreeEnsembleExecutionStrategy::kTreeMajorBatch, 1, 2, 8));

  const TreeEnsemblePlan exact(StumpForest(81), context, &registry);
  EXPECT_EQ(exact.profile_source(), TreeEnsembleProfileSource::kExact);
  EXPECT_EQ(exact.SelectExecution(64, 4).strategy, TreeEnsembleExecutionStrategy::kTreeMajorBatch);

  TreeEnsembleAttributes incompatible = StumpForest(81);
  incompatible.nodes_missing_value_tracks_true.assign(81, 0);
  incompatible.nodes_missing_value_tracks_true[0] = 1;
  const TreeEnsemblePlan portable(std::move(incompatible), context, &registry);
  EXPECT_NE(portable.model_signature(), baseline.model_signature());
  EXPECT_EQ(portable.profile_source(), TreeEnsembleProfileSource::kPortable);
  EXPECT_EQ(portable.SelectExecution(64, 4).strategy, TreeEnsembleExecutionStrategy::kRowParallel);

  const TreeEnsemblePlan other_processor(StumpForest(81), TreeEnsembleTuningContext{"other-cpu", 4},
                                         &registry);
  EXPECT_NE(other_processor.profile_source(), TreeEnsembleProfileSource::kExact);
  const TreeEnsemblePlan other_threads(StumpForest(81), TreeEnsembleTuningContext{"test-cpu", 2},
                                       &registry);
  EXPECT_NE(other_threads.profile_source(), TreeEnsembleProfileSource::kExact);
}

TEST(TreeEnsembleReference, ProfileLifecycleIsCapturedAndHotPathOwnsItsPolicy) {
  const TreeEnsembleTuningContext context{"test-cpu", 4};
  std::unique_ptr<TreeEnsemblePlan> captured;
  std::uint64_t captured_generation = 0;
  {
    TreeEnsembleTuningRegistry registry;
    const TreeEnsemblePlan key_plan(StumpForest(3), context, nullptr);
    registry.PutExact(key_plan.model_key(),
                      OneRegionPolicy(TreeEnsembleExecutionStrategy::kRowParallel, 4, 2));
    captured = std::make_unique<TreeEnsemblePlan>(StumpForest(3), context, &registry);
    captured_generation = captured->profile_generation();
    registry.PutExact(key_plan.model_key(),
                      OneRegionPolicy(TreeEnsembleExecutionStrategy::kTreeMajorBatch, 1, 2, 16));
    const TreeEnsemblePlan updated(StumpForest(3), context, &registry);
    EXPECT_GT(updated.profile_generation(), captured_generation);
    EXPECT_EQ(updated.SelectExecution(64, 4).strategy,
              TreeEnsembleExecutionStrategy::kTreeMajorBatch);
    EXPECT_EQ(captured->SelectExecution(64, 4).strategy,
              TreeEnsembleExecutionStrategy::kRowParallel);
  }
  EXPECT_EQ(captured->profile_generation(), captured_generation);
  EXPECT_EQ(captured->SelectExecution(64, 4).strategy, TreeEnsembleExecutionStrategy::kRowParallel);
}

TEST(TreeEnsembleReference, InvalidOrIncompatiblePoliciesUseExplicitSafeFallback) {
  TreeEnsembleTuningRegistry registry;
  TreeEnsembleTuningPolicy unordered =
      OneRegionPolicy(TreeEnsembleExecutionStrategy::kRowParallel, 1, 1);
  unordered.regions.insert(unordered.regions.begin(), unordered.regions.front());
  EXPECT_THROW(registry.PutPortable({}, unordered), std::invalid_argument);

  const TreeEnsembleTuningContext context{"test-cpu", 4};
  const TreeEnsemblePlan baseline(StumpForest(3), context, nullptr);
  TreeEnsembleTuningPolicy insufficient =
      OneRegionPolicy(TreeEnsembleExecutionStrategy::kRowParallel, 4, 2);
  insufficient.regions[0].maximum_rows = 50;
  insufficient.regions.push_back(
      {std::nullopt, TreeEnsembleExecutionStrategy::kTreeParallel, 128, 4, 1, 1, 1});
  registry.PutExact(baseline.model_key(), insufficient);
  const TreeEnsemblePlan workspace_fallback(StumpForest(3), context, &registry);
  EXPECT_EQ(workspace_fallback.profile_source(), TreeEnsembleProfileSource::kSafeFallback);
  EXPECT_EQ(workspace_fallback.SelectExecution(51, 4).strategy,
            TreeEnsembleExecutionStrategy::kRowParallel);

  TreeEnsembleTuningPolicy unsupported =
      OneRegionPolicy(TreeEnsembleExecutionStrategy::kTreeParallel, 4, 2);
  unsupported.layout = TreeEnsembleNodeLayout::kSplitSoa;
  registry.PutExact(baseline.model_key(), unsupported);
  const TreeEnsemblePlan layout_fallback(StumpForest(3), context, &registry);
  EXPECT_EQ(layout_fallback.profile_source(), TreeEnsembleProfileSource::kSafeFallback);
  EXPECT_LE(layout_fallback.tuning_policy().regions.size(), 4U);
}

TEST(TreeEnsembleReference, CalibrationRejectsInvalidCandidatesAndPersistsEvidenceAtomically) {
  const TreeEnsembleTuningContext context{"calibration-cpu", 4};
  const TreeEnsemblePlan key_plan(StumpForest(3), context, nullptr);
  const TreeEnsembleTuningPolicy fallback =
      OneRegionPolicy(TreeEnsembleExecutionStrategy::kTreeMajorBatch, 1, 2);
  TreeEnsembleTuningPolicy oversized =
      OneRegionPolicy(TreeEnsembleExecutionStrategy::kRowParallel, 1, 2);
  oversized.regions[0].workspace_bytes = 4096;
  const TreeEnsembleTuningPolicy incorrect =
      OneRegionPolicy(TreeEnsembleExecutionStrategy::kTreeParallel, 1, 2);
  const TreeEnsembleTuningPolicy winner =
      OneRegionPolicy(TreeEnsembleExecutionStrategy::kRowParallel, 1, 2);
  const std::vector<TreeEnsembleCalibrationCandidate> candidates{
      {"oversized", TreeEnsembleCalibrationStage::kLayout, oversized},
      {"incorrect", TreeEnsembleCalibrationStage::kScheduling, incorrect},
      {"winner", TreeEnsembleCalibrationStage::kScheduling, winner},
  };
  const std::filesystem::path evidence_path =
      std::filesystem::temp_directory_path() / "onnx_light_cpu_tree_calibration.txt";
  std::filesystem::remove(evidence_path);

  TreeEnsembleCalibrationOptions options;
  options.duration_budget_ns = 1'000'000'000;
  options.memory_budget_bytes = 1024;
  options.warmup_runs = 2;
  options.repetitions = 3;
  options.required_wins = 2;
  options.minimum_improvement = 0.05;
  options.evidence_path = evidence_path.string();
  std::size_t calls = 0;
  const auto measure = [&](const TreeEnsembleTuningPolicy &policy, std::size_t warmups,
                           std::size_t repetitions) {
    ++calls;
    EXPECT_EQ(warmups, 2U);
    EXPECT_EQ(repetitions, 3U);
    TreeEnsembleCalibrationMeasurement result;
    result.elapsed_ns = 100;
    const auto strategy = policy.regions[0].strategy;
    result.correct = strategy != TreeEnsembleExecutionStrategy::kTreeParallel;
    result.samples_ns.assign(
        repetitions, strategy == TreeEnsembleExecutionStrategy::kRowParallel ? 80.0 : 100.0);
    if (!result.correct) {
      result.failure = "output mismatch";
    }
    return result;
  };

  TreeEnsembleTuningRegistry registry;
  const auto report =
      registry.CalibrateExact(key_plan.model_key(), fallback, candidates, options, measure);
  EXPECT_TRUE(report.changed);
  EXPECT_TRUE(report.persisted);
  EXPECT_FALSE(report.budget_exhausted);
  EXPECT_EQ(report.selected_policy, winner);
  EXPECT_EQ(calls, 6U);
  ASSERT_TRUE(std::filesystem::exists(evidence_path));
  EXPECT_FALSE(std::filesystem::exists(evidence_path.string() + ".tmp.0"));
  std::ifstream first_stream(evidence_path);
  const std::string first((std::istreambuf_iterator<char>(first_stream)),
                          std::istreambuf_iterator<char>());
  EXPECT_NE(first.find("output mismatch"), std::string::npos);
  EXPECT_NE(first.find("memory budget exceeded"), std::string::npos);
  EXPECT_NE(first.find("onnx_light_cpu_tree_calibration_v1"), std::string::npos);

  TreeEnsembleTuningRegistry repeated_registry;
  const auto repeated_report = repeated_registry.CalibrateExact(key_plan.model_key(), fallback,
                                                                candidates, options, measure);
  EXPECT_TRUE(repeated_report.persisted);
  std::ifstream second_stream(evidence_path);
  const std::string second((std::istreambuf_iterator<char>(second_stream)),
                           std::istreambuf_iterator<char>());
  EXPECT_EQ(second, first);

  const auto inspection = registry.InspectExact(key_plan.model_key());
  ASSERT_TRUE(inspection.selected_policy.has_value());
  EXPECT_EQ(*inspection.selected_policy, winner);
  EXPECT_EQ(inspection.rejected_reasons.size(), 2U);
  const TreeEnsemblePlan calibrated(StumpForest(3), context, &registry);
  EXPECT_EQ(calibrated.SelectExecution(64, 4).strategy,
            TreeEnsembleExecutionStrategy::kRowParallel);

  registry.OverrideExact(key_plan.model_key(), incorrect);
  EXPECT_TRUE(registry.InspectExact(key_plan.model_key()).override_policy.has_value());
  EXPECT_EQ(TreeEnsemblePlan(StumpForest(3), context, &registry).SelectExecution(64, 4).strategy,
            TreeEnsembleExecutionStrategy::kTreeParallel);
  registry.PutPortable(key_plan.structural_buckets(), fallback);
  registry.ForcePortable(key_plan.model_key(), true);
  EXPECT_TRUE(registry.InspectExact(key_plan.model_key()).force_portable);
  EXPECT_EQ(TreeEnsemblePlan(StumpForest(3), context, &registry).SelectExecution(64, 4).strategy,
            TreeEnsembleExecutionStrategy::kTreeMajorBatch);

  std::filesystem::remove(evidence_path);
}

TEST(TreeEnsembleReference, CalibrationBudgetFailurePreservesActiveProfileAndCanBeDisabled) {
  const TreeEnsembleTuningContext context{"calibration-cpu", 4};
  const TreeEnsemblePlan key_plan(StumpForest(3), context, nullptr);
  const TreeEnsembleTuningPolicy active =
      OneRegionPolicy(TreeEnsembleExecutionStrategy::kTreeParallel, 1, 2);
  const TreeEnsembleTuningPolicy candidate =
      OneRegionPolicy(TreeEnsembleExecutionStrategy::kRowParallel, 1, 2);
  TreeEnsembleTuningRegistry registry;
  registry.PutExact(key_plan.model_key(), active);
  TreeEnsembleCalibrationOptions options;
  options.duration_budget_ns = 10;
  options.repetitions = 1;
  options.required_wins = 1;
  const auto report = registry.CalibrateExact(
      key_plan.model_key(), active,
      {{"candidate", TreeEnsembleCalibrationStage::kScheduling, candidate}}, options,
      [](const TreeEnsembleTuningPolicy &, std::size_t, std::size_t) {
        return TreeEnsembleCalibrationMeasurement{true, {1.0}, 10, {}};
      });
  EXPECT_TRUE(report.budget_exhausted);
  EXPECT_FALSE(report.changed);
  EXPECT_EQ(*registry.InspectExact(key_plan.model_key()).selected_policy, active);

  registry.SetCalibrationEnabled(false);
  std::size_t calls = 0;
  const auto disabled = registry.CalibrateExact(
      key_plan.model_key(), active,
      {{"candidate", TreeEnsembleCalibrationStage::kScheduling, candidate}}, options,
      [&](const TreeEnsembleTuningPolicy &, std::size_t, std::size_t) {
        ++calls;
        return TreeEnsembleCalibrationMeasurement{};
      });
  EXPECT_EQ(calls, 0U);
  ASSERT_EQ(disabled.evidence.size(), 1U);
  EXPECT_EQ(disabled.evidence[0].rejected_reason, "calibration disabled");
  EXPECT_FALSE(registry.InspectExact(key_plan.model_key()).calibration_enabled);
}

TEST(TreeEnsembleReference, EverySchedulingStrategyMatchesScalarAcrossThreadCounts) {
  const TreeEnsembleAttributes attributes = StumpForest(81);
  const TreeEnsembleReference reference(attributes);
  const TreeEnsemblePlan plan(attributes);
  for (const std::size_t rows : {1U, 49U, 50U, 51U, 129U}) {
    std::vector<double> input(rows);
    for (std::size_t row = 0; row < rows; ++row) {
      input[row] = (row % 2 == 0) ? -1.0 : 1.0;
    }
    const std::vector<double> expected = reference.Evaluate(input, rows);
    for (const int64_t threads : {1, 2, 4}) {
      ThreadedExecutor executor;
      onnx_light_cpu::ExecutionExecutorView view{&executor, threads, &ThreadedExecutor::Run};
      onnx_light_cpu::ExecutionExecutorScope scope(&view);
      EXPECT_EQ(plan.Evaluate(input, rows), expected) << "rows=" << rows << ", threads=" << threads;
      EXPECT_LE(executor.maximum_active.load(std::memory_order_relaxed),
                static_cast<std::size_t>(threads));
    }
  }

  const TreeEnsembleAttributes row_attributes = StumpForest(3);
  const TreeEnsembleReference row_reference(row_attributes);
  const TreeEnsemblePlan row_plan(row_attributes);
  std::vector<double> input(51, -1.0);
  ThreadedExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 4, &ThreadedExecutor::Run};
  onnx_light_cpu::ExecutionExecutorScope scope(&view);
  EXPECT_EQ(row_plan.SelectExecution(input.size()).strategy,
            TreeEnsembleExecutionStrategy::kRowParallel);
  EXPECT_EQ(row_plan.Evaluate(input, input.size()), row_reference.Evaluate(input, input.size()));
}

TEST(TreeEnsembleReference, ThresholdsSignedZeroInfinityAndMissingRouting) {
  TreeEnsembleAttributes attributes = Stump();
  attributes.value_type = TreeValueType::kFloat64;
  attributes.nodes_splits = {-0.0};
  attributes.nodes_modes = {TreeBranchMode::kEq};
  attributes.nodes_missing_value_tracks_true = {1};
  const double infinity = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<double> actual = TreeEnsembleReference(std::move(attributes))
                                         .Evaluate({-0.0, 0.0, -infinity, infinity, nan}, 5);
  EXPECT_EQ(actual, (std::vector<double>{1.0, 1.0, -1.0, -1.0, 1.0}));
}

TEST(TreeEnsembleReference, Float16RoundsAtTieToEvenBoundary) {
  TreeEnsembleAttributes attributes = Stump();
  attributes.value_type = TreeValueType::kFloat16;
  attributes.nodes_splits = {1.00048828125};
  const std::vector<double> actual =
      TreeEnsembleReference(std::move(attributes)).Evaluate({1.0, 1.00048828125, 1.0009765625}, 3);
  EXPECT_EQ(actual, (std::vector<double>{1.0, 1.0, -1.0}));
}

TEST(TreeEnsembleReference, MembershipUsesNonEmptyNanDelimitedSets) {
  TreeEnsembleAttributes attributes = Stump();
  attributes.nodes_modes = {TreeBranchMode::kMember};
  attributes.membership_values = {-0.0, 2.0, 2.0, std::numeric_limits<double>::quiet_NaN()};
  const std::vector<double> actual =
      TreeEnsembleReference(std::move(attributes)).Evaluate({0.0, 2.0, 3.0}, 3);
  EXPECT_EQ(actual, (std::vector<double>{1.0, 1.0, -1.0}));
}

TEST(TreeEnsembleReference, AggregatesMultipleTargetsAndAppliesTransforms) {
  TreeEnsembleAttributes attributes;
  attributes.n_features = 1;
  attributes.n_targets = 2;
  attributes.tree_roots = {0, 1, 2, 3};
  attributes.nodes_featureids = {0, 0, 0, 0};
  attributes.nodes_splits = {0, 0, 0, 0};
  attributes.nodes_modes = {TreeBranchMode::kLeq, TreeBranchMode::kLeq, TreeBranchMode::kLeq,
                            TreeBranchMode::kLeq};
  attributes.nodes_truenodeids = {0, 1, 2, 3};
  attributes.nodes_falsenodeids = {0, 1, 2, 3};
  attributes.nodes_trueleafs = {1, 1, 1, 1};
  attributes.nodes_falseleafs = {1, 1, 1, 1};
  attributes.leaf_targetids = {0, 0, 1, 1};
  attributes.leaf_weights = {2.0, 4.0, 1.0, 3.0};
  attributes.aggregate = TreeAggregate::kAverage;
  attributes.post_transform = TreePostTransform::kSoftmax;
  const std::vector<double> actual =
      TreeEnsembleReference(std::move(attributes)).Evaluate({0.0}, 1);
  ASSERT_EQ(actual.size(), 2U);
  EXPECT_NEAR(actual[0], 0.6224593, 1e-6);
  EXPECT_NEAR(actual[1], 0.3775407, 1e-6);
}

TEST(TreeEnsembleReference, AggregatesMinMaxWithoutZeroBiasAndRetainsAverageTreeDivisor) {
  TreeEnsembleAttributes average;
  average.n_features = 1;
  average.n_targets = 2;
  average.aggregate = TreeAggregate::kAverage;
  average.tree_roots = {0, 1};
  average.nodes_featureids = {0, 0};
  average.nodes_splits = {0.0, 0.0};
  average.nodes_modes = {TreeBranchMode::kLeq, TreeBranchMode::kLeq};
  average.nodes_truenodeids = {0, 2};
  average.nodes_falsenodeids = {1, 3};
  average.nodes_trueleafs = {1, 1};
  average.nodes_falseleafs = {1, 1};
  average.leaf_targetids = {0, 0, 1, 1};
  average.leaf_weights = {6.0, 4.0, 2.0, 1.0};
  const std::vector<double> average_actual = TreeEnsembleReference(average).Evaluate({0.0}, 1);
  EXPECT_EQ(average_actual, (std::vector<double>{3.0, 1.0}));

  TreeEnsembleAttributes minimum = average;
  minimum.aggregate = TreeAggregate::kMin;
  minimum.leaf_weights = {5.0, 10.0, 3.0, 7.0};
  const std::vector<double> minimum_actual = TreeEnsembleReference(minimum).Evaluate({0.0}, 1);
  EXPECT_EQ(minimum_actual, (std::vector<double>{5.0, 3.0}));

  TreeEnsembleAttributes maximum = average;
  maximum.aggregate = TreeAggregate::kMax;
  maximum.leaf_weights = {-5.0, -10.0, -3.0, -7.0};
  const std::vector<double> maximum_actual = TreeEnsembleReference(maximum).Evaluate({0.0}, 1);
  EXPECT_EQ(maximum_actual, (std::vector<double>{-5.0, -3.0}));

  TreeEnsembleAttributes biased_min = minimum;
  biased_min.base_values = {10.0, 2.0};
  biased_min.leaf_weights = {5.0, 12.0, 7.0, 9.0};
  EXPECT_EQ(TreeEnsembleReference(biased_min).Evaluate({0.0}, 1), (std::vector<double>{15.0, 9.0}));

  const TreeEnsemblePlan minimum_plan(minimum);
  EXPECT_EQ(minimum_plan.Evaluate({0.0}, 1), (std::vector<double>{5.0, 3.0}));
  const TreeEnsemblePlan maximum_plan(maximum);
  EXPECT_EQ(maximum_plan.Evaluate({0.0}, 1), (std::vector<double>{-5.0, -3.0}));
}

TEST(TreeEnsembleReference, RejectsMalformedAttributeLengthsAndIndices) {
  TreeEnsembleAttributes attributes = Stump();
  attributes.nodes_splits.clear();
  EXPECT_THROW(TreeEnsembleReference(std::move(attributes)), std::invalid_argument);

  attributes = Stump();
  attributes.nodes_featureids[0] = 1;
  EXPECT_THROW(TreeEnsembleReference(std::move(attributes)), std::invalid_argument);

  attributes = Stump();
  attributes.nodes_truenodeids[0] = 2;
  EXPECT_THROW(TreeEnsembleReference(std::move(attributes)), std::invalid_argument);

  attributes = Stump();
  attributes.leaf_targetids[0] = 1;
  EXPECT_THROW(TreeEnsembleReference(std::move(attributes)), std::invalid_argument);

  attributes = Stump();
  attributes.tree_roots[0] = 1;
  EXPECT_THROW(TreeEnsembleReference(std::move(attributes)), std::invalid_argument);
}

TEST(TreeEnsembleReference, RejectsCyclesSharedAndUnreachableNodes) {
  TreeEnsembleAttributes attributes = Stump();
  attributes.nodes_featureids = {0, 0};
  attributes.nodes_splits = {0, 0};
  attributes.nodes_modes = {TreeBranchMode::kLeq, TreeBranchMode::kLeq};
  attributes.nodes_truenodeids = {1, 0};
  attributes.nodes_falsenodeids = {0, 1};
  attributes.nodes_trueleafs = {0, 0};
  attributes.nodes_falseleafs = {1, 1};
  EXPECT_THROW(TreeEnsembleReference{attributes}, std::invalid_argument);

  attributes.nodes_truenodeids = {1, 0};
  attributes.nodes_falsenodeids = {1, 1};
  attributes.nodes_trueleafs = {0, 1};
  attributes.nodes_falseleafs[0] = 0;
  EXPECT_THROW(TreeEnsembleReference{attributes}, std::invalid_argument);

  attributes = Stump();
  attributes.nodes_featureids.push_back(0);
  attributes.nodes_splits.push_back(0);
  attributes.nodes_modes.push_back(TreeBranchMode::kLeq);
  attributes.nodes_truenodeids.push_back(0);
  attributes.nodes_falsenodeids.push_back(1);
  attributes.nodes_trueleafs.push_back(1);
  attributes.nodes_falseleafs.push_back(1);
  EXPECT_THROW(TreeEnsembleReference(std::move(attributes)), std::invalid_argument);
}

TEST(TreeEnsembleReference, RejectsMalformedMembershipDelimiters) {
  TreeEnsembleAttributes attributes = Stump();
  attributes.nodes_modes = {TreeBranchMode::kMember};
  EXPECT_THROW(TreeEnsembleReference{attributes}, std::invalid_argument);

  attributes.membership_values = {std::numeric_limits<double>::quiet_NaN()};
  EXPECT_THROW(TreeEnsembleReference{attributes}, std::invalid_argument);

  attributes.membership_values = {1.0};
  EXPECT_THROW(TreeEnsembleReference{attributes}, std::invalid_argument);

  attributes.membership_values = {1.0, std::numeric_limits<double>::quiet_NaN(), 2.0};
  EXPECT_THROW(TreeEnsembleReference(std::move(attributes)), std::invalid_argument);
}

TEST(TreeEnsembleReference, DeprecatedRegressorV5AdapterHandlesIntegerInputs) {
  TreeEnsembleRegressorAttributes attributes;
  attributes.tree = LegacyStump();
  attributes.n_targets = 2;
  attributes.aggregate = TreeAggregate::kSum;
  attributes.post_transform = TreePostTransform::kLogistic;
  attributes.target_treeids = {0, 0, 0, 0};
  attributes.target_nodeids = {1, 1, 2, 2};
  attributes.target_ids = {0, 1, 0, 1};
  attributes.target_weights = {1.0, -1.0, 2.0, -2.0};
  attributes.base_values = {0.5, 0.5};
  const std::vector<float> actual =
      onnx_light_cpu::reference::EvaluateTreeEnsembleRegressor(attributes, {-1, 1}, 2);
  ASSERT_EQ(actual.size(), 4U);
  EXPECT_NEAR(actual[0], 0.8175745f, 1e-6f);
  EXPECT_NEAR(actual[1], 0.3775407f, 1e-6f);
  EXPECT_NEAR(actual[2], 0.9241418f, 1e-6f);
  EXPECT_NEAR(actual[3], 0.1824255f, 1e-6f);
}

TEST(TreeEnsembleReference, DeprecatedClassifierComposesLabelsAndUsesStableTies) {
  TreeEnsembleClassifierAttributes attributes;
  attributes.tree = LegacyStump();
  attributes.class_treeids = {0, 0, 0, 0};
  attributes.class_nodeids = {1, 1, 2, 2};
  attributes.class_ids = {0, 1, 0, 1};
  attributes.class_weights = {1.0, 1.0, 0.0, 2.0};
  attributes.labels = ClassLabels(std::vector<std::string>{"left", "right"});
  const auto actual =
      onnx_light_cpu::reference::EvaluateTreeEnsembleClassifier(attributes, {-1, 1}, 2);
  EXPECT_TRUE(actual.integer_labels.empty());
  EXPECT_EQ(actual.string_labels, (std::vector<std::string>{"left", "right"}));
  EXPECT_EQ(actual.scores, (std::vector<float>{1.0f, 1.0f, 0.0f, 2.0f}));
}

TEST(TreeEnsembleReference, DeprecatedAdaptersRejectLabelsAndTargetMetadata) {
  TreeEnsembleClassifierAttributes classifier;
  classifier.tree = LegacyStump();
  classifier.labels = ClassLabels(std::vector<std::int64_t>{});
  EXPECT_THROW(onnx_light_cpu::reference::EvaluateTreeEnsembleClassifier(classifier, {0}, 1),
               std::invalid_argument);

  TreeEnsembleRegressorAttributes regressor;
  regressor.tree = LegacyStump();
  regressor.n_targets = 1;
  regressor.target_treeids = {0};
  regressor.target_nodeids = {1};
  regressor.target_ids = {1};
  regressor.target_weights = {1};
  EXPECT_THROW(onnx_light_cpu::reference::EvaluateTreeEnsembleRegressor(regressor, {-1}, 1),
               std::invalid_argument);
}

} // namespace
