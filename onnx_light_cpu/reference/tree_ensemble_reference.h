// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace onnx_light_cpu::reference {

enum class TreeBranchMode : std::uint8_t {
  kLeq = 0,
  kLt = 1,
  kGte = 2,
  kGt = 3,
  kEq = 4,
  kNeq = 5,
  kMember = 6,
};

enum class TreeAggregate : std::int64_t {
  kAverage = 0,
  kSum = 1,
  kMin = 2,
  kMax = 3,
};

enum class TreePostTransform : std::int64_t {
  kNone = 0,
  kSoftmax = 1,
  kLogistic = 2,
  kSoftmaxZero = 3,
  kProbit = 4,
};

enum class TreeValueType {
  kFloat16,
  kFloat32,
  kFloat64,
};

enum class TreeEnsembleExecutionStrategy {
  kRowParallel,
  kTreeParallel,
  kTreeMajorBatch,
};

enum class TreeEnsembleNodeLayout {
  kOrtCompactAosPointer,
  kCompactAosIndex,
  kSplitSoa,
};

enum class TreeEnsembleCountBucket : std::uint8_t {
  kOne,
  kTwoToFour,
  kFiveToSixteen,
  kSeventeenToEighty,
  kEightyOneToTwoHundredFiftySix,
  kMoreThanTwoHundredFiftySix,
};

enum class TreeEnsembleDepthBucket : std::uint8_t {
  kStump,
  kTwoToFour,
  kFiveToEight,
  kNineToSixteen,
  kMoreThanSixteen,
};

enum class TreeEnsembleBranchMix : std::uint8_t {
  kHomogeneous,
  kMixed,
};

enum class TreeEnsembleMembershipDensity : std::uint8_t {
  kNone,
  kSparse,
  kDense,
};

struct TreeEnsembleStructuralBuckets {
  TreeEnsembleCountBucket tree_count = TreeEnsembleCountBucket::kOne;
  TreeEnsembleDepthBucket depth = TreeEnsembleDepthBucket::kStump;
  TreeEnsembleCountBucket target_count = TreeEnsembleCountBucket::kOne;
  TreeEnsembleBranchMix branch_mode_mix = TreeEnsembleBranchMix::kHomogeneous;
  TreeEnsembleMembershipDensity membership_density = TreeEnsembleMembershipDensity::kNone;

  bool operator==(const TreeEnsembleStructuralBuckets &) const = default;
};

struct TreeEnsembleModelKey {
  std::string library = "onnx_light_cpu";
  std::string kernel = "TreeEnsemble";
  std::string domain = "ai.onnx.ml";
  std::int64_t opset = 5;
  std::string implementation = "prepared_tree_ensemble";
  TreeValueType input_type = TreeValueType::kFloat32;
  TreeValueType accumulator_type = TreeValueType::kFloat64;
  std::string processor = "portable";
  std::size_t threads = 1;
  std::string model_digest;

  bool operator==(const TreeEnsembleModelKey &) const = default;
};

struct TreeEnsembleExecutionRegion {
  std::optional<std::size_t> maximum_rows;
  TreeEnsembleExecutionStrategy strategy = TreeEnsembleExecutionStrategy::kTreeMajorBatch;
  std::size_t batch_rows = 1;
  std::size_t maximum_threads = 1;
  std::size_t row_chunk = 1;
  std::size_t tree_chunk = 1;
  std::size_t workspace_bytes = 0;
};

struct TreeEnsembleTuningPolicy {
  TreeEnsembleNodeLayout layout = TreeEnsembleNodeLayout::kOrtCompactAosPointer;
  std::vector<TreeEnsembleExecutionRegion> regions;
  std::size_t membership_linear_limit = 8;
  std::size_t membership_bitset_range_limit = 256;
  std::size_t traversal_prefetch_distance = 0;
};

enum class TreeEnsembleProfileSource : std::uint8_t {
  kSafeFallback,
  kPortable,
  kExact,
};

struct TreeEnsembleTuningContext {
  std::string processor = "portable";
  std::size_t threads = 0;
};

struct TreeEnsembleExecutionDecision {
  TreeEnsembleExecutionStrategy strategy = TreeEnsembleExecutionStrategy::kTreeMajorBatch;
  std::size_t batch_rows = 0;
  std::size_t participants = 1;
  std::size_t row_chunk = 1;
  std::size_t tree_chunk = 1;
  std::size_t workspace_bytes = 0;
};

/// Thread-safe profile store. Plans resolve it once during construction and
/// capture the selected policy and generation.
class TreeEnsembleTuningRegistry {
public:
  void PutExact(TreeEnsembleModelKey key, TreeEnsembleTuningPolicy policy);
  void PutPortable(TreeEnsembleStructuralBuckets buckets, TreeEnsembleTuningPolicy policy);

  std::uint64_t generation() const noexcept;

private:
  friend class TreeEnsemblePlan;
  struct ExactEntry {
    TreeEnsembleModelKey key;
    TreeEnsembleTuningPolicy policy;
  };
  struct PortableEntry {
    TreeEnsembleStructuralBuckets buckets;
    TreeEnsembleTuningPolicy policy;
  };

  mutable std::mutex mutex_;
  std::vector<ExactEntry> exact_;
  std::vector<PortableEntry> portable_;
  std::uint64_t generation_ = 0;
};

struct TreeEnsembleRegressorAttributes;
struct TreeEnsembleClassifierAttributes;

/// Direct representation of the ai.onnx.ml TreeEnsemble-5 attributes.
struct TreeEnsembleAttributes {
  std::int64_t n_features = 0;
  std::int64_t n_targets = 0;
  TreeValueType value_type = TreeValueType::kFloat32;
  TreeAggregate aggregate = TreeAggregate::kSum;
  TreePostTransform post_transform = TreePostTransform::kNone;

  std::vector<std::int64_t> tree_roots;
  std::vector<std::int64_t> nodes_featureids;
  std::vector<double> nodes_splits;
  std::vector<TreeBranchMode> nodes_modes;
  std::vector<std::int64_t> nodes_truenodeids;
  std::vector<std::int64_t> nodes_falsenodeids;
  std::vector<std::int64_t> nodes_trueleafs;
  std::vector<std::int64_t> nodes_falseleafs;
  std::vector<std::int64_t> nodes_missing_value_tracks_true;
  std::vector<double> membership_values;
  std::vector<std::int64_t> leaf_targetids;
  std::vector<double> leaf_weights;
  std::vector<double> base_values;
};

/// Lower a deprecated TreeEnsembleRegressor-5 graph into the canonical
/// TreeEnsemble-5 plan representation. The returned values are validated by the
/// same structural checks used by TreeEnsembleReference.
TreeEnsembleAttributes
LowerTreeEnsembleRegressor(const TreeEnsembleRegressorAttributes &attributes);

/// Lower a deprecated TreeEnsembleClassifier-5 graph into the canonical
/// TreeEnsemble-5 plan representation.
TreeEnsembleAttributes
LowerTreeEnsembleClassifier(const TreeEnsembleClassifierAttributes &attributes);

/// Scalar, allocation-free-per-path oracle for TreeEnsemble-5.
///
/// Construction validates the complete graph. Evaluation accepts values as
/// doubles and rounds inputs, attributes, intermediates, and outputs according
/// to value_type, which also permits exact float16 corpus generation without a
/// compiler-specific half type.
class TreeEnsembleReference {
public:
  explicit TreeEnsembleReference(TreeEnsembleAttributes attributes);

  const TreeEnsembleAttributes &attributes() const noexcept { return attributes_; }

  std::vector<double> Evaluate(const std::vector<double> &input, std::size_t rows) const;

private:
  TreeEnsembleAttributes attributes_;
  std::vector<std::vector<double>> membership_sets_;
};

/// Attribute representation shared by the deprecated version-5 classifier and
/// regressor schemas.
struct LegacyTreeAttributes {
  std::int64_t n_features = 0;
  std::vector<std::int64_t> nodes_treeids;
  std::vector<std::int64_t> nodes_nodeids;
  std::vector<std::int64_t> nodes_featureids;
  std::vector<double> nodes_values;
  std::vector<std::string> nodes_modes;
  std::vector<std::int64_t> nodes_truenodeids;
  std::vector<std::int64_t> nodes_falsenodeids;
  std::vector<std::int64_t> nodes_missing_value_tracks_true;
};

struct TreeEnsembleRegressorAttributes {
  LegacyTreeAttributes tree;
  std::int64_t n_targets = 0;
  TreeAggregate aggregate = TreeAggregate::kSum;
  TreePostTransform post_transform = TreePostTransform::kNone;
  std::vector<std::int64_t> target_treeids;
  std::vector<std::int64_t> target_nodeids;
  std::vector<std::int64_t> target_ids;
  std::vector<double> target_weights;
  std::vector<double> base_values;
};

using ClassLabels = std::variant<std::vector<std::int64_t>, std::vector<std::string>>;

struct TreeEnsembleClassifierAttributes {
  LegacyTreeAttributes tree;
  std::vector<std::int64_t> class_treeids;
  std::vector<std::int64_t> class_nodeids;
  std::vector<std::int64_t> class_ids;
  std::vector<double> class_weights;
  ClassLabels labels;
  TreePostTransform post_transform = TreePostTransform::kNone;
  std::vector<double> base_values;
};

struct TreeClassifierResult {
  std::vector<std::int64_t> integer_labels;
  std::vector<std::string> string_labels;
  std::vector<float> scores;
};

/// Canonical node layout used by a prepared TreeEnsemble plan.
struct TreeEnsembleNode {
  std::int64_t feature_id = 0;
  double split = 0.0;
  TreeBranchMode mode = TreeBranchMode::kLeq;
  std::int64_t true_child = 0;
  std::int64_t false_child = 0;
  bool true_is_leaf = false;
  bool false_is_leaf = false;
  bool missing_value_tracks_true = false;
  std::vector<double> members;
  std::vector<std::size_t> true_leaf_indices;
  std::vector<std::size_t> false_leaf_indices;
};

/// Canonical leaf entry used by a prepared TreeEnsemble plan.
struct TreeEnsembleLeaf {
  std::int64_t target_id = 0;
  double weight = 0.0;
};

/// Immutable prepared representation for the ai.onnx.ml TreeEnsemble-5 schema.
class TreeEnsemblePlan {
public:
  explicit TreeEnsemblePlan(TreeEnsembleAttributes attributes);
  TreeEnsemblePlan(TreeEnsembleAttributes attributes, TreeEnsembleTuningContext context,
                   const TreeEnsembleTuningRegistry *registry);
  explicit TreeEnsemblePlan(const TreeEnsembleRegressorAttributes &attributes);
  explicit TreeEnsemblePlan(const TreeEnsembleClassifierAttributes &attributes);

  std::vector<double> Evaluate(const std::vector<double> &input, std::size_t rows) const;
  TreeEnsembleExecutionDecision SelectExecution(std::size_t rows,
                                                std::size_t effective_threads = 0) const noexcept;

  const TreeEnsembleAttributes &attributes() const noexcept { return attributes_; }
  const std::vector<std::int64_t> &tree_roots() const noexcept { return tree_roots_; }
  const std::vector<TreeEnsembleNode> &nodes() const noexcept { return nodes_; }
  const std::vector<TreeEnsembleLeaf> &leaves() const noexcept { return leaves_; }
  const std::vector<double> &base_values() const noexcept { return base_values_; }
  const std::vector<std::vector<double>> &membership_sets() const noexcept {
    return membership_sets_;
  }
  const std::vector<std::size_t> &true_leaf_indices(std::size_t node_index) const noexcept {
    return nodes_[node_index].true_leaf_indices;
  }
  const std::vector<std::size_t> &false_leaf_indices(std::size_t node_index) const noexcept {
    return nodes_[node_index].false_leaf_indices;
  }
  std::size_t max_depth() const noexcept { return max_depth_; }
  std::size_t average_depth() const noexcept { return average_depth_; }
  const std::string &model_signature() const noexcept { return model_signature_; }
  const TreeEnsembleModelKey &model_key() const noexcept { return model_key_; }
  const TreeEnsembleStructuralBuckets &structural_buckets() const noexcept {
    return structural_buckets_;
  }
  const TreeEnsembleTuningPolicy &tuning_policy() const noexcept { return tuning_policy_; }
  TreeEnsembleProfileSource profile_source() const noexcept { return profile_source_; }
  std::uint64_t profile_generation() const noexcept { return profile_generation_; }
  std::size_t workspace_bytes() const noexcept { return workspace_bytes_; }
  bool uses_64bit_indices() const noexcept { return uses_64_bit_indices_; }

private:
  static std::string MakeModelSignature(const TreeEnsembleAttributes &attributes,
                                        const TreeEnsembleStructuralBuckets &buckets);

  TreeEnsembleAttributes attributes_;
  std::vector<std::int64_t> tree_roots_;
  std::vector<TreeEnsembleNode> nodes_;
  std::vector<TreeEnsembleLeaf> leaves_;
  std::vector<double> base_values_;
  std::vector<std::vector<double>> membership_sets_;
  std::size_t max_depth_ = 0;
  std::size_t average_depth_ = 0;
  std::string model_signature_;
  TreeEnsembleModelKey model_key_;
  TreeEnsembleStructuralBuckets structural_buckets_;
  TreeEnsembleTuningPolicy tuning_policy_;
  TreeEnsembleProfileSource profile_source_ = TreeEnsembleProfileSource::kSafeFallback;
  std::uint64_t profile_generation_ = 0;
  std::size_t workspace_bytes_ = 0;
  bool uses_64_bit_indices_ = false;
  bool uses_dynamic_safe_policy_ = false;
};

/// Version-5 deprecated schema adapters backed by the same scalar semantics.
std::vector<float> EvaluateTreeEnsembleRegressor(const TreeEnsembleRegressorAttributes &attributes,
                                                 const std::vector<double> &input,
                                                 std::size_t rows);

TreeClassifierResult
EvaluateTreeEnsembleClassifier(const TreeEnsembleClassifierAttributes &attributes,
                               const std::vector<double> &input, std::size_t rows);

} // namespace onnx_light_cpu::reference
