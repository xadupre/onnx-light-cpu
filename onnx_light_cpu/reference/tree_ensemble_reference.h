// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
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
};

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

/// Version-5 deprecated schema adapters backed by the same scalar semantics.
std::vector<float> EvaluateTreeEnsembleRegressor(const TreeEnsembleRegressorAttributes &attributes,
                                                 const std::vector<double> &input,
                                                 std::size_t rows);

TreeClassifierResult
EvaluateTreeEnsembleClassifier(const TreeEnsembleClassifierAttributes &attributes,
                               const std::vector<double> &input, std::size_t rows);

} // namespace onnx_light_cpu::reference
