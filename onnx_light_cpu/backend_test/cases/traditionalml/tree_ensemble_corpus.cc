// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/traditionalml/tree_ensemble_corpus.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace onnx_light_cpu::backend_test {
namespace {

TreeEnsembleAttributes MakeStump(TreeBranchMode mode, DataType type) {
  TreeEnsembleAttributes attributes;
  attributes.n_features = 1;
  attributes.n_targets = 1;
  attributes.value_type = type;
  attributes.tree_roots = {0};
  attributes.nodes_featureids = {0};
  attributes.nodes_splits = {0.0};
  attributes.nodes_modes = {mode};
  attributes.nodes_truenodeids = {0};
  attributes.nodes_falsenodeids = {1};
  attributes.nodes_trueleafs = {1};
  attributes.nodes_falseleafs = {1};
  attributes.nodes_missing_value_tracks_true = {0};
  attributes.leaf_targetids = {0, 0};
  attributes.leaf_weights = {1.0, -1.0};
  if (mode == TreeBranchMode::kMember) {
    attributes.membership_values = {-1.0, 0.0, 0.0, std::numeric_limits<double>::quiet_NaN()};
  }
  return attributes;
}

void AddCase(std::vector<TreeEnsembleCorpusCase> &cases, std::string name,
             TreeEnsembleAttributes attributes, std::vector<double> input, std::size_t rows) {
  TreeEnsembleOracle oracle(attributes);
  std::vector<double> expected = oracle.Evaluate(input, rows);
  cases.push_back({std::move(name), "ai.onnx.ml", "TreeEnsemble", 5, std::move(attributes),
                   std::move(input), rows, std::move(expected)});
}

const char *ModeName(TreeBranchMode mode) {
  switch (mode) {
  case TreeBranchMode::kLeq:
    return "leq";
  case TreeBranchMode::kLt:
    return "lt";
  case TreeBranchMode::kGte:
    return "gte";
  case TreeBranchMode::kGt:
    return "gt";
  case TreeBranchMode::kEq:
    return "eq";
  case TreeBranchMode::kNeq:
    return "neq";
  case TreeBranchMode::kMember:
    return "member";
  }
  return "unknown";
}

const char *AggregateName(TreeAggregate aggregate) {
  switch (aggregate) {
  case TreeAggregate::kAverage:
    return "average";
  case TreeAggregate::kSum:
    return "sum";
  case TreeAggregate::kMin:
    return "min";
  case TreeAggregate::kMax:
    return "max";
  }
  return "unknown";
}

const char *TransformName(TreePostTransform transform) {
  switch (transform) {
  case TreePostTransform::kNone:
    return "none";
  case TreePostTransform::kSoftmax:
    return "softmax";
  case TreePostTransform::kLogistic:
    return "logistic";
  case TreePostTransform::kSoftmaxZero:
    return "softmax_zero";
  case TreePostTransform::kProbit:
    return "probit";
  }
  return "unknown";
}

TreeEnsembleAttributes MakeAggregateModel(TreeAggregate aggregate, TreePostTransform transform) {
  TreeEnsembleAttributes attributes;
  attributes.n_features = 1;
  attributes.n_targets = 2;
  attributes.aggregate = aggregate;
  attributes.post_transform = transform;
  attributes.tree_roots = {0, 1};
  attributes.nodes_featureids = {0, 0};
  attributes.nodes_splits = {0.0, 1.0};
  attributes.nodes_modes = {TreeBranchMode::kLeq, TreeBranchMode::kGt};
  attributes.nodes_truenodeids = {0, 2};
  attributes.nodes_falsenodeids = {1, 3};
  attributes.nodes_trueleafs = {1, 1};
  attributes.nodes_falseleafs = {1, 1};
  attributes.leaf_targetids = {0, 0, 1, 1};
  if (transform == TreePostTransform::kProbit) {
    attributes.leaf_weights = {0.1, 0.2, 0.8, 0.7};
  } else {
    attributes.leaf_weights = {1.0, -2.0, 3.0, -4.0};
  }
  return attributes;
}

TreeEnsembleAttributes MakeDeepModel() {
  constexpr std::size_t kDepth = 64;
  TreeEnsembleAttributes attributes;
  attributes.n_features = 4096;
  attributes.n_targets = 1;
  attributes.tree_roots = {0};
  for (std::size_t index = 0; index < kDepth; ++index) {
    attributes.nodes_featureids.push_back(static_cast<std::int64_t>(index));
    attributes.nodes_splits.push_back(1.0);
    attributes.nodes_modes.push_back(TreeBranchMode::kLeq);
    attributes.nodes_truenodeids.push_back(index + 1 == kDepth
                                               ? static_cast<std::int64_t>(kDepth)
                                               : static_cast<std::int64_t>(index + 1));
    attributes.nodes_falsenodeids.push_back(static_cast<std::int64_t>(index));
    attributes.nodes_trueleafs.push_back(index + 1 == kDepth ? 1 : 0);
    attributes.nodes_falseleafs.push_back(1);
    attributes.leaf_targetids.push_back(0);
    attributes.leaf_weights.push_back(-static_cast<double>(index));
  }
  attributes.leaf_targetids.push_back(0);
  attributes.leaf_weights.push_back(64.0);
  return attributes;
}

} // namespace

std::vector<TreeEnsembleCorpusCase> GenerateTreeEnsembleV5Corpus() {
  std::vector<TreeEnsembleCorpusCase> cases;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();

  for (TreeBranchMode mode :
       {TreeBranchMode::kLeq, TreeBranchMode::kLt, TreeBranchMode::kGte, TreeBranchMode::kGt,
        TreeBranchMode::kEq, TreeBranchMode::kNeq, TreeBranchMode::kMember}) {
    TreeEnsembleAttributes attributes = MakeStump(mode, DataType::DOUBLE);
    attributes.nodes_missing_value_tracks_true = {
        static_cast<std::int64_t>(static_cast<std::uint8_t>(mode) % 2)};
    AddCase(cases, std::string("mode_") + ModeName(mode), std::move(attributes),
            {-infinity, -1.0, -0.0, 0.0, 1.0, infinity, nan}, 7);
  }

  for (DataType type : {DataType::FLOAT16, DataType::FLOAT, DataType::DOUBLE}) {
    TreeEnsembleAttributes attributes = MakeStump(TreeBranchMode::kLeq, type);
    attributes.nodes_splits = {1.00048828125};
    const char *name = type == DataType::FLOAT16
                           ? "float16_boundary"
                           : (type == DataType::FLOAT ? "float32_boundary" : "float64_boundary");
    AddCase(cases, name, std::move(attributes),
            {1.0, 1.00048828125, std::nextafter(1.00048828125, 2.0)}, 3);
  }

  for (TreeAggregate aggregate :
       {TreeAggregate::kAverage, TreeAggregate::kSum, TreeAggregate::kMin, TreeAggregate::kMax}) {
    for (TreePostTransform transform :
         {TreePostTransform::kNone, TreePostTransform::kSoftmax, TreePostTransform::kLogistic,
          TreePostTransform::kSoftmaxZero, TreePostTransform::kProbit}) {
      AddCase(cases,
              std::string("aggregate_") + AggregateName(aggregate) + "_" + TransformName(transform),
              MakeAggregateModel(aggregate, transform), {-1.0, 0.0, 1.0, 2.0}, 4);
    }
  }

  TreeEnsembleAttributes missing_false = MakeStump(TreeBranchMode::kNeq, DataType::FLOAT);
  missing_false.nodes_missing_value_tracks_true = {0};
  AddCase(cases, "missing_tracks_false", std::move(missing_false), {nan}, 1);
  TreeEnsembleAttributes missing_true = MakeStump(TreeBranchMode::kEq, DataType::FLOAT);
  missing_true.nodes_missing_value_tracks_true = {1};
  AddCase(cases, "missing_tracks_true", std::move(missing_true), {nan}, 1);

  TreeEnsembleAttributes large_members = MakeStump(TreeBranchMode::kMember, DataType::FLOAT);
  large_members.membership_values.clear();
  for (int value = 0; value < 1024; ++value) {
    large_members.membership_values.push_back(static_cast<double>(value));
  }
  large_members.membership_values.push_back(nan);
  AddCase(cases, "membership_large_duplicate_hit_miss", std::move(large_members),
          {0.0, 1023.0, 1024.0}, 3);

  TreeEnsembleAttributes multi_target =
      MakeAggregateModel(TreeAggregate::kSum, TreePostTransform::kNone);
  AddCase(cases, "multiple_roots_targets", std::move(multi_target), {-1.0, 2.0}, 2);

  TreeEnsembleAttributes empty = MakeStump(TreeBranchMode::kLeq, DataType::FLOAT);
  AddCase(cases, "empty_batch", std::move(empty), {}, 0);

  std::vector<double> deep_input(4096, 0.0);
  AddCase(cases, "depth_64_features_4096", MakeDeepModel(), std::move(deep_input), 1);
  return cases;
}

} // namespace onnx_light_cpu::backend_test
