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
using ml_cpu_compute::DataType;

namespace {

ml_cpu_compute::TreeEnsembleAttributes MakeStump(ml_cpu_compute::TreeBranchMode mode,
                                                 DataType type) {
  ml_cpu_compute::TreeEnsembleAttributes attributes;
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
  if (mode == ml_cpu_compute::TreeBranchMode::kMember) {
    attributes.membership_values = {-1.0, 0.0, 0.0, std::numeric_limits<double>::quiet_NaN()};
  }
  return attributes;
}

void AddCase(std::vector<TreeEnsembleCorpusCase> &cases, std::string name,
             ml_cpu_compute::TreeEnsembleAttributes attributes, std::vector<double> input,
             std::size_t rows) {
  ml_cpu_compute::TreeEnsembleOracle oracle(attributes);
  std::vector<double> expected = oracle.Evaluate(input, rows);
  cases.push_back({std::move(name), "ai.onnx.ml", "TreeEnsemble", 5, std::move(attributes),
                   std::move(input), rows, std::move(expected)});
}

const char *ModeName(ml_cpu_compute::TreeBranchMode mode) {
  switch (mode) {
  case ml_cpu_compute::TreeBranchMode::kLeq:
    return "leq";
  case ml_cpu_compute::TreeBranchMode::kLt:
    return "lt";
  case ml_cpu_compute::TreeBranchMode::kGte:
    return "gte";
  case ml_cpu_compute::TreeBranchMode::kGt:
    return "gt";
  case ml_cpu_compute::TreeBranchMode::kEq:
    return "eq";
  case ml_cpu_compute::TreeBranchMode::kNeq:
    return "neq";
  case ml_cpu_compute::TreeBranchMode::kMember:
    return "member";
  }
  return "unknown";
}

const char *AggregateName(ml_cpu_compute::TreeAggregate aggregate) {
  switch (aggregate) {
  case ml_cpu_compute::TreeAggregate::kAverage:
    return "average";
  case ml_cpu_compute::TreeAggregate::kSum:
    return "sum";
  case ml_cpu_compute::TreeAggregate::kMin:
    return "min";
  case ml_cpu_compute::TreeAggregate::kMax:
    return "max";
  }
  return "unknown";
}

const char *TransformName(ml_cpu_compute::TreePostTransform transform) {
  switch (transform) {
  case ml_cpu_compute::TreePostTransform::kNone:
    return "none";
  case ml_cpu_compute::TreePostTransform::kSoftmax:
    return "softmax";
  case ml_cpu_compute::TreePostTransform::kLogistic:
    return "logistic";
  case ml_cpu_compute::TreePostTransform::kSoftmaxZero:
    return "softmax_zero";
  case ml_cpu_compute::TreePostTransform::kProbit:
    return "probit";
  }
  return "unknown";
}

ml_cpu_compute::TreeEnsembleAttributes
MakeAggregateModel(ml_cpu_compute::TreeAggregate aggregate,
                   ml_cpu_compute::TreePostTransform transform) {
  ml_cpu_compute::TreeEnsembleAttributes attributes;
  attributes.n_features = 1;
  attributes.n_targets = 2;
  attributes.aggregate = aggregate;
  attributes.post_transform = transform;
  attributes.tree_roots = {0, 1};
  attributes.nodes_featureids = {0, 0};
  attributes.nodes_splits = {0.0, 1.0};
  attributes.nodes_modes = {ml_cpu_compute::TreeBranchMode::kLeq,
                            ml_cpu_compute::TreeBranchMode::kGt};
  attributes.nodes_truenodeids = {0, 2};
  attributes.nodes_falsenodeids = {1, 3};
  attributes.nodes_trueleafs = {1, 1};
  attributes.nodes_falseleafs = {1, 1};
  attributes.leaf_targetids = {0, 0, 1, 1};
  if (transform == ml_cpu_compute::TreePostTransform::kProbit) {
    attributes.leaf_weights = {0.1, 0.2, 0.8, 0.7};
  } else {
    attributes.leaf_weights = {1.0, -2.0, 3.0, -4.0};
  }
  return attributes;
}

ml_cpu_compute::TreeEnsembleAttributes MakeDeepModel() {
  constexpr std::size_t kDepth = 64;
  ml_cpu_compute::TreeEnsembleAttributes attributes;
  attributes.n_features = 4096;
  attributes.n_targets = 1;
  attributes.tree_roots = {0};
  for (std::size_t index = 0; index < kDepth; ++index) {
    attributes.nodes_featureids.push_back(static_cast<std::int64_t>(index));
    attributes.nodes_splits.push_back(1.0);
    attributes.nodes_modes.push_back(ml_cpu_compute::TreeBranchMode::kLeq);
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

  for (ml_cpu_compute::TreeBranchMode mode :
       {ml_cpu_compute::TreeBranchMode::kLeq, ml_cpu_compute::TreeBranchMode::kLt,
        ml_cpu_compute::TreeBranchMode::kGte, ml_cpu_compute::TreeBranchMode::kGt,
        ml_cpu_compute::TreeBranchMode::kEq, ml_cpu_compute::TreeBranchMode::kNeq,
        ml_cpu_compute::TreeBranchMode::kMember}) {
    ml_cpu_compute::TreeEnsembleAttributes attributes = MakeStump(mode, DataType::DOUBLE);
    attributes.nodes_missing_value_tracks_true = {
        static_cast<std::int64_t>(static_cast<std::uint8_t>(mode) % 2)};
    AddCase(cases, std::string("mode_") + ModeName(mode), std::move(attributes),
            {-infinity, -1.0, -0.0, 0.0, 1.0, infinity, nan}, 7);
  }

  for (DataType type : {DataType::FLOAT16, DataType::FLOAT, DataType::DOUBLE}) {
    ml_cpu_compute::TreeEnsembleAttributes attributes =
        MakeStump(ml_cpu_compute::TreeBranchMode::kLeq, type);
    attributes.nodes_splits = {1.00048828125};
    const char *name = type == DataType::FLOAT16
                           ? "float16_boundary"
                           : (type == DataType::FLOAT ? "float32_boundary" : "float64_boundary");
    AddCase(cases, name, std::move(attributes),
            {1.0, 1.00048828125, std::nextafter(1.00048828125, 2.0)}, 3);
  }

  for (ml_cpu_compute::TreeAggregate aggregate :
       {ml_cpu_compute::TreeAggregate::kAverage, ml_cpu_compute::TreeAggregate::kSum,
        ml_cpu_compute::TreeAggregate::kMin, ml_cpu_compute::TreeAggregate::kMax}) {
    for (ml_cpu_compute::TreePostTransform transform :
         {ml_cpu_compute::TreePostTransform::kNone, ml_cpu_compute::TreePostTransform::kSoftmax,
          ml_cpu_compute::TreePostTransform::kLogistic,
          ml_cpu_compute::TreePostTransform::kSoftmaxZero,
          ml_cpu_compute::TreePostTransform::kProbit}) {
      AddCase(cases,
              std::string("aggregate_") + AggregateName(aggregate) + "_" + TransformName(transform),
              MakeAggregateModel(aggregate, transform), {-1.0, 0.0, 1.0, 2.0}, 4);
    }
  }

  ml_cpu_compute::TreeEnsembleAttributes missing_false =
      MakeStump(ml_cpu_compute::TreeBranchMode::kNeq, DataType::FLOAT);
  missing_false.nodes_missing_value_tracks_true = {0};
  AddCase(cases, "missing_tracks_false", std::move(missing_false), {nan}, 1);
  ml_cpu_compute::TreeEnsembleAttributes missing_true =
      MakeStump(ml_cpu_compute::TreeBranchMode::kEq, DataType::FLOAT);
  missing_true.nodes_missing_value_tracks_true = {1};
  AddCase(cases, "missing_tracks_true", std::move(missing_true), {nan}, 1);

  ml_cpu_compute::TreeEnsembleAttributes large_members =
      MakeStump(ml_cpu_compute::TreeBranchMode::kMember, DataType::FLOAT);
  large_members.membership_values.clear();
  for (int value = 0; value < 1024; ++value) {
    large_members.membership_values.push_back(static_cast<double>(value));
  }
  large_members.membership_values.push_back(nan);
  AddCase(cases, "membership_large_duplicate_hit_miss", std::move(large_members),
          {0.0, 1023.0, 1024.0}, 3);

  ml_cpu_compute::TreeEnsembleAttributes multi_target = MakeAggregateModel(
      ml_cpu_compute::TreeAggregate::kSum, ml_cpu_compute::TreePostTransform::kNone);
  AddCase(cases, "multiple_roots_targets", std::move(multi_target), {-1.0, 2.0}, 2);

  ml_cpu_compute::TreeEnsembleAttributes empty =
      MakeStump(ml_cpu_compute::TreeBranchMode::kLeq, DataType::FLOAT);
  AddCase(cases, "empty_batch", std::move(empty), {}, 0);

  std::vector<double> deep_input(4096, 0.0);
  AddCase(cases, "depth_64_features_4096", MakeDeepModel(), std::move(deep_input), 1);
  return cases;
}

} // namespace onnx_light_cpu::backend_test
