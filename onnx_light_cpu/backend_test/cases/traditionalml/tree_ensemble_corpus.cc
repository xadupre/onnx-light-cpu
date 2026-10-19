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
using cpu_compute::DataType;

namespace {

cpu_compute::TreeEnsembleAttributes MakeStump(cpu_compute::TreeBranchMode mode, DataType type) {
  cpu_compute::TreeEnsembleAttributes attributes;
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
  if (mode == cpu_compute::TreeBranchMode::kMember) {
    attributes.membership_values = {-1.0, 0.0, 0.0, std::numeric_limits<double>::quiet_NaN()};
  }
  return attributes;
}

void AddCase(std::vector<TreeEnsembleCorpusCase> &cases, std::string name,
             cpu_compute::TreeEnsembleAttributes attributes, std::vector<double> input,
             std::size_t rows) {
  cpu_compute::TreeEnsembleOracle oracle(attributes);
  std::vector<double> expected = oracle.Evaluate(input, rows);
  cases.push_back({std::move(name), "ai.onnx.ml", "TreeEnsemble", 5, std::move(attributes),
                   std::move(input), rows, std::move(expected)});
}

const char *ModeName(cpu_compute::TreeBranchMode mode) {
  switch (mode) {
  case cpu_compute::TreeBranchMode::kLeq:
    return "leq";
  case cpu_compute::TreeBranchMode::kLt:
    return "lt";
  case cpu_compute::TreeBranchMode::kGte:
    return "gte";
  case cpu_compute::TreeBranchMode::kGt:
    return "gt";
  case cpu_compute::TreeBranchMode::kEq:
    return "eq";
  case cpu_compute::TreeBranchMode::kNeq:
    return "neq";
  case cpu_compute::TreeBranchMode::kMember:
    return "member";
  }
  return "unknown";
}

const char *AggregateName(cpu_compute::TreeAggregate aggregate) {
  switch (aggregate) {
  case cpu_compute::TreeAggregate::kAverage:
    return "average";
  case cpu_compute::TreeAggregate::kSum:
    return "sum";
  case cpu_compute::TreeAggregate::kMin:
    return "min";
  case cpu_compute::TreeAggregate::kMax:
    return "max";
  }
  return "unknown";
}

const char *TransformName(cpu_compute::TreePostTransform transform) {
  switch (transform) {
  case cpu_compute::TreePostTransform::kNone:
    return "none";
  case cpu_compute::TreePostTransform::kSoftmax:
    return "softmax";
  case cpu_compute::TreePostTransform::kLogistic:
    return "logistic";
  case cpu_compute::TreePostTransform::kSoftmaxZero:
    return "softmax_zero";
  case cpu_compute::TreePostTransform::kProbit:
    return "probit";
  }
  return "unknown";
}

cpu_compute::TreeEnsembleAttributes MakeAggregateModel(cpu_compute::TreeAggregate aggregate,
                                                       cpu_compute::TreePostTransform transform) {
  cpu_compute::TreeEnsembleAttributes attributes;
  attributes.n_features = 1;
  attributes.n_targets = 2;
  attributes.aggregate = aggregate;
  attributes.post_transform = transform;
  attributes.tree_roots = {0, 1};
  attributes.nodes_featureids = {0, 0};
  attributes.nodes_splits = {0.0, 1.0};
  attributes.nodes_modes = {cpu_compute::TreeBranchMode::kLeq, cpu_compute::TreeBranchMode::kGt};
  attributes.nodes_truenodeids = {0, 2};
  attributes.nodes_falsenodeids = {1, 3};
  attributes.nodes_trueleafs = {1, 1};
  attributes.nodes_falseleafs = {1, 1};
  attributes.leaf_targetids = {0, 0, 1, 1};
  if (transform == cpu_compute::TreePostTransform::kProbit) {
    attributes.leaf_weights = {0.1, 0.2, 0.8, 0.7};
  } else {
    attributes.leaf_weights = {1.0, -2.0, 3.0, -4.0};
  }
  return attributes;
}

cpu_compute::TreeEnsembleAttributes MakeDeepModel() {
  constexpr std::size_t kDepth = 64;
  cpu_compute::TreeEnsembleAttributes attributes;
  attributes.n_features = 4096;
  attributes.n_targets = 1;
  attributes.tree_roots = {0};
  for (std::size_t index = 0; index < kDepth; ++index) {
    attributes.nodes_featureids.push_back(static_cast<std::int64_t>(index));
    attributes.nodes_splits.push_back(1.0);
    attributes.nodes_modes.push_back(cpu_compute::TreeBranchMode::kLeq);
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

  for (cpu_compute::TreeBranchMode mode :
       {cpu_compute::TreeBranchMode::kLeq, cpu_compute::TreeBranchMode::kLt,
        cpu_compute::TreeBranchMode::kGte, cpu_compute::TreeBranchMode::kGt,
        cpu_compute::TreeBranchMode::kEq, cpu_compute::TreeBranchMode::kNeq,
        cpu_compute::TreeBranchMode::kMember}) {
    cpu_compute::TreeEnsembleAttributes attributes = MakeStump(mode, DataType::DOUBLE);
    attributes.nodes_missing_value_tracks_true = {
        static_cast<std::int64_t>(static_cast<std::uint8_t>(mode) % 2)};
    AddCase(cases, std::string("mode_") + ModeName(mode), std::move(attributes),
            {-infinity, -1.0, -0.0, 0.0, 1.0, infinity, nan}, 7);
  }

  for (DataType type : {DataType::FLOAT16, DataType::FLOAT, DataType::DOUBLE}) {
    cpu_compute::TreeEnsembleAttributes attributes =
        MakeStump(cpu_compute::TreeBranchMode::kLeq, type);
    attributes.nodes_splits = {1.00048828125};
    const char *name = type == DataType::FLOAT16
                           ? "float16_boundary"
                           : (type == DataType::FLOAT ? "float32_boundary" : "float64_boundary");
    AddCase(cases, name, std::move(attributes),
            {1.0, 1.00048828125, std::nextafter(1.00048828125, 2.0)}, 3);
  }

  for (cpu_compute::TreeAggregate aggregate :
       {cpu_compute::TreeAggregate::kAverage, cpu_compute::TreeAggregate::kSum,
        cpu_compute::TreeAggregate::kMin, cpu_compute::TreeAggregate::kMax}) {
    for (cpu_compute::TreePostTransform transform :
         {cpu_compute::TreePostTransform::kNone, cpu_compute::TreePostTransform::kSoftmax,
          cpu_compute::TreePostTransform::kLogistic, cpu_compute::TreePostTransform::kSoftmaxZero,
          cpu_compute::TreePostTransform::kProbit}) {
      AddCase(cases,
              std::string("aggregate_") + AggregateName(aggregate) + "_" + TransformName(transform),
              MakeAggregateModel(aggregate, transform), {-1.0, 0.0, 1.0, 2.0}, 4);
    }
  }

  cpu_compute::TreeEnsembleAttributes missing_false =
      MakeStump(cpu_compute::TreeBranchMode::kNeq, DataType::FLOAT);
  missing_false.nodes_missing_value_tracks_true = {0};
  AddCase(cases, "missing_tracks_false", std::move(missing_false), {nan}, 1);
  cpu_compute::TreeEnsembleAttributes missing_true =
      MakeStump(cpu_compute::TreeBranchMode::kEq, DataType::FLOAT);
  missing_true.nodes_missing_value_tracks_true = {1};
  AddCase(cases, "missing_tracks_true", std::move(missing_true), {nan}, 1);

  cpu_compute::TreeEnsembleAttributes large_members =
      MakeStump(cpu_compute::TreeBranchMode::kMember, DataType::FLOAT);
  large_members.membership_values.clear();
  for (int value = 0; value < 1024; ++value) {
    large_members.membership_values.push_back(static_cast<double>(value));
  }
  large_members.membership_values.push_back(nan);
  AddCase(cases, "membership_large_duplicate_hit_miss", std::move(large_members),
          {0.0, 1023.0, 1024.0}, 3);

  cpu_compute::TreeEnsembleAttributes multi_target =
      MakeAggregateModel(cpu_compute::TreeAggregate::kSum, cpu_compute::TreePostTransform::kNone);
  AddCase(cases, "multiple_roots_targets", std::move(multi_target), {-1.0, 2.0}, 2);

  cpu_compute::TreeEnsembleAttributes empty =
      MakeStump(cpu_compute::TreeBranchMode::kLeq, DataType::FLOAT);
  AddCase(cases, "empty_batch", std::move(empty), {}, 0);

  std::vector<double> deep_input(4096, 0.0);
  AddCase(cases, "depth_64_features_4096", MakeDeepModel(), std::move(deep_input), 1);
  return cases;
}

} // namespace onnx_light_cpu::backend_test
