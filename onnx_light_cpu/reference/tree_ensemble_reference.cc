// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/reference/tree_ensemble_reference.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace onnx_light_cpu::reference {
namespace {

[[noreturn]] void Invalid(const std::string &message) {
  throw std::invalid_argument("TreeEnsemble reference: " + message);
}

std::uint16_t FloatToHalfBits(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t sign = (bits >> 16) & 0x8000U;
  const std::uint32_t exponent = (bits >> 23) & 0xffU;
  const std::uint32_t mantissa = bits & 0x7fffffU;
  if (exponent == 0xffU) {
    return static_cast<std::uint16_t>(sign | (mantissa == 0 ? 0x7c00U : 0x7e00U));
  }
  const int half_exponent = static_cast<int>(exponent) - 127 + 15;
  if (half_exponent >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7c00U);
  }
  if (half_exponent <= 0) {
    if (half_exponent < -10) {
      return static_cast<std::uint16_t>(sign);
    }
    const std::uint32_t significand = mantissa | 0x800000U;
    const int shift = 14 - half_exponent;
    std::uint32_t rounded = significand >> shift;
    const std::uint32_t remainder = significand & ((1U << shift) - 1U);
    const std::uint32_t halfway = 1U << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (rounded & 1U) != 0)) {
      ++rounded;
    }
    return static_cast<std::uint16_t>(sign | rounded);
  }
  std::uint32_t rounded_mantissa = mantissa >> 13;
  const std::uint32_t remainder = mantissa & 0x1fffU;
  if (remainder > 0x1000U || (remainder == 0x1000U && (rounded_mantissa & 1U) != 0)) {
    ++rounded_mantissa;
    if (rounded_mantissa == 0x400U) {
      rounded_mantissa = 0;
      if (half_exponent + 1 >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7c00U);
      }
      return static_cast<std::uint16_t>(sign |
                                        (static_cast<std::uint32_t>(half_exponent + 1) << 10));
    }
  }
  return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(half_exponent) << 10) |
                                    rounded_mantissa);
}

float HalfBitsToFloat(std::uint16_t value) {
  const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000U) << 16;
  std::uint32_t exponent = (value >> 10) & 0x1fU;
  std::uint32_t mantissa = value & 0x3ffU;
  std::uint32_t bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      int unbiased = -14;
      while ((mantissa & 0x400U) == 0) {
        mantissa <<= 1;
        --unbiased;
      }
      mantissa &= 0x3ffU;
      bits = sign | (static_cast<std::uint32_t>(unbiased + 127) << 23) | (mantissa << 13);
    }
  } else if (exponent == 0x1fU) {
    bits = sign | 0x7f800000U | (mantissa << 13);
  } else {
    exponent = exponent - 15 + 127;
    bits = sign | (exponent << 23) | (mantissa << 13);
  }
  return std::bit_cast<float>(bits);
}

double RoundValue(double value, TreeValueType type) {
  switch (type) {
  case TreeValueType::kFloat16:
    return HalfBitsToFloat(FloatToHalfBits(static_cast<float>(value)));
  case TreeValueType::kFloat32:
    return static_cast<float>(value);
  case TreeValueType::kFloat64:
    return value;
  }
  Invalid("unknown value type");
}

bool Compare(TreeBranchMode mode, double value, double split, const std::vector<double> &members) {
  switch (mode) {
  case TreeBranchMode::kLeq:
    return value <= split;
  case TreeBranchMode::kLt:
    return value < split;
  case TreeBranchMode::kGte:
    return value >= split;
  case TreeBranchMode::kGt:
    return value > split;
  case TreeBranchMode::kEq:
    return value == split;
  case TreeBranchMode::kNeq:
    return value != split;
  case TreeBranchMode::kMember:
    return std::find(members.begin(), members.end(), value) != members.end();
  }
  Invalid("unknown branch mode");
}

void ApplyPostTransform(std::vector<double> &values, TreePostTransform transform) {
  if (transform == TreePostTransform::kNone) {
    return;
  }
  if (transform == TreePostTransform::kLogistic) {
    for (double &value : values) {
      const double magnitude = 1.0 / (1.0 + std::exp(-std::abs(value)));
      value = value < 0 ? 1.0 - magnitude : magnitude;
    }
    return;
  }
  if (transform == TreePostTransform::kProbit) {
    constexpr double kAlpha = 0.147;
    constexpr double kSqrt2 = 1.41421356;
    for (double &value : values) {
      const double x = value * 2.0 - 1.0;
      const double product = (1.0 - x) * (1.0 + x);
      if (product == 0.0) {
        value = 0.0;
        continue;
      }
      const double logarithm = std::log(product);
      const double v = 2.0 / (std::numbers::pi * kAlpha) + 0.5 * logarithm;
      const double inverse =
          std::copysign(std::sqrt(-v + std::sqrt(v * v - logarithm / kAlpha)), x);
      value = kSqrt2 * inverse;
    }
    return;
  }
  const double maximum = *std::max_element(values.begin(), values.end());
  double sum = 0.0;
  for (double &value : values) {
    if (transform == TreePostTransform::kSoftmaxZero && std::abs(value) <= 1e-7) {
      value *= std::exp(-maximum);
    } else {
      value = std::exp(value - maximum);
    }
    sum += value;
  }
  if (sum == 0.0) {
    std::fill(values.begin(), values.end(), 0.5);
  } else {
    for (double &value : values) {
      value /= sum;
    }
  }
}

void ValidateTransform(TreePostTransform transform) {
  const auto code = static_cast<std::int64_t>(transform);
  if (code < 0 || code > 4) {
    Invalid("post_transform is out of range");
  }
}

void ValidateAggregate(TreeAggregate aggregate) {
  const auto code = static_cast<std::int64_t>(aggregate);
  if (code < 0 || code > 3) {
    Invalid("aggregate_function is out of range");
  }
}

struct LegacyKey {
  std::int64_t tree;
  std::int64_t node;

  bool operator==(const LegacyKey &) const = default;
};

struct LegacyKeyHash {
  std::size_t operator()(const LegacyKey &key) const noexcept {
    return std::hash<std::int64_t>{}(key.tree) ^
           (std::hash<std::int64_t>{}(key.node) + 0x9e3779b9U);
  }
};

using LegacyIndex = std::unordered_map<LegacyKey, std::size_t, LegacyKeyHash>;

struct LegacyPrepared {
  LegacyIndex nodes;
  std::vector<std::int64_t> roots;
};

LegacyPrepared ValidateLegacy(const LegacyTreeAttributes &tree) {
  const std::size_t size = tree.nodes_treeids.size();
  if (tree.n_features <= 0 || size == 0) {
    Invalid("legacy tree must have features and nodes");
  }
  if (tree.nodes_nodeids.size() != size || tree.nodes_featureids.size() != size ||
      tree.nodes_values.size() != size || tree.nodes_modes.size() != size ||
      tree.nodes_truenodeids.size() != size || tree.nodes_falsenodeids.size() != size ||
      (!tree.nodes_missing_value_tracks_true.empty() &&
       tree.nodes_missing_value_tracks_true.size() != size)) {
    Invalid("legacy nodes_* attributes must have equal lengths");
  }

  LegacyPrepared prepared;
  std::unordered_map<std::int64_t, std::vector<std::size_t>> tree_nodes;
  std::vector<std::size_t> indegree(size);
  for (std::size_t index = 0; index < size; ++index) {
    const LegacyKey key{tree.nodes_treeids[index], tree.nodes_nodeids[index]};
    if (!prepared.nodes.emplace(key, index).second) {
      Invalid("duplicate legacy (tree_id, node_id)");
    }
    tree_nodes[key.tree].push_back(index);
    if (tree.nodes_featureids[index] < 0 || tree.nodes_featureids[index] >= tree.n_features) {
      Invalid("legacy feature id is out of range");
    }
    const std::string &mode = tree.nodes_modes[index];
    if (mode != "LEAF" && mode != "BRANCH_LEQ" && mode != "BRANCH_LT" && mode != "BRANCH_GTE" &&
        mode != "BRANCH_GT" && mode != "BRANCH_EQ" && mode != "BRANCH_NEQ") {
      Invalid("invalid legacy node mode " + mode);
    }
  }
  for (std::size_t index = 0; index < size; ++index) {
    if (tree.nodes_modes[index] == "LEAF") {
      continue;
    }
    for (std::int64_t child : {tree.nodes_truenodeids[index], tree.nodes_falsenodeids[index]}) {
      const auto it = prepared.nodes.find({tree.nodes_treeids[index], child});
      if (it == prepared.nodes.end()) {
        Invalid("legacy path does not terminate at a valid node");
      }
      ++indegree[it->second];
    }
  }
  for (const auto &[tree_id, indices] : tree_nodes) {
    std::size_t root_count = 0;
    for (std::size_t index : indices) {
      if (indegree[index] == 0) {
        prepared.roots.push_back(static_cast<std::int64_t>(index));
        ++root_count;
      }
    }
    if (root_count != 1) {
      Invalid("legacy tree must have exactly one root");
    }
    (void)tree_id;
  }
  std::sort(prepared.roots.begin(), prepared.roots.end(),
            [&](std::int64_t left, std::int64_t right) {
              return tree.nodes_treeids[static_cast<std::size_t>(left)] <
                     tree.nodes_treeids[static_cast<std::size_t>(right)];
            });
  if (prepared.roots.empty()) {
    Invalid("legacy tree has no roots");
  }

  std::vector<int> color(size, 0);
  std::vector<int> owner(size, -1);
  const auto visit = [&](auto &&self, std::size_t index, int tree_number) -> void {
    if (color[index] == 1) {
      Invalid("legacy tree contains a cycle");
    }
    if (owner[index] != -1 && owner[index] != tree_number) {
      Invalid("legacy trees share a node");
    }
    if (color[index] == 2) {
      Invalid("legacy tree contains a shared node");
    }
    owner[index] = tree_number;
    color[index] = 1;
    if (tree.nodes_modes[index] != "LEAF") {
      const std::int64_t tree_id = tree.nodes_treeids[index];
      for (std::int64_t child : {tree.nodes_truenodeids[index], tree.nodes_falsenodeids[index]}) {
        const auto it = prepared.nodes.find({tree_id, child});
        self(self, it->second, tree_number);
      }
    }
    color[index] = 2;
  };
  for (std::size_t tree_number = 0; tree_number < prepared.roots.size(); ++tree_number) {
    visit(visit, static_cast<std::size_t>(prepared.roots[tree_number]),
          static_cast<int>(tree_number));
  }
  if (std::find(owner.begin(), owner.end(), -1) != owner.end()) {
    Invalid("legacy tree contains an unreachable node");
  }
  return prepared;
}

std::vector<LegacyKey> EvaluateLegacyLeaves(const LegacyTreeAttributes &tree,
                                            const LegacyPrepared &prepared,
                                            const std::vector<double> &input, std::size_t rows) {
  if (input.size() != rows * static_cast<std::size_t>(tree.n_features)) {
    Invalid("legacy input shape does not match n_features");
  }
  std::vector<LegacyKey> leaves;
  leaves.reserve(rows * prepared.roots.size());
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::int64_t root : prepared.roots) {
      std::size_t index = static_cast<std::size_t>(root);
      while (tree.nodes_modes[index] != "LEAF") {
        const double value = input[row * static_cast<std::size_t>(tree.n_features) +
                                   static_cast<std::size_t>(tree.nodes_featureids[index])];
        bool go_true;
        if (std::isnan(value)) {
          go_true = !tree.nodes_missing_value_tracks_true.empty() &&
                    tree.nodes_missing_value_tracks_true[index] != 0;
        } else {
          const std::string &mode = tree.nodes_modes[index];
          const double split = tree.nodes_values[index];
          if (mode == "BRANCH_LEQ") {
            go_true = value <= split;
          } else if (mode == "BRANCH_LT") {
            go_true = value < split;
          } else if (mode == "BRANCH_GTE") {
            go_true = value >= split;
          } else if (mode == "BRANCH_GT") {
            go_true = value > split;
          } else if (mode == "BRANCH_EQ") {
            go_true = value == split;
          } else if (mode == "BRANCH_NEQ") {
            go_true = value != split;
          } else {
            Invalid("invalid legacy node mode " + mode);
          }
        }
        const std::int64_t child =
            go_true ? tree.nodes_truenodeids[index] : tree.nodes_falsenodeids[index];
        index = prepared.nodes.at({tree.nodes_treeids[index], child});
      }
      leaves.push_back({tree.nodes_treeids[index], tree.nodes_nodeids[index]});
    }
  }
  return leaves;
}

template <typename Id>
std::unordered_map<LegacyKey, std::vector<std::pair<Id, double>>, LegacyKeyHash>
MakeWeights(const std::vector<std::int64_t> &tree_ids, const std::vector<std::int64_t> &node_ids,
            const std::vector<Id> &ids, const std::vector<double> &weights) {
  if (tree_ids.size() != node_ids.size() || tree_ids.size() != ids.size() ||
      tree_ids.size() != weights.size()) {
    Invalid("legacy weight metadata arrays must have equal lengths");
  }
  std::unordered_map<LegacyKey, std::vector<std::pair<Id, double>>, LegacyKeyHash> result;
  for (std::size_t index = 0; index < tree_ids.size(); ++index) {
    result[{tree_ids[index], node_ids[index]}].emplace_back(ids[index], weights[index]);
  }
  return result;
}

template <typename Id>
void ValidateWeights(const LegacyTreeAttributes &tree, const LegacyPrepared &prepared,
                     const std::unordered_map<LegacyKey, std::vector<std::pair<Id, double>>,
                                              LegacyKeyHash> &weights) {
  for (const auto &[key, entries] : weights) {
    const auto node = prepared.nodes.find(key);
    if (node == prepared.nodes.end() || tree.nodes_modes[node->second] != "LEAF" ||
        entries.empty()) {
      Invalid("legacy weight metadata must refer to a leaf");
    }
  }
  for (std::size_t index = 0; index < tree.nodes_modes.size(); ++index) {
    if (tree.nodes_modes[index] == "LEAF" &&
        weights.find({tree.nodes_treeids[index], tree.nodes_nodeids[index]}) == weights.end()) {
      Invalid("legacy leaf has no weight metadata");
    }
  }
}

} // namespace

TreeEnsembleReference::TreeEnsembleReference(TreeEnsembleAttributes attributes)
    : attributes_(std::move(attributes)) {
  ValidateAggregate(attributes_.aggregate);
  ValidateTransform(attributes_.post_transform);
  if (attributes_.n_features <= 0 || attributes_.n_targets <= 0) {
    Invalid("n_features and n_targets must be positive");
  }
  const std::size_t size = attributes_.nodes_featureids.size();
  if (size == 0 || attributes_.tree_roots.empty()) {
    Invalid("nodes and tree_roots must not be empty");
  }
  if (attributes_.nodes_splits.size() != size || attributes_.nodes_modes.size() != size ||
      attributes_.nodes_truenodeids.size() != size ||
      attributes_.nodes_falsenodeids.size() != size || attributes_.nodes_trueleafs.size() != size ||
      attributes_.nodes_falseleafs.size() != size ||
      (!attributes_.nodes_missing_value_tracks_true.empty() &&
       attributes_.nodes_missing_value_tracks_true.size() != size)) {
    Invalid("nodes_* attributes must have equal lengths");
  }
  if (attributes_.leaf_targetids.size() != attributes_.leaf_weights.size() ||
      attributes_.leaf_weights.empty()) {
    Invalid("leaf attributes must have equal non-zero lengths");
  }
  for (std::size_t index = 0; index < size; ++index) {
    if (attributes_.nodes_featureids[index] < 0 ||
        attributes_.nodes_featureids[index] >= attributes_.n_features) {
      Invalid("feature id is out of range");
    }
    if (static_cast<std::uint8_t>(attributes_.nodes_modes[index]) > 6) {
      Invalid("node mode is out of range");
    }
    for (std::int64_t flag :
         {attributes_.nodes_trueleafs[index], attributes_.nodes_falseleafs[index]}) {
      if (flag != 0 && flag != 1) {
        Invalid("leaf flags must be zero or one");
      }
    }
    for (const auto [child, is_leaf] :
         {std::pair{attributes_.nodes_truenodeids[index], attributes_.nodes_trueleafs[index]},
          std::pair{attributes_.nodes_falsenodeids[index], attributes_.nodes_falseleafs[index]}}) {
      const std::size_t limit = is_leaf != 0 ? attributes_.leaf_weights.size() : size;
      if (child < 0 || static_cast<std::size_t>(child) >= limit) {
        Invalid(is_leaf != 0 ? "leaf index is out of range" : "node index is out of range");
      }
    }
  }
  for (std::int64_t target : attributes_.leaf_targetids) {
    if (target < 0 || target >= attributes_.n_targets) {
      Invalid("leaf target id is out of range");
    }
  }

  membership_sets_.resize(size);
  std::size_t cursor = 0;
  for (std::size_t index = 0; index < size; ++index) {
    if (attributes_.nodes_modes[index] != TreeBranchMode::kMember) {
      continue;
    }
    while (cursor < attributes_.membership_values.size() &&
           !std::isnan(attributes_.membership_values[cursor])) {
      membership_sets_[index].push_back(
          RoundValue(attributes_.membership_values[cursor++], attributes_.value_type));
    }
    if (membership_sets_[index].empty() || cursor == attributes_.membership_values.size()) {
      Invalid("each membership node requires a non-empty NaN-delimited set");
    }
    ++cursor;
  }
  if (cursor != attributes_.membership_values.size()) {
    Invalid("membership_values contains missing or extra delimiters");
  }

  std::vector<int> color(size, 0);
  std::vector<int> owner(size, -1);
  const auto visit = [&](auto &&self, std::size_t index, int tree_number) -> void {
    if (color[index] == 1) {
      Invalid("tree contains a cycle");
    }
    if (color[index] == 2 || (owner[index] != -1 && owner[index] != tree_number)) {
      Invalid("trees contain a shared internal node");
    }
    owner[index] = tree_number;
    color[index] = 1;
    for (const auto [child, is_leaf] :
         {std::pair{attributes_.nodes_truenodeids[index], attributes_.nodes_trueleafs[index]},
          std::pair{attributes_.nodes_falsenodeids[index], attributes_.nodes_falseleafs[index]}}) {
      if (is_leaf == 0) {
        self(self, static_cast<std::size_t>(child), tree_number);
      }
    }
    color[index] = 2;
  };
  for (std::size_t tree = 0; tree < attributes_.tree_roots.size(); ++tree) {
    const std::int64_t root = attributes_.tree_roots[tree];
    if (root < 0 || static_cast<std::size_t>(root) >= size) {
      Invalid("tree root is out of range");
    }
    visit(visit, static_cast<std::size_t>(root), static_cast<int>(tree));
  }
  if (std::find(owner.begin(), owner.end(), -1) != owner.end()) {
    Invalid("tree contains an unreachable internal node");
  }
}

std::vector<double> TreeEnsembleReference::Evaluate(const std::vector<double> &input,
                                                    std::size_t rows) const {
  const std::size_t features = static_cast<std::size_t>(attributes_.n_features);
  const std::size_t targets = static_cast<std::size_t>(attributes_.n_targets);
  if (input.size() != rows * features) {
    Invalid("input shape does not match n_features");
  }
  std::vector<double> output(rows * targets);
  for (std::size_t row = 0; row < rows; ++row) {
    std::vector<double> values(targets);
    std::vector<std::size_t> counts(targets);
    for (std::int64_t root : attributes_.tree_roots) {
      std::size_t node = static_cast<std::size_t>(root);
      std::size_t leaf;
      for (;;) {
        const double value = RoundValue(
            input[row * features + static_cast<std::size_t>(attributes_.nodes_featureids[node])],
            attributes_.value_type);
        bool go_true;
        if (std::isnan(value)) {
          go_true = !attributes_.nodes_missing_value_tracks_true.empty() &&
                    attributes_.nodes_missing_value_tracks_true[node] != 0;
        } else {
          go_true = Compare(attributes_.nodes_modes[node], value,
                            RoundValue(attributes_.nodes_splits[node], attributes_.value_type),
                            membership_sets_[node]);
        }
        const bool is_leaf =
            (go_true ? attributes_.nodes_trueleafs[node] : attributes_.nodes_falseleafs[node]) != 0;
        const std::int64_t next =
            go_true ? attributes_.nodes_truenodeids[node] : attributes_.nodes_falsenodeids[node];
        if (is_leaf) {
          leaf = static_cast<std::size_t>(next);
          break;
        }
        node = static_cast<std::size_t>(next);
      }
      const std::size_t target = static_cast<std::size_t>(attributes_.leaf_targetids[leaf]);
      const double weight = RoundValue(attributes_.leaf_weights[leaf], attributes_.value_type);
      if (counts[target] == 0 || attributes_.aggregate == TreeAggregate::kSum ||
          attributes_.aggregate == TreeAggregate::kAverage) {
        values[target] = RoundValue(values[target] + weight, attributes_.value_type);
      } else if (attributes_.aggregate == TreeAggregate::kMin) {
        values[target] = std::min(values[target], weight);
      } else {
        values[target] = std::max(values[target], weight);
      }
      ++counts[target];
    }
    if (attributes_.aggregate == TreeAggregate::kAverage) {
      for (double &value : values) {
        value = RoundValue(value / static_cast<double>(attributes_.tree_roots.size()),
                           attributes_.value_type);
      }
    }
    ApplyPostTransform(values, attributes_.post_transform);
    for (std::size_t target = 0; target < targets; ++target) {
      output[row * targets + target] = RoundValue(values[target], attributes_.value_type);
    }
  }
  return output;
}

std::vector<float> EvaluateTreeEnsembleRegressor(const TreeEnsembleRegressorAttributes &attributes,
                                                 const std::vector<double> &input,
                                                 std::size_t rows) {
  ValidateAggregate(attributes.aggregate);
  ValidateTransform(attributes.post_transform);
  if (attributes.n_targets <= 0) {
    Invalid("legacy regressor n_targets must be positive");
  }
  if (!attributes.base_values.empty() &&
      attributes.base_values.size() != static_cast<std::size_t>(attributes.n_targets)) {
    Invalid("legacy regressor base_values has the wrong length");
  }
  const LegacyPrepared prepared = ValidateLegacy(attributes.tree);
  const auto leaves = EvaluateLegacyLeaves(attributes.tree, prepared, input, rows);
  const auto weights = MakeWeights(attributes.target_treeids, attributes.target_nodeids,
                                   attributes.target_ids, attributes.target_weights);
  ValidateWeights(attributes.tree, prepared, weights);
  for (const auto &[key, entries] : weights) {
    (void)key;
    for (const auto &[target_id, weight] : entries) {
      (void)weight;
      if (target_id < 0 || target_id >= attributes.n_targets) {
        Invalid("legacy regressor target id is out of range");
      }
    }
  }
  const std::size_t targets = static_cast<std::size_t>(attributes.n_targets);
  std::vector<float> result(rows * targets);
  for (std::size_t row = 0; row < rows; ++row) {
    std::vector<std::size_t> counts(targets);
    for (std::size_t tree = 0; tree < prepared.roots.size(); ++tree) {
      const auto it = weights.find(leaves[row * prepared.roots.size() + tree]);
      if (it == weights.end()) {
        Invalid("legacy regressor leaf has no target metadata");
      }
      for (const auto &[target_id, weight] : it->second) {
        if (target_id < 0 || target_id >= attributes.n_targets) {
          Invalid("legacy regressor target id is out of range");
        }
        const std::size_t target = static_cast<std::size_t>(target_id);
        float &value = result[row * targets + target];
        if (counts[target] == 0 || attributes.aggregate == TreeAggregate::kSum ||
            attributes.aggregate == TreeAggregate::kAverage) {
          value += static_cast<float>(weight);
        } else if (attributes.aggregate == TreeAggregate::kMin) {
          value = std::min(value, static_cast<float>(weight));
        } else {
          value = std::max(value, static_cast<float>(weight));
        }
        ++counts[target];
      }
    }
    std::vector<double> transformed(targets);
    for (std::size_t target = 0; target < targets; ++target) {
      float &value = result[row * targets + target];
      if (attributes.aggregate == TreeAggregate::kAverage) {
        value /= static_cast<float>(prepared.roots.size());
      }
      if (!attributes.base_values.empty()) {
        value += static_cast<float>(attributes.base_values[target]);
      }
      transformed[target] = value;
    }
    ApplyPostTransform(transformed, attributes.post_transform);
    for (std::size_t target = 0; target < targets; ++target) {
      result[row * targets + target] = static_cast<float>(transformed[target]);
    }
  }
  return result;
}

TreeClassifierResult
EvaluateTreeEnsembleClassifier(const TreeEnsembleClassifierAttributes &attributes,
                               const std::vector<double> &input, std::size_t rows) {
  ValidateTransform(attributes.post_transform);
  const std::size_t class_count =
      std::visit([](const auto &labels) { return labels.size(); }, attributes.labels);
  if (class_count == 0) {
    Invalid("legacy classifier labels must not be empty");
  }
  if (!attributes.base_values.empty() && attributes.base_values.size() != class_count) {
    Invalid("legacy classifier base_values has the wrong length");
  }
  const LegacyPrepared prepared = ValidateLegacy(attributes.tree);
  const auto leaves = EvaluateLegacyLeaves(attributes.tree, prepared, input, rows);
  const auto weights = MakeWeights(attributes.class_treeids, attributes.class_nodeids,
                                   attributes.class_ids, attributes.class_weights);
  ValidateWeights(attributes.tree, prepared, weights);
  for (const auto &[key, entries] : weights) {
    (void)key;
    for (const auto &[class_id, weight] : entries) {
      (void)weight;
      if (class_id < 0 || static_cast<std::size_t>(class_id) >= class_count) {
        Invalid("legacy classifier class id is out of range");
      }
    }
  }
  TreeClassifierResult result;
  result.scores.resize(rows * class_count);
  for (std::size_t row = 0; row < rows; ++row) {
    std::vector<double> scores(class_count);
    if (!attributes.base_values.empty()) {
      scores = attributes.base_values;
    }
    for (std::size_t tree = 0; tree < prepared.roots.size(); ++tree) {
      const auto it = weights.find(leaves[row * prepared.roots.size() + tree]);
      if (it == weights.end()) {
        Invalid("legacy classifier leaf has no class metadata");
      }
      for (const auto &[class_id, weight] : it->second) {
        if (class_id < 0 || static_cast<std::size_t>(class_id) >= class_count) {
          Invalid("legacy classifier class id is out of range");
        }
        scores[static_cast<std::size_t>(class_id)] += weight;
      }
    }
    ApplyPostTransform(scores, attributes.post_transform);
    const std::size_t best =
        static_cast<std::size_t>(std::max_element(scores.begin(), scores.end()) - scores.begin());
    std::transform(scores.begin(), scores.end(),
                   result.scores.begin() + static_cast<std::ptrdiff_t>(row * class_count),
                   [](double value) { return static_cast<float>(value); });
    if (const auto *labels = std::get_if<std::vector<std::int64_t>>(&attributes.labels)) {
      result.integer_labels.push_back((*labels)[best]);
    } else {
      result.string_labels.push_back(std::get<std::vector<std::string>>(attributes.labels)[best]);
    }
  }
  return result;
}

} // namespace onnx_light_cpu::reference
