// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#if defined(_WIN32)
#define NOMINMAX
#endif

#include "onnx_light_cpu/impl/traditionalml/tree_ensemble.h"

#include "onnx_light_cpu/impl/execution.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace onnx_light_cpu {
namespace {

constexpr std::size_t kTreeParallelThreshold = 80;
constexpr std::size_t kTreeMajorBatchRows = 128;
constexpr std::size_t kRowParallelThreshold = 50;
constexpr std::size_t kMaximumExecutionRegions = 4;

std::size_t SaturatingMultiply(std::size_t left, std::size_t right) noexcept {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    return std::numeric_limits<std::size_t>::max();
  }
  return left * right;
}

std::size_t SaturatingAdd(std::size_t left, std::size_t right) noexcept {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    return std::numeric_limits<std::size_t>::max();
  }
  return left + right;
}

std::size_t RegionWorkspaceBytes(const TreeEnsembleExecutionRegion &region,
                                 std::size_t targets) noexcept {
  return SaturatingMultiply(
      SaturatingMultiply(SaturatingMultiply(region.maximum_threads, region.batch_rows), targets),
      sizeof(double) + sizeof(std::size_t));
}

[[noreturn]] void Invalid(const std::string &message) {
  throw std::invalid_argument("TreeEnsemble: " + message);
}

void ValidatePolicyShape(const TreeEnsembleTuningPolicy &policy) {
  if (policy.regions.empty() || policy.regions.size() > kMaximumExecutionRegions) {
    Invalid("execution policy must contain one to four regions");
  }
  std::size_t previous = 0;
  for (std::size_t index = 0; index < policy.regions.size(); ++index) {
    const TreeEnsembleExecutionRegion &region = policy.regions[index];
    const bool final = index + 1 == policy.regions.size();
    if (final != !region.maximum_rows.has_value()) {
      Invalid("only the final execution region must be unbounded");
    }
    if (region.maximum_rows.has_value()) {
      if (*region.maximum_rows == 0 || (index != 0 && *region.maximum_rows <= previous)) {
        Invalid("execution region boundaries must be strictly increasing");
      }
      previous = *region.maximum_rows;
    }
    if (region.batch_rows == 0 || region.maximum_threads == 0 || region.row_chunk == 0 ||
        region.tree_chunk == 0 || region.workspace_bytes == 0) {
      Invalid("execution region parameters and workspace must be positive");
    }
  }
}

void ValidateExactKey(const TreeEnsembleModelKey &key) {
  if (key.library != "onnx_light_cpu" || key.kernel != "TreeEnsemble" ||
      key.domain != "ai.onnx.ml" || key.opset != 5 ||
      key.implementation != "prepared_tree_ensemble" || key.processor.empty() || key.threads == 0 ||
      key.model_digest.empty()) {
    Invalid("exact profile key is incomplete or incompatible");
  }
}

TreeEnsembleCountBucket CountBucket(std::size_t count) noexcept {
  if (count <= 1) {
    return TreeEnsembleCountBucket::kOne;
  }
  if (count <= 4) {
    return TreeEnsembleCountBucket::kTwoToFour;
  }
  if (count <= 16) {
    return TreeEnsembleCountBucket::kFiveToSixteen;
  }
  if (count <= 80) {
    return TreeEnsembleCountBucket::kSeventeenToEighty;
  }
  if (count <= 256) {
    return TreeEnsembleCountBucket::kEightyOneToTwoHundredFiftySix;
  }
  return TreeEnsembleCountBucket::kMoreThanTwoHundredFiftySix;
}

TreeEnsembleDepthBucket DepthBucket(std::size_t depth) noexcept {
  if (depth <= 1) {
    return TreeEnsembleDepthBucket::kStump;
  }
  if (depth <= 4) {
    return TreeEnsembleDepthBucket::kTwoToFour;
  }
  if (depth <= 8) {
    return TreeEnsembleDepthBucket::kFiveToEight;
  }
  if (depth <= 16) {
    return TreeEnsembleDepthBucket::kNineToSixteen;
  }
  return TreeEnsembleDepthBucket::kMoreThanSixteen;
}

TreeEnsembleExecutionRegion MakeRegion(std::optional<std::size_t> maximum_rows,
                                       TreeEnsembleExecutionStrategy strategy,
                                       std::size_t batch_rows, std::size_t maximum_threads,
                                       std::size_t trees, std::size_t targets) {
  TreeEnsembleExecutionRegion region;
  region.maximum_rows = maximum_rows;
  region.strategy = strategy;
  region.batch_rows = batch_rows;
  region.maximum_threads = maximum_threads;
  region.row_chunk = 1;
  region.tree_chunk = std::max<std::size_t>(1, (trees + maximum_threads - 1) / maximum_threads);
  region.workspace_bytes = RegionWorkspaceBytes(region, targets);
  return region;
}

TreeEnsembleTuningPolicy MakeSafePolicy(std::size_t trees, std::size_t targets,
                                        std::size_t threads) {
  TreeEnsembleTuningPolicy policy;
  if (threads == 1) {
    policy.regions.push_back(MakeRegion(std::nullopt,
                                        TreeEnsembleExecutionStrategy::kTreeMajorBatch,
                                        kTreeMajorBatchRows, 1, trees, targets));
    return policy;
  }
  if (trees > kTreeParallelThreshold) {
    policy.regions.push_back(MakeRegion(1, TreeEnsembleExecutionStrategy::kTreeParallel,
                                        kTreeMajorBatchRows, threads, trees, targets));
    policy.regions.push_back(MakeRegion(kRowParallelThreshold,
                                        TreeEnsembleExecutionStrategy::kTreeMajorBatch,
                                        kTreeMajorBatchRows, 1, trees, targets));
    policy.regions.push_back(MakeRegion(std::nullopt, TreeEnsembleExecutionStrategy::kTreeParallel,
                                        kTreeMajorBatchRows, threads, trees, targets));
    return policy;
  }
  policy.regions.push_back(MakeRegion(kRowParallelThreshold,
                                      TreeEnsembleExecutionStrategy::kTreeMajorBatch,
                                      kTreeMajorBatchRows, 1, trees, targets));
  const bool use_tree_parallel = trees > threads || (targets > 1 && trees == threads);
  policy.regions.push_back(
      MakeRegion(std::nullopt,
                 use_tree_parallel ? TreeEnsembleExecutionStrategy::kTreeParallel
                                   : TreeEnsembleExecutionStrategy::kRowParallel,
                 use_tree_parallel ? kTreeMajorBatchRows : 1, threads, trees, targets));
  return policy;
}

bool PolicyCompatible(const TreeEnsembleTuningPolicy &policy, std::size_t targets,
                      std::size_t threads, DataType value_type, bool uses_64_bit_indices,
                      bool all_stumps, bool all_symmetric, bool has_hit_rates,
                      std::size_t active_targets) noexcept {
  if (uses_64_bit_indices && policy.layout != TreeEnsembleNodeLayout::kOrtCompactAosPointer) {
    return false;
  }
  if (policy.layout == TreeEnsembleNodeLayout::kPreorderHot && !has_hit_rates) {
    return false;
  }
  if (policy.traversal == TreeEnsembleTraversal::kStump && !all_stumps) {
    return false;
  }
  if (policy.traversal == TreeEnsembleTraversal::kSymmetric && !all_symmetric) {
    return false;
  }
  if (policy.optimized_float16 && value_type != DataType::FLOAT16) {
    return false;
  }
  const std::size_t accumulator_targets =
      policy.target_layout == TreeEnsembleTargetLayout::kSparse ? active_targets : targets;
  for (const TreeEnsembleExecutionRegion &region : policy.regions) {
    if (region.maximum_threads > threads ||
        RegionWorkspaceBytes(region, accumulator_targets) > region.workspace_bytes) {
      return false;
    }
  }
  return true;
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

double RoundValue(double value, DataType type) {
  switch (type) {
  case DataType::FLOAT16:
    return HalfBitsToFloat(FloatToHalfBits(static_cast<float>(value)));
  case DataType::FLOAT:
    return static_cast<float>(value);
  case DataType::DOUBLE:
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

TreeBranchMode ParseLegacyBranchMode(const std::string &mode) {
  if (mode == "BRANCH_LEQ") {
    return TreeBranchMode::kLeq;
  }
  if (mode == "BRANCH_LT") {
    return TreeBranchMode::kLt;
  }
  if (mode == "BRANCH_GTE") {
    return TreeBranchMode::kGte;
  }
  if (mode == "BRANCH_GT") {
    return TreeBranchMode::kGt;
  }
  if (mode == "BRANCH_EQ") {
    return TreeBranchMode::kEq;
  }
  if (mode == "BRANCH_NEQ") {
    return TreeBranchMode::kNeq;
  }
  Invalid("invalid legacy node mode " + mode);
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

double Median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  const std::size_t middle = samples.size() / 2;
  return samples.size() % 2 == 0 ? (samples[middle - 1] + samples[middle]) * 0.5 : samples[middle];
}

double MedianAbsoluteDeviation(const std::vector<double> &samples, double median) {
  std::vector<double> deviations;
  deviations.reserve(samples.size());
  for (double sample : samples) {
    deviations.push_back(std::abs(sample - median));
  }
  return Median(std::move(deviations));
}

std::size_t PolicyWorkspace(const TreeEnsembleTuningPolicy &policy) noexcept {
  std::size_t result = 0;
  for (const TreeEnsembleExecutionRegion &region : policy.regions) {
    result = std::max(result, region.workspace_bytes);
  }
  return result;
}

TreeEnsembleTuningPolicy ComposeCandidatePolicy(const TreeEnsembleTuningPolicy &current,
                                                const TreeEnsembleTuningPolicy &candidate,
                                                TreeEnsembleCalibrationStage stage) {
  TreeEnsembleTuningPolicy composed = current;
  const auto workspace_density = [](const TreeEnsembleExecutionRegion &region) {
    const std::size_t scale = SaturatingMultiply(region.maximum_threads, region.batch_rows);
    return region.workspace_bytes / scale + (region.workspace_bytes % scale != 0 ? 1U : 0U);
  };
  const auto resize_workspace = [&](TreeEnsembleExecutionRegion &region, std::size_t density) {
    region.workspace_bytes =
        SaturatingMultiply(density, SaturatingMultiply(region.maximum_threads, region.batch_rows));
  };
  switch (stage) {
  case TreeEnsembleCalibrationStage::kLayout:
    composed.layout = candidate.layout;
    composed.membership_linear_limit = candidate.membership_linear_limit;
    composed.membership_bitset_range_limit = candidate.membership_bitset_range_limit;
    break;
  case TreeEnsembleCalibrationStage::kTraversal:
    composed.traversal = candidate.traversal;
    composed.target_layout = candidate.target_layout;
    composed.optimized_float16 = candidate.optimized_float16;
    if (candidate.target_layout == TreeEnsembleTargetLayout::kSparse) {
      for (std::size_t index = 0;
           index < composed.regions.size() && index < candidate.regions.size(); ++index) {
        composed.regions[index].workspace_bytes = candidate.regions[index].workspace_bytes;
      }
    }
    break;
  case TreeEnsembleCalibrationStage::kScheduling: {
    const std::size_t density = workspace_density(current.regions.front());
    composed.regions = candidate.regions;
    for (TreeEnsembleExecutionRegion &region : composed.regions) {
      resize_workspace(region, density);
    }
  } break;
  case TreeEnsembleCalibrationStage::kBatch:
    for (std::size_t index = 0; index < composed.regions.size() && index < candidate.regions.size();
         ++index) {
      const std::size_t bytes_per_participant_row = workspace_density(composed.regions[index]);
      composed.regions[index].maximum_rows = candidate.regions[index].maximum_rows;
      composed.regions[index].batch_rows = candidate.regions[index].batch_rows;
      resize_workspace(composed.regions[index], bytes_per_participant_row);
    }
    break;
  case TreeEnsembleCalibrationStage::kChunk:
    for (std::size_t index = 0; index < composed.regions.size() && index < candidate.regions.size();
         ++index) {
      const std::size_t bytes_per_participant_row = workspace_density(composed.regions[index]);
      composed.regions[index].maximum_threads = candidate.regions[index].maximum_threads;
      composed.regions[index].row_chunk = candidate.regions[index].row_chunk;
      composed.regions[index].tree_chunk = candidate.regions[index].tree_chunk;
      resize_workspace(composed.regions[index], bytes_per_participant_row);
    }
    break;
  case TreeEnsembleCalibrationStage::kWorkspace:
    for (std::size_t index = 0; index < composed.regions.size() && index < candidate.regions.size();
         ++index) {
      composed.regions[index].workspace_bytes = std::max(composed.regions[index].workspace_bytes,
                                                         candidate.regions[index].workspace_bytes);
    }
    break;
  case TreeEnsembleCalibrationStage::kPrefetch:
    composed.traversal_prefetch_distance = candidate.traversal_prefetch_distance;
    break;
  }
  return composed;
}

void WritePolicy(std::ostream &stream, const TreeEnsembleTuningPolicy &policy) {
  stream << static_cast<int>(policy.layout) << ' ' << static_cast<int>(policy.traversal) << ' '
         << static_cast<int>(policy.target_layout) << ' ' << policy.optimized_float16 << ' '
         << policy.membership_linear_limit << ' ' << policy.membership_bitset_range_limit << ' '
         << policy.traversal_prefetch_distance << ' ' << policy.regions.size() << '\n';
  for (const TreeEnsembleExecutionRegion &region : policy.regions) {
    stream << (region.maximum_rows.has_value() ? 1 : 0) << ' ' << region.maximum_rows.value_or(0)
           << ' ' << static_cast<int>(region.strategy) << ' ' << region.batch_rows << ' '
           << region.maximum_threads << ' ' << region.row_chunk << ' ' << region.tree_chunk << ' '
           << region.workspace_bytes << '\n';
  }
}

class TemporaryFile {
public:
  explicit TemporaryFile(std::filesystem::path path) : path_(std::move(path)) {}

  ~TemporaryFile() {
    if (!released_) {
      std::error_code ignored;
      std::filesystem::remove(path_, ignored);
    }
  }

  const std::filesystem::path &path() const noexcept { return path_; }
  void Release() noexcept { released_ = true; }

private:
  std::filesystem::path path_;
  bool released_{false};
};

void ReplaceFileAtomically(const std::filesystem::path &source,
                           const std::filesystem::path &destination) {
#if defined(_WIN32)
  if (!MoveFileExW(source.c_str(), destination.c_str(),
                   MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    throw std::filesystem::filesystem_error(
        "unable to replace calibration evidence", source, destination,
        std::error_code(static_cast<int>(GetLastError()), std::system_category()));
  }
#else
  std::filesystem::rename(source, destination);
#endif
}

void PersistCalibrationEvidence(const TreeEnsembleCalibrationReport &report,
                                const std::string &path) {
  static std::atomic<std::uint64_t> sequence{0};
  const std::filesystem::path destination(path);
  TemporaryFile temporary(path + ".tmp." + std::to_string(sequence.fetch_add(1)));
  if (!destination.parent_path().empty()) {
    std::filesystem::create_directories(destination.parent_path());
  }
  std::ofstream stream(temporary.path(), std::ios::binary | std::ios::trunc);
  if (!stream) {
    throw std::runtime_error("unable to open temporary evidence file");
  }
  stream << "onnx_light_cpu_tree_calibration_v1\n"
         << std::quoted(report.key.processor) << ' ' << report.key.threads << ' '
         << std::quoted(report.key.model_digest) << '\n';
  WritePolicy(stream, report.selected_policy);
  stream << report.evidence.size() << '\n' << std::setprecision(17);
  for (const TreeEnsembleCalibrationEvidence &evidence : report.evidence) {
    stream << std::quoted(evidence.candidate) << ' ' << static_cast<int>(evidence.stage) << ' '
           << evidence.correct << ' ' << evidence.selected << ' ' << evidence.median_ns << ' '
           << evidence.dispersion_ns << ' ' << std::quoted(evidence.rejected_reason) << ' '
           << evidence.distribution_shift_correct << ' ' << evidence.peak_memory_bytes << ' '
           << evidence.distribution_shift_median_ns << ' ' << evidence.samples_ns.size();
    for (double sample : evidence.samples_ns) {
      stream << ' ' << sample;
    }
    stream << ' ' << evidence.distribution_shift_samples_ns.size();
    for (double sample : evidence.distribution_shift_samples_ns) {
      stream << ' ' << sample;
    }
    stream << '\n';
    WritePolicy(stream, evidence.policy);
  }
  stream.flush();
  if (!stream) {
    throw std::runtime_error("unable to write calibration evidence");
  }
  stream.close();
  ReplaceFileAtomically(temporary.path(), destination);
  temporary.Release();
}

} // namespace

void TreeEnsembleTuningRegistry::PutExact(TreeEnsembleModelKey key,
                                          TreeEnsembleTuningPolicy policy) {
  ValidatePolicyShape(policy);
  ValidateExactKey(key);
  std::lock_guard<std::mutex> lock(mutex_);
  const auto found = std::find_if(exact_.begin(), exact_.end(),
                                  [&](const ExactEntry &entry) { return entry.key == key; });
  if (found == exact_.end()) {
    exact_.push_back({std::move(key), std::move(policy)});
  } else {
    found->policy = std::move(policy);
  }
  ++generation_;
}

void TreeEnsembleTuningRegistry::PutPortable(TreeEnsembleStructuralBuckets buckets,
                                             TreeEnsembleTuningPolicy policy) {
  ValidatePolicyShape(policy);
  std::lock_guard<std::mutex> lock(mutex_);
  const auto found =
      std::find_if(portable_.begin(), portable_.end(),
                   [&](const PortableEntry &entry) { return entry.buckets == buckets; });
  if (found == portable_.end()) {
    portable_.push_back({buckets, std::move(policy)});
  } else {
    found->policy = std::move(policy);
  }
  ++generation_;
}

TreeEnsembleCalibrationReport TreeEnsembleTuningRegistry::CalibrateExact(
    const TreeEnsembleModelKey &key, const TreeEnsembleTuningPolicy &fallback,
    const std::vector<TreeEnsembleCalibrationCandidate> &candidates,
    const TreeEnsembleCalibrationOptions &options, const TreeEnsembleCalibrationMeasure &measure) {
  ValidateExactKey(key);
  ValidatePolicyShape(fallback);
  if (!measure || options.duration_budget_ns == 0 || options.memory_budget_bytes == 0 ||
      options.repetitions == 0 || options.required_wins == 0 ||
      !std::isfinite(options.minimum_improvement) || options.minimum_improvement < 0.0 ||
      options.minimum_improvement >= 1.0 ||
      !std::isfinite(options.maximum_distribution_shift_regression) ||
      options.maximum_distribution_shift_regression < 0.0) {
    Invalid("calibration options are invalid");
  }

  TreeEnsembleCalibrationReport report;
  report.key = key;
  report.selected_policy = fallback;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!calibration_enabled_) {
      for (const TreeEnsembleCalibrationCandidate &candidate : candidates) {
        report.evidence.push_back({candidate.name,
                                   candidate.stage,
                                   candidate.policy,
                                   {},
                                   0.0,
                                   0.0,
                                   false,
                                   false,
                                   "calibration disabled"});
      }
      return report;
    }
  }

  const auto calibration_start = std::chrono::steady_clock::now();
  std::uint64_t reported_duration = 0;
  const auto budget_available = [&]() {
    const auto actual = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - calibration_start)
                            .count();
    return actual >= 0 && static_cast<std::uint64_t>(actual) < options.duration_budget_ns &&
           reported_duration < options.duration_budget_ns;
  };
  const auto run = [&](const TreeEnsembleTuningPolicy &policy,
                       TreeEnsembleCalibrationEvidence &evidence) {
    if (!budget_available()) {
      evidence.rejected_reason = "duration budget exceeded";
      report.budget_exhausted = true;
      return false;
    }
    TreeEnsembleCalibrationMeasurement measurement;
    try {
      measurement = measure(policy, options.warmup_runs, options.repetitions);
    } catch (const std::exception &exception) {
      evidence.rejected_reason = std::string("measurement failed: ") + exception.what();
      return false;
    }
    if (measurement.elapsed_ns > options.duration_budget_ns - reported_duration) {
      reported_duration = options.duration_budget_ns;
    } else {
      reported_duration += measurement.elapsed_ns;
    }
    evidence.correct = measurement.correct;
    evidence.distribution_shift_correct = measurement.distribution_shift_correct;
    evidence.peak_memory_bytes = measurement.peak_memory_bytes;
    evidence.samples_ns.insert(evidence.samples_ns.end(), measurement.samples_ns.begin(),
                               measurement.samples_ns.end());
    evidence.distribution_shift_samples_ns.insert(evidence.distribution_shift_samples_ns.end(),
                                                  measurement.distribution_shift_samples_ns.begin(),
                                                  measurement.distribution_shift_samples_ns.end());
    if (!measurement.correct) {
      evidence.rejected_reason =
          measurement.failure.empty() ? "correctness validation failed" : measurement.failure;
      return false;
    }
    if (!measurement.distribution_shift_correct) {
      evidence.rejected_reason = measurement.distribution_shift_failure.empty()
                                     ? "distribution-shift validation failed"
                                     : measurement.distribution_shift_failure;
      return false;
    }
    if (measurement.peak_memory_bytes > options.memory_budget_bytes) {
      evidence.rejected_reason = "measured memory budget exceeded";
      return false;
    }
    if (measurement.samples_ns.size() != options.repetitions ||
        std::any_of(measurement.samples_ns.begin(), measurement.samples_ns.end(),
                    [](double sample) { return !std::isfinite(sample) || sample <= 0.0; })) {
      evidence.rejected_reason = "invalid timing samples";
      return false;
    }
    if (!budget_available()) {
      evidence.rejected_reason = "duration budget exceeded";
      report.budget_exhausted = true;
      return false;
    }
    return true;
  };

  for (int stage_code = static_cast<int>(TreeEnsembleCalibrationStage::kLayout);
       stage_code <= static_cast<int>(TreeEnsembleCalibrationStage::kPrefetch); ++stage_code) {
    const auto stage = static_cast<TreeEnsembleCalibrationStage>(stage_code);
    std::vector<const TreeEnsembleCalibrationCandidate *> stage_candidates;
    for (const TreeEnsembleCalibrationCandidate &candidate : candidates) {
      if (candidate.stage == stage) {
        stage_candidates.push_back(&candidate);
      }
    }
    if (stage_candidates.empty()) {
      continue;
    }

    const TreeEnsembleTuningPolicy stage_fallback = report.selected_policy;
    double best_median = std::numeric_limits<double>::infinity();
    const TreeEnsembleCalibrationCandidate *best = nullptr;
    std::size_t best_evidence = 0;
    for (const TreeEnsembleCalibrationCandidate *candidate : stage_candidates) {
      TreeEnsembleCalibrationEvidence evidence;
      evidence.candidate = candidate->name;
      evidence.stage = candidate->stage;
      evidence.policy = ComposeCandidatePolicy(stage_fallback, candidate->policy, stage);
      try {
        ValidatePolicyShape(evidence.policy);
      } catch (const std::invalid_argument &exception) {
        evidence.rejected_reason = exception.what();
        report.evidence.push_back(std::move(evidence));
        continue;
      }
      if (candidate->prepared_bytes > options.memory_budget_bytes ||
          PolicyWorkspace(evidence.policy) >
              options.memory_budget_bytes - candidate->prepared_bytes) {
        evidence.rejected_reason = "memory budget exceeded";
        report.evidence.push_back(std::move(evidence));
        continue;
      }

      bool won_every_repeat = true;
      for (std::size_t repeat = 0; repeat < options.required_wins; ++repeat) {
        TreeEnsembleCalibrationEvidence baseline;
        baseline.candidate = candidate->name + ":fallback";
        baseline.stage = stage;
        baseline.policy = stage_fallback;
        if (!run(stage_fallback, baseline)) {
          won_every_repeat = false;
          if (report.budget_exhausted) {
            evidence.rejected_reason = "duration budget exceeded";
          } else {
            evidence.rejected_reason = "fallback validation failed";
          }
          break;
        }
        baseline.median_ns = Median(baseline.samples_ns);
        baseline.dispersion_ns = MedianAbsoluteDeviation(baseline.samples_ns, baseline.median_ns);
        const std::vector<double> baseline_shift_samples = baseline.distribution_shift_samples_ns;
        report.evidence.push_back(std::move(baseline));
        const double fallback_median = report.evidence.back().median_ns;

        const std::size_t previous_sample_count = evidence.samples_ns.size();
        const std::size_t previous_shift_sample_count =
            evidence.distribution_shift_samples_ns.size();
        if (!run(evidence.policy, evidence)) {
          won_every_repeat = false;
          break;
        }
        if (candidate->requires_distribution_shift) {
          if (baseline_shift_samples.size() != options.repetitions ||
              evidence.distribution_shift_samples_ns.size() <
                  previous_shift_sample_count + options.repetitions) {
            evidence.rejected_reason = "missing distribution-shift timing samples";
            won_every_repeat = false;
            break;
          }
          const auto invalid_sample = [](double sample) {
            return !std::isfinite(sample) || sample <= 0.0;
          };
          if (std::any_of(baseline_shift_samples.begin(), baseline_shift_samples.end(),
                          invalid_sample) ||
              std::any_of(evidence.distribution_shift_samples_ns.begin() +
                              static_cast<std::ptrdiff_t>(previous_shift_sample_count),
                          evidence.distribution_shift_samples_ns.end(), invalid_sample)) {
            evidence.rejected_reason = "invalid distribution-shift timing samples";
            won_every_repeat = false;
            break;
          }
          const double shifted_fallback = Median(baseline_shift_samples);
          const std::vector<double> shifted_repeat(
              evidence.distribution_shift_samples_ns.end() -
                  static_cast<std::ptrdiff_t>(options.repetitions),
              evidence.distribution_shift_samples_ns.end());
          if (Median(shifted_repeat) >
              shifted_fallback * (1.0 + options.maximum_distribution_shift_regression)) {
            evidence.rejected_reason = "distribution-shift performance regression";
            won_every_repeat = false;
            break;
          }
        }
        const std::vector<double> repeat_samples(
            evidence.samples_ns.begin() + static_cast<std::ptrdiff_t>(previous_sample_count),
            evidence.samples_ns.end());
        if (Median(repeat_samples) > fallback_median * (1.0 - options.minimum_improvement)) {
          evidence.rejected_reason = "did not improve or retain the priority profile";
          won_every_repeat = false;
          break;
        }
      }
      if (!evidence.samples_ns.empty()) {
        evidence.median_ns = Median(evidence.samples_ns);
        evidence.dispersion_ns = MedianAbsoluteDeviation(evidence.samples_ns, evidence.median_ns);
      }
      if (!evidence.distribution_shift_samples_ns.empty()) {
        evidence.distribution_shift_median_ns = Median(evidence.distribution_shift_samples_ns);
      }
      const std::size_t evidence_index = report.evidence.size();
      report.evidence.push_back(std::move(evidence));
      if (won_every_repeat && report.evidence.back().median_ns < best_median) {
        best = candidate;
        best_median = report.evidence.back().median_ns;
        best_evidence = evidence_index;
      }
      if (report.budget_exhausted) {
        break;
      }
    }
    if (best != nullptr) {
      report.selected_policy = report.evidence[best_evidence].policy;
      report.evidence[best_evidence].selected = true;
    }
    if (report.budget_exhausted) {
      break;
    }
  }
  report.changed = !(report.selected_policy == fallback);

  if (!options.evidence_path.empty()) {
    PersistCalibrationEvidence(report, options.evidence_path);
    report.persisted = true;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = std::find_if(exact_.begin(), exact_.end(),
                              [&](const ExactEntry &entry) { return entry.key == key; });
    if (found == exact_.end()) {
      exact_.push_back(
          {key, report.changed ? std::optional(report.selected_policy) : std::nullopt});
      found = std::prev(exact_.end());
    } else if (report.changed) {
      found->policy = report.selected_policy;
    }
    found->evidence = report.evidence;
    ++generation_;
  }
  return report;
}

void TreeEnsembleTuningRegistry::OverrideExact(const TreeEnsembleModelKey &key,
                                               TreeEnsembleTuningPolicy policy) {
  ValidateExactKey(key);
  ValidatePolicyShape(policy);
  std::lock_guard<std::mutex> lock(mutex_);
  auto found = std::find_if(exact_.begin(), exact_.end(),
                            [&](const ExactEntry &entry) { return entry.key == key; });
  if (found == exact_.end()) {
    exact_.push_back({key, std::nullopt, std::move(policy)});
  } else {
    found->override_policy = std::move(policy);
  }
  ++generation_;
}

void TreeEnsembleTuningRegistry::ClearExactOverride(const TreeEnsembleModelKey &key) {
  ValidateExactKey(key);
  std::lock_guard<std::mutex> lock(mutex_);
  const auto found = std::find_if(exact_.begin(), exact_.end(),
                                  [&](const ExactEntry &entry) { return entry.key == key; });
  if (found != exact_.end() && found->override_policy.has_value()) {
    found->override_policy.reset();
    ++generation_;
  }
}

void TreeEnsembleTuningRegistry::ForcePortable(const TreeEnsembleModelKey &key, bool enabled) {
  ValidateExactKey(key);
  std::lock_guard<std::mutex> lock(mutex_);
  auto found = std::find_if(exact_.begin(), exact_.end(),
                            [&](const ExactEntry &entry) { return entry.key == key; });
  if (found == exact_.end()) {
    exact_.push_back({key, std::nullopt, std::nullopt, {}, enabled});
  } else if (found->force_portable != enabled) {
    found->force_portable = enabled;
  } else {
    return;
  }
  ++generation_;
}

void TreeEnsembleTuningRegistry::SetCalibrationEnabled(bool enabled) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (calibration_enabled_ != enabled) {
    calibration_enabled_ = enabled;
    ++generation_;
  }
}

TreeEnsembleTuningInspection
TreeEnsembleTuningRegistry::InspectExact(const TreeEnsembleModelKey &key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  TreeEnsembleTuningInspection inspection;
  inspection.calibration_enabled = calibration_enabled_;
  const auto found = std::find_if(exact_.begin(), exact_.end(),
                                  [&](const ExactEntry &entry) { return entry.key == key; });
  if (found == exact_.end()) {
    return inspection;
  }
  inspection.override_policy = found->override_policy;
  inspection.selected_policy =
      found->override_policy.has_value() ? found->override_policy : found->policy;
  inspection.evidence = found->evidence;
  inspection.force_portable = found->force_portable;
  for (const TreeEnsembleCalibrationEvidence &evidence : found->evidence) {
    if (!evidence.rejected_reason.empty()) {
      inspection.rejected_reasons.push_back(evidence.candidate + ": " + evidence.rejected_reason);
    }
  }
  return inspection;
}

std::uint64_t TreeEnsembleTuningRegistry::generation() const noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  return generation_;
}

std::string TreeEnsemblePlan::MakeModelSignature(const TreeEnsembleAttributes &attributes,
                                                 const TreeEnsembleStructuralBuckets &buckets) {
  std::uint64_t hash = UINT64_C(14695981039346656037);
  const auto append_u64 = [&](std::uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
      hash ^= static_cast<std::uint8_t>(value >> shift);
      hash *= UINT64_C(1099511628211);
    }
  };
  const auto append_scalar = [&](const auto &value) {
    using T = std::remove_cvref_t<decltype(value)>;
    if constexpr (std::is_enum_v<T>) {
      append_u64(static_cast<std::uint64_t>(value));
    } else if constexpr (std::is_floating_point_v<T>) {
      append_u64(std::bit_cast<std::uint64_t>(static_cast<double>(value)));
    } else {
      append_u64(static_cast<std::uint64_t>(value));
    }
  };
  const auto append_vector = [&](const auto &values) {
    append_u64(values.size());
    for (const auto &value : values) {
      append_scalar(value);
    }
  };

  append_scalar(attributes.n_features);
  append_scalar(attributes.n_targets);
  append_scalar(attributes.value_type);
  append_scalar(attributes.aggregate);
  append_scalar(attributes.post_transform);
  append_vector(attributes.tree_roots);
  append_vector(attributes.nodes_featureids);
  append_vector(attributes.nodes_modes);
  append_vector(attributes.nodes_truenodeids);
  append_vector(attributes.nodes_falsenodeids);
  append_vector(attributes.nodes_trueleafs);
  append_vector(attributes.nodes_falseleafs);
  append_vector(attributes.nodes_missing_value_tracks_true);
  append_vector(attributes.nodes_hitrates);
  append_vector(attributes.membership_values);
  append_vector(attributes.leaf_targetids);
  append_scalar(attributes.base_values.size());
  append_scalar(buckets.tree_count);
  append_scalar(buckets.depth);
  append_scalar(buckets.target_count);
  append_scalar(buckets.branch_mode_mix);
  append_scalar(buckets.membership_density);

  std::ostringstream stream;
  stream << "tree_ensemble_v5:" << std::hex << std::setfill('0') << std::setw(16) << hash;
  return stream.str();
}

TreeEnsemblePlan::TreeEnsemblePlan(TreeEnsembleAttributes attributes)
    : TreeEnsemblePlan(std::move(attributes), {}, nullptr) {
  uses_dynamic_safe_policy_ = true;
}

TreeEnsemblePlan::TreeEnsemblePlan(TreeEnsembleAttributes attributes,
                                   TreeEnsembleTuningContext context,
                                   const TreeEnsembleTuningRegistry *registry)
    : attributes_(std::move(attributes)) {
  TreeEnsembleOracle oracle(attributes_);
  (void)oracle;
  tree_roots_ = attributes_.tree_roots;
  base_values_ = attributes_.base_values;
  const std::size_t size = attributes_.nodes_featureids.size();
  nodes_.resize(size);
  membership_sets_.resize(size);
  if (size <= std::numeric_limits<std::uint32_t>::max() &&
      static_cast<std::uint64_t>(attributes_.n_features) <=
          std::numeric_limits<std::uint32_t>::max() &&
      attributes_.leaf_targetids.size() <= std::numeric_limits<std::uint32_t>::max()) {
    feature_ids32_.resize(size);
    true_children32_.resize(size);
    false_children32_.resize(size);
    compact_nodes_.resize(size);
  }
  prepared_splits_.resize(size);
  prepared_half_splits_.resize(size);
  prepared_modes_.resize(size);
  prepared_flags_.resize(size);
  std::size_t cursor = 0;
  for (std::size_t index = 0; index < size; ++index) {
    nodes_[index].feature_id = attributes_.nodes_featureids[index];
    nodes_[index].split = attributes_.nodes_splits[index];
    nodes_[index].mode = attributes_.nodes_modes[index];
    nodes_[index].true_child = attributes_.nodes_truenodeids[index];
    nodes_[index].false_child = attributes_.nodes_falsenodeids[index];
    nodes_[index].true_is_leaf = attributes_.nodes_trueleafs[index] != 0;
    nodes_[index].false_is_leaf = attributes_.nodes_falseleafs[index] != 0;
    nodes_[index].missing_value_tracks_true =
        !attributes_.nodes_missing_value_tracks_true.empty() &&
        attributes_.nodes_missing_value_tracks_true[index] != 0;
    if (!feature_ids32_.empty()) {
      feature_ids32_[index] = static_cast<std::uint32_t>(nodes_[index].feature_id);
      true_children32_[index] = static_cast<std::uint32_t>(nodes_[index].true_child);
      false_children32_[index] = static_cast<std::uint32_t>(nodes_[index].false_child);
    }
    prepared_splits_[index] = RoundValue(nodes_[index].split, attributes_.value_type);
    prepared_half_splits_[index] = static_cast<float>(prepared_splits_[index]);
    prepared_modes_[index] = static_cast<std::uint8_t>(nodes_[index].mode);
    prepared_flags_[index] = static_cast<std::uint8_t>(
        (nodes_[index].true_is_leaf ? 1U : 0U) | (nodes_[index].false_is_leaf ? 2U : 0U) |
        (nodes_[index].missing_value_tracks_true ? 4U : 0U));
    if (!compact_nodes_.empty()) {
      compact_nodes_[index] = {prepared_splits_[index], feature_ids32_[index],
                               true_children32_[index], false_children32_[index],
                               prepared_modes_[index],  prepared_flags_[index]};
    }
    if (nodes_[index].mode == TreeBranchMode::kMember) {
      std::vector<double> set;
      while (cursor < attributes_.membership_values.size() &&
             !std::isnan(attributes_.membership_values[cursor])) {
        set.push_back(RoundValue(attributes_.membership_values[cursor], attributes_.value_type));
        ++cursor;
      }
      if (set.empty() || cursor == attributes_.membership_values.size()) {
        Invalid("each membership node requires a non-empty NaN-delimited set");
      }
      membership_sets_[index] = std::move(set);
      nodes_[index].members = membership_sets_[index];
      ++cursor;
    }
    if (nodes_[index].true_is_leaf) {
      nodes_[index].true_leaf_indices.push_back(static_cast<std::size_t>(nodes_[index].true_child));
    }
    if (nodes_[index].false_is_leaf) {
      nodes_[index].false_leaf_indices.push_back(
          static_cast<std::size_t>(nodes_[index].false_child));
    }
  }
  if (cursor != attributes_.membership_values.size()) {
    Invalid("membership_values contains missing or extra delimiters");
  }
  leaves_.resize(attributes_.leaf_targetids.size());
  for (std::size_t index = 0; index < attributes_.leaf_targetids.size(); ++index) {
    leaves_[index].target_id = attributes_.leaf_targetids[index];
    leaves_[index].weight = RoundValue(attributes_.leaf_weights[index], attributes_.value_type);
  }
  target_to_active_.assign(static_cast<std::size_t>(attributes_.n_targets),
                           std::numeric_limits<std::size_t>::max());
  for (const TreeEnsembleLeaf &leaf : leaves_) {
    const std::size_t target = static_cast<std::size_t>(leaf.target_id);
    if (target_to_active_[target] == std::numeric_limits<std::size_t>::max()) {
      target_to_active_[target] = active_targets_.size();
      active_targets_.push_back(target);
    }
  }
  if (!attributes_.nodes_hitrates.empty() && !feature_ids32_.empty()) {
    std::vector<std::size_t> order;
    order.reserve(size);
    const auto append_preorder = [&](auto &&self, std::size_t node) -> void {
      order.push_back(node);
      std::vector<std::size_t> children;
      if (!nodes_[node].true_is_leaf) {
        children.push_back(static_cast<std::size_t>(nodes_[node].true_child));
      }
      if (!nodes_[node].false_is_leaf) {
        children.push_back(static_cast<std::size_t>(nodes_[node].false_child));
      }
      std::sort(children.begin(), children.end(), [&](std::size_t left, std::size_t right) {
        return attributes_.nodes_hitrates[left] > attributes_.nodes_hitrates[right];
      });
      for (std::size_t child : children) {
        self(self, child);
      }
    };
    for (std::int64_t root : tree_roots_) {
      append_preorder(append_preorder, static_cast<std::size_t>(root));
    }
    std::vector<std::uint32_t> remap(size);
    for (std::size_t index = 0; index < order.size(); ++index) {
      remap[order[index]] = static_cast<std::uint32_t>(index);
    }
    hot_tree_roots_.reserve(tree_roots_.size());
    for (std::int64_t root : tree_roots_) {
      hot_tree_roots_.push_back(remap[static_cast<std::size_t>(root)]);
    }
    hot_feature_ids_.resize(size);
    hot_true_children_.resize(size);
    hot_false_children_.resize(size);
    hot_splits_.resize(size);
    hot_half_splits_.resize(size);
    hot_modes_.resize(size);
    hot_flags_.resize(size);
    hot_membership_indices_.resize(size);
    for (std::size_t index = 0; index < order.size(); ++index) {
      const std::size_t original = order[index];
      hot_feature_ids_[index] = feature_ids32_[original];
      hot_true_children_[index] = nodes_[original].true_is_leaf ? true_children32_[original]
                                                                : remap[true_children32_[original]];
      hot_false_children_[index] = nodes_[original].false_is_leaf
                                       ? false_children32_[original]
                                       : remap[false_children32_[original]];
      hot_splits_[index] = prepared_splits_[original];
      hot_half_splits_[index] = prepared_half_splits_[original];
      hot_modes_[index] = prepared_modes_[original];
      hot_flags_[index] = prepared_flags_[original];
      hot_membership_indices_[index] = static_cast<std::uint32_t>(original);
    }
  }
  const auto depth = [&](auto &&self, std::size_t node) -> std::size_t {
    if (attributes_.nodes_trueleafs[node] != 0 && attributes_.nodes_falseleafs[node] != 0) {
      return 1U;
    }
    std::size_t best = 1U;
    if (attributes_.nodes_trueleafs[node] == 0) {
      best = std::max(
          best, 1U + self(self, static_cast<std::size_t>(attributes_.nodes_truenodeids[node])));
    }
    if (attributes_.nodes_falseleafs[node] == 0) {
      best = std::max(
          best, 1U + self(self, static_cast<std::size_t>(attributes_.nodes_falsenodeids[node])));
    }
    return best;
  };
  std::vector<int> visited(size, 0);
  const auto calculate_depth = [&](auto &&self, std::size_t node) -> std::size_t {
    if (visited[node] == 2) {
      return 0U;
    }
    visited[node] = 1;
    const std::size_t answer = depth(self, node);
    visited[node] = 2;
    return answer;
  };
  std::size_t total_depth = 0;
  for (std::int64_t root : tree_roots_) {
    const std::size_t depth_value =
        calculate_depth(calculate_depth, static_cast<std::size_t>(root));
    total_depth += depth_value;
    max_depth_ = std::max(max_depth_, depth_value);
  }
  average_depth_ = tree_roots_.empty() ? 0U : total_depth / tree_roots_.size();
  all_trees_are_stumps_ = max_depth_ == 1;
  all_trees_are_symmetric_ = true;
  for (std::int64_t root : tree_roots_) {
    std::vector<std::int64_t> level_features;
    std::vector<TreeBranchMode> level_modes;
    std::vector<double> level_splits;
    std::optional<std::size_t> leaf_depth;
    const auto symmetric = [&](auto &&self, std::size_t node, std::size_t level) -> bool {
      if (level == level_features.size()) {
        level_features.push_back(nodes_[node].feature_id);
        level_modes.push_back(nodes_[node].mode);
        level_splits.push_back(prepared_splits_[node]);
      } else if (level_features[level] != nodes_[node].feature_id ||
                 level_modes[level] != nodes_[node].mode ||
                 level_splits[level] != prepared_splits_[node]) {
        return false;
      }
      const auto check_child = [&](bool is_leaf, std::int64_t child) {
        if (is_leaf) {
          const std::size_t depth = level + 1;
          if (!leaf_depth.has_value()) {
            leaf_depth = depth;
          }
          return *leaf_depth == depth;
        }
        return self(self, static_cast<std::size_t>(child), level + 1);
      };
      return check_child(nodes_[node].true_is_leaf, nodes_[node].true_child) &&
             check_child(nodes_[node].false_is_leaf, nodes_[node].false_child);
    };
    if (!symmetric(symmetric, static_cast<std::size_t>(root), 0) ||
        leaf_depth.value_or(0) != max_depth_) {
      all_trees_are_symmetric_ = false;
      break;
    }
  }
  const auto maximum_index =
      std::max({static_cast<std::uint64_t>(attributes_.n_features),
                static_cast<std::uint64_t>(attributes_.n_targets),
                static_cast<std::uint64_t>(attributes_.nodes_featureids.size()),
                static_cast<std::uint64_t>(attributes_.leaf_targetids.size()),
                static_cast<std::uint64_t>(attributes_.tree_roots.size())});
  uses_64_bit_indices_ =
      maximum_index > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max());

  structural_buckets_.tree_count = CountBucket(tree_roots_.size());
  structural_buckets_.depth = DepthBucket(max_depth_);
  structural_buckets_.target_count = CountBucket(static_cast<std::size_t>(attributes_.n_targets));
  structural_buckets_.branch_mode_mix =
      std::adjacent_find(attributes_.nodes_modes.begin(), attributes_.nodes_modes.end(),
                         std::not_equal_to<>()) == attributes_.nodes_modes.end()
          ? TreeEnsembleBranchMix::kHomogeneous
          : TreeEnsembleBranchMix::kMixed;
  const std::size_t membership_nodes = static_cast<std::size_t>(std::count(
      attributes_.nodes_modes.begin(), attributes_.nodes_modes.end(), TreeBranchMode::kMember));
  if (membership_nodes == 0) {
    structural_buckets_.membership_density = TreeEnsembleMembershipDensity::kNone;
  } else {
    const std::size_t values = attributes_.membership_values.size() > membership_nodes
                                   ? attributes_.membership_values.size() - membership_nodes
                                   : 0;
    structural_buckets_.membership_density = values <= membership_nodes * 8
                                                 ? TreeEnsembleMembershipDensity::kSparse
                                                 : TreeEnsembleMembershipDensity::kDense;
  }

  model_signature_ = MakeModelSignature(attributes_, structural_buckets_);
  context.threads =
      context.threads == 0 ? static_cast<std::size_t>(ExecutionThreadCount()) : context.threads;
  context.threads = std::max<std::size_t>(context.threads, 1);
  model_key_.input_type = attributes_.value_type;
  model_key_.processor = std::move(context.processor);
  model_key_.threads = context.threads;
  model_key_.model_digest = model_signature_;
  tuning_policy_ = MakeSafePolicy(tree_roots_.size(),
                                  static_cast<std::size_t>(attributes_.n_targets), context.threads);

  if (registry != nullptr) {
    std::lock_guard<std::mutex> lock(registry->mutex_);
    profile_generation_ = registry->generation_;
    const auto exact = std::find_if(registry->exact_.begin(), registry->exact_.end(),
                                    [&](const TreeEnsembleTuningRegistry::ExactEntry &entry) {
                                      return entry.key == model_key_;
                                    });
    const TreeEnsembleTuningPolicy *exact_policy = nullptr;
    if (exact != registry->exact_.end() && !exact->force_portable) {
      if (exact->override_policy.has_value()) {
        exact_policy = &*exact->override_policy;
      } else if (exact->policy.has_value()) {
        exact_policy = &*exact->policy;
      }
    }
    if (exact_policy != nullptr) {
      if (PolicyCompatible(*exact_policy, static_cast<std::size_t>(attributes_.n_targets),
                           context.threads, attributes_.value_type, uses_64_bit_indices_,
                           all_trees_are_stumps_, all_trees_are_symmetric_,
                           !attributes_.nodes_hitrates.empty(), active_targets_.size())) {
        tuning_policy_ = *exact_policy;
        profile_source_ = TreeEnsembleProfileSource::kExact;
      }
    } else {
      const auto portable =
          std::find_if(registry->portable_.begin(), registry->portable_.end(),
                       [&](const TreeEnsembleTuningRegistry::PortableEntry &entry) {
                         return entry.buckets == structural_buckets_;
                       });
      if (portable != registry->portable_.end() &&
          PolicyCompatible(portable->policy, static_cast<std::size_t>(attributes_.n_targets),
                           context.threads, attributes_.value_type, uses_64_bit_indices_,
                           all_trees_are_stumps_, all_trees_are_symmetric_,
                           !attributes_.nodes_hitrates.empty(), active_targets_.size())) {
        tuning_policy_ = portable->policy;
        profile_source_ = TreeEnsembleProfileSource::kPortable;
      }
    }
  }
  ValidatePolicyShape(tuning_policy_);
  workspace_bytes_ = 1;
  for (const TreeEnsembleExecutionRegion &region : tuning_policy_.regions) {
    workspace_bytes_ = std::max(workspace_bytes_, region.workspace_bytes);
  }
}

TreeEnsemblePlan::TreeEnsemblePlan(const TreeEnsembleRegressorAttributes &attributes)
    : TreeEnsemblePlan(LowerTreeEnsembleRegressor(attributes)) {}

TreeEnsemblePlan::TreeEnsemblePlan(const TreeEnsembleClassifierAttributes &attributes)
    : TreeEnsemblePlan(LowerTreeEnsembleClassifier(attributes)) {}

std::size_t TreeEnsemblePlan::prepared_storage_bytes() const noexcept {
  std::size_t bytes = 0;
  const auto add_vector = [&](const auto &values) {
    using Value = typename std::remove_cvref_t<decltype(values)>::value_type;
    bytes = SaturatingAdd(bytes, SaturatingMultiply(values.capacity(), sizeof(Value)));
  };
  add_vector(attributes_.tree_roots);
  add_vector(attributes_.nodes_featureids);
  add_vector(attributes_.nodes_splits);
  add_vector(attributes_.nodes_modes);
  add_vector(attributes_.nodes_truenodeids);
  add_vector(attributes_.nodes_falsenodeids);
  add_vector(attributes_.nodes_trueleafs);
  add_vector(attributes_.nodes_falseleafs);
  add_vector(attributes_.nodes_missing_value_tracks_true);
  add_vector(attributes_.nodes_hitrates);
  add_vector(attributes_.membership_values);
  add_vector(attributes_.leaf_targetids);
  add_vector(attributes_.leaf_weights);
  add_vector(attributes_.base_values);
  add_vector(tree_roots_);
  add_vector(nodes_);
  for (const TreeEnsembleNode &node : nodes_) {
    add_vector(node.members);
    add_vector(node.true_leaf_indices);
    add_vector(node.false_leaf_indices);
  }
  add_vector(leaves_);
  add_vector(base_values_);
  add_vector(membership_sets_);
  for (const std::vector<double> &members : membership_sets_) {
    add_vector(members);
  }
  add_vector(tuning_policy_.regions);
  add_vector(feature_ids32_);
  add_vector(true_children32_);
  add_vector(false_children32_);
  add_vector(prepared_splits_);
  add_vector(prepared_half_splits_);
  add_vector(prepared_modes_);
  add_vector(prepared_flags_);
  add_vector(compact_nodes_);
  add_vector(hot_tree_roots_);
  add_vector(hot_feature_ids_);
  add_vector(hot_true_children_);
  add_vector(hot_false_children_);
  add_vector(hot_splits_);
  add_vector(hot_half_splits_);
  add_vector(hot_modes_);
  add_vector(hot_flags_);
  add_vector(hot_membership_indices_);
  add_vector(active_targets_);
  add_vector(target_to_active_);
  bytes = SaturatingAdd(bytes, model_signature_.capacity());
  bytes = SaturatingAdd(bytes, model_key_.library.capacity());
  bytes = SaturatingAdd(bytes, model_key_.kernel.capacity());
  bytes = SaturatingAdd(bytes, model_key_.domain.capacity());
  bytes = SaturatingAdd(bytes, model_key_.implementation.capacity());
  bytes = SaturatingAdd(bytes, model_key_.processor.capacity());
  bytes = SaturatingAdd(bytes, model_key_.model_digest.capacity());
  return bytes;
}

void TreeEnsemblePlan::CompactRuntimeStorage() {
  if (uses_64_bit_indices_ || compact_nodes_.empty()) {
    return;
  }
  const auto release = [](auto &values) {
    using Vector = std::remove_cvref_t<decltype(values)>;
    Vector().swap(values);
  };

  release(attributes_.tree_roots);
  release(attributes_.nodes_featureids);
  release(attributes_.nodes_splits);
  release(attributes_.nodes_modes);
  release(attributes_.nodes_truenodeids);
  release(attributes_.nodes_falsenodeids);
  release(attributes_.nodes_trueleafs);
  release(attributes_.nodes_falseleafs);
  release(attributes_.nodes_missing_value_tracks_true);
  release(attributes_.nodes_hitrates);
  release(attributes_.membership_values);
  release(attributes_.leaf_targetids);
  release(attributes_.leaf_weights);
  release(nodes_);
  release(feature_ids32_);
  release(true_children32_);
  release(false_children32_);
  release(prepared_splits_);
  release(prepared_half_splits_);
  release(prepared_modes_);
  release(prepared_flags_);
  release(hot_tree_roots_);
  release(hot_feature_ids_);
  release(hot_true_children_);
  release(hot_false_children_);
  release(hot_splits_);
  release(hot_half_splits_);
  release(hot_modes_);
  release(hot_flags_);
  release(hot_membership_indices_);
  if (structural_buckets_.membership_density == TreeEnsembleMembershipDensity::kNone) {
    release(membership_sets_);
  }
}

std::vector<TreeEnsembleCalibrationCandidate>
TreeEnsemblePlan::GenerateCalibrationCandidates() const {
  std::vector<TreeEnsembleCalibrationCandidate> candidates;
  const auto add = [&](std::string name, TreeEnsembleCalibrationStage stage,
                       TreeEnsembleTuningPolicy policy, bool distribution_shift = true) {
    candidates.push_back(
        {std::move(name), stage, std::move(policy), prepared_storage_bytes(), distribution_shift});
  };
  if (!uses_64_bit_indices_) {
    TreeEnsembleTuningPolicy index = tuning_policy_;
    index.layout = TreeEnsembleNodeLayout::kCompactAosIndex;
    add("compact_aos_index", TreeEnsembleCalibrationStage::kLayout, std::move(index));

    TreeEnsembleTuningPolicy soa = tuning_policy_;
    soa.layout = TreeEnsembleNodeLayout::kSplitSoa;
    add("split_soa", TreeEnsembleCalibrationStage::kLayout, std::move(soa));
    if (!attributes_.nodes_hitrates.empty()) {
      TreeEnsembleTuningPolicy hot = tuning_policy_;
      hot.layout = TreeEnsembleNodeLayout::kPreorderHot;
      add("preorder_hot", TreeEnsembleCalibrationStage::kLayout, std::move(hot));
    }
  }
  if (all_trees_are_stumps_) {
    TreeEnsembleTuningPolicy stump = tuning_policy_;
    stump.traversal = TreeEnsembleTraversal::kStump;
    add("stump", TreeEnsembleCalibrationStage::kTraversal, std::move(stump));
  }
  if (all_trees_are_symmetric_ && !all_trees_are_stumps_) {
    TreeEnsembleTuningPolicy symmetric = tuning_policy_;
    symmetric.traversal = TreeEnsembleTraversal::kSymmetric;
    add("symmetric_tree", TreeEnsembleCalibrationStage::kTraversal, std::move(symmetric));
  }
  TreeEnsembleTuningPolicy interleaved = tuning_policy_;
  for (TreeEnsembleExecutionRegion &region : interleaved.regions) {
    region.strategy = TreeEnsembleExecutionStrategy::kInterleavedRows;
    region.maximum_threads = 1;
  }
  add("interleaved_rows", TreeEnsembleCalibrationStage::kScheduling, std::move(interleaved));

  const std::size_t targets = static_cast<std::size_t>(attributes_.n_targets);
  if (targets >= 16 && active_targets_.size() * 4 <= targets) {
    TreeEnsembleTuningPolicy sparse = tuning_policy_;
    sparse.target_layout = TreeEnsembleTargetLayout::kSparse;
    for (TreeEnsembleExecutionRegion &region : sparse.regions) {
      region.workspace_bytes = RegionWorkspaceBytes(region, active_targets_.size());
    }
    add("sparse_target", TreeEnsembleCalibrationStage::kTraversal, std::move(sparse));
  }
  if (attributes_.value_type == DataType::FLOAT16) {
    TreeEnsembleTuningPolicy half = tuning_policy_;
    half.optimized_float16 = true;
    add("optimized_float16", TreeEnsembleCalibrationStage::kTraversal, std::move(half));
  }
  for (const std::size_t distance : {1U, 2U, 4U}) {
    TreeEnsembleTuningPolicy prefetch = tuning_policy_;
    prefetch.traversal_prefetch_distance = distance;
    add("prefetch_" + std::to_string(distance), TreeEnsembleCalibrationStage::kPrefetch,
        std::move(prefetch));
  }
  return candidates;
}

TreeEnsembleExecutionDecision
TreeEnsemblePlan::SelectExecution(std::size_t rows, std::size_t effective_threads) const noexcept {
  const std::size_t trees = tree_roots_.size();
  if (effective_threads == 0) {
    effective_threads = static_cast<std::size_t>(ExecutionThreadCount());
  }
  effective_threads = std::max<std::size_t>(effective_threads, 1);
  if (rows == 0) {
    return {};
  }
  const auto accumulator_bytes = [&](TreeEnsembleExecutionStrategy strategy) {
    if (attributes_.n_targets == 1 && attributes_.aggregate == TreeAggregate::kSum &&
        attributes_.post_transform == TreePostTransform::kNone &&
        (rows == 1 || rows >= kRowParallelThreshold) &&
        strategy != TreeEnsembleExecutionStrategy::kTreeParallel) {
      return std::size_t{0};
    }
    const bool values_only = (strategy == TreeEnsembleExecutionStrategy::kTreeMajorBatch ||
                              strategy == TreeEnsembleExecutionStrategy::kInterleavedRows) &&
                             (attributes_.aggregate == TreeAggregate::kSum ||
                              attributes_.aggregate == TreeAggregate::kAverage);
    return sizeof(double) + (values_only ? 0U : sizeof(std::size_t));
  };
  if (uses_dynamic_safe_policy_) {
    TreeEnsembleExecutionDecision fallback;
    if (effective_threads == 1) {
      fallback.strategy = TreeEnsembleExecutionStrategy::kTreeMajorBatch;
    } else if (rows == 1 && trees > kTreeParallelThreshold) {
      fallback.strategy = TreeEnsembleExecutionStrategy::kTreeParallel;
    } else if (rows <= kRowParallelThreshold) {
      fallback.strategy = TreeEnsembleExecutionStrategy::kTreeMajorBatch;
    } else if (trees > effective_threads ||
               (attributes_.n_targets > 1 && trees == effective_threads)) {
      fallback.strategy = TreeEnsembleExecutionStrategy::kTreeParallel;
    } else {
      fallback.strategy = TreeEnsembleExecutionStrategy::kRowParallel;
    }
    if (fallback.strategy == TreeEnsembleExecutionStrategy::kTreeParallel) {
      fallback.participants = std::min(effective_threads, trees);
      fallback.batch_rows = std::min(rows, kTreeMajorBatchRows);
    } else if (fallback.strategy == TreeEnsembleExecutionStrategy::kRowParallel) {
      fallback.participants = std::min(effective_threads, rows);
      fallback.batch_rows = 1;
    } else {
      fallback.batch_rows = std::min(rows, kTreeMajorBatchRows);
    }
    fallback.row_chunk = (rows + fallback.participants - 1) / fallback.participants;
    fallback.tree_chunk = (trees + fallback.participants - 1) / fallback.participants;
    fallback.workspace_bytes = SaturatingMultiply(
        SaturatingMultiply(SaturatingMultiply(fallback.participants, fallback.batch_rows),
                           static_cast<std::size_t>(attributes_.n_targets)),
        accumulator_bytes(fallback.strategy));
    return fallback;
  }
  if (effective_threads == 1) {
    TreeEnsembleExecutionDecision serial;
    serial.strategy = TreeEnsembleExecutionStrategy::kTreeMajorBatch;
    serial.batch_rows = std::min(rows, kTreeMajorBatchRows);
    serial.tree_chunk = std::max<std::size_t>(trees, 1);
    serial.workspace_bytes = SaturatingMultiply(
        SaturatingMultiply(serial.batch_rows,
                           tuning_policy_.target_layout == TreeEnsembleTargetLayout::kSparse
                               ? active_targets_.size()
                               : static_cast<std::size_t>(attributes_.n_targets)),
        accumulator_bytes(serial.strategy));
    return serial;
  }

  const TreeEnsembleExecutionRegion *selected = &tuning_policy_.regions.back();
  for (const TreeEnsembleExecutionRegion &region : tuning_policy_.regions) {
    if (!region.maximum_rows.has_value() || rows <= *region.maximum_rows) {
      selected = &region;
      break;
    }
  }

  TreeEnsembleExecutionDecision decision;
  decision.strategy = selected->strategy;
  decision.row_chunk = selected->row_chunk;
  decision.tree_chunk = selected->tree_chunk;
  const std::size_t participant_cap = std::min(effective_threads, selected->maximum_threads);
  if (selected->strategy == TreeEnsembleExecutionStrategy::kTreeParallel) {
    decision.participants = std::min(participant_cap, trees);
    decision.batch_rows = std::min(rows, selected->batch_rows);
  } else if (selected->strategy == TreeEnsembleExecutionStrategy::kRowParallel) {
    decision.participants = std::min(participant_cap, rows);
    decision.batch_rows = std::min(rows, selected->batch_rows);
  } else {
    decision.participants = 1;
    decision.batch_rows = std::min(rows, selected->batch_rows);
  }
  decision.workspace_bytes = SaturatingMultiply(
      SaturatingMultiply(SaturatingMultiply(decision.participants, decision.batch_rows),
                         tuning_policy_.target_layout == TreeEnsembleTargetLayout::kSparse
                             ? active_targets_.size()
                             : static_cast<std::size_t>(attributes_.n_targets)),
      accumulator_bytes(decision.strategy));
  return decision;
}

template <typename T>
void TreeEnsemblePlan::EvaluateIntoImpl(const T *input, std::size_t input_size, std::size_t rows,
                                        T *output) const {
#if defined(__GNUC__) || defined(__clang__)
  const T *__restrict input_data = input;
  T *__restrict output_data = output;
#else
  const T *input_data = input;
  T *output_data = output;
#endif
  const std::size_t features = static_cast<std::size_t>(attributes_.n_features);
  const std::size_t targets = static_cast<std::size_t>(attributes_.n_targets);
  if (input_size != rows * features) {
    Invalid("input shape does not match n_features");
  }
  if (rows == 0) {
    return;
  }
  const bool sparse_targets = tuning_policy_.target_layout == TreeEnsembleTargetLayout::kSparse;
  const std::size_t accumulator_targets = sparse_targets ? active_targets_.size() : targets;
  const bool homogeneous_leq =
      structural_buckets_.branch_mode_mix == TreeEnsembleBranchMix::kHomogeneous &&
      !tuning_policy_.optimized_float16 && !compact_nodes_.empty() &&
      static_cast<TreeBranchMode>(compact_nodes_[0].mode) == TreeBranchMode::kLeq;
  const bool homogeneous_member =
      structural_buckets_.branch_mode_mix == TreeEnsembleBranchMix::kHomogeneous &&
      !tuning_policy_.optimized_float16 && !compact_nodes_.empty() &&
      static_cast<TreeBranchMode>(compact_nodes_[0].mode) == TreeBranchMode::kMember;

  const auto find_leaf = [&](std::size_t row, std::size_t tree) {
    if (homogeneous_leq && !uses_64_bit_indices_) {
      std::size_t node = static_cast<std::size_t>(tree_roots_[tree]);
      if (all_trees_are_symmetric_) {
        for (std::size_t level = 0; level < max_depth_; ++level) {
          const TreeEnsembleCompactNode &current = compact_nodes_[node];
          const double value =
              input_data[row * features + static_cast<std::size_t>(current.feature_id)];
          const bool go_true = value <= current.split;
          node = static_cast<std::size_t>(go_true ? current.true_child : current.false_child);
        }
        return node;
      }
      for (;;) {
        const TreeEnsembleCompactNode &current = compact_nodes_[node];
        const double value =
            input_data[row * features + static_cast<std::size_t>(current.feature_id)];
        const bool go_true = value <= current.split;
        const bool is_leaf = (current.flags & (go_true ? 1U : 2U)) != 0;
        const std::uint32_t next = go_true ? current.true_child : current.false_child;
        if (is_leaf) {
          return static_cast<std::size_t>(next);
        }
        node = static_cast<std::size_t>(next);
      }
    }
    if (homogeneous_member && !uses_64_bit_indices_) {
      std::size_t node = static_cast<std::size_t>(tree_roots_[tree]);
      for (;;) {
        const TreeEnsembleCompactNode &current = compact_nodes_[node];
        const double value =
            input_data[row * features + static_cast<std::size_t>(current.feature_id)];
        const std::vector<double> &members = membership_sets_[node];
        const bool go_true = std::find(members.begin(), members.end(), value) != members.end();
        const bool is_leaf = (current.flags & (go_true ? 1U : 2U)) != 0;
        const std::uint32_t next = go_true ? current.true_child : current.false_child;
        if (is_leaf) {
          return static_cast<std::size_t>(next);
        }
        node = static_cast<std::size_t>(next);
      }
    }
    if (tuning_policy_.layout == TreeEnsembleNodeLayout::kOrtCompactAosPointer &&
        tuning_policy_.traversal == TreeEnsembleTraversal::kGeneral &&
        !tuning_policy_.optimized_float16 && tuning_policy_.traversal_prefetch_distance == 0 &&
        !uses_64_bit_indices_) {
      std::size_t portable_node = static_cast<std::size_t>(tree_roots_[tree]);
      for (;;) {
        const TreeEnsembleCompactNode &current = compact_nodes_[portable_node];
        const double portable_value =
            input_data[row * features + static_cast<std::size_t>(current.feature_id)];
        static const std::vector<double> empty_members;
        const std::vector<double> &members =
            membership_sets_.empty() ? empty_members : membership_sets_[portable_node];
        const bool portable_true = std::isnan(portable_value)
                                       ? (current.flags & 4U) != 0
                                       : Compare(static_cast<TreeBranchMode>(current.mode),
                                                 portable_value, current.split, members);
        const bool portable_leaf = (current.flags & (portable_true ? 1U : 2U)) != 0;
        const std::uint32_t portable_next =
            portable_true ? current.true_child : current.false_child;
        if (portable_leaf) {
          return static_cast<std::size_t>(portable_next);
        }
        portable_node = static_cast<std::size_t>(portable_next);
      }
    }
    const bool hot = tuning_policy_.layout == TreeEnsembleNodeLayout::kPreorderHot;
    std::size_t node = hot ? static_cast<std::size_t>(hot_tree_roots_[tree])
                           : static_cast<std::size_t>(tree_roots_[tree]);
    std::size_t symmetric_levels = max_depth_;
    for (;;) {
      const bool soa = tuning_policy_.layout == TreeEnsembleNodeLayout::kSplitSoa;
      const bool compact = tuning_policy_.layout == TreeEnsembleNodeLayout::kCompactAosIndex;
      const std::size_t feature = hot ? static_cast<std::size_t>(hot_feature_ids_[node])
                                  : compact
                                      ? static_cast<std::size_t>(compact_nodes_[node].feature_id)
                                  : soa ? static_cast<std::size_t>(feature_ids32_[node])
                                        : static_cast<std::size_t>(nodes_[node].feature_id);
      const std::uint8_t flags =
          hot       ? hot_flags_[node]
          : compact ? compact_nodes_[node].flags
          : soa     ? prepared_flags_[node]
                    : static_cast<std::uint8_t>((nodes_[node].true_is_leaf ? 1U : 0U) |
                                                (nodes_[node].false_is_leaf ? 2U : 0U) |
                                                (nodes_[node].missing_value_tracks_true ? 4U : 0U));
      const TreeBranchMode mode = hot       ? static_cast<TreeBranchMode>(hot_modes_[node])
                                  : compact ? static_cast<TreeBranchMode>(compact_nodes_[node].mode)
                                  : soa     ? static_cast<TreeBranchMode>(prepared_modes_[node])
                                            : nodes_[node].mode;
      const double value =
          tuning_policy_.optimized_float16
              ? static_cast<double>(HalfBitsToFloat(
                    FloatToHalfBits(static_cast<float>(input_data[row * features + feature]))))
              : RoundValue(input_data[row * features + feature], attributes_.value_type);
      const double split = tuning_policy_.optimized_float16
                               ? static_cast<double>(hot       ? hot_half_splits_[node]
                                                     : compact ? compact_nodes_[node].split
                                                               : prepared_half_splits_[node])
                               : (hot       ? hot_splits_[node]
                                  : compact ? compact_nodes_[node].split
                                            : prepared_splits_[node]);
      const std::size_t membership_index =
          hot ? static_cast<std::size_t>(hot_membership_indices_[node]) : node;
      const bool go_true = std::isnan(value)
                               ? (flags & 4U) != 0
                               : Compare(mode, value, split, membership_sets_[membership_index]);
      const bool is_leaf = go_true ? (flags & 1U) != 0 : (flags & 2U) != 0;
      const std::int64_t next =
          hot       ? static_cast<std::int64_t>(go_true ? hot_true_children_[node]
                                                        : hot_false_children_[node])
          : compact ? static_cast<std::int64_t>(go_true ? compact_nodes_[node].true_child
                                                        : compact_nodes_[node].false_child)
          : soa     ? static_cast<std::int64_t>(go_true ? true_children32_[node]
                                                        : false_children32_[node])
                    : (go_true ? nodes_[node].true_child : nodes_[node].false_child);
      if (tuning_policy_.traversal == TreeEnsembleTraversal::kSymmetric &&
          --symmetric_levels == 0) {
        return static_cast<std::size_t>(next);
      }
      if (tuning_policy_.traversal == TreeEnsembleTraversal::kStump || is_leaf) {
        return static_cast<std::size_t>(next);
      }
      node = static_cast<std::size_t>(next);
#if defined(__GNUC__) || defined(__clang__)
      const std::size_t prefetch = tuning_policy_.traversal_prefetch_distance;
      if (prefetch != 0 && node + prefetch < nodes_.size()) {
        if (soa) {
          __builtin_prefetch(&feature_ids32_[node + prefetch], 0, 1);
          __builtin_prefetch(&prepared_splits_[node + prefetch], 0, 1);
        } else if (hot) {
          __builtin_prefetch(&hot_feature_ids_[node + prefetch], 0, 1);
          __builtin_prefetch(&hot_splits_[node + prefetch], 0, 1);
        } else if (compact) {
          __builtin_prefetch(&compact_nodes_[node + prefetch], 0, 1);
        } else {
          __builtin_prefetch(&nodes_[node + prefetch], 0, 1);
        }
      }
#endif
    }
  };

  const auto initialize = [&](std::vector<double> &values, std::vector<std::size_t> &counts,
                              std::size_t active_rows, bool include_base) {
    std::fill(values.begin(), values.end(), 0.0);
    std::fill(counts.begin(), counts.end(), 0);
    if (include_base) {
      for (std::size_t row = 0; row < active_rows; ++row) {
        for (std::size_t slot = 0; slot < accumulator_targets; ++slot) {
          const std::size_t target = sparse_targets ? active_targets_[slot] : slot;
          if (target < attributes_.base_values.size()) {
            values[row * accumulator_targets + slot] = attributes_.base_values[target];
          }
        }
      }
    }
  };

  const auto accumulate = [&](std::vector<double> &values, std::vector<std::size_t> &counts,
                              std::size_t offset, std::size_t leaf) {
    const std::size_t target = static_cast<std::size_t>(leaves_[leaf].target_id);
    const std::size_t slot = sparse_targets ? target_to_active_[target] : target;
    const std::size_t index = offset * accumulator_targets + slot;
    const double weight = leaves_[leaf].weight;
    if (attributes_.aggregate == TreeAggregate::kSum ||
        attributes_.aggregate == TreeAggregate::kAverage) {
      values[index] = attributes_.value_type == DataType::FLOAT
                          ? static_cast<float>(values[index] + weight)
                          : values[index] + weight;
      ++counts[index];
      return;
    }
    const double bias =
        target < attributes_.base_values.size() ? attributes_.base_values[target] : 0.0;
    const double contribution = RoundValue(bias + weight, attributes_.value_type);
    if (counts[index] == 0 && (attributes_.aggregate == TreeAggregate::kMin ||
                               attributes_.aggregate == TreeAggregate::kMax)) {
      values[index] = contribution;
    } else if (attributes_.aggregate == TreeAggregate::kMin) {
      values[index] = std::min(values[index], contribution);
    } else {
      values[index] = std::max(values[index], contribution);
    }
    ++counts[index];
  };

  const auto finish = [&](std::vector<double> &values, std::size_t output_row) {
    if (attributes_.aggregate == TreeAggregate::kAverage) {
      for (double &value : values) {
        value = RoundValue(value / static_cast<double>(tree_roots_.size()), attributes_.value_type);
      }
    }
    ApplyPostTransform(values, attributes_.post_transform);
    for (std::size_t target = 0; target < targets; ++target) {
      output_data[output_row * targets + target] =
          static_cast<T>(RoundValue(values[target], attributes_.value_type));
    }
  };

  const auto dense_row = [&](const std::vector<double> &values, std::size_t offset,
                             std::vector<double> &dense) {
    if (!sparse_targets) {
      std::copy_n(values.begin() + static_cast<std::ptrdiff_t>(offset * targets), targets,
                  dense.begin());
      return;
    }
    std::fill(dense.begin(), dense.end(), 0.0);
    for (std::size_t target = 0; target < targets && target < attributes_.base_values.size();
         ++target) {
      dense[target] = attributes_.base_values[target];
    }
    for (std::size_t slot = 0; slot < accumulator_targets; ++slot) {
      dense[active_targets_[slot]] = values[offset * accumulator_targets + slot];
    }
  };

  const TreeEnsembleExecutionDecision decision = SelectExecution(rows);
  if (rows == 1 && decision.strategy != TreeEnsembleExecutionStrategy::kTreeParallel &&
      targets == 1 && !sparse_targets && attributes_.aggregate == TreeAggregate::kSum &&
      attributes_.post_transform == TreePostTransform::kNone) {
    double value = attributes_.base_values.empty() ? 0.0 : attributes_.base_values[0];
    for (std::size_t tree = 0; tree < tree_roots_.size(); ++tree) {
      value += leaves_[find_leaf(0, tree)].weight;
      if (attributes_.value_type == DataType::FLOAT) {
        value = static_cast<float>(value);
      }
    }
    output_data[0] = static_cast<T>(value);
    return;
  }
  if ((decision.strategy == TreeEnsembleExecutionStrategy::kTreeMajorBatch ||
       decision.strategy == TreeEnsembleExecutionStrategy::kInterleavedRows) &&
      targets == 1 && !sparse_targets && attributes_.aggregate == TreeAggregate::kSum &&
      attributes_.post_transform == TreePostTransform::kNone) {
    if (rows >= kRowParallelThreshold) {
      const double base = attributes_.base_values.empty() ? 0.0 : attributes_.base_values[0];
      for (std::size_t row = 0; row < rows; ++row) {
        double value = base;
        for (std::size_t tree = 0; tree < tree_roots_.size(); ++tree) {
          value += leaves_[find_leaf(row, tree)].weight;
          if (attributes_.value_type == DataType::FLOAT) {
            value = static_cast<float>(value);
          }
        }
        output_data[row] = static_cast<T>(value);
      }
      return;
    }
    std::vector<double> values(decision.batch_rows);
    const double base = attributes_.base_values.empty() ? 0.0 : attributes_.base_values[0];
    for (std::size_t batch = 0; batch < rows; batch += decision.batch_rows) {
      const std::size_t active_rows = std::min(decision.batch_rows, rows - batch);
      std::fill_n(values.begin(), active_rows, base);
      for (std::size_t tree = 0; tree < tree_roots_.size(); ++tree) {
        for (std::size_t row = 0; row < active_rows; ++row) {
          const double weight = leaves_[find_leaf(batch + row, tree)].weight;
          values[row] = attributes_.value_type == DataType::FLOAT
                            ? static_cast<float>(values[row] + weight)
                            : values[row] + weight;
        }
      }
      std::transform(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(active_rows),
                     output_data + batch, [](double value) { return static_cast<T>(value); });
    }
    return;
  }
  if ((decision.strategy == TreeEnsembleExecutionStrategy::kTreeMajorBatch ||
       decision.strategy == TreeEnsembleExecutionStrategy::kInterleavedRows) &&
      (attributes_.aggregate == TreeAggregate::kSum ||
       attributes_.aggregate == TreeAggregate::kAverage)) {
    std::vector<double> values(decision.batch_rows * accumulator_targets);
    std::vector<double> row_values(targets);
    for (std::size_t batch = 0; batch < rows; batch += decision.batch_rows) {
      const std::size_t active_rows = std::min(decision.batch_rows, rows - batch);
      std::fill_n(values.begin(), active_rows * accumulator_targets, 0.0);
      for (std::size_t row = 0; row < active_rows; ++row) {
        for (std::size_t slot = 0; slot < accumulator_targets; ++slot) {
          const std::size_t target = sparse_targets ? active_targets_[slot] : slot;
          if (target < attributes_.base_values.size()) {
            values[row * accumulator_targets + slot] = attributes_.base_values[target];
          }
        }
      }
      for (std::size_t tree = 0; tree < tree_roots_.size(); ++tree) {
        for (std::size_t row = 0; row < active_rows; ++row) {
          const TreeEnsembleLeaf &leaf = leaves_[find_leaf(batch + row, tree)];
          const std::size_t target = static_cast<std::size_t>(leaf.target_id);
          const std::size_t slot = sparse_targets ? target_to_active_[target] : target;
          const std::size_t index = row * accumulator_targets + slot;
          values[index] = attributes_.value_type == DataType::FLOAT
                              ? static_cast<float>(values[index] + leaf.weight)
                              : values[index] + leaf.weight;
        }
      }
      for (std::size_t row = 0; row < active_rows; ++row) {
        dense_row(values, row, row_values);
        finish(row_values, batch + row);
      }
    }
    return;
  }
  if (decision.strategy == TreeEnsembleExecutionStrategy::kRowParallel) {
    ExecuteRanges(static_cast<std::int64_t>(rows), static_cast<double>(kExecutionGrainSize),
                  [&](std::int64_t begin, std::int64_t end) {
                    std::vector<double> values(accumulator_targets);
                    std::vector<std::size_t> counts(accumulator_targets);
                    std::vector<double> output_values(targets);
                    for (std::size_t row = static_cast<std::size_t>(begin);
                         row < static_cast<std::size_t>(end); ++row) {
                      initialize(values, counts, 1, true);
                      for (std::size_t tree = 0; tree < tree_roots_.size(); ++tree) {
                        accumulate(values, counts, 0, find_leaf(row, tree));
                      }
                      if (sparse_targets) {
                        dense_row(values, 0, output_values);
                        finish(output_values, row);
                      } else {
                        finish(values, row);
                      }
                    }
                  });
    return;
  }

  if (decision.strategy == TreeEnsembleExecutionStrategy::kTreeMajorBatch ||
      decision.strategy == TreeEnsembleExecutionStrategy::kInterleavedRows) {
    std::vector<double> values(decision.batch_rows * accumulator_targets);
    std::vector<std::size_t> counts(decision.batch_rows * accumulator_targets);
    std::vector<double> row_values(targets);
    for (std::size_t batch = 0; batch < rows; batch += decision.batch_rows) {
      const std::size_t active_rows = std::min(decision.batch_rows, rows - batch);
      initialize(values, counts, active_rows, true);
      for (std::size_t tree = 0; tree < tree_roots_.size(); ++tree) {
        constexpr std::size_t kInterleave = 4;
        const std::size_t row_step =
            decision.strategy == TreeEnsembleExecutionStrategy::kInterleavedRows ? kInterleave : 1;
        for (std::size_t row_begin = 0; row_begin < active_rows; row_begin += row_step) {
          const std::size_t row_end = std::min(active_rows, row_begin + row_step);
          for (std::size_t row = row_begin; row < row_end; ++row) {
            accumulate(values, counts, row, find_leaf(batch + row, tree));
          }
        }
      }
      for (std::size_t row = 0; row < active_rows; ++row) {
        dense_row(values, row, row_values);
        finish(row_values, batch + row);
      }
    }
    return;
  }

  const std::size_t participants = decision.participants;
  std::vector<double> partial_values(participants * decision.batch_rows * accumulator_targets);
  std::vector<std::size_t> partial_counts(participants * decision.batch_rows * accumulator_targets);
  std::vector<double> row_values(accumulator_targets);
  std::vector<std::size_t> row_counts(accumulator_targets);
  std::vector<double> output_values(targets);
  for (std::size_t batch = 0; batch < rows; batch += decision.batch_rows) {
    const std::size_t active_rows = std::min(decision.batch_rows, rows - batch);
    initialize(partial_values, partial_counts, participants * active_rows, false);
    ExecuteRanges(static_cast<std::int64_t>(participants), static_cast<double>(kExecutionGrainSize),
                  [&](std::int64_t begin, std::int64_t end) {
                    for (std::size_t participant = static_cast<std::size_t>(begin);
                         participant < static_cast<std::size_t>(end); ++participant) {
                      const std::size_t tree_begin =
                          tree_roots_.size() * participant / participants;
                      const std::size_t tree_end =
                          tree_roots_.size() * (participant + 1) / participants;
                      for (std::size_t tree = tree_begin; tree < tree_end; ++tree) {
                        for (std::size_t row = 0; row < active_rows; ++row) {
                          accumulate(partial_values, partial_counts,
                                     participant * active_rows + row, find_leaf(batch + row, tree));
                        }
                      }
                    }
                  });
    for (std::size_t row = 0; row < active_rows; ++row) {
      initialize(row_values, row_counts, 1, true);
      for (std::size_t participant = 0; participant < participants; ++participant) {
        const std::size_t partial_offset = (participant * active_rows + row) * accumulator_targets;
        for (std::size_t slot = 0; slot < accumulator_targets; ++slot) {
          if (partial_counts[partial_offset + slot] == 0) {
            continue;
          }
          if (row_counts[slot] == 0 && (attributes_.aggregate == TreeAggregate::kMin ||
                                        attributes_.aggregate == TreeAggregate::kMax)) {
            row_values[slot] = partial_values[partial_offset + slot];
          } else if (attributes_.aggregate == TreeAggregate::kSum ||
                     attributes_.aggregate == TreeAggregate::kAverage) {
            row_values[slot] = RoundValue(row_values[slot] + partial_values[partial_offset + slot],
                                          attributes_.value_type);
          } else if (attributes_.aggregate == TreeAggregate::kMin) {
            row_values[slot] = std::min(row_values[slot], partial_values[partial_offset + slot]);
          } else {
            row_values[slot] = std::max(row_values[slot], partial_values[partial_offset + slot]);
          }
          row_counts[slot] += partial_counts[partial_offset + slot];
        }
      }
      if (sparse_targets) {
        dense_row(row_values, 0, output_values);
        finish(output_values, batch + row);
      } else {
        finish(row_values, batch + row);
      }
    }
  }
}

std::vector<double> TreeEnsemblePlan::Evaluate(const std::vector<double> &input,
                                               std::size_t rows) const {
  std::vector<double> output(rows * static_cast<std::size_t>(attributes_.n_targets));
  EvaluateIntoImpl(input.data(), input.size(), rows, output.data());
  return output;
}

void TreeEnsemblePlan::EvaluateInto(const float *input, std::size_t input_size, std::size_t rows,
                                    float *output) const {
  EvaluateIntoImpl(input, input_size, rows, output);
}

void TreeEnsemblePlan::EvaluateInto(const double *input, std::size_t input_size, std::size_t rows,
                                    double *output) const {
  EvaluateIntoImpl(input, input_size, rows, output);
}

TreeEnsembleAttributes
LowerTreeEnsembleRegressor(const TreeEnsembleRegressorAttributes &attributes) {
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
  const auto weights = MakeWeights(attributes.target_treeids, attributes.target_nodeids,
                                   attributes.target_ids, attributes.target_weights);
  ValidateWeights(attributes.tree, prepared, weights);
  TreeEnsembleAttributes lowered;
  lowered.n_features = attributes.tree.n_features;
  lowered.n_targets = attributes.n_targets;
  lowered.value_type = DataType::DOUBLE;
  lowered.aggregate = attributes.aggregate;
  lowered.post_transform = attributes.post_transform;
  lowered.base_values = attributes.base_values;
  lowered.tree_roots.reserve(prepared.roots.size());
  std::unordered_map<LegacyKey, std::size_t, LegacyKeyHash> internal_nodes;
  std::unordered_map<LegacyKey, std::vector<std::size_t>, LegacyKeyHash> leaf_map;
  for (std::size_t index = 0; index < attributes.tree.nodes_nodeids.size(); ++index) {
    const LegacyKey key{attributes.tree.nodes_treeids[index], attributes.tree.nodes_nodeids[index]};
    if (attributes.tree.nodes_modes[index] == "LEAF") {
      auto it = weights.find(key);
      if (it == weights.end()) {
        Invalid("legacy regressor leaf has no target metadata");
      }
      for (const auto &[target_id, weight] : it->second) {
        lowered.leaf_targetids.push_back(target_id);
        lowered.leaf_weights.push_back(weight);
        leaf_map[key].push_back(lowered.leaf_targetids.size() - 1);
      }
      continue;
    }
    internal_nodes[key] = lowered.nodes_featureids.size();
    lowered.nodes_featureids.push_back(attributes.tree.nodes_featureids[index]);
    lowered.nodes_splits.push_back(attributes.tree.nodes_values[index]);
    lowered.nodes_modes.push_back(ParseLegacyBranchMode(attributes.tree.nodes_modes[index]));
    lowered.nodes_truenodeids.push_back(0);
    lowered.nodes_falsenodeids.push_back(0);
    lowered.nodes_trueleafs.push_back(0);
    lowered.nodes_falseleafs.push_back(0);
    if (!attributes.tree.nodes_missing_value_tracks_true.empty()) {
      lowered.nodes_missing_value_tracks_true.push_back(
          attributes.tree.nodes_missing_value_tracks_true[index]);
    }
  }
  for (std::size_t index = 0; index < attributes.tree.nodes_nodeids.size(); ++index) {
    const LegacyKey key{attributes.tree.nodes_treeids[index], attributes.tree.nodes_nodeids[index]};
    if (attributes.tree.nodes_modes[index] == "LEAF") {
      continue;
    }
    const std::size_t canonical_index = internal_nodes.at(key);
    const auto attach_child = [&](std::int64_t child, std::int64_t &target, std::int64_t &is_leaf) {
      const auto child_key = LegacyKey{attributes.tree.nodes_treeids[index], child};
      const auto it = internal_nodes.find(child_key);
      if (it != internal_nodes.end()) {
        target = static_cast<std::int64_t>(it->second);
        is_leaf = 0;
      } else {
        const auto leaf_it = leaf_map.find(child_key);
        if (leaf_it == leaf_map.end()) {
          Invalid("legacy path does not terminate at a valid node");
        }
        is_leaf = 1;
        target = static_cast<std::int64_t>(leaf_it->second.front());
      }
    };
    attach_child(attributes.tree.nodes_truenodeids[index],
                 lowered.nodes_truenodeids[canonical_index],
                 lowered.nodes_trueleafs[canonical_index]);
    attach_child(attributes.tree.nodes_falsenodeids[index],
                 lowered.nodes_falsenodeids[canonical_index],
                 lowered.nodes_falseleafs[canonical_index]);
  }
  for (std::int64_t root : prepared.roots) {
    const LegacyKey key{attributes.tree.nodes_treeids[static_cast<std::size_t>(root)],
                        attributes.tree.nodes_nodeids[static_cast<std::size_t>(root)]};
    lowered.tree_roots.push_back(static_cast<std::int64_t>(internal_nodes.at(key)));
  }
  return lowered;
}

TreeEnsembleAttributes
LowerTreeEnsembleClassifier(const TreeEnsembleClassifierAttributes &attributes) {
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
  const auto weights = MakeWeights(attributes.class_treeids, attributes.class_nodeids,
                                   attributes.class_ids, attributes.class_weights);
  ValidateWeights(attributes.tree, prepared, weights);
  TreeEnsembleAttributes lowered;
  lowered.n_features = attributes.tree.n_features;
  lowered.n_targets = static_cast<std::int64_t>(class_count);
  lowered.value_type = DataType::DOUBLE;
  lowered.aggregate = TreeAggregate::kSum;
  lowered.post_transform = attributes.post_transform;
  lowered.base_values = attributes.base_values;
  std::unordered_map<LegacyKey, std::size_t, LegacyKeyHash> internal_nodes;
  std::unordered_map<LegacyKey, std::vector<std::size_t>, LegacyKeyHash> leaf_map;
  for (std::size_t index = 0; index < attributes.tree.nodes_nodeids.size(); ++index) {
    const LegacyKey key{attributes.tree.nodes_treeids[index], attributes.tree.nodes_nodeids[index]};
    if (attributes.tree.nodes_modes[index] == "LEAF") {
      const auto it = weights.find(key);
      if (it == weights.end()) {
        Invalid("legacy classifier leaf has no class metadata");
      }
      for (const auto &[class_id, weight] : it->second) {
        lowered.leaf_targetids.push_back(static_cast<std::int64_t>(class_id));
        lowered.leaf_weights.push_back(weight);
        leaf_map[key].push_back(lowered.leaf_targetids.size() - 1);
      }
      continue;
    }
    internal_nodes[key] = lowered.nodes_featureids.size();
    lowered.nodes_featureids.push_back(attributes.tree.nodes_featureids[index]);
    lowered.nodes_splits.push_back(attributes.tree.nodes_values[index]);
    lowered.nodes_modes.push_back(ParseLegacyBranchMode(attributes.tree.nodes_modes[index]));
    lowered.nodes_truenodeids.push_back(0);
    lowered.nodes_falsenodeids.push_back(0);
    lowered.nodes_trueleafs.push_back(0);
    lowered.nodes_falseleafs.push_back(0);
    if (!attributes.tree.nodes_missing_value_tracks_true.empty()) {
      lowered.nodes_missing_value_tracks_true.push_back(
          attributes.tree.nodes_missing_value_tracks_true[index]);
    }
  }
  for (std::size_t index = 0; index < attributes.tree.nodes_nodeids.size(); ++index) {
    const LegacyKey key{attributes.tree.nodes_treeids[index], attributes.tree.nodes_nodeids[index]};
    if (attributes.tree.nodes_modes[index] == "LEAF") {
      continue;
    }
    const std::size_t canonical_index = internal_nodes.at(key);
    const auto attach_child = [&](std::int64_t child, std::int64_t &target, std::int64_t &is_leaf) {
      const auto child_key = LegacyKey{attributes.tree.nodes_treeids[index], child};
      const auto it = internal_nodes.find(child_key);
      if (it != internal_nodes.end()) {
        target = static_cast<std::int64_t>(it->second);
        is_leaf = 0;
      } else {
        const auto leaf_it = leaf_map.find(child_key);
        if (leaf_it == leaf_map.end()) {
          Invalid("legacy path does not terminate at a valid node");
        }
        is_leaf = 1;
        target = static_cast<std::int64_t>(leaf_it->second.front());
      }
    };
    attach_child(attributes.tree.nodes_truenodeids[index],
                 lowered.nodes_truenodeids[canonical_index],
                 lowered.nodes_trueleafs[canonical_index]);
    attach_child(attributes.tree.nodes_falsenodeids[index],
                 lowered.nodes_falsenodeids[canonical_index],
                 lowered.nodes_falseleafs[canonical_index]);
  }
  for (std::int64_t root : prepared.roots) {
    const LegacyKey key{attributes.tree.nodes_treeids[static_cast<std::size_t>(root)],
                        attributes.tree.nodes_nodeids[static_cast<std::size_t>(root)]};
    lowered.tree_roots.push_back(static_cast<std::int64_t>(internal_nodes.at(key)));
  }
  return lowered;
}

TreeEnsembleOracle::TreeEnsembleOracle(TreeEnsembleAttributes attributes)
    : attributes_(std::move(attributes)) {
  ValidateAggregate(attributes_.aggregate);
  ValidateTransform(attributes_.post_transform);
  if (attributes_.n_features <= 0 || attributes_.n_targets <= 0) {
    Invalid("n_features and n_targets must be positive");
  }
  if (!attributes_.base_values.empty() &&
      attributes_.base_values.size() != static_cast<std::size_t>(attributes_.n_targets)) {
    Invalid("base_values has the wrong length");
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
       attributes_.nodes_missing_value_tracks_true.size() != size) ||
      (!attributes_.nodes_hitrates.empty() && attributes_.nodes_hitrates.size() != size)) {
    Invalid("nodes_* attributes must have equal lengths");
  }
  if (std::any_of(attributes_.nodes_hitrates.begin(), attributes_.nodes_hitrates.end(),
                  [](double value) { return !std::isfinite(value) || value < 0.0; })) {
    Invalid("nodes_hitrates must be finite and non-negative");
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

std::vector<double> TreeEnsembleOracle::Evaluate(const std::vector<double> &input,
                                                 std::size_t rows) const {
  const std::size_t features = static_cast<std::size_t>(attributes_.n_features);
  const std::size_t targets = static_cast<std::size_t>(attributes_.n_targets);
  if (input.size() != rows * features) {
    Invalid("input shape does not match n_features");
  }
  std::vector<double> output(rows * targets);
  for (std::size_t row = 0; row < rows; ++row) {
    std::vector<double> values(targets, 0.0);
    for (std::size_t target = 0; target < targets && target < attributes_.base_values.size();
         ++target) {
      values[target] = attributes_.base_values[target];
    }
    std::vector<std::size_t> counts(targets, 0);
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
      const double bias =
          target < attributes_.base_values.size() ? attributes_.base_values[target] : 0.0;
      const double contribution = RoundValue(bias + weight, attributes_.value_type);
      if (counts[target] == 0) {
        if (attributes_.aggregate == TreeAggregate::kMin ||
            attributes_.aggregate == TreeAggregate::kMax) {
          values[target] = contribution;
        } else {
          values[target] = RoundValue(values[target] + weight, attributes_.value_type);
        }
      } else if (attributes_.aggregate == TreeAggregate::kSum ||
                 attributes_.aggregate == TreeAggregate::kAverage) {
        values[target] = RoundValue(values[target] + weight, attributes_.value_type);
      } else if (attributes_.aggregate == TreeAggregate::kMin) {
        values[target] = std::min(values[target], contribution);
      } else {
        values[target] = std::max(values[target], contribution);
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
        const std::size_t target = static_cast<std::size_t>(target_id);
        float &value = result[row * targets + target];
        const float bias = !attributes.base_values.empty()
                               ? static_cast<float>(attributes.base_values[target])
                               : 0.0f;
        const float contribution = bias + static_cast<float>(weight);
        if (counts[target] == 0) {
          if (attributes.aggregate == TreeAggregate::kMin ||
              attributes.aggregate == TreeAggregate::kMax) {
            value = contribution;
          } else {
            value += static_cast<float>(weight);
          }
        } else if (attributes.aggregate == TreeAggregate::kSum ||
                   attributes.aggregate == TreeAggregate::kAverage) {
          value += static_cast<float>(weight);
        } else if (attributes.aggregate == TreeAggregate::kMin) {
          value = std::min(value, contribution);
        } else {
          value = std::max(value, contribution);
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

} // namespace onnx_light_cpu
