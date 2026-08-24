// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/traditionalml/include_traditionalml_cases.h"
#include "onnx_light_cpu/backend_test/cases/traditionalml/tree_ensemble_corpus.h"

#include "onnx_light_cpu/impl/traditionalml/tree_ensemble.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/memory/simple_tensor.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace onnx_light_cpu::backend_test {
namespace {

namespace bt_ns = ONNX_LIGHT_NAMESPACE::core::backend_test;
namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

using bt_ns::Expect;
using bt_ns::IoData;
using ONNX_LIGHT_NAMESPACE::AttributeProto;
using ONNX_LIGHT_NAMESPACE::NodeProto;
using ONNX_LIGHT_NAMESPACE::TensorProto;
using rt_ns::DataType;
using rt_ns::DefaultOpset;
using rt_ns::OpsetId;
using rt_ns::Tensor;
constexpr std::int64_t kMlOpsetVersion = 5;

void AddInt(NodeProto &node, const char *name, std::int64_t value) {
  AttributeProto *attribute = node.add_attribute();
  attribute->set_name(name);
  attribute->set_type(AttributeProto::AttributeType::INT);
  attribute->set_i(value);
}

void AddInts(NodeProto &node, const char *name, const std::vector<std::int64_t> &values) {
  AttributeProto *attribute = node.add_attribute();
  attribute->set_name(name);
  attribute->set_type(AttributeProto::AttributeType::INTS);
  for (std::int64_t value : values) {
    attribute->add_ints(value);
  }
}

void AddString(NodeProto &node, const char *name, const std::string &value) {
  AttributeProto *attribute = node.add_attribute();
  attribute->set_name(name);
  attribute->set_type(AttributeProto::AttributeType::STRING);
  attribute->set_s(value);
}

void AddStrings(NodeProto &node, const char *name, const std::vector<std::string> &values) {
  AttributeProto *attribute = node.add_attribute();
  attribute->set_name(name);
  attribute->set_type(AttributeProto::AttributeType::STRINGS);
  for (const std::string &value : values) {
    *attribute->add_strings() = ONNX_LIGHT_NAMESPACE::utils::String(value);
  }
}

void AddFloats(NodeProto &node, const char *name, const std::vector<double> &values) {
  AttributeProto *attribute = node.add_attribute();
  attribute->set_name(name);
  attribute->set_type(AttributeProto::AttributeType::FLOATS);
  for (double value : values) {
    attribute->add_floats(static_cast<float>(value));
  }
}

template <typename T>
void AddTensor(NodeProto &node, const char *name, TensorProto::DataType data_type,
               const std::vector<T> &values) {
  AttributeProto *attribute = node.add_attribute();
  attribute->set_name(name);
  attribute->set_type(AttributeProto::AttributeType::TENSOR);
  TensorProto *tensor = attribute->add_t();
  tensor->set_data_type(data_type);
  tensor->add_dims(static_cast<std::uint64_t>(values.size()));
  std::vector<std::uint8_t> raw(values.size() * sizeof(T));
  std::memcpy(raw.data(), values.data(), raw.size());
  tensor->set_raw_data(ONNX_LIGHT_NAMESPACE::utils::ByteSpan(raw));
}

void AddTypedTensor(NodeProto &node, const char *name, TreeValueType type,
                    const std::vector<double> &values) {
  if (type == TreeValueType::kFloat64) {
    AddTensor(node, name, TensorProto::DOUBLE, values);
    return;
  }
  std::vector<float> floats(values.begin(), values.end());
  if (type == TreeValueType::kFloat32) {
    AddTensor(node, name, TensorProto::FLOAT, floats);
    return;
  }
  std::vector<std::uint16_t> halves;
  halves.reserve(values.size());
  for (float value : floats) {
    halves.push_back(rt_ns::FloatToFloat16Bits(value));
  }
  AddTensor(node, name, TensorProto::FLOAT16, halves);
}

void AddModes(NodeProto &node, const std::vector<TreeBranchMode> &modes) {
  std::vector<std::uint8_t> values;
  values.reserve(modes.size());
  for (TreeBranchMode mode : modes) {
    values.push_back(static_cast<std::uint8_t>(mode));
  }
  AddTensor(node, "nodes_modes", TensorProto::UINT8, values);
}

NodeProto MakeTreeEnsembleNode(const TreeEnsembleAttributes &attributes) {
  NodeProto node;
  node.set_op_type("TreeEnsemble");
  node.set_domain("ai.onnx.ml");
  node.add_input("X");
  node.add_output("Y");
  AddInts(node, "tree_roots", attributes.tree_roots);
  AddInts(node, "nodes_featureids", attributes.nodes_featureids);
  AddTypedTensor(node, "nodes_splits", attributes.value_type, attributes.nodes_splits);
  AddModes(node, attributes.nodes_modes);
  AddInts(node, "nodes_truenodeids", attributes.nodes_truenodeids);
  AddInts(node, "nodes_falsenodeids", attributes.nodes_falsenodeids);
  AddInts(node, "nodes_trueleafs", attributes.nodes_trueleafs);
  AddInts(node, "nodes_falseleafs", attributes.nodes_falseleafs);
  if (!attributes.nodes_missing_value_tracks_true.empty()) {
    AddInts(node, "nodes_missing_value_tracks_true", attributes.nodes_missing_value_tracks_true);
  }
  if (!attributes.membership_values.empty()) {
    AddTypedTensor(node, "membership_values", attributes.value_type, attributes.membership_values);
  }
  AddInts(node, "leaf_targetids", attributes.leaf_targetids);
  AddTypedTensor(node, "leaf_weights", attributes.value_type, attributes.leaf_weights);
  if (!attributes.base_values.empty()) {
    AddTypedTensor(node, "base_values", attributes.value_type, attributes.base_values);
  }
  AddInt(node, "n_targets", attributes.n_targets);
  AddInt(node, "aggregate_function", static_cast<std::int64_t>(attributes.aggregate));
  AddInt(node, "post_transform", static_cast<std::int64_t>(attributes.post_transform));
  return node;
}

std::vector<float> ToFloat(const std::vector<double> &values) {
  return std::vector<float>(values.begin(), values.end());
}

Tensor MakeTypedTensor(const std::vector<double> &values, const std::vector<std::int64_t> &shape,
                       TreeValueType type) {
  if (type == TreeValueType::kFloat64) {
    return Tensor::FromDouble("", shape, values);
  }
  const std::vector<float> floats = ToFloat(values);
  if (type == TreeValueType::kFloat16) {
    return rt_ns::MakeFloat16Tensor("", shape, floats);
  }
  return Tensor::FromFloat("", shape, floats);
}

void AddLegacyTree(NodeProto &node, const LegacyTreeAttributes &tree) {
  AddInts(node, "nodes_treeids", tree.nodes_treeids);
  AddInts(node, "nodes_nodeids", tree.nodes_nodeids);
  AddInts(node, "nodes_featureids", tree.nodes_featureids);
  AddFloats(node, "nodes_values", tree.nodes_values);
  AddStrings(node, "nodes_modes", tree.nodes_modes);
  AddInts(node, "nodes_truenodeids", tree.nodes_truenodeids);
  AddInts(node, "nodes_falsenodeids", tree.nodes_falsenodeids);
  if (!tree.nodes_missing_value_tracks_true.empty()) {
    AddInts(node, "nodes_missing_value_tracks_true", tree.nodes_missing_value_tracks_true);
  }
}

LegacyTreeAttributes MakeLegacyStump() {
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

TreeEnsembleRegressorAttributes MakeRegressor() {
  TreeEnsembleRegressorAttributes attributes;
  attributes.tree = MakeLegacyStump();
  attributes.n_targets = 2;
  attributes.target_treeids = {0, 0, 0, 0};
  attributes.target_nodeids = {1, 1, 2, 2};
  attributes.target_ids = {0, 1, 0, 1};
  attributes.target_weights = {1.0, -1.0, 2.0, -2.0};
  attributes.base_values = {0.5, 0.5};
  attributes.post_transform = TreePostTransform::kLogistic;
  return attributes;
}

TreeEnsembleAttributes MakeBenchmarkForest(std::size_t trees, std::int64_t features) {
  constexpr std::size_t kDepth = 4;
  constexpr std::size_t kInternalNodes = (1U << kDepth) - 1;
  constexpr std::size_t kLeaves = 1U << kDepth;
  TreeEnsembleAttributes attributes;
  attributes.n_features = features;
  attributes.n_targets = 1;
  attributes.value_type = TreeValueType::kFloat32;
  for (std::size_t tree = 0; tree < trees; ++tree) {
    const std::size_t node_offset = tree * kInternalNodes;
    const std::size_t leaf_offset = tree * kLeaves;
    attributes.tree_roots.push_back(static_cast<std::int64_t>(node_offset));
    for (std::size_t node = 0; node < kInternalNodes; ++node) {
      attributes.nodes_featureids.push_back(
          static_cast<std::int64_t>((tree + node) % static_cast<std::size_t>(features)));
      attributes.nodes_splits.push_back(static_cast<double>(static_cast<int>(node % 7) - 3) * 0.25);
      attributes.nodes_modes.push_back(TreeBranchMode::kLeq);
      for (bool true_branch : {true, false}) {
        const std::size_t child = 2 * node + (true_branch ? 1 : 2);
        const bool leaf = child >= kInternalNodes;
        const std::int64_t child_id = static_cast<std::int64_t>(
            leaf ? leaf_offset + child - kInternalNodes : node_offset + child);
        if (true_branch) {
          attributes.nodes_truenodeids.push_back(child_id);
          attributes.nodes_trueleafs.push_back(leaf ? 1 : 0);
        } else {
          attributes.nodes_falsenodeids.push_back(child_id);
          attributes.nodes_falseleafs.push_back(leaf ? 1 : 0);
        }
      }
    }
    for (std::size_t leaf = 0; leaf < kLeaves; ++leaf) {
      attributes.leaf_targetids.push_back(0);
      attributes.leaf_weights.push_back(
          static_cast<double>(static_cast<int>((leaf + tree) % 11) - 5) /
          (static_cast<double>(trees) * 8.0));
    }
  }
  return attributes;
}

struct TreeEnsembleBenchmarkSpec {
  std::size_t trees;
  std::int64_t features;
  std::size_t rows;
};

constexpr TreeEnsembleBenchmarkSpec kTreeEnsembleBenchmarkCases[] = {
    {10, 4, 1}, {100, 64, 8}, {1000, 1024, 32}, {10000, 4096, 1}, {10000, 4096, 128},
};

NodeProto MakeRegressorNode(const TreeEnsembleRegressorAttributes &attributes) {
  NodeProto node;
  node.set_op_type("TreeEnsembleRegressor");
  node.set_domain("ai.onnx.ml");
  node.add_input("X");
  node.add_output("Y");
  AddLegacyTree(node, attributes.tree);
  AddInt(node, "n_targets", attributes.n_targets);
  AddInts(node, "target_treeids", attributes.target_treeids);
  AddInts(node, "target_nodeids", attributes.target_nodeids);
  AddInts(node, "target_ids", attributes.target_ids);
  AddFloats(node, "target_weights", attributes.target_weights);
  AddFloats(node, "base_values", attributes.base_values);
  AddString(node, "aggregate_function", "SUM");
  AddString(node, "post_transform", "LOGISTIC");
  return node;
}

TreeEnsembleClassifierAttributes MakeClassifier(bool string_labels) {
  TreeEnsembleClassifierAttributes attributes;
  attributes.tree = MakeLegacyStump();
  attributes.class_treeids = {0, 0, 0, 0};
  attributes.class_nodeids = {1, 1, 2, 2};
  attributes.class_ids = {0, 1, 0, 1};
  attributes.class_weights = {1.0, 1.0, 0.0, 2.0};
  if (string_labels) {
    attributes.labels = std::vector<std::string>{"left", "right"};
  } else {
    attributes.labels = std::vector<std::int64_t>{10, 20};
  }
  return attributes;
}

NodeProto MakeClassifierNode(const TreeEnsembleClassifierAttributes &attributes) {
  NodeProto node;
  node.set_op_type("TreeEnsembleClassifier");
  node.set_domain("ai.onnx.ml");
  node.add_input("X");
  node.add_output("Y");
  node.add_output("Z");
  AddLegacyTree(node, attributes.tree);
  AddInts(node, "class_treeids", attributes.class_treeids);
  AddInts(node, "class_nodeids", attributes.class_nodeids);
  AddInts(node, "class_ids", attributes.class_ids);
  AddFloats(node, "class_weights", attributes.class_weights);
  if (const auto *labels = std::get_if<std::vector<std::int64_t>>(&attributes.labels)) {
    AddInts(node, "classlabels_int64s", *labels);
  } else {
    AddStrings(node, "classlabels_strings", std::get<std::vector<std::string>>(attributes.labels));
  }
  AddString(node, "post_transform", "NONE");
  return node;
}

template <typename T> Tensor MakeNumericInput(const std::vector<T> &values) {
  return Tensor::From<T>("", {static_cast<std::int64_t>(values.size()), 1}, values);
}

} // namespace

void RegisterCpuTreeEnsembleCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId ml_opset("ai.onnx.ml", kMlOpsetVersion);
  if (mode == TestMode::BENCHMARK) {
    for (const TreeEnsembleBenchmarkSpec &spec : kTreeEnsembleBenchmarkCases) {
      TreeEnsembleAttributes attributes = MakeBenchmarkForest(spec.trees, spec.features);
      NodeProto node = MakeTreeEnsembleNode(attributes);
      const std::string name = "test_cpu_treeensemble_t" + std::to_string(spec.trees) + "_f" +
                               std::to_string(spec.features) + "_b" + std::to_string(spec.rows) +
                               "_benchmark";
      Expect(registry, std::move(node), name, {DefaultOpset(13), ml_opset},
             [attributes = std::move(attributes), spec]() mutable -> IoData {
               std::vector<double> values(spec.rows * static_cast<std::size_t>(spec.features));
               for (std::size_t index = 0; index < values.size(); ++index) {
                 values[index] = static_cast<double>(static_cast<int>(index % 17) - 8) * 0.25;
               }
               TreeEnsembleOracle oracle(attributes);
               std::vector<double> expected = oracle.Evaluate(values, spec.rows);
               Tensor input = Tensor::FromFloat(
                   "", {static_cast<std::int64_t>(spec.rows), spec.features}, ToFloat(values));
               Tensor output = Tensor::FromFloat("", {static_cast<std::int64_t>(spec.rows), 1},
                                                 ToFloat(expected));
               return IoData{{std::move(input)}, {std::move(output)}};
             });
    }
    return;
  }
  if (mode != TestMode::TEST) {
    return;
  }
  for (TreeEnsembleCorpusCase test_case : GenerateTreeEnsembleV5Corpus()) {
    NodeProto node = MakeTreeEnsembleNode(test_case.attributes);
    const std::string name = "test_cpu_treeensemble_v5_" + test_case.name;
    Expect(registry, std::move(node), name, {DefaultOpset(13), ml_opset},
           [test_case = std::move(test_case)]() mutable -> IoData {
             const std::vector<std::int64_t> input_shape = {
                 static_cast<std::int64_t>(test_case.rows), test_case.attributes.n_features};
             const std::vector<std::int64_t> output_shape = {
                 static_cast<std::int64_t>(test_case.rows), test_case.attributes.n_targets};
             Tensor input =
                 MakeTypedTensor(test_case.input, input_shape, test_case.attributes.value_type);
             Tensor expected =
                 MakeTypedTensor(test_case.expected, output_shape, test_case.attributes.value_type);
             return IoData{{std::move(input)}, {std::move(expected)}};
           });
  }
}

void RegisterCpuTreeEnsembleRegressorCases(std::vector<TestCase> &registry, TestMode mode) {
  if (mode != TestMode::TEST) {
    return;
  }
  const OpsetId ml_opset("ai.onnx.ml", kMlOpsetVersion);
  const TreeEnsembleRegressorAttributes attributes = MakeRegressor();
  const std::vector<double> input_values = {-1.0, 1.0};
  const std::vector<float> expected = EvaluateTreeEnsembleRegressor(attributes, input_values, 2);
  for (DataType type : {DataType::FLOAT, DataType::DOUBLE, DataType::INT32, DataType::INT64}) {
    NodeProto node = MakeRegressorNode(attributes);
    const std::string suffix = type == DataType::FLOAT    ? "float"
                               : type == DataType::DOUBLE ? "double"
                               : type == DataType::INT32  ? "int32"
                                                          : "int64";
    Expect(registry, std::move(node), "test_cpu_treeensembleregressor_v5_" + suffix,
           {DefaultOpset(13), ml_opset}, [type, expected]() mutable -> IoData {
             Tensor input = type == DataType::FLOAT    ? MakeNumericInput<float>({-1.0f, 1.0f})
                            : type == DataType::DOUBLE ? MakeNumericInput<double>({-1.0, 1.0})
                            : type == DataType::INT32  ? MakeNumericInput<std::int32_t>({-1, 1})
                                                       : MakeNumericInput<std::int64_t>({-1, 1});
             Tensor output = Tensor::FromFloat("", {2, 2}, expected);
             return IoData{{std::move(input)}, {std::move(output)}};
           });
  }
}

void RegisterCpuTreeEnsembleClassifierCases(std::vector<TestCase> &registry, TestMode mode) {
  if (mode != TestMode::TEST) {
    return;
  }
  const OpsetId ml_opset("ai.onnx.ml", kMlOpsetVersion);
  for (bool string_labels : {false, true}) {
    const TreeEnsembleClassifierAttributes attributes = MakeClassifier(string_labels);
    const TreeClassifierResult expected =
        EvaluateTreeEnsembleClassifier(attributes, {-1.0, 1.0}, 2);
    NodeProto node = MakeClassifierNode(attributes);
    const std::string suffix = string_labels ? "strings" : "int64";
    Expect(registry, std::move(node), "test_cpu_treeensembleclassifier_v5_" + suffix,
           {DefaultOpset(13), ml_opset}, [string_labels, expected]() mutable -> IoData {
             Tensor input = Tensor::FromFloat("", {2, 1}, {-1.0f, 1.0f});
             Tensor labels = string_labels ? Tensor::FromStrings("", {2}, expected.string_labels)
                                           : Tensor::FromInt64("", {2}, expected.integer_labels);
             Tensor scores = Tensor::FromFloat("", {2, 2}, expected.scores);
             return IoData{{std::move(input)}, {std::move(labels), std::move(scores)}};
           });
  }
}

} // namespace onnx_light_cpu::backend_test
