// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/patterns/com_microsoft/patterns.h"

#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/builder/pattern_registry.h"
#include "onnx_proto/onnx_helper.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace onnx_light_cpu {
namespace {

namespace builder_ns = ONNX_LIGHT_NAMESPACE::core::builder;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

using builder_ns::BuilderError;
using builder_ns::GraphGraph;
using builder_ns::MatchResult;
using ONNX_LIGHT_NAMESPACE::AttributeProto;
using ONNX_LIGHT_NAMESPACE::FindAttribute;
using ONNX_LIGHT_NAMESPACE::MakeNode;
using ONNX_LIGHT_NAMESPACE::NodeProto;
using ONNX_LIGHT_NAMESPACE::TensorProto;

bool IsDefaultNode(const NodeProto *node, const char *op_type, std::size_t inputs) {
  if (node == nullptr || node->op_type() != op_type || node->input_size() != inputs ||
      node->output_size() != 1) {
    return false;
  }
  return node->domain().empty() || node->domain() == "ai.onnx";
}

bool HasOnlyConsumer(GraphGraph &graph, const NodeProto &node, const NodeProto *consumer) {
  if (node.output_size() != 1 || graph.IsOutput(node.output(0)) ||
      graph.IsUsedBySubgraph(node.output(0))) {
    return false;
  }
  const auto &consumers = graph.NextNodes(node.output(0));
  return consumers.size() == 1 && consumers[0] == consumer;
}

bool ExactGelu(const NodeProto &node) {
  const AttributeProto *approximate = FindAttribute(node, "approximate");
  return approximate == nullptr || (approximate->type() == AttributeProto::AttributeType::STRING &&
                                    approximate->s() == "none");
}

bool IsFloating(sym_ns::TensorType type) {
  return type == sym_ns::TensorType::kFloat16 || type == sym_ns::TensorType::kFloat ||
         type == sym_ns::TensorType::kDouble || type == sym_ns::TensorType::kBfloat16;
}

int64_t GetIntAttributeOrDefault(const NodeProto &node, const char *name, int64_t default_value) {
  const AttributeProto *attribute = FindAttribute(node, name);
  return attribute == nullptr ? default_value : attribute->i();
}

bool ReadSingleInt64(const TensorProto &tensor, int64_t &value) {
  bool single_element = true;
  for (int64_t dimension : tensor.dims()) {
    single_element = single_element && dimension == 1;
  }
  if (tensor.data_type() != TensorProto::DataType::INT64 || !single_element) {
    return false;
  }
  if (!tensor.int64_data().empty()) {
    value = tensor.int64_data()[0];
    return true;
  }
  if (tensor.is_raw_data() && tensor.ref_raw_data().size() == sizeof(int64_t)) {
    const auto &raw = tensor.ref_raw_data();
    uint64_t bits = 0;
    for (std::size_t i = 0; i < sizeof(int64_t); ++i) {
      bits |= static_cast<uint64_t>(raw[i]) << (8 * i);
    }
    value = static_cast<int64_t>(bits);
    return true;
  }
  return false;
}

bool ReadsAxis(GraphGraph &graph, const NodeProto &node, int64_t expected) {
  if (node.input_size() == 2) {
    const TensorProto *axes = graph.GetComputedConstant(node.input(1));
    int64_t axis = 0;
    return axes != nullptr && ReadSingleInt64(*axes, axis) && axis == expected;
  }
  const AttributeProto *axes = FindAttribute(node, "axes");
  return axes != nullptr && axes->type() == AttributeProto::AttributeType::INTS &&
         axes->ints().size() == 1 && axes->ints()[0] == expected;
}

bool KeepDims(const NodeProto &node, int64_t expected) {
  const AttributeProto *keepdims = FindAttribute(node, "keepdims");
  return keepdims == nullptr
             ? expected == 1
             : keepdims->type() == AttributeProto::AttributeType::INT && keepdims->i() == expected;
}

bool IsRank3GroupedAttention(GraphGraph &graph, const NodeProto &node) {
  if (!IsDefaultNode(&node, "Attention", 3) || !graph.HasShape(node.input(0)) ||
      !graph.HasShape(node.input(1)) || !graph.HasShape(node.input(2)) ||
      graph.GetShape(node.input(0)).Shape().Rank() != 3 ||
      graph.GetShape(node.input(1)).Shape().Rank() != 3 ||
      graph.GetShape(node.input(2)).Shape().Rank() != 3) {
    return false;
  }
  const AttributeProto *q_heads = FindAttribute(node, "q_num_heads");
  const AttributeProto *kv_heads = FindAttribute(node, "kv_num_heads");
  return q_heads != nullptr && kv_heads != nullptr && q_heads->i() > 0 && kv_heads->i() > 0 &&
         q_heads->i() != kv_heads->i() && q_heads->i() % kv_heads->i() == 0;
}

NodeProto MakeInt64Constant(const std::string &name, int64_t value) {
  NodeProto node = MakeNode("Constant", {}, {name});
  AttributeProto *attribute = node.add_attribute();
  attribute->set_name("value");
  attribute->set_type(AttributeProto::AttributeType::TENSOR);
  TensorProto &tensor = attribute->ref_t();
  tensor.set_data_type(TensorProto::DataType::INT64);
  tensor.ref_int64_data().push_back(value);
  return node;
}

NodeProto MakeCast(const std::string &input, const std::string &output, const std::string &name) {
  NodeProto node = MakeNode("Cast", {input}, {output}, "", name.c_str());
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "to",
                                     static_cast<int64_t>(TensorProto::DataType::INT32));
  return node;
}

} // namespace

std::set<std::string> BiasGeluFusionPattern::FastOpType() const { return {"Gelu"}; }

MatchResult BiasGeluFusionPattern::Match(GraphGraph &graph, const NodeProto &candidate) const {
  if (!IsDefaultNode(&candidate, "Gelu", 1) || !ExactGelu(candidate)) {
    return NoMatch(candidate, "candidate is not an exact default-domain Gelu");
  }
  const NodeProto *add = graph.NodeBefore(candidate.input(0));
  if (!IsDefaultNode(add, "Add", 2) || !HasOnlyConsumer(graph, *add, &candidate)) {
    return NoMatch(candidate, "Gelu input is not an exclusively consumed two-input Add");
  }
  if (!graph.HasShape(add->input(0)) || !graph.HasShape(add->input(1)) ||
      !graph.HasType(add->input(0)) ||
      graph.GetType(add->input(0)) != graph.GetType(add->input(1)) ||
      !IsFloating(graph.GetType(add->input(0)))) {
    return NoMatch(candidate, "Add input shapes or floating-point types are unavailable");
  }
  const auto &left = graph.GetShape(add->input(0)).Shape();
  const auto &right = graph.GetShape(add->input(1)).Shape();
  const bool left_bias = left.Rank() == 1 && right.Rank() > 1 && left[0] == right[right.Rank() - 1];
  const bool right_bias =
      right.Rank() == 1 && left.Rank() >= 1 && right[0] == left[left.Rank() - 1];
  if (!left_bias && !right_bias) {
    return NoMatch(candidate, "Add does not broadcast a rank-1 last-dimension bias");
  }
  return MatchResult{this, {add, &candidate}, nullptr};
}

ONNX_LIGHT_NAMESPACE::utils::RepeatedProtoField<NodeProto>
BiasGeluFusionPattern::Apply(GraphGraph &graph, const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr ||
      Match(graph, *nodes[1]).pattern == nullptr) {
    throw BuilderError("BiasGeluFusionPattern::Apply received an invalid match.");
  }
  const NodeProto &add = *nodes[0];
  const NodeProto &gelu = *nodes[1];
  const auto &left = graph.GetShape(add.input(0)).Shape();
  const bool left_bias = left.Rank() == 1 && graph.GetShape(add.input(1)).Shape().Rank() > 1;
  const std::string &input = add.input(left_bias ? 1 : 0);
  const std::string &bias = add.input(left_bias ? 0 : 1);
  graph.Builder().SetOpsetVersion(kMicrosoftDomain, 1);
  ONNX_LIGHT_NAMESPACE::utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(MakeNode("BiasGelu", {input, bias}, {gelu.output(0)}, kMicrosoftDomain,
                                  ("BiasGeluFusion--" + gelu.name()).c_str()));
  return replacements;
}

std::set<std::string> CDistFusionPattern::FastOpType() const { return {"ReduceSum"}; }

MatchResult CDistFusionPattern::Match(GraphGraph &graph, const NodeProto &candidate) const {
  if (!IsDefaultNode(&candidate, "ReduceSum", candidate.input_size()) ||
      (candidate.input_size() != 1 && candidate.input_size() != 2) ||
      !ReadsAxis(graph, candidate, -1) || !KeepDims(candidate, 0)) {
    return NoMatch(candidate, "candidate is not ReduceSum over the last axis");
  }
  const NodeProto *square = graph.NodeBefore(candidate.input(0));
  if (!IsDefaultNode(square, "Mul", 2) || square->input(0) != square->input(1) ||
      !HasOnlyConsumer(graph, *square, &candidate)) {
    return NoMatch(candidate, "ReduceSum input is not an exclusively consumed self-Mul");
  }
  const NodeProto *sub = graph.NodeBefore(square->input(0));
  if (!IsDefaultNode(sub, "Sub", 2) || !HasOnlyConsumer(graph, *sub, square)) {
    return NoMatch(candidate, "squared value is not produced by an exclusively consumed Sub");
  }
  const NodeProto *a_unsqueeze = graph.NodeBefore(sub->input(0));
  const NodeProto *b_unsqueeze = graph.NodeBefore(sub->input(1));
  if (!IsDefaultNode(a_unsqueeze, "Unsqueeze",
                     a_unsqueeze == nullptr ? 0 : a_unsqueeze->input_size()) ||
      !IsDefaultNode(b_unsqueeze, "Unsqueeze",
                     b_unsqueeze == nullptr ? 0 : b_unsqueeze->input_size()) ||
      (a_unsqueeze->input_size() != 1 && a_unsqueeze->input_size() != 2) ||
      (b_unsqueeze->input_size() != 1 && b_unsqueeze->input_size() != 2) ||
      !ReadsAxis(graph, *a_unsqueeze, 1) || !ReadsAxis(graph, *b_unsqueeze, 0) ||
      !HasOnlyConsumer(graph, *a_unsqueeze, sub) || !HasOnlyConsumer(graph, *b_unsqueeze, sub)) {
    return NoMatch(candidate, "Sub inputs are not exclusive Unsqueeze(A, 1) and Unsqueeze(B, 0)");
  }
  if (!graph.HasShape(a_unsqueeze->input(0)) || !graph.HasShape(b_unsqueeze->input(0)) ||
      graph.GetShape(a_unsqueeze->input(0)).Shape().Rank() != 2 ||
      graph.GetShape(b_unsqueeze->input(0)).Shape().Rank() != 2) {
    return NoMatch(candidate, "CDist source inputs are not known rank-2 tensors");
  }
  return MatchResult{this, {a_unsqueeze, b_unsqueeze, sub, square, &candidate}, nullptr};
}

ONNX_LIGHT_NAMESPACE::utils::RepeatedProtoField<NodeProto>
CDistFusionPattern::Apply(GraphGraph &graph, const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 5 || nodes[0] == nullptr || nodes[1] == nullptr || nodes[4] == nullptr ||
      Match(graph, *nodes[4]).pattern == nullptr) {
    throw BuilderError("CDistFusionPattern::Apply received an invalid match.");
  }
  graph.Builder().SetOpsetVersion(kMicrosoftDomain, 1);
  NodeProto replacement =
      MakeNode("CDist", {nodes[0]->input(0), nodes[1]->input(0)}, {nodes[4]->output(0)},
               kMicrosoftDomain, ("CDistFusion--" + nodes[4]->name()).c_str());
  ONNX_LIGHT_NAMESPACE::AddAttribute(replacement, "metric", std::string("sqeuclidean"));
  ONNX_LIGHT_NAMESPACE::utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(std::move(replacement));
  return replacements;
}

std::set<std::string> GroupQueryAttentionFusionPattern::FastOpType() const { return {"Attention"}; }

MatchResult GroupQueryAttentionFusionPattern::Match(GraphGraph &graph,
                                                    const NodeProto &candidate) const {
  if (!IsRank3GroupedAttention(graph, candidate)) {
    return NoMatch(candidate, "candidate is not a rank-3 grouped ai.onnx::Attention");
  }
  return MatchResult{this, {&candidate}, nullptr};
}

ONNX_LIGHT_NAMESPACE::utils::RepeatedProtoField<NodeProto>
GroupQueryAttentionFusionPattern::Apply(GraphGraph &graph,
                                        const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 1 || nodes[0] == nullptr || Match(graph, *nodes[0]).pattern == nullptr) {
    throw BuilderError("GroupQueryAttentionFusionPattern::Apply received an invalid match.");
  }
  const NodeProto &attention = *nodes[0];
  const std::string prefix = "GroupQueryAttentionFusion--" + attention.name();
  const std::string key_shape = prefix + "-key-shape";
  const std::string sequence_length = prefix + "-sequence-length";
  const std::string batch_shape = prefix + "-batch-shape";
  const std::string batch_size = prefix + "-batch-size";
  const std::string one = prefix + "-one";
  const std::string zero = prefix + "-zero";
  const std::string axes = prefix + "-axes";
  const std::string batch_dims = prefix + "-batch-dims";
  const std::string seqlen_value = prefix + "-seqlen-value";
  const std::string seqlen_value_int32 = prefix + "-seqlen-value-int32";
  const std::string sequence_length_int32 = prefix + "-sequence-length-int32";
  const std::string seqlens_k = prefix + "-seqlens-k";
  ONNX_LIGHT_NAMESPACE::utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(MakeInt64Constant(zero, 0));
  replacements.push_back(MakeInt64Constant(one, 1));
  replacements.push_back(MakeInt64Constant(axes, 0));
  replacements.push_back(
      MakeNode("Shape", {attention.input(1)}, {key_shape}, "", (prefix + "-ShapeKey").c_str()));
  replacements.push_back(MakeNode("Gather", {key_shape, one}, {sequence_length}, "",
                                  (prefix + "-GatherSequence").c_str()));
  replacements.push_back(
      MakeNode("Shape", {attention.input(0)}, {batch_shape}, "", (prefix + "-ShapeQuery").c_str()));
  replacements.push_back(
      MakeNode("Gather", {batch_shape, zero}, {batch_size}, "", (prefix + "-GatherBatch").c_str()));
  replacements.push_back(MakeNode("Unsqueeze", {batch_size, axes}, {batch_dims}, "",
                                  (prefix + "-UnsqueezeBatch").c_str()));
  replacements.push_back(MakeNode("Sub", {sequence_length, one}, {seqlen_value}, "",
                                  (prefix + "-SubtractOne").c_str()));
  replacements.push_back(
      MakeCast(seqlen_value, seqlen_value_int32, prefix + "-CastSequenceMinusOne"));
  replacements.push_back(
      MakeCast(sequence_length, sequence_length_int32, prefix + "-CastSequenceLength"));
  replacements.push_back(MakeNode("Expand", {seqlen_value_int32, batch_dims}, {seqlens_k}, "",
                                  (prefix + "-ExpandSequence").c_str()));
  NodeProto gqa = MakeNode("GroupQueryAttention",
                           {attention.input(0), attention.input(1), attention.input(2), "", "",
                            seqlens_k, sequence_length_int32},
                           {attention.output(0)}, kMicrosoftDomain, prefix.c_str());
  ONNX_LIGHT_NAMESPACE::AddAttribute(gqa, "num_heads",
                                     FindAttribute(attention, "q_num_heads")->i());
  ONNX_LIGHT_NAMESPACE::AddAttribute(gqa, "kv_num_heads",
                                     FindAttribute(attention, "kv_num_heads")->i());
  ONNX_LIGHT_NAMESPACE::AddAttribute(gqa, "causal",
                                     GetIntAttributeOrDefault(attention, "is_causal", 0));
  if (const AttributeProto *scale = FindAttribute(attention, "scale"); scale != nullptr) {
    ONNX_LIGHT_NAMESPACE::AddAttribute(gqa, "scale", scale->f());
  }
  if (const AttributeProto *softcap = FindAttribute(attention, "softcap"); softcap != nullptr) {
    ONNX_LIGHT_NAMESPACE::AddAttribute(gqa, "softcap", softcap->f());
  }
  replacements.push_back(std::move(gqa));
  graph.Builder().SetOpsetVersion(kMicrosoftDomain, 1);
  return replacements;
}

void RegisterCustomOperatorPatterns() {
  static std::once_flag once;
  std::call_once(once, [] {
    builder_ns::RegisterPattern("MicrosoftBiasGelu",
                                [] { return std::make_unique<BiasGeluFusionPattern>(); });
    builder_ns::RegisterPattern("MicrosoftCDist",
                                [] { return std::make_unique<CDistFusionPattern>(); });
    builder_ns::RegisterPattern("MicrosoftGroupQueryAttention", [] {
      return std::make_unique<GroupQueryAttentionFusionPattern>();
    });
  });
}

} // namespace onnx_light_cpu
