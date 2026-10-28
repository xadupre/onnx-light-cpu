// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/patterns/com_microsoft/patterns.h"

#include "onnx_core/builder/graph_graph.h"
#include "onnx_proto/onnx_helper.h"

#include <algorithm>
#include <bit>
#include <limits>
#include <unordered_set>

namespace onnx_light_cpu {
namespace {
namespace builder = ONNX_LIGHT_NAMESPACE::core::builder;
namespace sym = ONNX_LIGHT_NAMESPACE::core::symbolic;
using ONNX_LIGHT_NAMESPACE::AddAttribute;
using ONNX_LIGHT_NAMESPACE::AttributeProto;
using ONNX_LIGHT_NAMESPACE::FindAttribute;
using ONNX_LIGHT_NAMESPACE::MakeNode;
using ONNX_LIGHT_NAMESPACE::NodeProto;
using ONNX_LIGHT_NAMESPACE::TensorProto;

bool Local(builder::GraphGraph &graph, const std::string &domain, const std::string &op,
           const std::string &overload = "") {
  return graph.Builder().Shapes().HasLocalFunction(
      ONNX_LIGHT_NAMESPACE::core::shapes::ShapesContext::LocalFunctionKey(domain, op, overload));
}

bool Op(builder::GraphGraph &graph, const NodeProto *node, const char *op, int inputs,
        int outputs = 1) {
  return node != nullptr && node->op_type() == op &&
         (node->domain().empty() || node->domain() == "ai.onnx") && node->overload().empty() &&
         node->input_size() == inputs && node->output_size() == outputs &&
         std::all_of(node->output().begin(), node->output().end(),
                     [](const std::string &output) { return !output.empty(); }) &&
         !Local(graph, node->domain(), node->op_type(), node->overload());
}

bool Attributes(const NodeProto &node, std::initializer_list<const char *> allowed) {
  std::unordered_set<std::string> seen;
  for (const auto &attribute : node.attribute()) {
    bool found = false;
    for (const auto *name : allowed) {
      found = found || attribute.name() == name;
    }
    if (!found || !seen.insert(attribute.name()).second || !attribute.ref_attr_name().empty()) {
      return false;
    }
  }
  return true;
}

bool Int(const NodeProto &node, const char *name, int64_t fallback, int64_t expected) {
  const auto *attribute = FindAttribute(node, name);
  return attribute == nullptr ? fallback == expected
                              : attribute->type() == AttributeProto::AttributeType::INT &&
                                    attribute->i() == expected;
}

bool Axis(const NodeProto &node) { return Int(node, "axis", -1, -1) || Int(node, "axis", -1, 1); }

bool Constant(builder::GraphGraph &graph, const std::string &name,
              const std::vector<int64_t> &expected) {
  const auto *tensor = graph.GetComputedConstant(name);
  if (tensor == nullptr || tensor->data_type() != TensorProto::DataType::INT64 ||
      tensor->dims().size() != 1 || tensor->dims(0) != static_cast<int64_t>(expected.size())) {
    return false;
  }
  if (!tensor->is_raw_data()) {
    return tensor->int64_data().size() == expected.size() &&
           std::equal(expected.begin(), expected.end(), tensor->int64_data().begin());
  }
  const auto &raw = tensor->ref_raw_data();
  if (raw.size() != expected.size() * sizeof(int64_t)) {
    return false;
  }
  for (std::size_t i = 0; i < expected.size(); ++i) {
    uint64_t bits = 0;
    for (std::size_t j = 0; j < sizeof(int64_t); ++j) {
      bits |= static_cast<uint64_t>(static_cast<uint8_t>(raw[i * sizeof(int64_t) + j])) << (8 * j);
    }
    if (std::bit_cast<int64_t>(bits) != expected[i]) {
      return false;
    }
  }
  return true;
}

bool Reshape(builder::GraphGraph &graph, const NodeProto *node, const std::vector<int64_t> &shape) {
  return Op(graph, node, "Reshape", 2) && Attributes(*node, {"allowzero"}) &&
         Int(*node, "allowzero", 0, 0) && Constant(graph, node->input(1), shape);
}

bool Sum(builder::GraphGraph &graph, const NodeProto *node, int64_t keepdims, bool rank_two) {
  return Op(graph, node, "ReduceSum", 2) &&
         Attributes(*node, {"keepdims", "noop_with_empty_axes"}) &&
         Int(*node, "keepdims", 1, keepdims) && Int(*node, "noop_with_empty_axes", 0, 0) &&
         (Constant(graph, node->input(1), {1}) ||
          Constant(graph, node->input(1), {rank_two ? -1 : -2}));
}

bool Grouped(builder::GraphGraph &graph, const NodeProto *node) {
  return node != nullptr &&
         (Op(graph, node, "GroupedMatMul", 3) || Op(graph, node, "GroupedMatMul", 4)) &&
         node->attribute().empty();
}

bool FloatShape(builder::GraphGraph &graph, const std::string &name, const sym::SymShape &shape) {
  return graph.HasType(name) && graph.GetType(name) == sym::TensorType::kFloat &&
         graph.HasShape(name) && graph.GetShape(name).Shape() == shape;
}

struct MoEMatch {
  std::vector<const NodeProto *> nodes;
  const NodeProto *first = nullptr;
  const NodeProto *second = nullptr;
  const NodeProto *softmax = nullptr;
  int64_t experts = 0;
  int64_t k = 0;
  bool normalize = false;
  std::string activation = "identity";
};

bool Recognize(builder::GraphGraph &graph, const NodeProto &candidate, MoEMatch &match) {
  const auto &opsets = graph.Builder().OpsetVersions();
  const auto microsoft = opsets.find("com.microsoft");
  if (graph.Builder().OpsetVersion("") < 29 ||
      (microsoft != opsets.end() && microsoft->second != 1) ||
      Local(graph, "com.microsoft", "MoE") || Local(graph, "", "Gather") ||
      Local(graph, "", "Transpose") || !Sum(graph, &candidate, 0, false)) {
    return false;
  }
  const auto *mul = graph.NodeBefore(candidate.input(0));
  if (!Op(graph, mul, "Mul", 2) || !mul->attribute().empty()) {
    return false;
  }
  const NodeProto *unsqueeze = nullptr;
  const NodeProto *restore = nullptr;
  for (int side = 0; side < 2; ++side) {
    const auto *node = graph.NodeBefore(mul->input(side));
    if (Op(graph, node, "Unsqueeze", 2) && node->attribute().empty() &&
        (Constant(graph, node->input(1), {-1}) || Constant(graph, node->input(1), {2}))) {
      unsqueeze = node;
      restore = graph.NodeBefore(mul->input(1 - side));
      break;
    }
  }
  if (unsqueeze == nullptr || !Op(graph, restore, "Reshape", 2)) {
    return false;
  }
  const auto *second = graph.NodeBefore(restore->input(0));
  if (!Grouped(graph, second)) {
    return false;
  }
  const auto *flat = graph.NodeBefore(second->input(0));
  const auto *indices = graph.NodeBefore(second->input(2));
  if (!Op(graph, flat, "Reshape", 2) || !Reshape(graph, indices, {-1, 1})) {
    return false;
  }
  const auto *activation = graph.NodeBefore(flat->input(0));
  const NodeProto *first = activation;
  if (Op(graph, activation, "Relu", 1) && activation->attribute().empty()) {
    match.activation = "relu";
    first = graph.NodeBefore(activation->input(0));
  } else if (Op(graph, activation, "Gelu", 1) && Attributes(*activation, {"approximate"})) {
    const auto *approximate = FindAttribute(*activation, "approximate");
    if (approximate == nullptr || approximate->type() != AttributeProto::AttributeType::STRING ||
        approximate->s() != "tanh") {
      return false;
    }
    match.activation = "gelu";
    first = graph.NodeBefore(activation->input(0));
  } else if (Op(graph, activation, "Identity", 1) && activation->attribute().empty()) {
    first = graph.NodeBefore(activation->input(0));
  } else {
    activation = nullptr;
  }
  if (!Grouped(graph, first) || indices->input(0) != first->input(2)) {
    return false;
  }
  const auto *topk = graph.NodeBefore(first->input(2));
  if (!Op(graph, topk, "TopK", 2, 2) || topk->output(1) != first->input(2) ||
      !Attributes(*topk, {"axis", "largest", "sorted"}) || !Axis(*topk) ||
      !Int(*topk, "largest", 1, 1) || !Int(*topk, "sorted", 1, 1)) {
    return false;
  }
  const auto *softmax = graph.NodeBefore(topk->input(0));
  if (!Op(graph, softmax, "Softmax", 1) || !Attributes(*softmax, {"axis"}) || !Axis(*softmax)) {
    return false;
  }
  const auto &x = first->input(0);
  const auto &logits = softmax->input(0);
  if (!graph.HasShape(x) || !graph.HasShape(logits) || !graph.HasShape(first->input(1))) {
    return false;
  }
  const auto &input = graph.GetShape(x).Shape();
  const auto &router = graph.GetShape(logits).Shape();
  const auto &weight = graph.GetShape(first->input(1)).Shape();
  if (input.Rank() != 2 || router.Rank() != 2 || weight.Rank() != 3 || router[0] != input[0] ||
      !input[1].IsInt() || !router[1].IsInt() || !weight[2].IsInt()) {
    return false;
  }
  const int64_t hidden = input[1].AsInt(), experts = router[1].AsInt(),
                intermediate = weight[2].AsInt();
  for (const auto dimension : {hidden, experts, intermediate}) {
    if (dimension <= 0 || dimension > std::numeric_limits<int>::max()) {
      return false;
    }
  }
  // TopK's K input is a one-element INT64 tensor, not a scalar.
  const auto *k = graph.GetComputedConstant(topk->input(1));
  if (k == nullptr || k->dims().size() != 1 || k->dims(0) != 1) {
    return false;
  }
  int64_t selected = 0;
  if (!k->is_raw_data() && k->int64_data().size() == 1) {
    selected = k->int64_data()[0];
  } else if (k->is_raw_data() && k->ref_raw_data().size() == sizeof(int64_t)) {
    uint64_t bits = 0;
    for (std::size_t j = 0; j < sizeof(int64_t); ++j) {
      bits |= static_cast<uint64_t>(static_cast<uint8_t>(k->ref_raw_data()[j])) << (8 * j);
    }
    selected = std::bit_cast<int64_t>(bits);
  }
  if (selected <= 0 || selected > experts || !Constant(graph, topk->input(1), {selected}) ||
      !FloatShape(graph, x, input) || !FloatShape(graph, logits, router) ||
      !FloatShape(graph, first->input(1), sym::SymShape({experts, hidden, intermediate})) ||
      !FloatShape(graph, second->input(1), sym::SymShape({experts, intermediate, hidden})) ||
      !Reshape(graph, flat, {-1, intermediate}) ||
      !Reshape(graph, restore, {-1, selected, hidden})) {
    return false;
  }
  for (const auto *node : {first, second}) {
    if (!graph.IsConstant(node->input(1)) ||
        (node->input_size() == 4 && !node->input(3).empty() && !graph.IsConstant(node->input(3)))) {
      return false;
    }
    if (node->input_size() == 4 && !node->input(3).empty() &&
        !FloatShape(graph, node->input(3),
                    sym::SymShape({experts, node == first ? intermediate : hidden}))) {
      return false;
    }
  }
  const auto *div = graph.NodeBefore(unsqueeze->input(0));
  const NodeProto *sum = nullptr;
  if (Op(graph, div, "Div", 2) && div->attribute().empty()) {
    sum = graph.NodeBefore(div->input(1));
    if (div->input(0) != topk->output(0) || !Sum(graph, sum, 1, true) ||
        sum->input(0) != topk->output(0)) {
      return false;
    }
    match.normalize = true;
  } else if (unsqueeze->input(0) != topk->output(0)) {
    return false;
  } else {
    div = nullptr;
  }
  match.nodes = {softmax, topk};
  if (sum != nullptr) {
    match.nodes.push_back(sum);
    match.nodes.push_back(div);
  }
  match.nodes.push_back(first);
  if (activation != nullptr) {
    match.nodes.push_back(activation);
  }
  for (const auto *node : {flat, indices, second, restore, unsqueeze, mul, &candidate}) {
    match.nodes.push_back(node);
  }
  const std::unordered_set<const NodeProto *> members(match.nodes.begin(), match.nodes.end());
  if (members.size() != match.nodes.size()) {
    return false;
  }
  const auto internal = [&](const std::string &input) {
    return members.contains(graph.NodeBefore(input));
  };
  if (internal(x) || internal(logits)) {
    return false;
  }
  for (const auto *node : {first, second}) {
    if (internal(node->input(1)) ||
        (node->input_size() == 4 && !node->input(3).empty() && internal(node->input(3)))) {
      return false;
    }
  }
  for (const auto *node : match.nodes) {
    if (node == &candidate) {
      continue;
    }
    for (const auto &output : node->output()) {
      if (output.empty() || graph.IsOutput(output) || graph.IsUsedBySubgraph(output)) {
        return false;
      }
      for (const auto *consumer : graph.NextNodes(output)) {
        if (!members.contains(consumer)) {
          return false;
        }
      }
    }
  }
  match.first = first;
  match.second = second;
  match.softmax = softmax;
  match.experts = experts;
  match.k = selected;
  return true;
}
} // namespace

std::set<std::string> MoEFusionPattern::FastOpType() const { return {"ReduceSum"}; }

builder::MatchResult MoEFusionPattern::Match(builder::GraphGraph &graph,
                                             const NodeProto &candidate) const {
  MoEMatch match;
  if (!Recognize(graph, candidate, match)) {
    return NoMatch(candidate, "not an isolated FLOAT GroupedMatMul MoE block");
  }
  return builder::MatchResult{this, match.nodes, &candidate};
}

ONNX_LIGHT_NAMESPACE::utils::RepeatedProtoField<NodeProto>
MoEFusionPattern::Apply(builder::GraphGraph &graph,
                        const std::vector<const NodeProto *> &nodes) const {
  MoEMatch match;
  if (nodes.empty() || nodes.back() == nullptr || !Recognize(graph, *nodes.back(), match) ||
      match.nodes != nodes) {
    throw builder::BuilderError("MoEFusionPattern::Apply: invalid match.");
  }
  std::unordered_set<std::string> reserved;
  const auto unique = [&](const std::string &prefix) {
    auto name = prefix;
    for (std::size_t suffix = 0; graph.Builder().HasName(name) || reserved.contains(name);
         ++suffix) {
      name = prefix + "_" + std::to_string(suffix);
    }
    reserved.insert(name);
    return name;
  };
  TensorProto reverse;
  reverse.set_name(unique("moe_reverse_experts"));
  reverse.set_data_type(TensorProto::DataType::INT64);
  reverse.add_dims(match.experts);
  for (int64_t expert = match.experts; expert > 0; --expert) {
    reverse.ref_int64_data().push_back(expert - 1);
  }
  const auto order = graph.Builder().MakeInitializer(reverse);
  ONNX_LIGHT_NAMESPACE::utils::RepeatedProtoField<NodeProto> replacements;
  const auto gather = [&](const std::string &input, int64_t axis) {
    const auto output = unique(input + "_moe_reversed");
    auto node = MakeNode("Gather", {input, order}, {output});
    AddAttribute(node, "axis", axis);
    replacements.push_back(std::move(node));
    return output;
  };
  const auto transpose = [&](const std::string &input) {
    const auto reversed = gather(input, 0);
    const auto output = unique(input + "_moe_transposed");
    auto node = MakeNode("Transpose", {reversed}, {output});
    AddAttribute(node, "perm", std::vector<int64_t>{0, 2, 1});
    replacements.push_back(std::move(node));
    return output;
  };
  const auto logits = gather(match.softmax->input(0), 1);
  const auto w1 = transpose(match.first->input(1));
  const auto w2 = transpose(match.second->input(1));
  const auto bias = [&](const NodeProto &node) {
    return node.input_size() == 4 && !node.input(3).empty() ? gather(node.input(3), 0)
                                                            : std::string{};
  };
  const auto b1 = bias(*match.first);
  const auto b2 = bias(*match.second);
  auto moe =
      MakeNode("MoE", {match.first->input(0), logits, w1, b1, w2, b2}, {nodes.back()->output(0)});
  moe.set_domain("com.microsoft");
  AddAttribute(moe, "k", match.k);
  AddAttribute(moe, "activation_type", match.activation);
  AddAttribute(moe, "normalize_routing_weights", static_cast<int64_t>(match.normalize));
  replacements.push_back(std::move(moe));
  graph.Builder().SetOpsetVersion("com.microsoft", 1);
  return replacements;
}
} // namespace onnx_light_cpu
