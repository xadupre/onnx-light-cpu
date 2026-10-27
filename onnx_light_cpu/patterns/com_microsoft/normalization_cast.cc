// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/patterns/com_microsoft/patterns.h"

#include "ml_cpu_compute/math/half_conversion.h"
#include "onnx_light_cpu/kernels/com_microsoft/naive_skip_simplified_layer_normalization_kernel.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/builder/graph_graph.h"
#include "onnx_proto/onnx_helper.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace onnx_light_cpu {
namespace {

namespace builder = ONNX_LIGHT_NAMESPACE::core::builder;
namespace sym = ONNX_LIGHT_NAMESPACE::core::symbolic;
using ONNX_LIGHT_NAMESPACE::AttributeProto;
using ONNX_LIGHT_NAMESPACE::FindAttribute;
using ONNX_LIGHT_NAMESPACE::NodeProto;
using ONNX_LIGHT_NAMESPACE::TensorProto;

bool HasType(builder::GraphGraph &graph, const std::string &name, sym::TensorType type) {
  return graph.HasType(name) && graph.GetType(name) == type;
}

bool IsCast(const NodeProto *node, TensorProto::DataType to) {
  if (node == nullptr || (!node->domain().empty() && node->domain() != "ai.onnx") ||
      node->op_type() != "Cast" || node->input_size() != 1 || node->output_size() != 1 ||
      node->attribute().size() != 1) {
    return false;
  }
  const auto *attribute = FindAttribute(*node, "to");
  return attribute != nullptr && attribute->type() == AttributeProto::AttributeType::INT &&
         attribute->i() == static_cast<int64_t>(to);
}

const NodeProto *NarrowingConsumer(builder::GraphGraph &graph, const std::string &name) {
  if (graph.IsOutput(name) || graph.IsUsedBySubgraph(name) ||
      !HasType(graph, name, sym::TensorType::kFloat)) {
    return nullptr;
  }
  const auto &consumers = graph.NextNodes(name);
  if (consumers.size() != 1 || !IsCast(consumers[0], TensorProto::DataType::FLOAT16) ||
      !HasType(graph, consumers[0]->output(0), sym::TensorType::kFloat16)) {
    return nullptr;
  }
  return consumers[0];
}

// Compare bits as well as values to preserve signed zero. Nonfinite weights are
// deliberately excluded because NaN payload conversion need not round-trip.
bool ExactHalfConstant(const TensorProto *tensor, TensorProto *converted = nullptr) {
  if (tensor == nullptr || tensor->data_type() != TensorProto::DataType::FLOAT ||
      tensor->dims().size() != 1 || tensor->dims(0) <= 0 ||
      static_cast<uint64_t>(tensor->dims(0)) >
          std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    return false;
  }
  const auto count = static_cast<std::size_t>(tensor->dims(0));
  if (tensor->is_raw_data() ? tensor->ref_raw_data().size() != count * sizeof(float)
                            : tensor->float_data().size() != count) {
    return false;
  }
  if (converted != nullptr) {
    converted->set_data_type(TensorProto::DataType::FLOAT16);
    converted->add_dims(tensor->dims(0));
  }
  for (std::size_t i = 0; i < count; ++i) {
    float value;
    if (tensor->is_raw_data()) {
      const auto &raw = tensor->ref_raw_data();
      uint32_t bits = 0;
      for (std::size_t byte = 0; byte < sizeof(float); ++byte) {
        bits |= static_cast<uint32_t>(static_cast<uint8_t>(raw[i * sizeof(float) + byte]))
                << (8 * byte);
      }
      value = std::bit_cast<float>(bits);
    } else {
      value = tensor->float_data()[i];
    }
    const auto half = ml_cpu_compute::detail::FloatToFloat16Bits(value);
    if (!std::isfinite(value) || std::bit_cast<uint32_t>(ml_cpu_compute::detail::Float16BitsToFloat(
                                     half)) != std::bit_cast<uint32_t>(value)) {
      return false;
    }
    if (converted != nullptr) {
      converted->ref_int32_data().push_back(half);
    }
  }
  return true;
}

bool HalfSource(builder::GraphGraph &graph, const std::string &name) {
  const auto *cast = graph.NodeBefore(name);
  return IsCast(cast, TensorProto::DataType::FLOAT) &&
         HasType(graph, cast->input(0), sym::TensorType::kFloat16);
}

bool SupportedShapes(builder::GraphGraph &graph, const NodeProto &node) {
  for (const auto &name : node.input()) {
    if (!name.empty() && !graph.HasShape(name)) {
      return false;
    }
  }
  const auto &input = graph.GetShape(node.input(0)).Shape();
  const auto &skip = graph.GetShape(node.input(1)).Shape();
  if ((input.Rank() != 2 && input.Rank() != 3) || !input[input.Rank() - 1].IsInt()) {
    return false;
  }
  const auto width = input[input.Rank() - 1].AsInt();
  if (width <= 0 || width > std::numeric_limits<int>::max()) {
    return false;
  }
  if (skip != input) {
    if (input.Rank() != 3 || (skip.Rank() != 2 && skip.Rank() != 3) ||
        skip[skip.Rank() - 1] != input[2] || skip[skip.Rank() - 2] != input[1] ||
        (skip.Rank() == 3 && skip[0] != sym::SymDim(1))) {
      return false;
    }
  }
  for (std::size_t i = 2; i < node.input_size(); ++i) {
    if (!node.input(i).empty()) {
      const auto &shape = graph.GetShape(node.input(i)).Shape();
      if (shape.Rank() != 1 || shape[0] != input[input.Rank() - 1]) {
        return false;
      }
    }
  }
  return true;
}

} // namespace

std::set<std::string> NormalizationCastEliminationPattern::FastOpType() const {
  return {"SkipSimplifiedLayerNormalization"};
}

builder::MatchResult NormalizationCastEliminationPattern::Match(builder::GraphGraph &graph,
                                                                const NodeProto &candidate) const {
  if (kernel_name_ != NaiveSkipSimplifiedLayerNormalizationKernel::kName ||
      graph.Builder().device() != sym::Device::kCPU) {
    return NoMatch(candidate, "no compatible CPU kernel execution contract");
  }
  if (candidate.domain() != kMicrosoftDomain ||
      candidate.op_type() != "SkipSimplifiedLayerNormalization" ||
      graph.Builder().OpsetVersion(kMicrosoftDomain) != 1 || candidate.input_size() < 3 ||
      candidate.input_size() > 4 || candidate.output_size() < 1 || candidate.output_size() > 4 ||
      candidate.output(0).empty() || candidate.attribute().size() > 1) {
    return NoMatch(candidate, "not a supported Microsoft skip RMS normalization schema");
  }
  for (const auto &attribute : candidate.attribute()) {
    if (attribute.name() != "epsilon" || attribute.type() != AttributeProto::AttributeType::FLOAT ||
        !std::isfinite(attribute.f()) || attribute.f() < 0.0F) {
      return NoMatch(candidate, "unsupported normalization attribute");
    }
  }
  for (std::size_t i = 0; i < candidate.input_size(); ++i) {
    const auto &name = candidate.input(i);
    if (i == 3 && name.empty()) {
      continue;
    }
    if (!HasType(graph, name, sym::TensorType::kFloat)) {
      return NoMatch(candidate, "normalization input is not FLOAT");
    }
    if (!HalfSource(graph, name) &&
        (i < 2 || !ExactHalfConstant(graph.GetComputedConstant(name)))) {
      return NoMatch(candidate, "input cannot be represented in FLOAT16 without rounding");
    }
  }
  if (!SupportedShapes(graph, candidate)) {
    return NoMatch(candidate, "unsupported input or weight shapes");
  }
  for (std::size_t i = 1; i < candidate.output_size() && i < 3; ++i) {
    if (!candidate.output(i).empty() &&
        !HasType(graph, candidate.output(i), sym::TensorType::kFloat)) {
      return NoMatch(candidate, "statistics must remain FLOAT");
    }
  }
  const auto *output_cast = NarrowingConsumer(graph, candidate.output(0));
  if (output_cast == nullptr) {
    return NoMatch(candidate, "normalized output is not exclusively narrowed to FLOAT16");
  }
  const NodeProto *sum_cast = nullptr;
  if (candidate.output_size() == 4 && !candidate.output(3).empty() &&
      graph.IsUsed(candidate.output(3))) {
    sum_cast = NarrowingConsumer(graph, candidate.output(3));
    if (sum_cast == nullptr) {
      return NoMatch(candidate, "live FLOAT residual cannot be replaced by a rounded sum");
    }
  }
  // Input casts are deliberately outside the match. Cleanup removes only those
  // that become unused, preserving other consumers and subgraph captures.
  return builder::MatchResult{this, {&candidate, output_cast, sum_cast}, nullptr};
}

ONNX_LIGHT_NAMESPACE::utils::RepeatedProtoField<NodeProto>
NormalizationCastEliminationPattern::Apply(builder::GraphGraph &graph,
                                           const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 3 || nodes[0] == nullptr || Match(graph, *nodes[0]).nodes != nodes) {
    throw builder::BuilderError("NormalizationCastEliminationPattern::Apply: invalid match.");
  }
  ONNX_LIGHT_NAMESPACE::utils::RepeatedProtoField<NodeProto> replacements;
  NodeProto replacement = *nodes[0];
  for (std::size_t i = 0; i < replacement.input_size(); ++i) {
    const auto name = replacement.input(i);
    if (name.empty()) {
      continue;
    }
    if (HalfSource(graph, name)) {
      replacement.ref_input()[i] = graph.NodeBefore(name)->input(0);
    } else {
      TensorProto weight;
      if (!ExactHalfConstant(graph.GetComputedConstant(name), &weight)) {
        throw builder::BuilderError("NormalizationCastEliminationPattern: inexact weight.");
      }
      const auto weight_name = graph.Builder().UniqueName(name + "_fp16");
      auto constant = ONNX_LIGHT_NAMESPACE::MakeNode("Constant", {}, {weight_name});
      auto *attribute = constant.add_attribute();
      attribute->set_name("value");
      attribute->set_type(AttributeProto::AttributeType::TENSOR);
      attribute->ref_t() = std::move(weight);
      replacements.push_back(std::move(constant));
      replacement.ref_input()[i] = weight_name;
    }
  }
  replacement.ref_output()[0] = nodes[1]->output(0);
  if (replacement.output_size() == 4) {
    replacement.ref_output()[3] = nodes[2] == nullptr ? "" : nodes[2]->output(0);
  }
  replacements.push_back(std::move(replacement));
  return replacements;
}

} // namespace onnx_light_cpu
