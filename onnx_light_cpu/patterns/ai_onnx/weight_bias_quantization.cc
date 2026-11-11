// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/patterns/weight_bias_quantization.h"

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>

namespace onnx_light_cpu {
namespace {

namespace builder = ONNX_LIGHT_NAMESPACE::core::builder;
namespace rt = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym = ONNX_LIGHT_NAMESPACE::core::symbolic;
using ONNX_LIGHT_NAMESPACE::AttributeProto;
using ONNX_LIGHT_NAMESPACE::NodeProto;
using ONNX_LIGHT_NAMESPACE::TensorProto;
using Nodes = ONNX_LIGHT_NAMESPACE::utils::RepeatedProtoField<NodeProto>;

class NearestRounding {
public:
  NearestRounding() : previous_(std::fegetround()) { std::fesetround(FE_TONEAREST); }
  ~NearestRounding() { std::fesetround(previous_); }

private:
  int previous_;
};

bool StandardNode(builder::GraphGraph &graph, const NodeProto *node) {
  return node != nullptr && (node->domain().empty() || node->domain() == "ai.onnx") &&
         node->overload().empty() &&
         !graph.Builder().Shapes().HasLocalFunction(
             ONNX_LIGHT_NAMESPACE::core::shapes::ShapesContext::LocalFunctionKey(
                 node->domain(), node->op_type(), node->overload()));
}

std::optional<rt::Tensor> Constant(builder::GraphGraph &graph, const std::string &name) {
  const auto *proto = graph.GetComputedConstant(name);
  if (proto == nullptr) {
    return std::nullopt;
  }
  return rt::TensorFromProto(*proto);
}

std::optional<float> FloatScalar(builder::GraphGraph &graph, const std::string &name) {
  const auto tensor = Constant(graph, name);
  if (!tensor || tensor->data_type != TensorProto::FLOAT || !tensor->shape.empty() ||
      tensor->element_count() != 1 || !std::isfinite(tensor->AsFloat()[0])) {
    return std::nullopt;
  }
  return tensor->AsFloat()[0];
}

bool PositiveScale(float scale) { return std::isnormal(scale) && scale > 0.0F; }

bool QuantizationAttributes(const NodeProto &node) {
  for (const auto &attribute : node.attribute()) {
    if (attribute.name() == "axis" && attribute.type() == AttributeProto::INT) {
      continue; // Axis has no effect on a scalar scale.
    }
    if (attribute.name() != "block_size" || attribute.type() != AttributeProto::INT ||
        attribute.i() != 0) {
      return false;
    }
  }
  return true;
}

std::optional<float> QuantizationScale(builder::GraphGraph &graph, const NodeProto *node,
                                       const std::string &op, bool weight = false) {
  if (!StandardNode(graph, node) || node->op_type() != op || node->input_size() < 2 ||
      node->input_size() > 3 || node->output_size() != 1 || !QuantizationAttributes(*node)) {
    return std::nullopt;
  }
  const auto scale = FloatScalar(graph, node->input(1));
  if (!scale || !PositiveScale(*scale)) {
    return std::nullopt;
  }
  int type = TensorProto::UINT8;
  if (op == "DequantizeLinear") {
    if (!graph.HasType(node->input(0))) {
      return std::nullopt;
    }
    const auto input_type = graph.GetType(node->input(0));
    if (input_type != sym::TensorType::kInt8 && input_type != sym::TensorType::kUint8) {
      return std::nullopt;
    }
    type = input_type == sym::TensorType::kInt8 ? TensorProto::INT8 : TensorProto::UINT8;
  }
  if (node->input_size() == 3 && !node->input(2).empty()) {
    const auto zero = Constant(graph, node->input(2));
    if (!zero || !zero->shape.empty() || zero->element_count() != 1 ||
        (op == "DequantizeLinear" && zero->data_type != type)) {
      return std::nullopt;
    }
    type = zero->data_type;
    if (type != TensorProto::INT8 && type != TensorProto::UINT8) {
      return std::nullopt;
    }
    if (weight && (type != TensorProto::INT8 || zero->As<int8_t>()[0] != 0)) {
      return std::nullopt;
    }
  }
  if ((type != TensorProto::INT8 && type != TensorProto::UINT8) ||
      (weight && type != TensorProto::INT8)) {
    return std::nullopt;
  }
  return scale;
}

bool OutputPath(builder::GraphGraph &graph, const NodeProto &node) {
  const auto *current = &node;
  for (;;) {
    if (current->output_size() != 1 || graph.IsOutput(current->output(0)) ||
        graph.IsUsedBySubgraph(current->output(0))) {
      return false;
    }
    const auto &consumers = graph.NextNodes(current->output(0));
    if (consumers.size() != 1 || !StandardNode(graph, consumers[0]) ||
        consumers[0]->input_size() == 0 || consumers[0]->input(0) != current->output(0)) {
      return false;
    }
    current = consumers[0];
    if (current->op_type() == "QuantizeLinear") {
      return QuantizationScale(graph, current, "QuantizeLinear").has_value();
    }
    if (!current->attribute().empty()) {
      return false;
    }
    if (current->op_type() == "Relu" && current->input_size() == 1) {
      continue;
    }
    if (current->op_type() != "Clip" || current->input_size() < 1 || current->input_size() > 3) {
      return false;
    }
    float bounds[2] = {-std::numeric_limits<float>::infinity(),
                       std::numeric_limits<float>::infinity()};
    for (int i = 1; i < current->input_size(); ++i) {
      if (!current->input(i).empty()) {
        const auto bound = FloatScalar(graph, current->input(i));
        if (!bound) {
          return false;
        }
        bounds[i - 1] = *bound;
      }
    }
    if (bounds[0] > bounds[1]) {
      return false;
    }
  }
}

bool FiniteFloat(const rt::Tensor &tensor) {
  if (tensor.data_type != TensorProto::FLOAT || tensor.element_count() == 0) {
    return false;
  }
  for (int64_t i = 0; i < tensor.element_count(); ++i) {
    if (!std::isfinite(tensor.AsFloat()[i])) {
      return false;
    }
  }
  return true;
}

std::optional<std::vector<int64_t>> Shape(builder::GraphGraph &graph, const std::string &name) {
  if (!graph.HasShape(name)) {
    return std::nullopt;
  }
  std::vector<int64_t> shape;
  const auto &dims = graph.GetShape(name).Shape();
  for (std::size_t i = 0; i < dims.Rank(); ++i) {
    if (!dims[i].IsInt() || dims[i].AsInt() <= 0) {
      return std::nullopt;
    }
    shape.push_back(dims[i].AsInt());
  }
  return shape;
}

std::optional<int64_t> Channels(const NodeProto &node, const std::vector<int64_t> &input,
                                const rt::Shape &weight) {
  if (node.op_type() == "Gemm") {
    if (input.size() != 2 || weight.size() != 2) {
      return std::nullopt;
    }
    int64_t trans_a = 0, trans_b = 0;
    for (const auto &attribute : node.attribute()) {
      if (attribute.name() == "alpha" || attribute.name() == "beta") {
        if (attribute.type() != AttributeProto::FLOAT || attribute.f() != 1.0F) {
          return std::nullopt;
        }
      } else if ((attribute.name() == "transA" || attribute.name() == "transB") &&
                 attribute.type() == AttributeProto::INT &&
                 (attribute.i() == 0 || attribute.i() == 1)) {
        (attribute.name() == "transA" ? trans_a : trans_b) = attribute.i();
      } else {
        return std::nullopt;
      }
    }
    return input[trans_a ? 0 : 1] == weight[trans_b ? 1 : 0]
               ? std::optional<int64_t>(weight[trans_b ? 0 : 1])
               : std::nullopt;
  }
  if ((input.size() != 3 && input.size() != 4) || input.size() != weight.size()) {
    return std::nullopt;
  }
  const auto spatial = input.size() - 2;
  for (const auto &attribute : node.attribute()) {
    if (attribute.name() == "group" && attribute.type() == AttributeProto::INT &&
        attribute.i() == 1) {
      continue;
    }
    if (attribute.name() == "auto_pad" && attribute.type() == AttributeProto::STRING &&
        attribute.s() == "NOTSET") {
      continue;
    }
    const auto &name = attribute.name();
    if ((name != "kernel_shape" && name != "strides" && name != "dilations" && name != "pads") ||
        attribute.type() != AttributeProto::INTS ||
        attribute.ints().size() != (name == "pads" ? 2 * spatial : spatial)) {
      return std::nullopt;
    }
    for (std::size_t i = 0; i < attribute.ints().size(); ++i) {
      if (attribute.ints()[i] < (name == "pads" ? 0 : 1) ||
          (name == "kernel_shape" && attribute.ints()[i] != weight[i + 2])) {
        return std::nullopt;
      }
    }
  }
  const bool transpose = node.op_type() == "ConvTranspose";
  return input[1] == weight[transpose ? 0 : 1] ? std::optional<int64_t>(weight[transpose ? 1 : 0])
                                               : std::nullopt;
}

struct Plan {
  bool quantize_weight;
  float weight_scale;
  std::optional<float> bias_scale;
};

std::optional<Plan> MakePlan(builder::GraphGraph &graph, const NodeProto &node) {
  if (graph.Builder().device() != sym::Device::kCPU || graph.Builder().OpsetVersion("") < 13 ||
      !StandardNode(graph, &node) ||
      (node.op_type() != "Conv" && node.op_type() != "ConvTranspose" && node.op_type() != "Gemm") ||
      node.input_size() < 2 || node.input_size() > 3 || !OutputPath(graph, node)) {
    return std::nullopt;
  }
  for (const auto &name : node.input()) {
    if (!name.empty() && (!graph.HasType(name) || graph.GetType(name) != sym::TensorType::kFloat)) {
      return std::nullopt;
    }
  }
  const auto activation_scale =
      QuantizationScale(graph, graph.NodeBefore(node.input(0)), "DequantizeLinear");
  const auto input_shape = Shape(graph, node.input(0));
  if (!activation_scale || !input_shape) {
    return std::nullopt;
  }
  const auto *weight_dq = graph.NodeBefore(node.input(1));
  const bool already_quantized =
      StandardNode(graph, weight_dq) && weight_dq->op_type() == "DequantizeLinear";
  if (already_quantized && !QuantizationScale(graph, weight_dq, "DequantizeLinear", true)) {
    return std::nullopt;
  }
  auto weight = Constant(graph, already_quantized ? weight_dq->input(0) : node.input(1));
  if (!weight) {
    return std::nullopt;
  }
  for (const auto dim : weight->shape) {
    if (dim <= 0) {
      return std::nullopt;
    }
  }
  const auto channels = Channels(node, *input_shape, weight->shape);
  if (!channels) {
    return std::nullopt;
  }
  float weight_scale = 1.0F;
  if (already_quantized) {
    const auto scale = QuantizationScale(graph, weight_dq, "DequantizeLinear", true);
    if (!scale || weight->data_type != TensorProto::INT8) {
      return std::nullopt;
    }
    weight_scale = *scale;
  } else {
    if (!FiniteFloat(*weight)) {
      return std::nullopt;
    }
    float maximum = 0.0F;
    for (int64_t i = 0; i < weight->element_count(); ++i) {
      maximum = std::max(maximum, std::abs(weight->AsFloat()[i]));
    }
    weight_scale = maximum == 0.0F ? 1.0F : maximum / 127.0F;
    if (!PositiveScale(weight_scale) || !std::isfinite(127.0F * weight_scale)) {
      return std::nullopt;
    }
  }
  Plan plan{!already_quantized, weight_scale, std::nullopt};
  if (node.input_size() == 3 && !node.input(2).empty()) {
    const auto *bias_source = graph.NodeBefore(node.input(2));
    if (bias_source != nullptr && bias_source->op_type() == "DequantizeLinear") {
      return std::nullopt;
    }
    const auto bias = Constant(graph, node.input(2));
    const float scale = *activation_scale * weight_scale;
    if (!bias || !FiniteFloat(*bias) || bias->shape.size() != 1 || bias->shape[0] != *channels ||
        !PositiveScale(scale)) {
      return std::nullopt;
    }
    const rt::KernelContext context(rt::DefaultOpset(graph.Builder().OpsetVersion("")));
    const auto divisor = rt::Tensor::FromFloat("", {}, {scale});
    const auto rounded = ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::Round(context)(
        ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::Div(context)(*bias, divisor));
    for (int64_t i = 0; i < rounded.element_count(); ++i) {
      const double value = rounded.AsFloat()[i];
      if (!std::isfinite(value) || value < std::numeric_limits<int32_t>::min() ||
          value > std::numeric_limits<int32_t>::max()) {
        return std::nullopt;
      }
    }
    plan.bias_scale = scale;
  }
  if (already_quantized && !plan.bias_scale) {
    return std::nullopt;
  }
  return plan;
}

class WeightBiasQuantizationPattern final : public builder::PatternOptimization {
public:
  WeightBiasQuantizationPattern()
      : PatternOptimization(0, "CPUWeightBiasQuantization", sym::Device::kCPU) {}

  std::set<std::string> FastOpType() const override { return {"Conv", "ConvTranspose", "Gemm"}; }

  builder::MatchResult Match(builder::GraphGraph &graph, const NodeProto &node) const override {
    return MakePlan(graph, node) ? builder::MatchResult{this, {&node}, nullptr}
                                 : NoMatch(node, "unsupported CPU weight/bias quantization");
  }

  Nodes Apply(builder::GraphGraph &graph,
              const std::vector<const NodeProto *> &nodes) const override {
    const auto plan = MakePlan(graph, *nodes.at(0));
    if (!plan) {
      throw builder::BuilderError("WeightBiasQuantization: invalid match");
    }
    Nodes result;
    auto replacement = *nodes[0];
    auto &builder = graph.Builder();
    const auto initializer = [&](TensorProto tensor, const std::string &prefix) {
      auto name = prefix;
      for (std::size_t suffix = 0; builder.HasName(name); ++suffix) {
        name = prefix + "_" + std::to_string(suffix);
      }
      tensor.set_name(name);
      return std::string(builder.MakeInitializer(tensor));
    };
    const auto scalar = [&](float value) {
      TensorProto tensor;
      tensor.set_data_type(TensorProto::FLOAT);
      tensor.ref_float_data().push_back(value);
      return initializer(tensor, "weight_bias_scale");
    };
    const auto emit = [&](const std::string &op, const std::vector<std::string> &inputs) {
      NodeProto node;
      node.set_op_type(op);
      node.set_name(builder.UniqueName("weight_bias_" + op));
      for (const auto &input : inputs) {
        node.add_input(input);
      }
      const auto output = builder.UniqueName("weight_bias_value");
      node.add_output(output);
      result.push_back(std::move(node));
      return output;
    };
    const auto materialize = [&](const std::string &input, const std::string &prefix) {
      if (graph.NodeBefore(input) == nullptr) {
        return input;
      }
      const auto *value = graph.GetComputedConstant(input);
      if (value == nullptr) {
        throw builder::BuilderError("WeightBiasQuantization: missing computed constant");
      }
      return initializer(*value, prefix);
    };
    if (plan->quantize_weight) {
      const auto scale = scalar(plan->weight_scale);
      TensorProto zero;
      zero.set_data_type(TensorProto::INT8);
      zero.ref_int32_data().push_back(0);
      const auto zero_name = initializer(zero, "weight_bias_zero");
      const auto weight = materialize(replacement.input(1), "weight_bias_weight");
      const auto quantized = emit("QuantizeLinear", {weight, scale, zero_name});
      replacement.ref_input()[1] = emit("DequantizeLinear", {quantized, scale, zero_name});
    }
    if (plan->bias_scale) {
      const auto scale = scalar(*plan->bias_scale);
      const auto bias = materialize(replacement.input(2), "weight_bias_bias");
      const auto divided = emit("Div", {bias, scale});
      const auto rounded = emit("Round", {divided});
      const auto quantized = emit("Cast", {rounded});
      ONNX_LIGHT_NAMESPACE::AddAttribute(result.back(), "to", int64_t{TensorProto::INT32});
      replacement.ref_input()[2] = emit("DequantizeLinear", {quantized, scale});
    }
    result.push_back(std::move(replacement));
    return result;
  }
};

} // namespace

ONNX_LIGHT_NAMESPACE::RewritingProto PrepareWeightBiasQuantization(builder::GraphBuilder &graph) {
  if (graph.device() != sym::Device::kCPU) {
    return ONNX_LIGHT_NAMESPACE::RewritingProto();
  }
  NearestRounding rounding;
  ONNX_LIGHT_NAMESPACE::onnx_kernels::RegisterKernelFunctions();
  std::vector<std::unique_ptr<builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<WeightBiasQuantizationPattern>());
  builder::GraphGraph optimizer(graph, std::move(patterns));
  builder::ConstantFoldingOptions folding;
  folding.excluded_ops.insert({"", "DequantizeLinear"});
  std::vector<std::string> computed_values;
  for (const auto &node : graph.Nodes()) {
    for (const auto &name : node.output()) {
      if (optimizer.IsConstant(name)) {
        computed_values.push_back(name);
      }
    }
  }
  if (!computed_values.empty()) {
    builder::GraphBuilder evaluated(graph.ToModel(), {}, graph.LocalRegistry());
    evaluated.ConstantFold(folding);
    builder::GraphGraph evaluated_graph(evaluated);
    for (const auto &name : computed_values) {
      if (const auto *value = evaluated_graph.GetComputedConstant(name)) {
        optimizer.SetComputedConstant(name, *value);
      }
    }
  }
  return optimizer.Optimize(-1, nullptr, folding, false, {}, false, false, false);
}

} // namespace onnx_light_cpu
