// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/gradient/com_microsoft/gradients.h"

#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/gradient/grad_common.h"
#include "onnx_proto/onnx_helper.h"

#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

namespace onnx_light_cpu {
namespace {

namespace grad_ns = ONNX_LIGHT_NAMESPACE::core::gradient;

using ONNX_LIGHT_NAMESPACE::AttributeProto;
using ONNX_LIGHT_NAMESPACE::FindAttribute;
using ONNX_LIGHT_NAMESPACE::FunctionProto;
using ONNX_LIGHT_NAMESPACE::NodeProto;
using ONNX_LIGHT_NAMESPACE::TensorProto;

std::string AddConstant(FunctionProto &func, int &counter, const char *prefix, float value) {
  const std::string output = grad_ns::NewGradName(prefix, counter);
  NodeProto &node = func.add_node("Constant", {}, {output});
  AttributeProto *attribute = node.add_attribute();
  attribute->set_name("value");
  attribute->set_type(AttributeProto::AttributeType::TENSOR);
  TensorProto &tensor = attribute->ref_t();
  tensor.set_data_type(TensorProto::DataType::FLOAT);
  tensor.ref_float_data().push_back(value);
  return output;
}

std::string AddAxes(FunctionProto &func, int &counter, const char *prefix,
                    std::vector<int64_t> values) {
  const std::string output = grad_ns::NewGradName(prefix, counter);
  NodeProto &node = func.add_node("Constant", {}, {output});
  AttributeProto *attribute = node.add_attribute();
  attribute->set_name("value");
  attribute->set_type(AttributeProto::AttributeType::TENSOR);
  TensorProto &tensor = attribute->ref_t();
  tensor.set_data_type(TensorProto::DataType::INT64);
  tensor.ref_dims().push_back(static_cast<int64_t>(values.size()));
  for (int64_t value : values) {
    tensor.ref_int64_data().push_back(value);
  }
  return output;
}

std::string AddInt64Constant(FunctionProto &func, int &counter, const char *prefix, int64_t value) {
  const std::string output = grad_ns::NewGradName(prefix, counter);
  NodeProto &node = func.add_node("Constant", {}, {output});
  AttributeProto *attribute = node.add_attribute();
  attribute->set_name("value");
  attribute->set_type(AttributeProto::AttributeType::TENSOR);
  TensorProto &tensor = attribute->ref_t();
  tensor.set_data_type(TensorProto::DataType::INT64);
  tensor.ref_int64_data().push_back(value);
  return output;
}

std::string Unary(FunctionProto &func, int &counter, const char *op_type, const std::string &input,
                  const char *prefix) {
  const std::string output = grad_ns::NewGradName(prefix, counter);
  func.add_node(op_type, {input}, {output});
  return output;
}

std::string Binary(FunctionProto &func, int &counter, const char *op_type, const std::string &left,
                   const std::string &right, const char *prefix) {
  const std::string output = grad_ns::NewGradName(prefix, counter);
  func.add_node(op_type, {left, right}, {output});
  return output;
}

std::string CastConstantLike(FunctionProto &func, int &counter, const std::string &like,
                             const char *prefix, float value) {
  return Binary(func, counter, "CastLike", AddConstant(func, counter, prefix, value), like,
                "cast_constant");
}

std::string CastToFloat(FunctionProto &func, int &counter, const std::string &input,
                        const char *prefix) {
  const std::string output = grad_ns::NewGradName(prefix, counter);
  NodeProto &node = func.add_node("Cast", {input}, {output});
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "to",
                                     static_cast<int64_t>(TensorProto::DataType::FLOAT));
  return output;
}

std::string Unsqueeze(FunctionProto &func, int &counter, const std::string &input, int64_t axis,
                      const char *prefix) {
  const std::string axes = AddAxes(func, counter, "axes", {axis});
  return Binary(func, counter, "Unsqueeze", input, axes, prefix);
}

std::string ReduceSum(FunctionProto &func, int &counter, const std::string &input, int64_t axis,
                      const char *prefix) {
  const std::string axes = AddAxes(func, counter, "axes", {axis});
  const std::string output = grad_ns::NewGradName(prefix, counter);
  NodeProto &node = func.add_node("ReduceSum", {input, axes}, {output});
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "keepdims", int64_t{0});
  return output;
}

std::string Reshape(FunctionProto &func, int &counter, const std::string &input,
                    std::vector<int64_t> shape, const char *prefix) {
  return Binary(func, counter, "Reshape", input, AddAxes(func, counter, "shape", std::move(shape)),
                prefix);
}

std::string Transpose(FunctionProto &func, int &counter, const std::string &input,
                      std::vector<int64_t> perm, const char *prefix) {
  const std::string output = grad_ns::NewGradName(prefix, counter);
  NodeProto &node = func.add_node("Transpose", {input}, {output});
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "perm", std::move(perm));
  return output;
}

std::string ReduceSumKeepDims(FunctionProto &func, int &counter, const std::string &input,
                              int64_t axis, const char *prefix) {
  const std::string axes = AddAxes(func, counter, "axes", {axis});
  const std::string output = grad_ns::NewGradName(prefix, counter);
  NodeProto &node = func.add_node("ReduceSum", {input, axes}, {output});
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "keepdims", int64_t{1});
  return output;
}

int64_t GetIntAttributeOrDefault(const NodeProto &node, const char *name, int64_t default_value) {
  const AttributeProto *attribute = FindAttribute(node, name);
  return attribute == nullptr ? default_value : attribute->i();
}

float GetFloatAttributeOrDefault(const NodeProto &node, const char *name, float default_value) {
  const AttributeProto *attribute = FindAttribute(node, name);
  return attribute == nullptr ? default_value : attribute->f();
}

std::string GetStringAttributeOrDefault(const NodeProto &node, const char *name,
                                        const char *default_value) {
  const AttributeProto *attribute = FindAttribute(node, name);
  return attribute == nullptr ? default_value : std::string(attribute->s());
}

bool HasInput(const NodeProto &node, int index) {
  return node.input_size() > index && !node.input(index).empty();
}

bool IsSupportedGroupQueryAttention(const NodeProto &node, int64_t &num_heads,
                                    int64_t &kv_num_heads) {
  if (node.input_size() < 7 || node.input_size() > 16 || node.output_size() != 1 ||
      node.output(0).empty() || !HasInput(node, 0) || !HasInput(node, 1) || !HasInput(node, 2) ||
      !HasInput(node, 5) || !HasInput(node, 6) || HasInput(node, 3) || HasInput(node, 4)) {
    return false;
  }
  for (int index = 7; index < node.input_size(); ++index) {
    if (HasInput(node, index)) {
      return false;
    }
  }
  num_heads = GetIntAttributeOrDefault(node, "num_heads", 0);
  kv_num_heads = GetIntAttributeOrDefault(node, "kv_num_heads", 0);
  return num_heads > 0 && kv_num_heads > 0 && num_heads % kv_num_heads == 0 &&
         GetIntAttributeOrDefault(node, "do_rotary", 0) == 0 &&
         GetIntAttributeOrDefault(node, "sliding_window_cache", 0) == 0 &&
         GetIntAttributeOrDefault(node, "smooth_softmax", 0) == 0 &&
         GetIntAttributeOrDefault(node, "qk_output", 0) == 0 &&
         GetIntAttributeOrDefault(node, "kv_cache_bit_width", 0) == 0 &&
         GetIntAttributeOrDefault(node, "local_window_size", -1) == -1 &&
         GetStringAttributeOrDefault(node, "k_quant_type", "NONE") == "NONE" &&
         GetStringAttributeOrDefault(node, "v_quant_type", "NONE") == "NONE" &&
         GetFloatAttributeOrDefault(node, "softcap", 0.0f) == 0.0f;
}

bool GradGroupQueryAttention(const NodeProto &node, const std::string &output_grad,
                             std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                             FunctionProto &func) {
  int64_t num_heads = 0;
  int64_t kv_num_heads = 0;
  if (!IsSupportedGroupQueryAttention(node, num_heads, kv_num_heads)) {
    return false;
  }
  const int64_t group_size = num_heads / kv_num_heads;
  const std::string &query = node.input(0);
  const std::string &key = node.input(1);
  const std::string &value = node.input(2);
  const std::string float_query = CastToFloat(func, counter, query, "float_query");
  const std::string float_key = CastToFloat(func, counter, key, "float_key");
  const std::string float_value = CastToFloat(func, counter, value, "float_value");
  const std::string float_output_grad =
      CastToFloat(func, counter, output_grad, "float_output_grad");

  const std::string probabilities = grad_ns::NewGradName("attention_probabilities", counter);
  const std::string ignored = grad_ns::NewGradName("attention_output", counter);
  NodeProto &attention = func.add_node("Attention", {float_query, float_key, float_value},
                                       {ignored, "", "", probabilities});
  ONNX_LIGHT_NAMESPACE::AddAttribute(attention, "q_num_heads", num_heads);
  ONNX_LIGHT_NAMESPACE::AddAttribute(attention, "kv_num_heads", kv_num_heads);
  ONNX_LIGHT_NAMESPACE::AddAttribute(attention, "is_causal",
                                     GetIntAttributeOrDefault(node, "causal", 1));
  ONNX_LIGHT_NAMESPACE::AddAttribute(attention, "qk_matmul_output_mode", int64_t{3});
  if (const AttributeProto *scale = FindAttribute(node, "scale"); scale != nullptr) {
    ONNX_LIGHT_NAMESPACE::AddAttribute(attention, "scale", scale->f());
  }
  if (const AttributeProto *softcap = FindAttribute(node, "softcap"); softcap != nullptr) {
    ONNX_LIGHT_NAMESPACE::AddAttribute(attention, "softcap", softcap->f());
  }

  const std::string dy = Transpose(
      func, counter, Reshape(func, counter, float_output_grad, {0, 0, num_heads, -1}, "reshape_dy"),
      {0, 2, 1, 3}, "transpose_dy");
  const std::string key_heads =
      Reshape(func, counter, float_key, {0, 0, kv_num_heads, -1}, "reshape_key");
  const std::string value_heads =
      Reshape(func, counter, float_value, {0, 0, kv_num_heads, -1}, "reshape_value");
  const std::string group_repeats =
      AddAxes(func, counter, "group_repeats", {1, 1, 1, group_size, 1});
  const auto expand_heads = [&](const std::string &input, const char *prefix) {
    const std::string unsqueezed =
        Unsqueeze(func, counter, input, 3, (std::string(prefix) + "_unsqueeze").c_str());
    const std::string tiled = Binary(func, counter, "Tile", unsqueezed, group_repeats,
                                     (std::string(prefix) + "_tile").c_str());
    return Transpose(func, counter,
                     Reshape(func, counter, tiled, {0, 0, num_heads, -1},
                             (std::string(prefix) + "_reshape").c_str()),
                     {0, 2, 1, 3}, (std::string(prefix) + "_transpose").c_str());
  };
  const std::string expanded_key = expand_heads(key_heads, "expanded_key");
  const std::string expanded_value = expand_heads(value_heads, "expanded_value");

  const std::string dvalue_heads =
      Binary(func, counter, "MatMul",
             Transpose(func, counter, probabilities, {0, 1, 3, 2}, "transpose_probabilities"), dy,
             "dvalue_heads");
  const std::string dprobabilities = Binary(
      func, counter, "MatMul", dy,
      Transpose(func, counter, expanded_value, {0, 1, 3, 2}, "transpose_value"), "dprobabilities");
  const std::string dscore =
      Binary(func, counter, "Mul", probabilities,
             Binary(func, counter, "Sub", dprobabilities,
                    ReduceSumKeepDims(func, counter,
                                      Binary(func, counter, "Mul", dprobabilities, probabilities,
                                             "weighted_probabilities"),
                                      -1, "sum_weighted_probabilities"),
                    "centered_probabilities"),
             "dscore");
  std::string scale;
  if (const AttributeProto *scale_attribute = FindAttribute(node, "scale");
      scale_attribute != nullptr) {
    scale = AddConstant(func, counter, "scale", scale_attribute->f());
  } else {
    const std::string query_shape = Unary(func, counter, "Shape", query, "query_shape");
    const std::string hidden_size =
        Binary(func, counter, "Gather", query_shape,
               AddInt64Constant(func, counter, "hidden_axis", 2), "hidden_size");
    const std::string head_dim =
        Binary(func, counter, "Div", hidden_size,
               AddInt64Constant(func, counter, "num_heads", num_heads), "head_dim");
    scale = Unary(func, counter, "Reciprocal",
                  Unary(func, counter, "Sqrt",
                        CastToFloat(func, counter, head_dim, "float_head_dim"), "sqrt_head_dim"),
                  "scale");
  }
  const std::string dquery = Binary(
      func, counter, "Mul", Binary(func, counter, "MatMul", dscore, expanded_key, "dquery_heads"),
      scale, "scaled_dquery");
  const std::string dkey_heads = Binary(
      func, counter, "Mul",
      Binary(func, counter, "MatMul",
             Transpose(func, counter, dscore, {0, 1, 3, 2}, "transpose_dscore"),
             Transpose(func, counter,
                       Reshape(func, counter, float_query, {0, 0, num_heads, -1}, "reshape_query"),
                       {0, 2, 1, 3}, "transpose_query"),
             "dkey_heads"),
      scale, "scaled_dkey");
  const auto reduce_groups = [&](const std::string &input, const char *prefix) {
    return Reshape(func, counter,
                   ReduceSum(func, counter,
                             Reshape(func, counter,
                                     Transpose(func, counter, input, {0, 2, 1, 3},
                                               (std::string(prefix) + "_transpose").c_str()),
                                     {0, 0, kv_num_heads, group_size, -1},
                                     (std::string(prefix) + "_grouped").c_str()),
                             3, (std::string(prefix) + "_reduce").c_str()),
                   {0, 0, -1}, prefix);
  };
  const std::string dquery_output =
      Reshape(func, counter, Transpose(func, counter, dquery, {0, 2, 1, 3}, "transpose_dquery"),
              {0, 0, -1}, "dquery_output");
  grad_ns::AccumulateGrad(Binary(func, counter, "CastLike", dquery_output, query, "typed_dquery"),
                          grad_accum[query], counter, func);
  grad_ns::AccumulateGrad(Binary(func, counter, "CastLike",
                                 reduce_groups(dkey_heads, "dkey_output"), key, "typed_dkey"),
                          grad_accum[key], counter, func);
  grad_ns::AccumulateGrad(Binary(func, counter, "CastLike",
                                 reduce_groups(dvalue_heads, "dvalue_output"), value,
                                 "typed_dvalue"),
                          grad_accum[value], counter, func);
  return true;
}

bool GradBiasGelu(const NodeProto &node, const std::string &output_grad,
                  std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                  FunctionProto &func) {
  if (node.input_size() != 2 || node.output_size() != 1) {
    return false;
  }
  const std::string &a = node.input(0);
  const std::string &bias = node.input(1);
  const std::string z = Binary(func, counter, "Add", a, bias, "z");
  const std::string inv_sqrt_two =
      CastConstantLike(func, counter, z, "inv_sqrt_two", 0.70710678118654752440f);
  const std::string scaled = Binary(func, counter, "Mul", z, inv_sqrt_two, "scaled");
  const std::string erf = Unary(func, counter, "Erf", scaled, "erf");
  const std::string one = CastConstantLike(func, counter, z, "one", 1.0f);
  const std::string half = CastConstantLike(func, counter, z, "half", 0.5f);
  const std::string cdf = Binary(
      func, counter, "Mul", Binary(func, counter, "Add", one, erf, "one_plus_erf"), half, "cdf");
  const std::string z_squared = Binary(func, counter, "Mul", z, z, "z_squared");
  const std::string minus_half = CastConstantLike(func, counter, z, "minus_half", -0.5f);
  const std::string density_exp =
      Unary(func, counter, "Exp",
            Binary(func, counter, "Mul", z_squared, minus_half, "negative_half_z_squared"),
            "density_exp");
  const std::string inv_sqrt_two_pi =
      CastConstantLike(func, counter, z, "inv_sqrt_two_pi", 0.39894228040143267794f);
  const std::string density = Binary(func, counter, "Mul", density_exp, inv_sqrt_two_pi, "density");
  const std::string derivative =
      Binary(func, counter, "Add", cdf, Binary(func, counter, "Mul", z, density, "z_density"),
             "derivative");
  const std::string dz = Binary(func, counter, "Mul", output_grad, derivative, "bias_gelu_grad");
  grad_ns::AccumulateGrad(dz, grad_accum[a], counter, func);

  const std::string a_shape = Unary(func, counter, "Shape", a, "a_shape");
  const std::string rank = Unary(func, counter, "Size", a_shape, "a_rank");
  const std::string zero = AddInt64Constant(func, counter, "axis_zero", 0);
  const std::string axis_one = AddInt64Constant(func, counter, "axis_one", 1);
  const std::string last_axis = Binary(func, counter, "Sub", rank, axis_one, "last_axis");
  const std::string reduction_axes = grad_ns::NewGradName("bias_reduction_axes", counter);
  func.add_node("Range", {zero, last_axis, axis_one}, {reduction_axes});
  const std::string dbias = grad_ns::NewGradName("bias_grad", counter);
  NodeProto &reduce = func.add_node("ReduceSum", {dz, reduction_axes}, {dbias});
  ONNX_LIGHT_NAMESPACE::AddAttribute(reduce, "keepdims", int64_t{0});
  ONNX_LIGHT_NAMESPACE::AddAttribute(reduce, "noop_with_empty_axes", int64_t{1});
  grad_ns::AccumulateGrad(dbias, grad_accum[bias], counter, func);
  return true;
}

bool GradCDist(const NodeProto &node, const std::string &output_grad,
               std::unordered_map<std::string, std::string> &grad_accum, int &counter,
               FunctionProto &func) {
  if (node.input_size() != 2 || node.output_size() != 1) {
    return false;
  }
  const std::string &a = node.input(0);
  const std::string &b = node.input(1);
  const std::string a_unsqueezed = Unsqueeze(func, counter, a, 1, "a_unsqueezed");
  const std::string b_unsqueezed = Unsqueeze(func, counter, b, 0, "b_unsqueezed");
  const std::string difference =
      Binary(func, counter, "Sub", a_unsqueezed, b_unsqueezed, "difference");
  std::string pair_gradient = output_grad;
  const AttributeProto *metric = FindAttribute(node, "metric");
  const std::string metric_value = metric == nullptr ? "sqeuclidean" : std::string(metric->s());
  if (metric_value == "euclidean") {
    const std::string distance = node.output(0);
    const std::string zero = CastConstantLike(func, counter, distance, "zero", 0.0f);
    const std::string is_zero = Binary(func, counter, "Equal", distance, zero, "is_zero");
    const std::string divided =
        Binary(func, counter, "Div", output_grad, distance, "distance_scaled_grad");
    pair_gradient = grad_ns::NewGradName("safe_distance_scaled_grad", counter);
    func.add_node("Where", {is_zero, zero, divided}, {pair_gradient});
  } else {
    const std::string two = CastConstantLike(func, counter, output_grad, "two", 2.0f);
    pair_gradient = Binary(func, counter, "Mul", output_grad, two, "twice_output_grad");
  }
  const std::string weighted =
      Binary(func, counter, "Mul", difference,
             Unsqueeze(func, counter, pair_gradient, 2, "pair_gradient"), "weighted_difference");
  grad_ns::AccumulateGrad(ReduceSum(func, counter, weighted, 1, "a_grad"), grad_accum[a], counter,
                          func);
  const std::string b_grad =
      Unary(func, counter, "Neg", ReduceSum(func, counter, weighted, 0, "b_grad_sum"), "b_grad");
  grad_ns::AccumulateGrad(b_grad, grad_accum[b], counter, func);
  return true;
}

} // namespace

void RegisterCustomOperatorGradients(grad_ns::GradRegistry &registry) {
  grad_ns::RegisterGradientFunction(kMicrosoftDomain, "BiasGelu", GradBiasGelu, registry);
  grad_ns::RegisterGradientFunction(kMicrosoftDomain, "CDist", GradCDist, registry);
  grad_ns::RegisterGradientFunction(kMicrosoftDomain, "GroupQueryAttention",
                                    GradGroupQueryAttention, registry);
}

} // namespace onnx_light_cpu
