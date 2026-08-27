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
}

} // namespace onnx_light_cpu
