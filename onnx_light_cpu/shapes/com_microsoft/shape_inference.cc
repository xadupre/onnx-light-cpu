// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/shapes/com_microsoft/shape_inference.h"

#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_core/symbolic/sym_tensor.h"

#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace onnx_light_cpu {
namespace {

namespace shapes_ns = ONNX_LIGHT_NAMESPACE::core::shapes;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

using shapes_ns::ShapesContext;
using sym_ns::SymShape;
using sym_ns::SymTensor;

void RequireInputs(const ShapesContext &ctx, const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                   const char *op_type) {
  if (node.input_size() != 2 || node.output_size() != 1 || !ctx.Has(node.input(0)) ||
      !ctx.Has(node.input(1))) {
    throw std::invalid_argument(std::string("ComputeShape") + op_type +
                                ": expected two known inputs and one output.");
  }
}

void ConstrainEqual(ShapesContext &ctx, const sym_ns::SymDim &left, const sym_ns::SymDim &right,
                    const char *message) {
  if (left.IsInt() && right.IsInt()) {
    if (left.AsInt() != right.AsInt()) {
      throw std::invalid_argument(message);
    }
    return;
  }
  if (left.IsExpr() && right.IsExpr() && left.AsExpr() != right.AsExpr()) {
    ctx.AddConstraint(left.AsExpr(), right.AsExpr());
  }
}

} // namespace

std::vector<OperatorSupportRegistration> CollectOperatorSupport() {
  return {
      {kMicrosoftDomain,
       "BiasGelu",
       "onnx_light_cpu::ComputeShapeBiasGelu",
       "onnx_light_cpu::ComputePeakMemoryBiasGelu",
       {"onnx_light_cpu::BiasGeluFusionPattern"},
       true},
      {kMicrosoftDomain,
       "CDist",
       "onnx_light_cpu::ComputeShapeCDist",
       "onnx_light_cpu::ComputePeakMemoryCDist",
       {"onnx_light_cpu::CDistFusionPattern"},
       true},
  };
}

void ComputeShapeCDist(ShapesContext &ctx, const ONNX_LIGHT_NAMESPACE::NodeProto &node) {
  RequireInputs(ctx, node, "CDist");
  const SymTensor &a = ctx.Get(node.input(0));
  const SymTensor &b = ctx.Get(node.input(1));
  if (a.Dtype() != b.Dtype()) {
    throw std::invalid_argument("ComputeShapeCDist: input element types must match.");
  }
  if (a.Shape().Rank() != 2 || b.Shape().Rank() != 2) {
    throw std::invalid_argument("ComputeShapeCDist: both inputs must have rank 2.");
  }
  ConstrainEqual(ctx, a.Shape()[1], b.Shape()[1],
                 "ComputeShapeCDist: feature dimensions must match.");
  SymShape output_shape;
  output_shape.PushBack(a.Shape()[0]);
  output_shape.PushBack(b.Shape()[0]);
  ctx.Set(node.output(0), SymTensor(nullptr, a.Dtype(), std::move(output_shape)));
}

void ComputeShapeBiasGelu(ShapesContext &ctx, const ONNX_LIGHT_NAMESPACE::NodeProto &node) {
  RequireInputs(ctx, node, "BiasGelu");
  const SymTensor &a = ctx.Get(node.input(0));
  const SymTensor &bias = ctx.Get(node.input(1));
  if (a.Dtype() != bias.Dtype()) {
    throw std::invalid_argument("ComputeShapeBiasGelu: input element types must match.");
  }
  if (a.Shape().Rank() == 0 || bias.Shape().Rank() != 1) {
    throw std::invalid_argument(
        "ComputeShapeBiasGelu: A must have positive rank and B must have rank 1.");
  }
  ConstrainEqual(ctx, a.Shape()[a.Shape().Rank() - 1], bias.Shape()[0],
                 "ComputeShapeBiasGelu: bias length must match the last input dimension.");
  ctx.Set(node.output(0), SymTensor(nullptr, a.Dtype(), a.Shape()));
}

int64_t ComputePeakMemoryCDist(sym_ns::Device, const std::vector<SymShape> &) { return 0; }

int64_t ComputePeakMemoryBiasGelu(sym_ns::Device, const std::vector<SymShape> &) { return 0; }

void RegisterMicrosoftShapeAndMemoryFunctions() {
  static std::once_flag once;
  std::call_once(once, [] {
    shapes_ns::RegisterComputeShapeFn(kMicrosoftDomain, "CDist", ComputeShapeCDist);
    shapes_ns::RegisterComputeShapeFn(kMicrosoftDomain, "BiasGelu", ComputeShapeBiasGelu);
    shapes_ns::RegisterComputePeakMemoryFn(kMicrosoftDomain, "CDist", sym_ns::Device::kCPU,
                                           ComputePeakMemoryCDist);
    shapes_ns::RegisterComputePeakMemoryFn(kMicrosoftDomain, "BiasGelu", sym_ns::Device::kCPU,
                                           ComputePeakMemoryBiasGelu);
  });
}

} // namespace onnx_light_cpu
