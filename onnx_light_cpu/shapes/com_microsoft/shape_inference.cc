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
      {kMicrosoftDomain,
       "GroupQueryAttention",
       "onnx_light_cpu::ComputeShapeGroupQueryAttention",
       "onnx_light_cpu::ComputePeakMemoryGroupQueryAttention",
       {"onnx_light_cpu::GroupQueryAttentionFusionPattern"},
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

void ComputeShapeGroupQueryAttention(ShapesContext &ctx,
                                     const ONNX_LIGHT_NAMESPACE::NodeProto &node) {
  if (node.input_size() < 7 || node.output_size() != 1 || !ctx.Has(node.input(0)) ||
      !ctx.Has(node.input(1)) || !ctx.Has(node.input(2))) {
    throw std::invalid_argument("ComputeShapeGroupQueryAttention: expected known Q/K/V inputs and "
                                "one output.");
  }
  const SymTensor &query = ctx.Get(node.input(0));
  const SymTensor &key = ctx.Get(node.input(1));
  const SymTensor &value = ctx.Get(node.input(2));
  if (query.Dtype() != key.Dtype() || query.Dtype() != value.Dtype()) {
    throw std::invalid_argument(
        "ComputeShapeGroupQueryAttention: query, key, and value types must match.");
  }
  if (query.Shape().Rank() != 3 || key.Shape().Rank() != 3 || value.Shape().Rank() != 3) {
    throw std::invalid_argument(
        "ComputeShapeGroupQueryAttention: query, key, and value must have rank 3.");
  }
  ConstrainEqual(ctx, query.Shape()[0], key.Shape()[0],
                 "ComputeShapeGroupQueryAttention: query and key batch dimensions must match.");
  ConstrainEqual(ctx, query.Shape()[0], value.Shape()[0],
                 "ComputeShapeGroupQueryAttention: query and value batch dimensions must match.");
  ConstrainEqual(ctx, key.Shape()[1], value.Shape()[1],
                 "ComputeShapeGroupQueryAttention: key and value sequence dimensions must match.");
  ctx.Set(node.output(0), SymTensor(nullptr, query.Dtype(), query.Shape()));
}

int64_t ComputePeakMemoryCDist(sym_ns::Device, const std::vector<SymShape> &) { return 0; }

int64_t ComputePeakMemoryBiasGelu(sym_ns::Device, const std::vector<SymShape> &) { return 0; }

int64_t ComputePeakMemoryGroupQueryAttention(sym_ns::Device, const std::vector<SymShape> &) {
  // The supported GroupQueryAttention path delegates to Attention's online-softmax
  // implementation, which does not materialize a full attention-score tensor.
  return 0;
}

void RegisterMicrosoftShapeAndMemoryFunctions() {
  static std::once_flag once;
  std::call_once(once, [] {
    shapes_ns::RegisterComputeShapeFn(kMicrosoftDomain, "CDist", ComputeShapeCDist);
    shapes_ns::RegisterComputeShapeFn(kMicrosoftDomain, "BiasGelu", ComputeShapeBiasGelu);
    shapes_ns::RegisterComputeShapeFn(kMicrosoftDomain, "GroupQueryAttention",
                                      ComputeShapeGroupQueryAttention);
    shapes_ns::RegisterComputePeakMemoryFn(kMicrosoftDomain, "CDist", sym_ns::Device::kCPU,
                                           ComputePeakMemoryCDist);
    shapes_ns::RegisterComputePeakMemoryFn(kMicrosoftDomain, "BiasGelu", sym_ns::Device::kCPU,
                                           ComputePeakMemoryBiasGelu);
    shapes_ns::RegisterComputePeakMemoryFn(kMicrosoftDomain, "GroupQueryAttention",
                                           sym_ns::Device::kCPU,
                                           ComputePeakMemoryGroupQueryAttention);
  });
}

} // namespace onnx_light_cpu
