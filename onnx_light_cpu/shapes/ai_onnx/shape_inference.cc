// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/shapes/ai_onnx/shape_inference.h"

#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_core/shapes/shape_broadcast.h"

#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace onnx_light_cpu {
namespace {

namespace shapes_ns = ONNX_LIGHT_NAMESPACE::core::shapes;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;
using sym_ns::SymDim;
using sym_ns::SymShape;
using sym_ns::SymTensor;
using sym_ns::TensorType;

void CheckFloatType(TensorType type) {
  if (type != TensorType::kUndefined && type != TensorType::kFloat &&
      type != TensorType::kFloat16 && type != TensorType::kDouble &&
      type != TensorType::kBfloat16) {
    throw std::invalid_argument("SimplifiedLayerNormalization: expected a floating-point tensor.");
  }
}

void CheckScaleBroadcast(shapes_ns::ShapesContext &ctx, const SymShape &x, const SymShape &scale) {
  if (scale.Rank() > x.Rank()) {
    throw std::invalid_argument("SimplifiedLayerNormalization: Scale rank exceeds X rank.");
  }
  for (size_t i = 0; i < scale.Rank(); ++i) {
    if (scale[i].IsInt() && scale[i].AsInt() < 0) {
      throw std::invalid_argument(
          "SimplifiedLayerNormalization: Scale dimensions must be nonnegative.");
    }
  }
  // Reuse the shared helper for concrete broadcast validation, then constrain
  // unidirectional broadcasting without incorrectly forcing symbolic Scale dimensions != 1.
  shapes_ns::BroadcastShapes(x, scale);
  for (size_t i = 0; i < scale.Rank(); ++i) {
    const auto &s = scale[i];
    const auto &d = x[x.Rank() - scale.Rank() + i];
    if (s == d || (s.IsInt() && s.AsInt() == 1)) {
      continue;
    }
    if (s.IsInt()) {
      if (d.IsInt()) {
        throw std::invalid_argument("SimplifiedLayerNormalization: Scale cannot expand X.");
      }
      ctx.AddConstraint(d.AsExpr(), s.ToString());
    } else if (d.IsInt() && d.AsInt() == 1) {
      ctx.AddConstraint(s.AsExpr(), "1");
    } else {
      // Scale must equal either one or the corresponding X dimension.
      ctx.AddConstraint("(" + s.AsExpr() + "-1)*(" + s.AsExpr() + "-(" + d.ToString() + "))", "0");
    }
  }
}

} // namespace

void ComputeShapeSimplifiedLayerNormalization(shapes_ns::ShapesContext &ctx,
                                              const ONNX_LIGHT_NAMESPACE::NodeProto &node) {
  if (node.input_size() != 2 || node.input(0).empty() || node.input(1).empty() ||
      node.output_size() < 1 || node.output_size() > 2 || node.output(0).empty()) {
    throw std::invalid_argument(
        "SimplifiedLayerNormalization: expected X, Scale, Y, and optional inv_std_var.");
  }
  int64_t axis = -1;
  int64_t stash_type = 1;
  for (const auto &attribute : node.attribute()) {
    if (attribute.name() == "axis") {
      axis = attribute.i();
    } else if (attribute.name() == "stash_type") {
      stash_type = attribute.i();
    }
  }
  if (stash_type != 1 && stash_type != 11) {
    throw std::invalid_argument("SimplifiedLayerNormalization: stash_type must be 1 or 11.");
  }
  if (!ctx.Has(node.input(0)) || !ctx.Has(node.input(1))) {
    return;
  }
  const auto &x = ctx.Get(node.input(0));
  const auto &scale = ctx.Get(node.input(1));
  CheckFloatType(x.Dtype());
  CheckFloatType(scale.Dtype());
  const auto rank = static_cast<int64_t>(x.Shape().Rank());
  if (rank < 1 || axis < -rank || axis >= rank) {
    throw std::invalid_argument("SimplifiedLayerNormalization: axis is outside X's positive rank.");
  }
  if (axis < 0) {
    axis += rank;
  }
  for (int64_t i = 0; i < rank; ++i) {
    const auto &dim = x.Shape()[i];
    if (dim.IsInt() && (dim.AsInt() < 0 || (i >= axis && dim.AsInt() == 0))) {
      throw std::invalid_argument(
          "SimplifiedLayerNormalization: dimensions must be nonnegative and the suffix nonempty.");
    }
    if (dim.IsExpr()) {
      ctx.AddLessEqualConstraint(i >= axis ? "1" : "0", dim.AsExpr());
    }
  }
  CheckScaleBroadcast(ctx, x.Shape(), scale.Shape());
  SymShape statistics_shape = x.Shape();
  for (int64_t i = axis; i < rank; ++i) {
    statistics_shape[i] = SymDim(int64_t{1});
  }
  ctx.Set(node.output(0), SymTensor(nullptr, scale.Dtype(), x.Shape()));
  if (node.output_size() == 2 && !node.output(1).empty()) {
    ctx.Set(node.output(1),
            SymTensor(nullptr, stash_type == 1 ? TensorType::kFloat : TensorType::kDouble,
                      std::move(statistics_shape)));
  }
}

int64_t ComputePeakMemorySimplifiedLayerNormalization(sym_ns::Device,
                                                      const std::vector<SymShape> &) {
  return 0;
}

void RegisterExperimentalShapeAndMemoryFunctions() {
  static std::once_flag once;
  std::call_once(once, [] {
    shapes_ns::RegisterComputeShapeFn("ai.onnx", "SimplifiedLayerNormalization",
                                      ComputeShapeSimplifiedLayerNormalization);
    shapes_ns::RegisterComputePeakMemoryFn("ai.onnx", "SimplifiedLayerNormalization",
                                           sym_ns::Device::kCPU,
                                           ComputePeakMemorySimplifiedLayerNormalization);
  });
}

} // namespace onnx_light_cpu
