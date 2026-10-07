// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/shapes/ai_onnx/shape_inference.h"

#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_core/shapes/shape_broadcast.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

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

void ComputeShapeRotaryEmbedding(shapes_ns::ShapesContext &ctx,
                                 const ONNX_LIGHT_NAMESPACE::NodeProto &node) {
  const bool microsoft = node.domain() == kMicrosoftDomain;
  if ((microsoft && node.input_size() != 4) ||
      (!microsoft && node.input_size() != 3 && node.input_size() != 4) || node.output_size() != 1 ||
      node.output(0).empty()) {
    throw std::invalid_argument("RotaryEmbedding: invalid input or output count.");
  }
  const auto &input_name = node.input(0);
  const auto &cos_name = node.input(microsoft ? 2 : 1);
  const auto &sin_name = node.input(microsoft ? 3 : 2);
  const auto &position_name = node.input(microsoft ? 1 : (node.input_size() == 4 ? 3 : 0));
  if (input_name.empty() || cos_name.empty() || sin_name.empty() ||
      (microsoft && position_name.empty()) || !ctx.Has(input_name) || !ctx.Has(cos_name) ||
      !ctx.Has(sin_name) ||
      (!position_name.empty() && position_name != input_name && !ctx.Has(position_name))) {
    return;
  }
  const auto &x = ctx.Get(input_name);
  const auto &cos = ctx.Get(cos_name);
  const auto &sin = ctx.Get(sin_name);
  const auto type = x.Dtype();
  if ((type != TensorType::kUndefined && type != TensorType::kFloat &&
       type != TensorType::kFloat16 && type != TensorType::kBfloat16) ||
      (cos.Dtype() != TensorType::kUndefined && cos.Dtype() != type) ||
      (sin.Dtype() != TensorType::kUndefined && sin.Dtype() != type)) {
    throw std::invalid_argument("RotaryEmbedding: input and caches must share a float type.");
  }
  const auto &shape = x.Shape();
  if (shape.Rank() != 3 && shape.Rank() != 4) {
    throw std::invalid_argument("RotaryEmbedding: input rank must be 3 or 4.");
  }
  const std::size_t seq_axis = shape.Rank() == 4 ? 2 : 1;
  std::int64_t heads = 0;
  std::int64_t rotary_dim = 0;
  for (const auto &attribute : node.attribute()) {
    if (attribute.name() == "num_heads") {
      heads = attribute.i();
    } else if (attribute.name() == "rotary_embedding_dim") {
      rotary_dim = attribute.i();
    }
  }
  if (rotary_dim < 0 || (rotary_dim != 0 && rotary_dim % 2 != 0) ||
      (shape.Rank() == 3 && heads <= 0 && (!microsoft || rotary_dim != 0))) {
    throw std::invalid_argument("RotaryEmbedding: invalid rotary dimension or head count.");
  }
  const auto &last = shape[shape.Rank() - 1];
  if (last.IsInt()) {
    const auto head = shape.Rank() == 4 ? last.AsInt() : heads > 0 ? last.AsInt() / heads : 0;
    if ((shape.Rank() == 3 && heads > 0 && last.AsInt() % heads != 0) ||
        (head != 0 && (head % 2 != 0 || (rotary_dim != 0 && rotary_dim > head)))) {
      throw std::invalid_argument("RotaryEmbedding: invalid head size.");
    }
  }
  const bool has_positions = microsoft || (node.input_size() == 4 && !position_name.empty());
  if (cos.Shape().Rank() != (has_positions ? 2 : 3) || sin.Shape().Rank() != cos.Shape().Rank()) {
    throw std::invalid_argument("RotaryEmbedding: invalid cache rank.");
  }
  auto equal = [&](const SymDim &a, const SymDim &b) {
    if (a.IsInt() && b.IsInt() && a.AsInt() != b.AsInt()) {
      throw std::invalid_argument("RotaryEmbedding: incompatible dimensions.");
    }
    if (a.IsExpr() || b.IsExpr()) {
      ctx.AddConstraint(a.ToString(), b.ToString());
    }
  };
  for (std::size_t i = 0; i < cos.Shape().Rank(); ++i) {
    equal(cos.Shape()[i], sin.Shape()[i]);
  }
  if (rotary_dim != 0) {
    const auto &cache_width = cos.Shape()[cos.Shape().Rank() - 1];
    const auto full_width = last.IsInt() && (shape.Rank() == 4 || heads > 0)
                                ? (shape.Rank() == 4 ? last.AsInt() : last.AsInt() / heads) / 2
                                : 0;
    if (microsoft && full_width > 0 && cache_width.IsExpr()) {
      const auto &width = cache_width.AsExpr();
      ctx.AddConstraint("(" + width + "-" + std::to_string(rotary_dim / 2) + ")*(" + width + "-" +
                            std::to_string(full_width) + ")",
                        "0");
    } else if (microsoft && full_width > 0 && cache_width.IsInt() &&
               cache_width.AsInt() == full_width) {
      // Microsoft's CPU kernel also accepts full-head-width caches for partial rotation.
    } else {
      equal(cache_width, SymDim(rotary_dim / 2));
    }
  } else if (shape.Rank() == 4 && last.IsInt()) {
    equal(cos.Shape()[cos.Shape().Rank() - 1], SymDim(last.AsInt() / 2));
  } else if (shape.Rank() == 3 && heads > 0 && last.IsInt()) {
    equal(cos.Shape()[cos.Shape().Rank() - 1], SymDim(last.AsInt() / heads / 2));
  }
  if (!has_positions) {
    equal(cos.Shape()[0], shape[0]);
    equal(cos.Shape()[1], shape[seq_axis]);
  } else if (ctx.Has(position_name)) {
    const auto &positions = ctx.Get(position_name);
    if (positions.Dtype() != TensorType::kUndefined && positions.Dtype() != TensorType::kInt64) {
      throw std::invalid_argument("RotaryEmbedding: position_ids must be INT64.");
    }
    if (positions.Shape().Rank() == 2) {
      equal(positions.Shape()[0], shape[0]);
      equal(positions.Shape()[1], shape[seq_axis]);
    } else if (!microsoft ||
               (positions.Shape().Rank() != 0 &&
                (positions.Shape().Rank() != 1 ||
                 (positions.Shape()[0].IsInt() && positions.Shape()[0].AsInt() != 1)))) {
      throw std::invalid_argument("RotaryEmbedding: invalid position_ids shape.");
    }
  }
  ctx.Set(node.output(0), SymTensor(nullptr, type, shape));
}

int64_t ComputePeakMemoryRotaryEmbedding(sym_ns::Device, const std::vector<SymShape> &) {
  return 0;
}

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
    shapes_ns::RegisterComputeShapeFn("ai.onnx", "RotaryEmbedding", ComputeShapeRotaryEmbedding);
    shapes_ns::RegisterComputePeakMemoryFn("ai.onnx", "RotaryEmbedding", sym_ns::Device::kCPU,
                                           ComputePeakMemoryRotaryEmbedding);
    shapes_ns::RegisterComputeShapeFn("ai.onnx", "SimplifiedLayerNormalization",
                                      ComputeShapeSimplifiedLayerNormalization);
    shapes_ns::RegisterComputePeakMemoryFn("ai.onnx", "SimplifiedLayerNormalization",
                                           sym_ns::Device::kCPU,
                                           ComputePeakMemorySimplifiedLayerNormalization);
  });
}

} // namespace onnx_light_cpu
