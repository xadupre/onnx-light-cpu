// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "ml_cpu_compute/com_microsoft/attention.h"
#include "ml_cpu_compute/checked_arithmetic.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"
#include "onnx_light_cpu/shapes/com_microsoft/shape_inference.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace onnx_light_cpu {
namespace {
namespace shapes_ns = ONNX_LIGHT_NAMESPACE::core::shapes;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;
namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
using sym_ns::SymDim;
using sym_ns::SymShape;
using sym_ns::SymTensor;
using NodeProto = ONNX_LIGHT_NAMESPACE::NodeProto;
bool Wired(const NodeProto &node, int slot, bool output = false) {
  return output ? node.output_size() > slot && !node.output(slot).empty()
                : node.input_size() > slot && !node.input(slot).empty();
}
void Fail(const char *message, const char *context = "ComputeShapeMicrosoftAttention") {
  throw std::invalid_argument(std::string(context) + ": " + message);
}
SymDim Equal(shapes_ns::ShapesContext &ctx, const SymDim &a, const SymDim &b) {
  if (a.IsInt() && b.IsInt()) {
    if (a.AsInt() != b.AsInt())
      Fail("incompatible dimensions.");
  } else if (a.ToString() != b.ToString())
    ctx.AddConstraint(a.ToString(), b.ToString());
  return a.IsInt() ? a : b.IsInt() ? b : a;
}
SymDim Divide(const SymDim &a, int64_t divisor,
              const char *context = "ComputeShapeMicrosoftAttention") {
  if (a.IsInt()) {
    if (a.AsInt() <= 0 || a.AsInt() % divisor)
      Fail("hidden width must be a positive multiple of heads (and three if QKV omitted).",
           context);
    return SymDim(a.AsInt() / divisor);
  }
  return divisor == 1 ? a : SymDim("(" + a.ToString() + ")/" + std::to_string(divisor));
}
SymDim Add(const SymDim &a, const SymDim &b) {
  if (a.IsInt() && b.IsInt())
    return SymDim(ml_cpu_compute::CheckedIndexAdd(a.AsInt(), b.AsInt(), "MicrosoftAttention",
                                                  "symbolic total length"));
  if (a.IsInt() && a.AsInt() == 0)
    return b;
  return SymDim("(" + a.ToString() + ")+(" + b.ToString() + ")");
}
std::pair<SymDim, SymDim> Widths(const NodeProto &node, const SymDim &width,
                                 shapes_ns::ShapesContext *ctx,
                                 const char *context = "ComputeShapeMicrosoftAttention") {
  if (const auto *sizes = rt_ns::FindAttribute(node, "qkv_hidden_sizes")) {
    if (sizes->ints().size() != 3 || sizes->ints(0) <= 0 || sizes->ints(1) != sizes->ints(0) ||
        sizes->ints(2) <= 0)
      Fail("qkv_hidden_sizes must be positive [Q,Q,V].", context);
    const auto sum = ml_cpu_compute::CheckedIndexAdd(
        ml_cpu_compute::CheckedIndexMultiply(2, sizes->ints(0), "MicrosoftAttention", "QK"),
        sizes->ints(2), "MicrosoftAttention", "QKV");
    if (ctx)
      Equal(*ctx, width, SymDim(sum));
    else if (!width.IsInt() || width.AsInt() != sum)
      Fail("weights width disagrees with qkv_hidden_sizes.", context);
    return {SymDim(sizes->ints(0)), SymDim(sizes->ints(2))};
  }
  const auto q = Divide(width, 3, context);
  return {q, q};
}
} // namespace

void ComputeShapeMicrosoftAttention(shapes_ns::ShapesContext &ctx, const NodeProto &node) {
  const auto attr = [&](const char *name, int64_t fallback = 0) {
    return rt_ns::GetAttributeIntOrDefault(node, name, fallback);
  };
  if (!Wired(node, 0) || !Wired(node, 1) || !Wired(node, 0, true) || node.input_size() > 7 ||
      node.output_size() > 2 || Wired(node, 6) || attr("past_present_share_buffer") != 0 ||
      attr("do_rotary") != 0 || attr("rotary_embedding_dim") != 0)
    Fail("input/weights/output required; shared-buffer and rotary modes unsupported.");
  for (int slot = 0; slot < node.input_size(); ++slot) {
    if (!Wired(node, slot))
      continue;
    if (!ctx.Has(node.input(slot)))
      Fail("input shape must be known.");
    const auto &tensor = ctx.Get(node.input(slot));
    if (tensor.Dtype() != (slot == 3 ? sym_ns::TensorType::kInt32 : sym_ns::TensorType::kFloat))
      Fail("FLOAT inputs and int32 mask required.");
    for (int64_t axis = 0; axis < tensor.Shape().Rank(); ++axis) {
      const auto &dim = tensor.Shape()[axis];
      if (dim.IsInt() && dim.AsInt() < 0)
        Fail("dimensions must be nonnegative.");
    }
  }
  const auto &x = ctx.Get(node.input(0)).Shape(), &w = ctx.Get(node.input(1)).Shape();
  if (x.Rank() != 3 || w.Rank() != 2)
    Fail("input must be rank 3 and weights rank 2.");
  if ((x[2].IsInt() && x[2].AsInt() <= 0) || (w[0].IsInt() && w[0].AsInt() <= 0))
    Fail("input hidden width must be positive.");
  Equal(ctx, x[2], w[0]);
  const auto heads = attr("num_heads"), causal = attr("unidirectional");
  if (heads <= 0 || (causal != 0 && causal != 1))
    Fail("invalid num_heads or unidirectional.");
  auto [q, v] = Widths(node, w[1], &ctx);
  auto h = Divide(q, heads);
  Divide(v, heads);
  for (const char *name : {"scale", "mask_filter_value"}) {
    if (const auto *a = rt_ns::FindAttribute(node, name); a && !std::isfinite(a->f()))
      Fail("scale and mask_filter_value must be finite.");
  }
  const auto batch = x[0], sequence = x[1];
  if (Wired(node, 2)) {
    const auto &bias = ctx.Get(node.input(2)).Shape();
    if (bias.Rank() != 1)
      Fail("bias must be rank 1.");
    Equal(ctx, bias[0], w[1]);
  }
  const bool cache = Wired(node, 4), present = Wired(node, 1, true);
  if (cache && !present)
    Fail("past requires present.");
  if (cache || present)
    Equal(ctx, q, v);
  SymDim p(int64_t{0});
  if (cache) {
    const auto &past = ctx.Get(node.input(4)).Shape();
    if (past.Rank() != 5)
      Fail("past must have rank 5.");
    Equal(ctx, past[0], SymDim(int64_t{2}));
    Equal(ctx, past[1], batch);
    Equal(ctx, past[2], SymDim(heads));
    h = Equal(ctx, h, past[4]);
    p = past[3];
  }
  const auto total = Add(p, sequence);
  if (Wired(node, 3)) {
    const auto &mask = ctx.Get(node.input(3)).Shape();
    const auto rank = mask.Rank();
    if (rank == 4 && causal)
      Fail("4D masks require unidirectional=0.");
    if (rank < 1 || rank > 4)
      Fail("mask must have rank 1 through 4.");
    if (rank == 1) {
      if (mask[0].IsInt() && batch.IsInt() && mask[0].AsInt() != batch.AsInt() &&
          mask[0].AsInt() != ml_cpu_compute::CheckedIndexMultiply(
                                 2, batch.AsInt(), "MicrosoftAttention", "mask length"))
        Fail("1D mask must have B or 2B elements.");
    } else {
      Equal(ctx, mask[0], batch);
      if (rank == 4) {
        Equal(ctx, mask[1], SymDim(int64_t{1}));
        Equal(ctx, mask[2], mask[3]);
        if (mask[2].IsInt() && total.IsInt() && mask[2].AsInt() < total.AsInt())
          Fail("4D mask capacity must be at least P+S.");
      } else {
        if (rank == 3)
          Equal(ctx, mask[1], sequence);
        Equal(ctx, mask[rank - 1], total);
      }
    }
  }
  if (Wired(node, 5)) {
    const auto &ab = ctx.Get(node.input(5)).Shape();
    if (ab.Rank() != 4)
      Fail("attention_bias must have rank 4.");
    if (ab[0].IsInt() && ab[0].AsInt() != 1)
      Equal(ctx, ab[0], batch);
    if (ab[1].IsInt() && ab[1].AsInt() != 1)
      Equal(ctx, ab[1], SymDim(heads));
    Equal(ctx, ab[2], sequence);
    Equal(ctx, ab[3], total);
  }
  SymShape output;
  for (const auto &dim : {batch, sequence, v})
    output.PushBack(dim);
  ctx.Set(node.output(0), SymTensor(nullptr, sym_ns::TensorType::kFloat, std::move(output)));
  if (present) {
    SymShape state;
    for (const auto &dim : {SymDim(int64_t{2}), batch, SymDim(heads), total, h})
      state.PushBack(dim);
    ctx.Set(node.output(1), SymTensor(nullptr, sym_ns::TensorType::kFloat, std::move(state)));
  }
}

shapes_ns::PeakMemoryEstimate
ComputePeakMemoryMicrosoftAttention(sym_ns::Device, const NodeProto &node,
                                    const std::vector<SymShape> &shapes) {
  constexpr const char *context = "ComputePeakMemoryMicrosoftAttention";
  if (shapes.size() < 2 || shapes[0].Rank() != 3 || shapes[1].Rank() != 2)
    return shapes_ns::UnknownPeakMemory(kMicrosoftDomain, "Attention");
  if (Wired(node, 4) && (shapes.size() <= 4 || shapes[4].Rank() != 5))
    return shapes_ns::UnknownPeakMemory(kMicrosoftDomain, "Attention");
  for (int slot : {0, 1, 4}) {
    if (slot >= static_cast<int>(shapes.size()) || (slot == 4 && !Wired(node, 4)))
      continue;
    for (int64_t axis = 0; axis < shapes[slot].Rank(); ++axis) {
      const auto &dim = shapes[slot][axis];
      if (!dim.IsInt())
        return shapes_ns::UnknownPeakMemory(kMicrosoftDomain, "Attention");
      if (dim.AsInt() < 0)
        Fail("memory dimensions must be nonnegative.", context);
    }
  }
  const auto b = shapes[0][0].AsInt(), s = shapes[0][1].AsInt();
  const auto [q, v] = Widths(node, shapes[1][1], nullptr, context);
  const auto heads = rt_ns::GetAttributeIntOrDefault(node, "num_heads", 0);
  if (heads <= 0)
    Fail("invalid head count.", context);
  const auto h = Divide(q, heads, context).AsInt();
  if (shapes[0][2].AsInt() <= 0)
    Fail("input hidden width must be positive.", context);
  const auto p = Wired(node, 4) ? shapes.at(4)[3].AsInt() : 0;
  const auto t = ml_cpu_compute::CheckedIndexAdd(p, s, "MicrosoftAttention", "total length");
  const bool additive = Wired(node, 3) || Wired(node, 5);
  ml_cpu_compute::AttentionDescriptor descriptor;
  descriptor.q_num_heads = descriptor.kv_num_heads = heads;
  descriptor.is_causal =
      !additive && rt_ns::GetAttributeIntOrDefault(node, "unidirectional", 0) != 0;
  descriptor.has_past_key = descriptor.has_past_value = Wired(node, 4);
  descriptor.has_attn_mask = additive;
  const std::vector<int64_t> qs{b, s, q.AsInt()}, vs{b, s, v.AsInt()}, ms{b, heads, s, t};
  const std::vector<int64_t> ps =
      Wired(node, 4) ? std::vector<int64_t>{b, heads, p, h} : std::vector<int64_t>{};
  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank3, qs, qs,
                                     vs, additive ? ms : std::vector<int64_t>{},
                                     additive ? ml_cpu_compute::AttentionMaskKind::kAdditive
                                              : ml_cpu_compute::AttentionMaskKind::kNone,
                                     ps, ps);
  const auto bytes = ml_cpu_compute::CheckedAdd(
      ml_cpu_compute::MicrosoftAttentionPeakMemoryBytes(plan, shapes[0][2].AsInt()),
      additive
          ? ml_cpu_compute::CheckedByteSize(
                ml_cpu_compute::CheckedDimension(ml_cpu_compute::CheckedShapeIndexProduct(
                                                     ms, "MicrosoftAttention", "mask elements"),
                                                 "MicrosoftAttention", "mask elements"),
                sizeof(float), "MicrosoftAttention", "mask scratch")
          : 0,
      "MicrosoftAttention", "scratch bytes");
  return static_cast<int64_t>(
      ml_cpu_compute::CheckedStride(bytes, "MicrosoftAttention", "scratch estimate"));
}
} // namespace onnx_light_cpu
