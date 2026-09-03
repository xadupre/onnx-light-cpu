// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/shapes/com_microsoft/shape_inference.h"

#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_core/symbolic/sym_tensor.h"

#include <cstdint>
#include <limits>
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
  if (left.IsExpr() && right.IsExpr()) {
    if (left.AsExpr() != right.AsExpr()) {
      ctx.AddConstraint(left.AsExpr(), right.AsExpr());
    }
  } else if (left.IsExpr()) {
    ctx.AddConstraint(left.AsExpr(), std::to_string(right.AsInt()));
  } else {
    ctx.AddConstraint(std::to_string(left.AsInt()), right.AsExpr());
  }
}

// Merges two dimensions required to be equal, preserving a concrete value
// when available and recording equality between distinct symbolic values.
sym_ns::SymDim MergeGqaDim(ShapesContext &ctx, const sym_ns::SymDim &left,
                           const sym_ns::SymDim &right, const char *what) {
  ConstrainEqual(ctx, left, right,
                 (std::string("ComputeShapeGroupQueryAttention: ") + what + " mismatch.").c_str());
  if (left.IsInt()) {
    return left;
  }
  return right.IsInt() ? right : left;
}

// Returns `hidden / heads` as a static dimension when `hidden` is static
// (throwing when it is not an exact multiple), otherwise a fresh symbolic
// placeholder.
sym_ns::SymDim DivideGqaDim(const sym_ns::SymDim &hidden, std::int64_t heads, const char *what) {
  if (!hidden.IsInt()) {
    return sym_ns::SymDim(std::string("?"));
  }
  if (heads <= 0 || hidden.AsInt() % heads != 0) {
    throw std::invalid_argument(std::string("ComputeShapeGroupQueryAttention: ") + what +
                                " must be a positive multiple of the head count.");
  }
  return sym_ns::SymDim(hidden.AsInt() / heads);
}

// Returns `heads * head` when `head` is static, otherwise a fresh symbolic
// placeholder.
sym_ns::SymDim MultiplyGqaDim(std::int64_t heads, const sym_ns::SymDim &head) {
  if (head.IsInt()) {
    return sym_ns::SymDim(heads * head.AsInt());
  }
  return sym_ns::SymDim(std::string("?"));
}

// Returns the symbolic sum used by present-cache outputs.
sym_ns::SymDim AddGqaDims(const sym_ns::SymDim &left, const sym_ns::SymDim &right) {
  if (left.IsInt() && right.IsInt()) {
    return sym_ns::SymDim(left.AsInt() + right.AsInt());
  }
  if (left.IsInt() && left.AsInt() == 0) {
    return right;
  }
  if (right.IsInt() && right.AsInt() == 0) {
    return left;
  }
  return sym_ns::SymDim("(" + left.ToString() + ")+(" + right.ToString() + ")");
}

std::int64_t GetGqaIntAttribute(const ONNX_LIGHT_NAMESPACE::NodeProto &node, const char *name,
                                std::int64_t fallback) {
  for (const auto &attribute : node.attribute()) {
    if (attribute.name() == name) {
      return attribute.i();
    }
  }
  return fallback;
}

std::string GetStringAttribute(const ONNX_LIGHT_NAMESPACE::NodeProto &node, const char *name,
                               const char *fallback) {
  for (const auto &attribute : node.attribute()) {
    if (attribute.name() == name) {
      return attribute.s();
    }
  }
  return fallback;
}

void RequireRank4(const SymShape &shape, const char *name) {
  if (shape.Rank() != 4) {
    throw std::invalid_argument(std::string("ComputeShapeGroupQueryAttention: '") + name +
                                "' must have rank 4.");
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
      {kMicrosoftDomain,
       "LinearAttention",
       "onnx_light_cpu::ComputeShapeLinearAttention",
       "onnx_light_cpu::ComputePeakMemoryLinearAttention",
       {"onnx_light_cpu::LinearAttentionFusionPattern"},
       false},
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
  if (node.input_size() < 7 || node.output_size() < 1 || node.output_size() > 3 ||
      !ctx.Has(node.input(0)) || !ctx.Has(node.input(1)) || !ctx.Has(node.input(2))) {
    throw std::invalid_argument("ComputeShapeGroupQueryAttention: expected known Q/K/V inputs and "
                                "one to three outputs.");
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
  const std::int64_t num_heads = GetGqaIntAttribute(node, "num_heads", 0);
  const std::int64_t kv_num_heads = GetGqaIntAttribute(node, "kv_num_heads", 0);
  if (num_heads <= 0 || kv_num_heads <= 0 || num_heads % kv_num_heads != 0) {
    throw std::invalid_argument("ComputeShapeGroupQueryAttention: num_heads and kv_num_heads must "
                                "be positive, and num_heads must be a multiple of kv_num_heads.");
  }

  const SymShape &q_shape = query.Shape();
  const SymShape &k_shape = key.Shape();
  const SymShape &v_shape = value.Shape();

  sym_ns::SymDim batch = MergeGqaDim(ctx, q_shape[0], k_shape[0], "batch");
  batch = MergeGqaDim(ctx, batch, v_shape[0], "batch");
  const sym_ns::SymDim &q_seq_len = q_shape[1];
  const sym_ns::SymDim kv_seq_len = MergeGqaDim(ctx, k_shape[1], v_shape[1], "kv_sequence_length");
  MergeGqaDim(ctx, q_seq_len, kv_seq_len, "sequence_length");

  sym_ns::SymDim head_dim = DivideGqaDim(q_shape[2], num_heads, "query hidden size");
  head_dim = MergeGqaDim(ctx, head_dim, DivideGqaDim(k_shape[2], kv_num_heads, "key hidden size"),
                         "head_size");
  sym_ns::SymDim v_head_dim = DivideGqaDim(v_shape[2], kv_num_heads, "value hidden size");

  const bool has_past_key = node.input_size() > 3 && !node.input(3).empty();
  const bool has_past_value = node.input_size() > 4 && !node.input(4).empty();
  if (has_past_key != has_past_value) {
    throw std::invalid_argument(
        "ComputeShapeGroupQueryAttention: past_key and past_value must be used together.");
  }
  sym_ns::SymDim past_length(int64_t{0});
  if (has_past_key) {
    if (!ctx.Has(node.input(3)) || !ctx.Has(node.input(4))) {
      throw std::invalid_argument(
          "ComputeShapeGroupQueryAttention: past_key and past_value shapes must be known.");
    }
    const SymShape &past_k_shape = ctx.Get(node.input(3)).Shape();
    const SymShape &past_v_shape = ctx.Get(node.input(4)).Shape();
    RequireRank4(past_k_shape, "past_key");
    RequireRank4(past_v_shape, "past_value");
    batch = MergeGqaDim(ctx, batch, past_k_shape[0], "batch");
    batch = MergeGqaDim(ctx, batch, past_v_shape[0], "batch");
    MergeGqaDim(ctx, past_k_shape[1], sym_ns::SymDim(kv_num_heads), "past_key head count");
    MergeGqaDim(ctx, past_v_shape[1], sym_ns::SymDim(kv_num_heads), "past_value head count");
    head_dim = MergeGqaDim(ctx, head_dim, past_k_shape[3], "head_size");
    v_head_dim = MergeGqaDim(ctx, v_head_dim, past_v_shape[3], "value head_size");
    past_length = MergeGqaDim(ctx, past_k_shape[2], past_v_shape[2], "past_sequence_length");
  }
  const sym_ns::SymDim total_seq_len = AddGqaDims(kv_seq_len, past_length);

  // Output 0: Y = (batch, q_sequence_length, num_heads * v_head_dim).
  {
    SymShape out_shape;
    out_shape.PushBack(batch);
    out_shape.PushBack(q_seq_len);
    out_shape.PushBack(MultiplyGqaDim(num_heads, v_head_dim));
    ctx.Set(node.output(0), SymTensor(nullptr, query.Dtype(), std::move(out_shape)));
  }

  const bool has_present_key = node.output_size() > 1 && !node.output(1).empty();
  const bool has_present_value = node.output_size() > 2 && !node.output(2).empty();
  if (has_present_key != has_present_value) {
    throw std::invalid_argument("ComputeShapeGroupQueryAttention: present_key and present_value "
                                "must be requested together.");
  }
  // Output 1: present_key = (batch, kv_num_heads, total_seq_len, head_size).
  if (has_present_key) {
    SymShape pk_shape;
    pk_shape.PushBack(batch);
    pk_shape.PushBack(sym_ns::SymDim(kv_num_heads));
    pk_shape.PushBack(total_seq_len);
    pk_shape.PushBack(head_dim);
    ctx.Set(node.output(1), SymTensor(nullptr, query.Dtype(), std::move(pk_shape)));
  }
  // Output 2: present_value = (batch, kv_num_heads, total_seq_len, v_head_size).
  if (has_present_value) {
    SymShape pv_shape;
    pv_shape.PushBack(batch);
    pv_shape.PushBack(sym_ns::SymDim(kv_num_heads));
    pv_shape.PushBack(total_seq_len);
    pv_shape.PushBack(v_head_dim);
    ctx.Set(node.output(2), SymTensor(nullptr, value.Dtype(), std::move(pv_shape)));
  }
}

void ComputeShapeLinearAttention(ShapesContext &ctx, const ONNX_LIGHT_NAMESPACE::NodeProto &node) {
  if (node.input_size() < 3 || node.input_size() > 6 || node.output_size() != 2 ||
      node.output(0).empty() || node.output(1).empty() || !ctx.Has(node.input(0)) ||
      !ctx.Has(node.input(1)) || !ctx.Has(node.input(2))) {
    throw std::invalid_argument(
        "ComputeShapeLinearAttention: expected known Q/K/V inputs and two non-empty outputs.");
  }
  const SymTensor &query = ctx.Get(node.input(0));
  const SymTensor &key = ctx.Get(node.input(1));
  const SymTensor &value = ctx.Get(node.input(2));
  if (query.Dtype() != key.Dtype() || query.Dtype() != value.Dtype()) {
    throw std::invalid_argument(
        "ComputeShapeLinearAttention: query, key, and value types must match.");
  }
  if (query.Dtype() != sym_ns::TensorType::kFloat) {
    throw std::invalid_argument("ComputeShapeLinearAttention: only FLOAT tensors are supported.");
  }
  if (query.Shape().Rank() != 3 || key.Shape().Rank() != 3 || value.Shape().Rank() != 3) {
    throw std::invalid_argument(
        "ComputeShapeLinearAttention: query, key, and value must have rank 3.");
  }
  const std::int64_t q_heads = GetGqaIntAttribute(node, "q_num_heads", 0);
  const std::int64_t kv_heads = GetGqaIntAttribute(node, "kv_num_heads", 0);
  if (q_heads <= 0 || kv_heads <= 0 || q_heads > std::numeric_limits<std::int32_t>::max() ||
      kv_heads > std::numeric_limits<std::int32_t>::max() ||
      (q_heads % kv_heads != 0 && kv_heads % q_heads != 0)) {
    throw std::invalid_argument(
        "ComputeShapeLinearAttention: the larger head count must be divisible by the smaller.");
  }
  if (GetGqaIntAttribute(node, "state_window", 0) != 0) {
    throw std::invalid_argument(
        "ComputeShapeLinearAttention: nonzero state_window is not supported.");
  }
  const std::string rule = GetStringAttribute(node, "update_rule", "gated_delta");
  const bool uses_decay = rule == "gated" || rule == "gated_delta";
  const bool uses_beta = rule == "delta" || rule == "gated_delta";
  if (rule != "linear" && rule != "gated" && rule != "delta" && rule != "gated_delta") {
    throw std::invalid_argument("ComputeShapeLinearAttention: invalid update_rule.");
  }
  const bool has_past = node.input_size() > 3 && !node.input(3).empty();
  const bool has_decay = node.input_size() > 4 && !node.input(4).empty();
  const bool has_beta = node.input_size() > 5 && !node.input(5).empty();
  if ((has_past && !ctx.Has(node.input(3))) || (has_decay && !ctx.Has(node.input(4))) ||
      (has_beta && !ctx.Has(node.input(5)))) {
    throw std::invalid_argument(
        "ComputeShapeLinearAttention: wired optional input shapes must be known.");
  }
  if ((uses_decay && !has_decay) || (uses_beta && !has_beta)) {
    throw std::invalid_argument(
        "ComputeShapeLinearAttention: update_rule requires its decay and beta inputs.");
  }

  const SymShape &q_shape = query.Shape();
  const SymShape &k_shape = key.Shape();
  const SymShape &v_shape = value.Shape();
  sym_ns::SymDim batch = MergeGqaDim(ctx, q_shape[0], k_shape[0], "batch");
  batch = MergeGqaDim(ctx, batch, v_shape[0], "batch");
  sym_ns::SymDim sequence = MergeGqaDim(ctx, q_shape[1], k_shape[1], "sequence_length");
  sequence = MergeGqaDim(ctx, sequence, v_shape[1], "sequence_length");
  const sym_ns::SymDim key_head_size = DivideGqaDim(q_shape[2], q_heads, "query hidden size");
  const sym_ns::SymDim value_head_size = DivideGqaDim(v_shape[2], kv_heads, "value hidden size");
  if (key_head_size.IsInt() && k_shape[2].IsInt()) {
    const std::int64_t dk = key_head_size.AsInt();
    const std::int64_t key_hidden = k_shape[2].AsInt();
    if (dk <= 0 || key_hidden <= 0 || key_hidden % dk != 0 || kv_heads % (key_hidden / dk) != 0) {
      throw std::invalid_argument(
          "ComputeShapeLinearAttention: key heads must divide kv_num_heads.");
    }
  }

  if (has_past) {
    const SymTensor &past = ctx.Get(node.input(3));
    if (past.Dtype() != sym_ns::TensorType::kFloat || past.Shape().Rank() != 4) {
      throw std::invalid_argument(
          "ComputeShapeLinearAttention: past_state must be a rank-4 FLOAT tensor.");
    }
    batch = MergeGqaDim(ctx, batch, past.Shape()[0], "past_state batch");
    MergeGqaDim(ctx, past.Shape()[1], sym_ns::SymDim(kv_heads), "past_state heads");
    MergeGqaDim(ctx, past.Shape()[2], key_head_size, "past_state key head size");
    MergeGqaDim(ctx, past.Shape()[3], value_head_size, "past_state value head size");
  }
  if (has_decay) {
    const SymTensor &decay = ctx.Get(node.input(4));
    if (decay.Dtype() != sym_ns::TensorType::kFloat || decay.Shape().Rank() != 3) {
      throw std::invalid_argument(
          "ComputeShapeLinearAttention: decay must be a rank-3 FLOAT tensor.");
    }
    MergeGqaDim(ctx, batch, decay.Shape()[0], "decay batch");
    MergeGqaDim(ctx, sequence, decay.Shape()[1], "decay sequence");
    if (decay.Shape()[2].IsInt()) {
      const std::int64_t hidden = decay.Shape()[2].AsInt();
      const bool per_head = hidden == kv_heads;
      const bool per_dimension =
          key_head_size.IsInt() && hidden == kv_heads * key_head_size.AsInt();
      if (!per_head && !per_dimension) {
        throw std::invalid_argument(
            "ComputeShapeLinearAttention: decay width must be H_kv or H_kv*d_k.");
      }
    }
  }
  if (has_beta) {
    const SymTensor &beta = ctx.Get(node.input(5));
    if (beta.Dtype() != sym_ns::TensorType::kFloat || beta.Shape().Rank() != 3) {
      throw std::invalid_argument(
          "ComputeShapeLinearAttention: beta must be a rank-3 FLOAT tensor.");
    }
    MergeGqaDim(ctx, batch, beta.Shape()[0], "beta batch");
    MergeGqaDim(ctx, sequence, beta.Shape()[1], "beta sequence");
    if (beta.Shape()[2].IsInt() && beta.Shape()[2].AsInt() != 1 &&
        beta.Shape()[2].AsInt() != kv_heads) {
      throw std::invalid_argument("ComputeShapeLinearAttention: beta width must be 1 or H_kv.");
    }
  }

  SymShape output_shape;
  output_shape.PushBack(batch);
  output_shape.PushBack(sequence);
  output_shape.PushBack(q_heads <= kv_heads ? v_shape[2]
                                            : MultiplyGqaDim(q_heads, value_head_size));
  ctx.Set(node.output(0), SymTensor(nullptr, query.Dtype(), std::move(output_shape)));

  SymShape state_shape;
  state_shape.PushBack(batch);
  state_shape.PushBack(sym_ns::SymDim(kv_heads));
  state_shape.PushBack(key_head_size);
  state_shape.PushBack(value_head_size);
  ctx.Set(node.output(1), SymTensor(nullptr, query.Dtype(), std::move(state_shape)));
}

int64_t ComputePeakMemoryCDist(sym_ns::Device, const std::vector<SymShape> &) { return 0; }

int64_t ComputePeakMemoryBiasGelu(sym_ns::Device, const std::vector<SymShape> &) { return 0; }

int64_t ComputePeakMemoryGroupQueryAttention(sym_ns::Device, const std::vector<SymShape> &) {
  // The supported GroupQueryAttention path delegates to Attention's online-softmax
  // implementation, which does not materialize a full attention-score tensor.
  return 0;
}

int64_t ComputePeakMemoryLinearAttention(sym_ns::Device, const std::vector<SymShape> &) {
  return 0;
}

void RegisterMicrosoftShapeAndMemoryFunctions() {
  static std::once_flag once;
  std::call_once(once, [] {
    shapes_ns::RegisterComputeShapeFn(kMicrosoftDomain, "CDist", ComputeShapeCDist);
    shapes_ns::RegisterComputeShapeFn(kMicrosoftDomain, "BiasGelu", ComputeShapeBiasGelu);
    shapes_ns::RegisterComputeShapeFn(kMicrosoftDomain, "GroupQueryAttention",
                                      ComputeShapeGroupQueryAttention);
    shapes_ns::RegisterComputeShapeFn(kMicrosoftDomain, "LinearAttention",
                                      ComputeShapeLinearAttention);
    shapes_ns::RegisterComputePeakMemoryFn(kMicrosoftDomain, "CDist", sym_ns::Device::kCPU,
                                           ComputePeakMemoryCDist);
    shapes_ns::RegisterComputePeakMemoryFn(kMicrosoftDomain, "BiasGelu", sym_ns::Device::kCPU,
                                           ComputePeakMemoryBiasGelu);
    shapes_ns::RegisterComputePeakMemoryFn(kMicrosoftDomain, "GroupQueryAttention",
                                           sym_ns::Device::kCPU,
                                           ComputePeakMemoryGroupQueryAttention);
    shapes_ns::RegisterComputePeakMemoryFn(kMicrosoftDomain, "LinearAttention",
                                           sym_ns::Device::kCPU, ComputePeakMemoryLinearAttention);
  });
}

} // namespace onnx_light_cpu
