// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/shapes/com_microsoft/shape_inference.h"

#include "cpu_compute/checked_arithmetic.h"
#include "cpu_compute/com_microsoft/matmul_nbits.h"
#include "cpu_compute/com_microsoft/moe.h"
#include "onnx_light_cpu/kernels/com_microsoft/matmul_nbits_int3.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"
#include "onnx_light_cpu/shapes/ai_onnx/shape_inference.h"

#include "onnx_core/runtime/quantization.h"
#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_core/symbolic/sym_tensor.h"

#if defined(ONNX_LIGHT_CPU_HAS_ONNX_SHAPE)
#include "onnx_extensions/shapes/shapes/nn/shape_nn.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
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

struct EncodedMatMulNBitsShape {
  std::int64_t k;
  std::int64_t n;
  std::int64_t bits;
  std::int64_t block_size;
  std::int32_t data_type;
  bool has_zero_points;
};

EncodedMatMulNBitsShape InspectEncodedMatMulNBitsShape(const ShapesContext &ctx,
                                                       const std::string &name) {
  const auto &encoded = ctx.GetEncodedValue(name);
  const auto layout = ctx.GetEncodedLayout(name);
  if (encoded.has_parameter_ref() || layout.root == nullptr || layout.record_count != 1 ||
      !encoded.has_logical_type() || !encoded.logical_type().has_tensor_type()) {
    throw std::invalid_argument(
        "ComputeShapeMatMulNBits: expected one self-contained encoded weight record.");
  }
  const auto &logical = encoded.logical_type().tensor_type();
  if (!logical.has_shape() || logical.shape().dim().size() != 2 ||
      !logical.shape().dim(0).has_dim_value() || !logical.shape().dim(1).has_dim_value() ||
      logical.shape().dim(0).dim_value() <= 0 || logical.shape().dim(1).dim_value() <= 0) {
    throw std::invalid_argument(
        "ComputeShapeMatMulNBits: encoded weights require a concrete positive [K, N] shape.");
  }

  const auto &root = *layout.root;
  const std::string format = root.name().value();
  if (format == kMatMulNBitsInt3Format) {
    const auto int3 = InspectMatMulNBitsInt3(encoded, root);
    return {int3.k, int3.n, 3, 32, int3.data_type, int3.has_zero_points};
  }
  const std::string prefix = "onnx_light.quantization.v1/ort_matmulnbits_int";
  const std::int64_t format_bits = format == prefix + "2"   ? 2
                                   : format == prefix + "4" ? 4
                                   : format == prefix + "8" ? 8
                                                            : 0;
  if (format_bits == 0 || !root.has_structure() ||
      (root.structure().field().size() != 3 && root.structure().field().size() != 4)) {
    throw std::invalid_argument("ComputeShapeMatMulNBits: unsupported encoded weight layout.");
  }

  const auto &parameters = root.structure().field(0);
  if (parameters.name() != "parameters" || !parameters.has_constant() ||
      parameters.constant().data_type() != ONNX_LIGHT_NAMESPACE::TensorProto::INT64 ||
      parameters.constant().dims().size() != 1 || parameters.constant().dims(0) != 2 ||
      parameters.constant().int64_data().size() != 2 ||
      parameters.constant().int64_data(0) != format_bits ||
      parameters.constant().int64_data(1) <= 0) {
    throw std::invalid_argument("ComputeShapeMatMulNBits: invalid encoded weight parameters.");
  }

  const std::int64_t k = logical.shape().dim(0).dim_value();
  const std::int64_t n = logical.shape().dim(1).dim_value();
  const std::int64_t block_size = parameters.constant().int64_data(1);
  if (!cpu_compute::MatMulNBitsBlockSizeSupported(format_bits, block_size)) {
    throw std::invalid_argument(
        "ComputeShapeMatMulNBits: encoded block size exceeds supported bounds.");
  }
  const std::int64_t groups = k / block_size + (k % block_size != 0 ? 1 : 0);
  const std::int64_t blob_size = block_size * format_bits / 8;
  const auto has_tensor_shape = [](const auto &field, const char *field_name,
                                   std::int32_t data_type, std::int64_t dim0, std::int64_t dim1,
                                   std::int64_t dim2 = -1) {
    if (field.name() != field_name || !field.has_type() || !field.type().has_tensor_type()) {
      return false;
    }
    const auto &tensor = field.type().tensor_type();
    const std::size_t rank = dim2 < 0 ? 2 : 3;
    return tensor.elem_type() == data_type && tensor.has_shape() &&
           tensor.shape().dim().size() == rank && tensor.shape().dim(0).has_dim_value() &&
           tensor.shape().dim(0).dim_value() == dim0 && tensor.shape().dim(1).has_dim_value() &&
           tensor.shape().dim(1).dim_value() == dim1 &&
           (dim2 < 0 ||
            (tensor.shape().dim(2).has_dim_value() && tensor.shape().dim(2).dim_value() == dim2));
  };
  const bool has_zeros = root.structure().field().size() == 4;
  bool valid_zeros = true;
  if (has_zeros) {
    const auto &zero = root.structure().field(3);
    const auto per_byte = 8 / format_bits;
    valid_zeros = has_tensor_shape(zero, "zero_points", ONNX_LIGHT_NAMESPACE::TensorProto::UINT8, n,
                                   groups / per_byte + (groups % per_byte != 0)) ||
                  has_tensor_shape(zero, "zero_points", logical.elem_type(), n, groups);
  }
  if (!valid_zeros ||
      !has_tensor_shape(root.structure().field(1), "B", ONNX_LIGHT_NAMESPACE::TensorProto::UINT8, n,
                        groups, blob_size) ||
      !has_tensor_shape(root.structure().field(2), "scales", logical.elem_type(), n, groups)) {
    throw std::invalid_argument(
        "ComputeShapeMatMulNBits: encoded weight fields do not match their logical tensor.");
  }
  return {k, n, format_bits, block_size, logical.elem_type(), root.structure().field().size() == 4};
}

void ValidateStructuredMatMulNBitsType(const ShapesContext &ctx, const std::string &name,
                                       std::int64_t k, std::int64_t n, std::int64_t bits,
                                       sym_ns::TensorType data_type) {
  const auto &declared = ctx.GetType(name);
  if (!declared.has_struct_type()) {
    return;
  }
  const auto &root = ctx.ResolveStructType(declared.struct_type());
  const bool is_int3 = root.name().value() == kMatMulNBitsInt3Format;
  if (!is_int3) {
    if (bits == 3) {
      throw std::invalid_argument("ComputeShapeMatMulNBits: INT3 requires an INT3 structured B.");
    }
    return;
  }
  if (bits != 3 || !root.has_structure() ||
      (root.structure().field().size() != 3 && root.structure().field().size() != 4)) {
    throw std::invalid_argument("ComputeShapeMatMulNBits: invalid INT3 structured B.");
  }

  const auto &fields = root.structure().field();
  const auto &parameters = fields[0];
  if (parameters.name() != "parameters" || !parameters.has_constant() ||
      parameters.constant().data_type() != ONNX_LIGHT_NAMESPACE::TensorProto::INT64 ||
      parameters.constant().dims().size() != 1 || parameters.constant().dims(0) != 2 ||
      parameters.constant().int64_data().size() != 2 || parameters.constant().int64_data(0) != 3 ||
      parameters.constant().int64_data(1) != 32) {
    throw std::invalid_argument("ComputeShapeMatMulNBits: invalid INT3 parameters.");
  }

  const std::int64_t blocks = k / 32 + (k % 32 != 0 ? 1 : 0);
  const auto matches = [n, blocks](const auto &field, const char *field_name,
                                   std::int32_t elem_type, std::int64_t third = -1) {
    if (field.name() != field_name || !field.has_type() || !field.type().has_tensor_type()) {
      return false;
    }
    const auto &tensor = field.type().tensor_type();
    const auto &dimensions = tensor.shape().dim();
    const std::size_t rank = third < 0 ? 2 : 3;
    return tensor.elem_type() == elem_type && tensor.has_shape() && dimensions.size() == rank &&
           dimensions[0].has_dim_value() && dimensions[0].dim_value() == n &&
           dimensions[1].has_dim_value() && dimensions[1].dim_value() == blocks &&
           (third < 0 || (dimensions[2].has_dim_value() && dimensions[2].dim_value() == third));
  };
  const std::int32_t scale_type =
      data_type == sym_ns::TensorType::kFloat     ? ONNX_LIGHT_NAMESPACE::TensorProto::FLOAT
      : data_type == sym_ns::TensorType::kFloat16 ? ONNX_LIGHT_NAMESPACE::TensorProto::FLOAT16
                                                  : ONNX_LIGHT_NAMESPACE::TensorProto::BFLOAT16;
  if (!matches(fields[1], "B", ONNX_LIGHT_NAMESPACE::TensorProto::UINT8, 12) ||
      !matches(fields[2], "scales", scale_type) ||
      (fields.size() == 4 &&
       !matches(fields[3], "zero_points", ONNX_LIGHT_NAMESPACE::TensorProto::UINT8))) {
    throw std::invalid_argument("ComputeShapeMatMulNBits: invalid INT3 structured fields.");
  }
}

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

std::int64_t GetRequiredIntAttribute(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                                     const char *name) {
  for (const auto &attribute : node.attribute()) {
    if (attribute.name() == name &&
        attribute.type() == ONNX_LIGHT_NAMESPACE::AttributeProto::AttributeType::INT) {
      return attribute.i();
    }
  }
  throw std::invalid_argument(std::string("ComputeShapeMatMulNBits: missing required attribute '") +
                              name + "'.");
}

std::int64_t GetIntAttribute(const ONNX_LIGHT_NAMESPACE::NodeProto &node, const char *name,
                             std::int64_t fallback) {
  for (const auto &attribute : node.attribute()) {
    if (attribute.name() == name) {
      return attribute.i();
    }
  }
  return fallback;
}

bool HasInput(const ONNX_LIGHT_NAMESPACE::NodeProto &node, std::size_t index) {
  return node.input_size() > index && !node.input(index).empty();
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
       "CausalConvWithState",
       "onnx_light_cpu::ComputeShapeCausalConvWithState",
       "onnx_light_cpu::ComputePeakMemoryCausalConvWithState",
       {},
       false},
      {kMicrosoftDomain,
       "GroupQueryAttention",
       "onnx_light_cpu::ComputeShapeGroupQueryAttention",
       "onnx_light_cpu::ComputePeakMemoryGroupQueryAttention",
       {"onnx_light_cpu::GroupQueryAttentionFusionPattern"},
       true},
      {kMicrosoftDomain,
       "MultiHeadAttention",
       "onnx_light_cpu::ComputeShapeMultiHeadAttention",
       "onnx_light_cpu::ComputePeakMemoryMultiHeadAttention",
       {},
       false},
      {kMicrosoftDomain,
       "LinearAttention",
       "onnx_light_cpu::ComputeShapeLinearAttention",
       "onnx_light_cpu::ComputePeakMemoryLinearAttention",
       {"onnx_light_cpu::LinearAttentionFusionPattern"},
       false},
      {kMicrosoftDomain,
       "MatMulNBits",
       "onnx_light_cpu::ComputeShapeMatMulNBits",
       "onnx_light_cpu::ComputePeakMemoryMatMulNBits",
       {"onnx_light_cpu::MatMulNBitsBiasFusionPattern"},
       true},
      {kMicrosoftDomain,
       "MoE",
       "onnx_light_cpu::ComputeShapeMoE",
       "onnx_light_cpu::ComputePeakMemoryMoE",
       {},
       false},
      {kMicrosoftDomain,
       "QMoE",
       "onnx_light_cpu::ComputeShapeQMoE",
       "onnx_light_cpu::ComputePeakMemoryQMoE",
       {},
       false},
      {kMicrosoftDomain,
       "RotaryEmbedding",
       "onnx_light_cpu::ComputeShapeRotaryEmbedding",
       "onnx_light_cpu::ComputePeakMemoryRotaryEmbedding",
       {},
       false},
      {"ai.onnx",
       "RotaryEmbedding",
       "onnx_light_cpu::ComputeShapeRotaryEmbedding",
       "onnx_light_cpu::ComputePeakMemoryRotaryEmbedding",
       {},
       false},
      {"ai.onnx",
       "SimplifiedLayerNormalization",
       "onnx_light_cpu::ComputeShapeSimplifiedLayerNormalization",
       "onnx_light_cpu::ComputePeakMemorySimplifiedLayerNormalization",
       {},
       false},
      {kMicrosoftDomain,
       "SkipLayerNormalization",
       "onnx_light_cpu::ComputeShapeSkipLayerNormalization",
       "onnx_light_cpu::ComputePeakMemorySkipLayerNormalization",
       {},
       false},
      {kMicrosoftDomain,
       "SkipSimplifiedLayerNormalization",
       "onnx_light_cpu::ComputeShapeSkipSimplifiedLayerNormalization",
       "onnx_light_cpu::ComputePeakMemorySkipSimplifiedLayerNormalization",
       {},
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

void ComputeShapeMoE(ShapesContext &ctx, const ONNX_LIGHT_NAMESPACE::NodeProto &node) {
  if (node.input_size() < 5 || node.input_size() > 8 || node.output_size() != 1 ||
      node.output(0).empty()) {
    throw std::invalid_argument(
        "ComputeShapeMoE: expected five to eight input slots and one output.");
  }
  for (int index : {0, 1, 2, 4}) {
    if (node.input(index).empty() || !ctx.Has(node.input(index))) {
      throw std::invalid_argument("ComputeShapeMoE: missing required input.");
    }
  }
  for (int index : {6, 7}) {
    if (HasInput(node, index)) {
      throw std::invalid_argument("ComputeShapeMoE: FC3 is not supported by the CPU kernel.");
    }
  }
  const auto &input = ctx.Get(node.input(0));
  const auto &router = ctx.Get(node.input(1));
  const auto &fc1 = ctx.Get(node.input(2));
  const auto &fc2 = ctx.Get(node.input(4));
  const auto type = input.Dtype();
  if ((type != ONNX_LIGHT_NAMESPACE::core::schema::TensorType::kFloat &&
       type != ONNX_LIGHT_NAMESPACE::core::schema::TensorType::kFloat16 &&
       type != ONNX_LIGHT_NAMESPACE::core::schema::TensorType::kBfloat16) ||
      router.Dtype() != type || fc1.Dtype() != type || fc2.Dtype() != type ||
      (input.Shape().Rank() != 2 && input.Shape().Rank() != 3) || router.Shape().Rank() != 2 ||
      fc1.Shape().Rank() != 3 || fc2.Shape().Rank() != 3) {
    throw std::invalid_argument("ComputeShapeMoE: invalid input types or ranks.");
  }
  const auto &shape = input.Shape();
  const auto &hidden = shape[shape.Rank() - 1];
  const auto &experts = router.Shape()[1];
  const auto &intermediate = fc2.Shape()[2];
  const auto k = GetGqaIntAttribute(node, "k", 1);
  const auto activation = GetStringAttribute(node, "activation_type", "relu");
  const auto fusion = GetGqaIntAttribute(node, "swiglu_fusion", 0);
  if (k < 1 || (experts.IsInt() && k > experts.AsInt()) ||
      (activation == "swiglu" && fusion != 1) || (activation != "swiglu" && fusion != 0) ||
      (activation != "relu" && activation != "gelu" && activation != "silu" &&
       activation != "identity" && activation != "swiglu") ||
      GetGqaIntAttribute(node, "use_sparse_mixer", 0) != 0) {
    throw std::invalid_argument("ComputeShapeMoE: unsupported attributes or expert count.");
  }
  for (const auto &dim : {hidden, experts, intermediate}) {
    if (dim.IsInt() && dim.AsInt() <= 0) {
      throw std::invalid_argument(
          "ComputeShapeMoE: feature and expert dimensions must be positive.");
    }
  }
  const sym_ns::SymDim rows =
      shape.Rank() == 2
          ? shape[0]
          : sym_ns::SymDim("(" + shape[0].ToString() + ")*(" + shape[1].ToString() + ")");
  if (shape.Rank() == 3 && shape[0].IsInt() && shape[1].IsInt()) {
    ConstrainEqual(ctx,
                   sym_ns::SymDim(cpu_compute::CheckedIndexMultiply(
                       shape[0].AsInt(), shape[1].AsInt(), "MoE", "token count")),
                   router.Shape()[0], "ComputeShapeMoE: router token count mismatch.");
  } else {
    ConstrainEqual(ctx, rows, router.Shape()[0], "ComputeShapeMoE: router token count mismatch.");
  }
  ConstrainEqual(ctx, experts, fc1.Shape()[0], "ComputeShapeMoE: FC1 experts mismatch.");
  ConstrainEqual(ctx, experts, fc2.Shape()[0], "ComputeShapeMoE: FC2 experts mismatch.");
  ConstrainEqual(ctx, hidden, fc1.Shape()[2], "ComputeShapeMoE: FC1 hidden mismatch.");
  ConstrainEqual(ctx, hidden, fc2.Shape()[1], "ComputeShapeMoE: FC2 hidden mismatch.");
  const sym_ns::SymDim width = activation != "swiglu" ? intermediate
                               : intermediate.IsInt()
                                   ? sym_ns::SymDim(cpu_compute::CheckedIndexMultiply(
                                         intermediate.AsInt(), 2, "MoE", "FC1 width"))
                                   : sym_ns::SymDim("2*(" + intermediate.ToString() + ")");
  ConstrainEqual(ctx, width, fc1.Shape()[1], "ComputeShapeMoE: FC1 width mismatch.");
  for (const auto [index, second] : {std::pair{3, width}, std::pair{5, hidden}}) {
    if (HasInput(node, index)) {
      if (!ctx.Has(node.input(index))) {
        throw std::invalid_argument("ComputeShapeMoE: missing bias.");
      }
      const auto &bias = ctx.Get(node.input(index));
      if (bias.Dtype() != type || bias.Shape().Rank() != 2) {
        throw std::invalid_argument("ComputeShapeMoE: invalid bias type or rank.");
      }
      ConstrainEqual(ctx, experts, bias.Shape()[0], "ComputeShapeMoE: bias experts mismatch.");
      ConstrainEqual(ctx, second, bias.Shape()[1], "ComputeShapeMoE: bias width mismatch.");
    }
  }
  ctx.Set(node.output(0), SymTensor(nullptr, type, shape));
}

void ComputeShapeQMoE(ShapesContext &ctx, const ONNX_LIGHT_NAMESPACE::NodeProto &node) {
  using ONNX_LIGHT_NAMESPACE::core::schema::TensorType;
  if (node.input_size() < 7 || node.input_size() > 14 || node.output_size() != 1 ||
      node.output(0).empty()) {
    throw std::invalid_argument("ComputeShapeQMoE: expected seven to fourteen input slots.");
  }
  for (int index : {0, 1, 2, 3, 5, 6}) {
    if (!HasInput(node, index) || !ctx.Has(node.input(index))) {
      throw std::invalid_argument("ComputeShapeQMoE: missing required input.");
    }
  }
  const auto &input = ctx.Get(node.input(0));
  const auto &router = ctx.Get(node.input(1));
  const auto type = input.Dtype();
  if ((type != TensorType::kFloat && type != TensorType::kFloat16 &&
       type != TensorType::kBfloat16) ||
      router.Dtype() != type || (input.Shape().Rank() != 2 && input.Shape().Rank() != 3) ||
      router.Shape().Rank() != 2) {
    throw std::invalid_argument("ComputeShapeQMoE: invalid activation/router types or ranks.");
  }
  const auto bits = GetGqaIntAttribute(node, "expert_weight_bits", 4);
  const auto block = GetGqaIntAttribute(node, "block_size", 0);
  const auto k = GetGqaIntAttribute(node, "k", 1);
  const auto fusion = GetGqaIntAttribute(node, "swiglu_fusion", 0);
  const auto activation = GetStringAttribute(node, "activation_type", "relu");
  const auto normalize = GetGqaIntAttribute(node, "normalize_routing_weights", 0);
  const bool third = HasInput(node, 8);
  for (const auto &attribute : node.attribute()) {
    if (((attribute.name() == "activation_alpha" || attribute.name() == "activation_beta") &&
         !std::isfinite(attribute.f())) ||
        (attribute.name() == "swiglu_limit" && (std::isnan(attribute.f()) || attribute.f() < 0))) {
      throw std::invalid_argument("ComputeShapeQMoE: invalid activation parameters.");
    }
  }
  if ((bits != 4 && bits != 8) || block < 0 ||
      (block != 0 && (block < 16 || (block & (block - 1)) != 0)) || k < 1 ||
      (normalize != 0 && normalize != 1) ||
      (activation != "relu" && activation != "gelu" && activation != "silu" &&
       activation != "identity" && activation != "swiglu") ||
      (activation == "swiglu" ? fusion != 1 : fusion != 0) || (third && fusion != 0) ||
      HasInput(node, 9) != third || (!third && HasInput(node, 10)) || HasInput(node, 11) ||
      HasInput(node, 12) || HasInput(node, 13) ||
      GetGqaIntAttribute(node, "use_sparse_mixer", 0) != 0 ||
      GetGqaIntAttribute(node, "weights_prepacked", 0) != 0) {
    throw std::invalid_argument("ComputeShapeQMoE: unsupported quantization or attributes.");
  }
  const auto &shape = input.Shape();
  const auto hidden = shape[shape.Rank() - 1];
  const auto experts = router.Shape()[1];
  if ((hidden.IsInt() && hidden.AsInt() <= 0) ||
      (experts.IsInt() && (experts.AsInt() <= 0 || k > experts.AsInt()))) {
    throw std::invalid_argument("ComputeShapeQMoE: invalid hidden or expert count.");
  }
  const auto &fc2 = ctx.Get(node.input(5));
  if (fc2.Shape().Rank() != 3 || ctx.Get(node.input(2)).Shape().Rank() != 3) {
    throw std::invalid_argument("ComputeShapeQMoE: projections must have rank 3.");
  }
  const auto divide_up = [](const sym_ns::SymDim &dim, int64_t divisor) {
    return dim.IsInt()
               ? sym_ns::SymDim(dim.AsInt() / divisor + (dim.AsInt() % divisor != 0))
               : sym_ns::SymDim("ceil((" + dim.ToString() + ")/" + std::to_string(divisor) + ")");
  };
  const auto compatible = [](const sym_ns::SymDim &left, const sym_ns::SymDim &right) {
    return !left.IsInt() || !right.IsInt() || left.AsInt() == right.AsInt();
  };
  const bool fused = activation == "swiglu";
  bool output_major = fused;
  bool rowwise = false;
  bool layout_ambiguous = false;
  auto width = fc2.Shape()[1];
  auto intermediate = width;
  struct Layout {
    bool output_major;
    bool rowwise;
    sym_ns::SymDim intermediate;
    sym_ns::SymDim width;
  };
  const auto fits = [&](bool major, bool row_scales) -> std::optional<Layout> {
    const auto candidate_width = major ? ctx.Get(node.input(2)).Shape()[1]
                                 : fused && fc2.Shape()[1].IsInt()
                                     ? sym_ns::SymDim(cpu_compute::CheckedIndexMultiply(
                                           fc2.Shape()[1].AsInt(), 2, "QMoE", "width"))
                                 : fused ? sym_ns::SymDim("2*(" + fc2.Shape()[1].ToString() + ")")
                                         : fc2.Shape()[1];
    if (candidate_width.IsInt() &&
        (candidate_width.AsInt() <= 0 || (fused && candidate_width.AsInt() % 2 != 0))) {
      return std::nullopt;
    }
    const auto candidate = fused ? divide_up(candidate_width, 2) : candidate_width;
    const auto match_projection = [&](int wi, int si, const sym_ns::SymDim &out,
                                      const sym_ns::SymDim &in) {
      const auto &w = ctx.Get(node.input(wi));
      const auto &s = ctx.Get(node.input(si));
      const auto leading = major ? out : in;
      const auto last = major ? in : out;
      return w.Shape().Rank() == 3 && s.Shape().Rank() == (block == 0 ? 2 : 3) &&
             compatible(experts, w.Shape()[0]) && compatible(leading, w.Shape()[1]) &&
             compatible(divide_up(last, 8 / bits), w.Shape()[2]) &&
             compatible(experts, s.Shape()[0]) &&
             compatible(block == 0 && !row_scales ? last : leading, s.Shape()[1]) &&
             (block == 0 || compatible(divide_up(last, block), s.Shape()[2]));
    };
    if (!match_projection(2, 3, candidate_width, hidden) ||
        !match_projection(5, 6, hidden, candidate) ||
        (third && !match_projection(8, 9, candidate, hidden))) {
      return std::nullopt;
    }
    const auto bias_matches = [&](int bias_index, const sym_ns::SymDim &outputs) {
      if (!HasInput(node, bias_index)) {
        return true;
      }
      const auto &bias = ctx.Get(node.input(bias_index));
      return bias.Shape().Rank() == 2 && compatible(experts, bias.Shape()[0]) &&
             compatible(outputs, bias.Shape()[1]);
    };
    if (!bias_matches(4, candidate_width) || !bias_matches(7, hidden) ||
        (third && !bias_matches(10, candidate))) {
      return std::nullopt;
    }
    return Layout{major, row_scales, candidate_width, candidate};
  };
  // Tensor quantization uses raw CPU output-major, per-row scales only.
  const auto output_layout = fits(true, block == 0);
  const std::optional<Layout> input_layout = block == 0 ? std::nullopt : fits(false, false);
  if (!output_layout && !input_layout) {
    throw std::invalid_argument("ComputeShapeQMoE: incompatible expert layouts.");
  }
  if (output_layout && input_layout) {
    layout_ambiguous = true;
  } else {
    const auto &layout = output_layout ? *output_layout : *input_layout;
    output_major = layout.output_major;
    rowwise = layout.rowwise;
    intermediate = layout.intermediate;
    width = layout.width;
  }
  const auto rows =
      shape.Rank() == 2 ? shape[0]
      : shape[0].IsInt() && shape[1].IsInt()
          ? sym_ns::SymDim(cpu_compute::CheckedIndexMultiply(shape[0].AsInt(), shape[1].AsInt(),
                                                             "QMoE", "tokens"))
          : sym_ns::SymDim("(" + shape[0].ToString() + ")*(" + shape[1].ToString() + ")");
  ConstrainEqual(ctx, rows, router.Shape()[0], "ComputeShapeQMoE: router rows mismatch.");
  const auto projection = [&](int weight_index, int scale_index, int bias_index,
                              const sym_ns::SymDim &outputs, const sym_ns::SymDim &inputs) {
    const auto &weight = ctx.Get(node.input(weight_index));
    const auto &scales = ctx.Get(node.input(scale_index));
    const auto leading = output_major ? outputs : inputs;
    const auto last = output_major ? inputs : outputs;
    if (weight.Dtype() != TensorType::kUint8 || weight.Shape().Rank() != 3 ||
        (scales.Dtype() != TensorType::kFloat && scales.Dtype() != TensorType::kFloat16 &&
         scales.Dtype() != TensorType::kBfloat16) ||
        scales.Shape().Rank() != (block == 0 ? 2 : 3)) {
      throw std::invalid_argument("ComputeShapeQMoE: invalid projection types or ranks.");
    }
    ConstrainEqual(ctx, experts, weight.Shape()[0], "ComputeShapeQMoE: weight experts mismatch.");
    ConstrainEqual(ctx, experts, scales.Shape()[0], "ComputeShapeQMoE: scale experts mismatch.");
    if (HasInput(node, bias_index)) {
      const auto &bias = ctx.Get(node.input(bias_index));
      if (bias.Dtype() != type || bias.Shape().Rank() != 2) {
        throw std::invalid_argument("ComputeShapeQMoE: invalid bias type or rank.");
      }
      ConstrainEqual(ctx, experts, bias.Shape()[0], "ComputeShapeQMoE: bias experts mismatch.");
    }
    if (layout_ambiguous) {
      return;
    }
    ConstrainEqual(ctx, leading, weight.Shape()[1], "ComputeShapeQMoE: weight rows mismatch.");
    ConstrainEqual(ctx, divide_up(last, 8 / bits), weight.Shape()[2],
                   "ComputeShapeQMoE: packed width mismatch.");
    ConstrainEqual(ctx, block == 0 && !rowwise ? last : leading, scales.Shape()[1],
                   "ComputeShapeQMoE: scale rows mismatch.");
    if (block != 0) {
      ConstrainEqual(ctx, divide_up(last, block), scales.Shape()[2],
                     "ComputeShapeQMoE: scale blocks mismatch.");
    }
    if (HasInput(node, bias_index)) {
      const auto &bias = ctx.Get(node.input(bias_index));
      ConstrainEqual(ctx, outputs, bias.Shape()[1], "ComputeShapeQMoE: bias width mismatch.");
    }
  };
  projection(2, 3, 4, intermediate, hidden);
  projection(5, 6, 7, hidden, width);
  if (third) {
    projection(8, 9, 10, width, hidden);
  }
  ctx.Set(node.output(0), SymTensor(nullptr, type, shape));
}

void ComputeShapeMatMulNBits(ShapesContext &ctx, const ONNX_LIGHT_NAMESPACE::NodeProto &node) {
  const bool has_encoded_value =
      node.input_size() >= 2 && HasInput(node, 1) && ctx.HasEncodedValue(node.input(1));
  const bool has_structured_type = node.input_size() >= 2 && HasInput(node, 1) &&
                                   ctx.HasType(node.input(1)) &&
                                   ctx.GetType(node.input(1)).has_struct_type();
  const bool encoded_weights = has_encoded_value || has_structured_type;
  if (node.input_size() < 2 || node.input_size() > 6 || node.output_size() != 1 ||
      node.output(0).empty() || !HasInput(node, 0) || !HasInput(node, 1) ||
      (!encoded_weights && !HasInput(node, 2)) || !ctx.Has(node.input(0)) ||
      (!encoded_weights && (!ctx.Has(node.input(1)) || !ctx.Has(node.input(2)))) ||
      (encoded_weights && (HasInput(node, 2) || HasInput(node, 3))) || HasInput(node, 4) ||
      (HasInput(node, 3) && !ctx.Has(node.input(3))) ||
      (HasInput(node, 5) && !ctx.Has(node.input(5)))) {
    throw std::invalid_argument(
        "ComputeShapeMatMulNBits: expected A, B, optional scales, zero_points and bias, and no "
        "g_idx; scales and zero_points must be omitted for encoded B.");
  }
  const std::int64_t k = GetRequiredIntAttribute(node, "K");
  const std::int64_t n = GetRequiredIntAttribute(node, "N");
  const std::int64_t bits = GetIntAttribute(node, "bits", 4);
  const std::int64_t block_size = GetRequiredIntAttribute(node, "block_size");
  const std::int64_t accuracy_level = GetIntAttribute(node, "accuracy_level", 0);
  const std::int64_t weight_prepacked = GetIntAttribute(node, "weight_prepacked", 0);
  if (k <= 0 || n <= 0 || !cpu_compute::MatMulNBitsBlockSizeSupported(bits, block_size) ||
      (accuracy_level != 0 && accuracy_level != 4) || weight_prepacked != 0) {
    throw std::invalid_argument("ComputeShapeMatMulNBits: unsupported bits, block_size, "
                                "accuracy_level or weight_prepacked.");
  }
  const SymTensor &a = ctx.Get(node.input(0));
  const bool supported_type = a.Dtype() == sym_ns::TensorType::kFloat ||
                              a.Dtype() == sym_ns::TensorType::kFloat16 ||
                              a.Dtype() == sym_ns::TensorType::kBfloat16;
  if (!supported_type) {
    throw std::invalid_argument(
        "ComputeShapeMatMulNBits: A must have FLOAT, FLOAT16, or BFLOAT16 type.");
  }
  if (a.Shape().Rank() == 0) {
    throw std::invalid_argument("ComputeShapeMatMulNBits: A must have positive rank.");
  }
  if (has_structured_type) {
    ValidateStructuredMatMulNBitsType(ctx, node.input(1), k, n, bits, a.Dtype());
  }
  const std::int64_t blocks = k / block_size + (k % block_size != 0);
  ConstrainEqual(ctx, a.Shape()[a.Shape().Rank() - 1], sym_ns::SymDim(k),
                 "ComputeShapeMatMulNBits: A's final dimension must equal K.");
  if (has_encoded_value) {
    const auto inputs = InspectEncodedMatMulNBitsShape(ctx, node.input(1));
    const auto encoded_data_type = inputs.data_type == ONNX_LIGHT_NAMESPACE::TensorProto::FLOAT
                                       ? sym_ns::TensorType::kFloat
                                   : inputs.data_type == ONNX_LIGHT_NAMESPACE::TensorProto::FLOAT16
                                       ? sym_ns::TensorType::kFloat16
                                   : inputs.data_type == ONNX_LIGHT_NAMESPACE::TensorProto::BFLOAT16
                                       ? sym_ns::TensorType::kBfloat16
                                       : sym_ns::TensorType::kUndefined;
    if (inputs.k != k || inputs.n != n || inputs.bits != bits || inputs.block_size != block_size) {
      throw std::invalid_argument(
          "ComputeShapeMatMulNBits: EncodedValueProto geometry does not match the node "
          "attributes.");
    }
    if (encoded_data_type != a.Dtype()) {
      throw std::invalid_argument("ComputeShapeMatMulNBits: encoded scales must match A's type.");
    }
  } else if (!has_structured_type) {
    if (bits == 3) {
      throw std::invalid_argument("ComputeShapeMatMulNBits: INT3 requires encoded B.");
    }
    const SymTensor &b = ctx.Get(node.input(1));
    const SymTensor &scales = ctx.Get(node.input(2));
    if (scales.Dtype() != a.Dtype() || b.Dtype() != sym_ns::TensorType::kUint8) {
      throw std::invalid_argument(
          "ComputeShapeMatMulNBits: scales must match A's type and B must be UINT8.");
    }
    if (b.Shape().Rank() != 3 || (scales.Shape().Rank() != 1 && scales.Shape().Rank() != 2)) {
      throw std::invalid_argument("ComputeShapeMatMulNBits: invalid input rank.");
    }
    ConstrainEqual(ctx, b.Shape()[0], sym_ns::SymDim(n),
                   "ComputeShapeMatMulNBits: B dimension 0 must equal N.");
    ConstrainEqual(ctx, b.Shape()[1], sym_ns::SymDim(blocks),
                   "ComputeShapeMatMulNBits: B dimension 1 must equal ceil(K / block_size).");
    ConstrainEqual(ctx, b.Shape()[2], sym_ns::SymDim(block_size * bits / 8),
                   "ComputeShapeMatMulNBits: B dimension 2 must equal block_size * bits / 8.");
    if (scales.Shape().Rank() == 1) {
      ConstrainEqual(ctx, scales.Shape()[0],
                     sym_ns::SymDim(cpu_compute::CheckedIndexMultiply(n, blocks, "MatMulNBits",
                                                                      "scale count")),
                     "ComputeShapeMatMulNBits: flat scales has the wrong length.");
    } else {
      ConstrainEqual(ctx, scales.Shape()[0], sym_ns::SymDim(n),
                     "ComputeShapeMatMulNBits: scales dimension 0 must equal N.");
      ConstrainEqual(ctx, scales.Shape()[1], sym_ns::SymDim(blocks),
                     "ComputeShapeMatMulNBits: scales dimension 1 has the wrong block count.");
    }
  }
  if (HasInput(node, 3)) {
    const auto &zero = ctx.Get(node.input(3));
    const bool packed = zero.Dtype() == sym_ns::TensorType::kUint8;
    if ((!packed && zero.Dtype() != a.Dtype()) ||
        (zero.Shape().Rank() != 1 && zero.Shape().Rank() != 2)) {
      throw std::invalid_argument("ComputeShapeMatMulNBits: invalid zero_points type or rank.");
    }
    const auto per_byte = 8 / bits;
    const auto stride = packed ? blocks / per_byte + (blocks % per_byte != 0) : blocks;
    if (zero.Shape().Rank() == 1) {
      ConstrainEqual(ctx, zero.Shape()[0],
                     sym_ns::SymDim(cpu_compute::CheckedIndexMultiply(n, stride, "MatMulNBits",
                                                                      "zero point count")),
                     "ComputeShapeMatMulNBits: invalid flat zero_points length.");
    } else {
      ConstrainEqual(ctx, zero.Shape()[0], sym_ns::SymDim(n),
                     "ComputeShapeMatMulNBits: zero_points dimension 0 must equal N.");
      ConstrainEqual(ctx, zero.Shape()[1], sym_ns::SymDim(stride),
                     "ComputeShapeMatMulNBits: invalid zero_points stride.");
    }
  }
  if (HasInput(node, 5)) {
    const SymTensor &bias = ctx.Get(node.input(5));
    if (bias.Dtype() != a.Dtype() || bias.Shape().Rank() != 1) {
      throw std::invalid_argument(
          "ComputeShapeMatMulNBits: bias must be rank 1 and have A's type.");
    }
    ConstrainEqual(ctx, bias.Shape()[0], sym_ns::SymDim(n),
                   "ComputeShapeMatMulNBits: bias length must equal N.");
  }
  SymShape output = a.Shape();
  output[output.Rank() - 1] = sym_ns::SymDim(n);
  ctx.Set(node.output(0), SymTensor(nullptr, a.Dtype(), std::move(output)));
}

static void ComputeShapeSkipLayerNormalizationCommon(ShapesContext &ctx,
                                                     const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                                                     bool simplified) {
  const char *op = simplified ? "SkipSimplifiedLayerNormalization" : "SkipLayerNormalization";
  const int maximum_inputs = simplified ? 4 : 5;
  if (node.input_size() < 3 || node.input_size() > maximum_inputs || node.input(0).empty() ||
      node.input(1).empty() || node.input(2).empty() || node.output_size() < 1 ||
      node.output_size() > 4 || node.output(0).empty()) {
    throw std::invalid_argument(std::string(op) +
                                ": expected input, skip, gamma, optional affine inputs, output, "
                                "and optional statistics and residual outputs.");
  }
  for (const auto &attribute : node.attribute()) {
    if (attribute.name() == "epsilon" && (!std::isfinite(attribute.f()) || attribute.f() < 0.0F)) {
      throw std::invalid_argument(std::string(op) + ": epsilon must be finite and nonnegative.");
    }
  }
  for (int i = 0; i < node.input_size(); ++i) {
    if (!node.input(i).empty() && !ctx.Has(node.input(i))) {
      throw std::invalid_argument(std::string(op) + ": missing descriptor for input slot " +
                                  std::to_string(i) + " ('" + node.input(i) + "').");
    }
  }
  const auto &input = ctx.Get(node.input(0));
  const auto &skip = ctx.Get(node.input(1));
  const auto &gamma = ctx.Get(node.input(2));
  const auto &shape = input.Shape();
  if (shape.Rank() != 2 && shape.Rank() != 3) {
    throw std::invalid_argument(std::string(op) + ": input must have rank 2 or 3.");
  }
  const auto &hidden = shape[shape.Rank() - 1];
  if (hidden.IsInt() && hidden.AsInt() > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string(op) + ": H must fit int.");
  }
  if (hidden.IsExpr()) {
    ctx.AddLessEqualConstraint(hidden.AsExpr(), std::to_string(std::numeric_limits<int>::max()));
  }
  for (int i = 0; i < node.input_size(); ++i) {
    if (node.input(i).empty()) {
      continue;
    }
    const auto &tensor = ctx.Get(node.input(i));
    if ((tensor.Dtype() != sym_ns::TensorType::kFloat &&
         tensor.Dtype() != sym_ns::TensorType::kFloat16 &&
         tensor.Dtype() != sym_ns::TensorType::kBfloat16) ||
        tensor.Dtype() != input.Dtype()) {
      throw std::invalid_argument(std::string(op) +
                                  ": all inputs must have matching FLOAT, FLOAT16, or BFLOAT16 "
                                  "types.");
    }
    for (size_t dim_index = 0; dim_index < tensor.Shape().Rank(); ++dim_index) {
      const auto &dim = tensor.Shape()[dim_index];
      const int64_t lower = dim_index + 1 == tensor.Shape().Rank() ? 1 : 0;
      if (dim.IsInt() && dim.AsInt() < lower) {
        throw std::invalid_argument(std::string(op) +
                                    ": dimensions must be nonnegative and H positive.");
      }
      if (dim.IsExpr()) {
        ctx.AddLessEqualConstraint(std::to_string(lower), dim.AsExpr());
      }
    }
  }
  if (gamma.Shape().Rank() != 1) {
    throw std::invalid_argument(std::string(op) + ": gamma must have rank 1.");
  }
  ConstrainEqual(ctx, shape[shape.Rank() - 1], gamma.Shape()[0],
                 (std::string(op) + ": gamma length must match H.").c_str());
  for (int slot = 3; slot < node.input_size(); ++slot) {
    if (!node.input(slot).empty()) {
      const auto &affine = ctx.Get(node.input(slot));
      const char *name = !simplified && slot == 3 ? "beta" : "bias";
      if (affine.Shape().Rank() != 1) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must have rank 1.");
      }
      ConstrainEqual(ctx, shape[shape.Rank() - 1], affine.Shape()[0],
                     (std::string(op) + ": " + name + " length must match H.").c_str());
    }
  }
  const auto &skip_shape = skip.Shape();
  if (skip_shape.Rank() != shape.Rank() && !(shape.Rank() == 3 && skip_shape.Rank() == 2)) {
    throw std::invalid_argument(std::string(op) +
                                ": skip rank must match input or omit its batch axis.");
  }
  const size_t offset = shape.Rank() - skip_shape.Rank();
  for (size_t i = 0; i < skip_shape.Rank(); ++i) {
    const auto &s = skip_shape[i];
    const auto &d = shape[offset + i];
    if (shape.Rank() == 3 && skip_shape.Rank() == 3 && i == 0) {
      if (s == d || (s.IsInt() && s.AsInt() == 1)) {
        continue;
      }
      if (s.IsExpr() && !(d.IsInt() && d.AsInt() == 1)) {
        // Batch broadcasting permits either one or the input batch size.
        ctx.AddConstraint("(" + s.AsExpr() + "-1)*(" + s.AsExpr() + "-(" + d.ToString() + "))",
                          "0");
        continue;
      }
    }
    ConstrainEqual(ctx, d, s, (std::string(op) + ": skip dimensions must match input.").c_str());
  }
  ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), shape));
  for (int slot = 1; slot < node.output_size() && slot < 3; ++slot) {
    if (!node.output(slot).empty()) {
      auto statistics_shape = shape;
      statistics_shape[statistics_shape.Rank() - 1] = sym_ns::SymDim(1);
      ctx.Set(node.output(slot),
              SymTensor(nullptr, sym_ns::TensorType::kFloat, std::move(statistics_shape)));
    }
  }
  if (node.output_size() == 4 && !node.output(3).empty()) {
    ctx.Set(node.output(3), SymTensor(nullptr, input.Dtype(), shape));
  }
}

void ComputeShapeSkipLayerNormalization(ShapesContext &ctx,
                                        const ONNX_LIGHT_NAMESPACE::NodeProto &node) {
  ComputeShapeSkipLayerNormalizationCommon(ctx, node, false);
}

void ComputeShapeSkipSimplifiedLayerNormalization(ShapesContext &ctx,
                                                  const ONNX_LIGHT_NAMESPACE::NodeProto &node) {
  ComputeShapeSkipLayerNormalizationCommon(ctx, node, true);
}

void ComputeShapeMultiHeadAttention(ShapesContext &ctx,
                                    const ONNX_LIGHT_NAMESPACE::NodeProto &node) {
  const auto fail = [](const char *message) {
    throw std::invalid_argument(std::string("ComputeShapeMultiHeadAttention: ") + message);
  };
  if (!HasInput(node, 0) || !HasInput(node, 1) || !HasInput(node, 2) || node.output_size() < 1 ||
      node.output_size() > 3 || node.output(0).empty() || node.input_size() > 10 ||
      HasInput(node, 8) || HasInput(node, 9) || (node.output_size() > 3 && !node.output(3).empty()))
    fail("Q/K/V and output are required; shared buffers and qk are unsupported.");
  for (int i = 0; i < node.input_size(); ++i)
    if (HasInput(node, i) && !ctx.Has(node.input(i)))
      fail("input shape is unknown.");
  const auto &q = ctx.Get(node.input(0));
  const auto &k = ctx.Get(node.input(1));
  const auto &v = ctx.Get(node.input(2));
  const auto dtype = q.Dtype();
  if ((dtype != sym_ns::TensorType::kFloat && dtype != sym_ns::TensorType::kFloat16 &&
       dtype != sym_ns::TensorType::kBfloat16) ||
      dtype != k.Dtype() || dtype != v.Dtype())
    fail("Q/K/V floating types must match.");
  const auto &qs = q.Shape(), &ks = k.Shape(), &vs = v.Shape();
  if (qs.Rank() != 3 || (ks.Rank() != 3 && ks.Rank() != 4) || vs.Rank() != ks.Rank())
    fail("Q must be rank 3 and K/V rank 3 or 4; packed rank 5 is unsupported.");
  const bool cross = ks.Rank() == 4;
  const int64_t heads = GetIntAttribute(node, "num_heads", 0);
  if (heads <= 0 || GetIntAttribute(node, "unidirectional", 0) < 0 ||
      GetIntAttribute(node, "unidirectional", 0) > 1)
    fail("invalid num_heads or unidirectional.");
  auto batch = qs[0];
  batch = MergeGqaDim(ctx, batch, ks[0], "batch");
  batch = MergeGqaDim(ctx, batch, vs[0], "batch");
  const auto per_head = [heads](const sym_ns::SymDim &width, const char *what) {
    if (width.IsInt())
      return DivideGqaDim(width, heads, what);
    return heads == 1 ? width
                      : sym_ns::SymDim("(" + width.ToString() + ")/" + std::to_string(heads));
  };
  auto head = per_head(qs[2], "query width");
  sym_ns::SymDim value_head(std::string("?"));
  sym_ns::SymDim length = ks[cross ? 2 : 1];
  if (cross) {
    MergeGqaDim(ctx, ks[1], sym_ns::SymDim(heads), "key heads");
    MergeGqaDim(ctx, vs[1], sym_ns::SymDim(heads), "value heads");
    head = MergeGqaDim(ctx, head, ks[3], "key head size");
    value_head = vs[3];
    MergeGqaDim(ctx, length, vs[2], "KV length");
  } else {
    MergeGqaDim(ctx, length, vs[1], "KV length");
    head = MergeGqaDim(ctx, head, per_head(ks[2], "key width"), "head size");
    value_head = per_head(vs[2], "value width");
  }
  if (HasInput(node, 3)) {
    if (ctx.Get(node.input(3)).Dtype() != dtype || ctx.Get(node.input(3)).Shape().Rank() != 1)
      fail("invalid projection bias.");
    const auto &bd = ctx.Get(node.input(3)).Shape()[0];
    if (head.IsInt() && value_head.IsInt())
      MergeGqaDim(
          ctx, bd,
          sym_ns::SymDim(cpu_compute::CheckedIndexMultiply(
              heads,
              cpu_compute::CheckedIndexAdd(
                  cpu_compute::CheckedIndexMultiply(2, head.AsInt(), "MultiHeadAttention", "bias"),
                  value_head.AsInt(), "MultiHeadAttention", "bias"),
              "MultiHeadAttention", "bias")),
          "projection bias");
  }
  if (HasInput(node, 6) != HasInput(node, 7))
    fail("past K/V must be paired.");
  sym_ns::SymDim past(int64_t{0});
  if (HasInput(node, 6)) {
    if (cross)
      fail("past cache with rank-4 cross-attention is unsupported.");
    const auto &pk = ctx.Get(node.input(6)), &pv = ctx.Get(node.input(7));
    if (pk.Dtype() != dtype || pv.Dtype() != dtype || pk.Shape().Rank() != 4 ||
        pv.Shape().Rank() != 4)
      fail("invalid past cache.");
    batch = MergeGqaDim(ctx, batch, pk.Shape()[0], "past batch");
    batch = MergeGqaDim(ctx, batch, pv.Shape()[0], "past batch");
    MergeGqaDim(ctx, pk.Shape()[1], sym_ns::SymDim(heads), "past heads");
    MergeGqaDim(ctx, pv.Shape()[1], sym_ns::SymDim(heads), "past heads");
    head = MergeGqaDim(ctx, head, pk.Shape()[3], "past key size");
    value_head = MergeGqaDim(ctx, value_head, pv.Shape()[3], "past value size");
    past = MergeGqaDim(ctx, pk.Shape()[2], pv.Shape()[2], "past length");
  }
  const auto total = AddGqaDims(past, length);
  if (HasInput(node, 4)) {
    const auto &mask = ctx.Get(node.input(4));
    if (mask.Dtype() != sym_ns::TensorType::kInt32 ||
        (mask.Shape().Rank() != 1 && mask.Shape().Rank() != 2 && mask.Shape().Rank() != 3))
      fail("invalid int32 mask.");
    MergeGqaDim(ctx, mask.Shape()[0], batch, "mask batch");
    if (mask.Shape().Rank() == 3)
      MergeGqaDim(ctx, mask.Shape()[1], qs[1], "mask query length");
    if (mask.Shape().Rank() >= 2)
      MergeGqaDim(ctx, mask.Shape()[mask.Shape().Rank() - 1], total, "mask total length");
  }
  if (HasInput(node, 5)) {
    const auto &ab = ctx.Get(node.input(5));
    if (ab.Dtype() != dtype || ab.Shape().Rank() != 4)
      fail("invalid attention bias.");
    for (int i = 0; i < 2; ++i) {
      const auto &dim = ab.Shape()[i];
      const auto expected = i == 0 ? batch : sym_ns::SymDim(heads);
      if (dim.IsInt() && dim.AsInt() == 1)
        continue;
      if (dim.IsInt() || (expected.IsInt() && expected.AsInt() == 1))
        MergeGqaDim(ctx, dim, expected, "attention bias");
    }
    MergeGqaDim(ctx, ab.Shape()[2], qs[1], "attention bias S");
    MergeGqaDim(ctx, ab.Shape()[3], total, "attention bias T");
  }
  SymShape out;
  out.PushBack(batch);
  out.PushBack(qs[1]);
  if (!cross) {
    out.PushBack(vs[2]);
  } else if (value_head.IsInt()) {
    out.PushBack(MultiplyGqaDim(heads, value_head));
  } else {
    out.PushBack(heads == 1
                     ? value_head
                     : sym_ns::SymDim(std::to_string(heads) + "*(" + value_head.ToString() + ")"));
  }
  ctx.Set(node.output(0), SymTensor(nullptr, dtype, std::move(out)));
  const bool present_k = node.output_size() > 1 && !node.output(1).empty();
  const bool present_v = node.output_size() > 2 && !node.output(2).empty();
  if (present_k || present_v) {
    SymShape pk, pv;
    for (auto *shape : {&pk, &pv}) {
      shape->PushBack(batch);
      shape->PushBack(sym_ns::SymDim(heads));
      shape->PushBack(total);
    }
    pk.PushBack(head);
    pv.PushBack(value_head);
    if (present_k)
      ctx.Set(node.output(1), SymTensor(nullptr, dtype, std::move(pk)));
    if (present_v)
      ctx.Set(node.output(2), SymTensor(nullptr, dtype, std::move(pv)));
  }
}

void ComputeShapeGroupQueryAttention(ShapesContext &ctx,
                                     const ONNX_LIGHT_NAMESPACE::NodeProto &node) {
  if (node.input_size() < 7 || node.output_size() < 1 || node.output_size() > 3 ||
      !ctx.Has(node.input(0))) {
    throw std::invalid_argument(
        "ComputeShapeGroupQueryAttention: expected a known query and one to three outputs.");
  }
  if (HasInput(node, 1) != HasInput(node, 2)) {
    throw std::invalid_argument(
        "ComputeShapeGroupQueryAttention: key and value must both be wired or both be empty.");
  }
  const bool packed_qkv = !HasInput(node, 1);
  if (!packed_qkv && (!ctx.Has(node.input(1)) || !ctx.Has(node.input(2)))) {
    throw std::invalid_argument("ComputeShapeGroupQueryAttention: key and value must be known.");
  }
  const SymTensor &query = ctx.Get(node.input(0));
  const SymTensor *key = packed_qkv ? nullptr : &ctx.Get(node.input(1));
  const SymTensor *value = packed_qkv ? nullptr : &ctx.Get(node.input(2));
  if (!packed_qkv && (query.Dtype() != key->Dtype() || query.Dtype() != value->Dtype())) {
    throw std::invalid_argument(
        "ComputeShapeGroupQueryAttention: query, key, and value types must match.");
  }
  if (query.Shape().Rank() != 3 ||
      (!packed_qkv && (key->Shape().Rank() != 3 || value->Shape().Rank() != 3))) {
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
  sym_ns::SymDim batch = q_shape[0];
  const sym_ns::SymDim &q_seq_len = q_shape[1];
  sym_ns::SymDim kv_seq_len = q_seq_len;
  const std::int64_t heads =
      packed_qkv ? cpu_compute::CheckedIndexAdd(
                       num_heads,
                       cpu_compute::CheckedIndexMultiply(2, kv_num_heads, "GroupQueryAttention",
                                                         "packed head count"),
                       "GroupQueryAttention", "packed head count")
                 : num_heads;
  if (packed_qkv && q_shape[2].IsInt() && q_shape[2].AsInt() <= 0) {
    throw std::invalid_argument(
        "ComputeShapeGroupQueryAttention: packed hidden size must be positive.");
  }
  sym_ns::SymDim head_dim = DivideGqaDim(q_shape[2], heads, "query hidden size");
  sym_ns::SymDim v_head_dim = head_dim;
  if (!packed_qkv) {
    const SymShape &k_shape = key->Shape();
    const SymShape &v_shape = value->Shape();
    batch = MergeGqaDim(ctx, batch, k_shape[0], "batch");
    batch = MergeGqaDim(ctx, batch, v_shape[0], "batch");
    kv_seq_len = MergeGqaDim(ctx, k_shape[1], v_shape[1], "kv_sequence_length");
    MergeGqaDim(ctx, q_seq_len, kv_seq_len, "sequence_length");
    head_dim = MergeGqaDim(ctx, head_dim, DivideGqaDim(k_shape[2], kv_num_heads, "key hidden size"),
                           "head_size");
    v_head_dim = DivideGqaDim(v_shape[2], kv_num_heads, "value hidden size");
  }

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
    ctx.Set(node.output(2), SymTensor(nullptr, query.Dtype(), std::move(pv_shape)));
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
  bool key_heads_ordinary = false;
  if (key_head_size.IsInt() && k_shape[2].IsInt()) {
    const std::int64_t dk = key_head_size.AsInt();
    const std::int64_t key_hidden = k_shape[2].AsInt();
    if (dk <= 0 || key_hidden <= 0 || key_hidden % dk != 0 || kv_heads % (key_hidden / dk) != 0) {
      throw std::invalid_argument(
          "ComputeShapeLinearAttention: key heads must divide kv_num_heads.");
    }
    key_heads_ordinary = (key_hidden / dk) == kv_heads;
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

#if defined(ONNX_LIGHT_CPU_HAS_ONNX_SHAPE)
  // Ordinary grouping (q_num_heads a multiple of kv_num_heads) with no shared
  // physical key heads matches exactly what onnx-light's own shape function
  // already computes; reuse it instead of re-deriving the output/state
  // shapes below. Inverse grouping and shared key heads aren't supported by
  // onnx-light's function, so those keep the bespoke computation.
  if (q_heads % kv_heads == 0 && key_heads_ordinary) {
    ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::nn::ComputeShapeLinearAttention(
        ctx, node, node.input(0).c_str(), node.input(1).c_str(), node.input(2).c_str(),
        has_past ? node.input(3).c_str() : nullptr);
    return;
  }
#endif

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

void ComputeShapeCausalConvWithState(ShapesContext &ctx,
                                     const ONNX_LIGHT_NAMESPACE::NodeProto &node) {
  constexpr const char *error = "ComputeShapeCausalConvWithState: invalid input geometry.";
  if (node.input_size() < 2 || node.input_size() > 4 || node.output_size() != 2 ||
      node.output(0).empty() || node.output(1).empty() || !ctx.Has(node.input(0)) ||
      !ctx.Has(node.input(1))) {
    throw std::invalid_argument(error);
  }
  const auto &input = ctx.Get(node.input(0));
  const auto &weight = ctx.Get(node.input(1));
  const auto dtype = input.Dtype();
  if ((dtype != sym_ns::TensorType::kFloat && dtype != sym_ns::TensorType::kFloat16 &&
       dtype != sym_ns::TensorType::kBfloat16) ||
      weight.Dtype() != dtype || weight.Shape().Rank() != 3 ||
      GetIntAttribute(node, "ndim", 1) != 1 || GetIntAttribute(node, "state_window", 0) != 0 ||
      GetIntAttribute(node, "dilation", 1) < 1 ||
      GetIntAttribute(node, "dilation", 1) > std::numeric_limits<int>::max() ||
      (GetIntAttribute(node, "channels_last", 0) != 0 &&
       GetIntAttribute(node, "channels_last", 0) != 1)) {
    throw std::invalid_argument(error);
  }
  const auto &activation = GetStringAttribute(node, "activation", "none");
  if (activation != "none" && activation != "silu" && activation != "swish") {
    throw std::invalid_argument(error);
  }
  const bool channels_last = GetIntAttribute(node, "channels_last", 0) == 1;
  const auto &in_shape = input.Shape();
  const auto &w_shape = weight.Shape();
  if ((!channels_last && in_shape.Rank() != 3) || (channels_last && in_shape.Rank() < 3) ||
      (w_shape[2].IsInt() && w_shape[2].AsInt() < 1)) {
    throw std::invalid_argument(error);
  }
  ConstrainEqual(ctx, w_shape[1], sym_ns::SymDim(1), error);
  sym_ns::SymDim channels = in_shape[1];
  if (channels_last) {
    channels = in_shape[2];
    for (std::size_t i = 3; i < in_shape.Rank(); ++i) {
      channels =
          channels.IsInt() && in_shape[i].IsInt()
              ? sym_ns::SymDim(cpu_compute::CheckedIndexMultiply(
                    channels.AsInt(), in_shape[i].AsInt(), "CausalConvWithState", "channels"))
              : sym_ns::SymDim("(" + channels.ToString() + ")*(" + in_shape[i].ToString() + ")");
    }
  }
  ConstrainEqual(ctx, w_shape[0], channels, error);
  if (HasInput(node, 2)) {
    if (!ctx.Has(node.input(2)) || ctx.Get(node.input(2)).Dtype() != dtype ||
        ctx.Get(node.input(2)).Shape().Rank() != 1) {
      throw std::invalid_argument(error);
    }
    ConstrainEqual(ctx, ctx.Get(node.input(2)).Shape()[0], channels, error);
  }
  const std::int64_t dilation = GetIntAttribute(node, "dilation", 1);
  const auto &kernel = w_shape[2];
  const sym_ns::SymDim pad =
      kernel.IsInt() ? sym_ns::SymDim(cpu_compute::CheckedIndexMultiply(
                           kernel.AsInt() - 1, dilation, "CausalConvWithState", "state length"))
                     : sym_ns::SymDim("(" + kernel.ToString() + "-1)*" + std::to_string(dilation));
  SymShape state;
  for (std::size_t i = 0; i < in_shape.Rank(); ++i) {
    state.PushBack(i == (channels_last ? 1U : 2U) ? pad : in_shape[i]);
  }
  if (HasInput(node, 3)) {
    if (!ctx.Has(node.input(3)) || ctx.Get(node.input(3)).Dtype() != dtype ||
        ctx.Get(node.input(3)).Shape().Rank() != state.Rank()) {
      throw std::invalid_argument(error);
    }
    for (std::size_t i = 0; i < state.Rank(); ++i) {
      ConstrainEqual(ctx, ctx.Get(node.input(3)).Shape()[i], state[i], error);
    }
  }
  ctx.Set(node.output(0), SymTensor(nullptr, dtype, in_shape));
  ctx.Set(node.output(1), SymTensor(nullptr, dtype, std::move(state)));
}

int64_t ComputePeakMemoryCausalConvWithState(sym_ns::Device, const std::vector<SymShape> &) {
  return 0;
}

int64_t ComputePeakMemoryCDist(sym_ns::Device, const std::vector<SymShape> &) { return 0; }

int64_t ComputePeakMemoryBiasGelu(sym_ns::Device, const std::vector<SymShape> &) { return 0; }

int64_t ComputePeakMemoryMatMulNBits(sym_ns::Device, const std::vector<SymShape> &) { return 0; }

int64_t ComputePeakMemoryMoE(sym_ns::Device, const std::vector<SymShape> &shapes) {
  const std::size_t fc2_index = shapes.size() > 4 && shapes[4].Rank() == 3 ? 4 : 3;
  if (shapes.size() <= fc2_index || (shapes[0].Rank() != 2 && shapes[0].Rank() != 3) ||
      shapes[1].Rank() != 2 || shapes[2].Rank() != 3 || shapes[fc2_index].Rank() != 3) {
    return 0;
  }
  const auto &batch = shapes[0][0];
  const auto &sequence = shapes[0].Rank() == 3 ? shapes[0][1] : sym_ns::SymDim(1);
  const auto &experts = shapes[1][1];
  const auto &hidden = shapes[0][shapes[0].Rank() - 1];
  const auto &width = shapes[2][1];
  const auto &intermediate = shapes[fc2_index][2];
  if (!batch.IsInt() || !sequence.IsInt() || !experts.IsInt() || !hidden.IsInt() ||
      !width.IsInt() || !intermediate.IsInt()) {
    return 0;
  }
  const std::size_t tokens = cpu_compute::CheckedMultiply(
      cpu_compute::CheckedDimension(batch.AsInt(), "MoE", "batch"),
      cpu_compute::CheckedDimension(sequence.AsInt(), "MoE", "sequence"), "MoE", "token count");
  const std::size_t bytes = cpu_compute::ComputeMoEMaxScratchBytes(
      tokens, cpu_compute::CheckedDimension(hidden.AsInt(), "MoE", "hidden"),
      cpu_compute::CheckedDimension(experts.AsInt(), "MoE", "experts"),
      cpu_compute::CheckedDimension(width.AsInt(), "MoE", "fc1 width"),
      cpu_compute::CheckedDimension(intermediate.AsInt(), "MoE", "intermediate"));
  if (bytes > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error("ComputePeakMemoryMoE: scratch-memory size overflow.");
  }
  return static_cast<std::int64_t>(bytes);
}

int64_t ComputePeakMemoryQMoE(sym_ns::Device, const std::vector<SymShape> &shapes) {
  if (shapes.size() < 4 || (shapes[0].Rank() != 2 && shapes[0].Rank() != 3) ||
      shapes[1].Rank() != 2) {
    return 0;
  }
  for (const auto &shape : shapes) {
    for (std::size_t index = 0; index < shape.Rank(); ++index) {
      if (!shape[index].IsInt()) {
        return 0;
      }
      cpu_compute::CheckedDimension(shape[index].AsInt(), "QMoE", "scratch dimension");
    }
  }
  const auto hidden =
      cpu_compute::CheckedDimension(shapes[0][shapes[0].Rank() - 1].AsInt(), "QMoE", "hidden");
  const auto experts = cpu_compute::CheckedDimension(shapes[1][1].AsInt(), "QMoE", "experts");
  std::size_t tokens = cpu_compute::CheckedDimension(shapes[0][0].AsInt(), "QMoE", "tokens");
  if (shapes[0].Rank() == 3) {
    tokens = cpu_compute::CheckedMultiply(
        tokens, cpu_compute::CheckedDimension(shapes[0][1].AsInt(), "QMoE", "sequence"), "QMoE",
        "tokens");
  }
  if (tokens == 0) {
    return 0;
  }
  std::size_t values =
      cpu_compute::CheckedProduct({tokens, hidden, 2}, "QMoE", "activation scratch");
  values = cpu_compute::CheckedAdd(
      values, cpu_compute::CheckedMultiply(tokens, experts, "QMoE", "router scratch"), "QMoE",
      "value scratch");
  std::size_t width = hidden;
  // Shapes do not carry bit width or optional-slot metadata. Bounding every
  // rank-3 tensor as INT4 weights also handles compacted optional input lists.
  for (std::size_t index = 2; index < shapes.size(); ++index) {
    const auto &shape = shapes[index];
    if (shape.Rank() == 3) {
      const auto rows = cpu_compute::CheckedDimension(shape[1].AsInt(), "QMoE", "projection rows");
      const auto columns = cpu_compute::CheckedMultiply(
          cpu_compute::CheckedDimension(shape[2].AsInt(), "QMoE", "packed columns"), 2, "QMoE",
          "columns");
      width = std::max(width, std::max(rows, columns));
      values = cpu_compute::CheckedAdd(
          values,
          cpu_compute::CheckedProduct(
              {cpu_compute::CheckedDimension(shape[0].AsInt(), "QMoE", "weight experts"), rows,
               columns},
              "QMoE", "dequantized projection"),
          "QMoE", "value scratch");
    }
  }
  values = cpu_compute::CheckedAdd(
      values, cpu_compute::CheckedProduct({experts, width, 3}, "QMoE", "bias scratch"), "QMoE",
      "value scratch");
  const auto workspace = cpu_compute::CheckedAdd(
      cpu_compute::CheckedMultiply(width, 3, "QMoE", "workspace"), hidden, "QMoE", "workspace");
  values = cpu_compute::CheckedAdd(
      values, cpu_compute::CheckedMultiply(tokens, workspace, "QMoE", "split expert scratch"),
      "QMoE", "value scratch");
  auto bytes = cpu_compute::CheckedByteSize(values, sizeof(float), "QMoE", "value bytes");
  bytes = cpu_compute::CheckedAdd(
      bytes, cpu_compute::ComputeMoEMaxScratchBytes(tokens, hidden, experts, width, width), "QMoE",
      "scratch bytes");
  // Split FC3 execution has one route array per token, unlike bounded fused MoE tiles.
  bytes = cpu_compute::CheckedAdd(
      bytes,
      cpu_compute::CheckedProduct({tokens, experts, sizeof(cpu_compute::MoERoute)}, "QMoE",
                                  "split routes"),
      "QMoE", "scratch bytes");
  if (bytes > static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
    throw std::overflow_error("ComputePeakMemoryQMoE: scratch size overflows int64_t.");
  }
  return static_cast<int64_t>(bytes);
}

int64_t ComputePeakMemorySkipLayerNormalization(sym_ns::Device, const std::vector<SymShape> &) {
  return 0;
}

int64_t ComputePeakMemorySkipSimplifiedLayerNormalization(sym_ns::Device,
                                                          const std::vector<SymShape> &) {
  return 0;
}

int64_t ComputePeakMemoryGroupQueryAttention(sym_ns::Device,
                                             const std::vector<SymShape> &input_shapes) {
  if (input_shapes.empty() || input_shapes[0].Rank() != 3) {
    return 0;
  }

  const bool has_key = input_shapes.size() > 1 && input_shapes[1].Rank() == 3;
  const bool has_value = input_shapes.size() > 2 && input_shapes[2].Rank() == 3;
  if (has_key != has_value) {
    return 0;
  }
  constexpr int64_t kConservativeElementBytes = sizeof(float);
  bool known = true;
  const auto element_count = [&](const SymShape &shape, std::int64_t rank) {
    if (shape.Rank() != rank) {
      known = false;
      return int64_t{0};
    }
    int64_t count = 1;
    for (std::int64_t index = 0; index < rank; ++index) {
      const auto &dim = shape[index];
      if (!dim.IsInt()) {
        known = false;
        return int64_t{0};
      }
      if (dim.AsInt() < 0) {
        throw std::invalid_argument(
            "ComputePeakMemoryGroupQueryAttention: dimensions must be non-negative.");
      }
      count = cpu_compute::CheckedIndexMultiply(
          count, dim.AsInt(), "ComputePeakMemoryGroupQueryAttention", "scratch element count");
    }
    return count;
  };

  const int64_t query_elements = element_count(input_shapes[0], 3);
  const bool packed_qkv = !has_key;
  const int64_t key_elements = has_key ? element_count(input_shapes[1], 3) : 0;
  const int64_t value_elements = has_value ? element_count(input_shapes[2], 3) : 0;
  int64_t past_elements = 0;
  int64_t past_length = 0;
  for (std::size_t index : {std::size_t{3}, std::size_t{4}}) {
    if (input_shapes.size() > index && input_shapes[index].Rank() == 4) {
      if (past_length == 0 && input_shapes[index][2].IsInt()) {
        past_length = input_shapes[index][2].AsInt();
      }
      past_elements = cpu_compute::CheckedIndexAdd(
          past_elements, element_count(input_shapes[index], 4),
          "ComputePeakMemoryGroupQueryAttention", "past scratch element count");
    }
  }
  if (!known) {
    return 0;
  }

  // The shared hook must cover the NAIVE implementation. Its longest-lived
  // temporaries include unpacked Q/K/V for packed input, optional rotated Q/K,
  // and concatenated full K/V. Without node attributes, the complete packed
  // width is a conservative upper bound for both Q+K and K+V.
  const int64_t unpack_elements = packed_qkv ? query_elements : 0;
  const int64_t rotary_elements =
      packed_qkv ? query_elements
                 : cpu_compute::CheckedIndexAdd(query_elements, key_elements,
                                                "ComputePeakMemoryGroupQueryAttention",
                                                "rotary scratch element count");
  const int64_t current_kv_elements =
      packed_qkv ? query_elements
                 : cpu_compute::CheckedIndexAdd(key_elements, value_elements,
                                                "ComputePeakMemoryGroupQueryAttention",
                                                "current KV scratch element count");
  const int64_t score_elements = cpu_compute::CheckedIndexAdd(
      input_shapes[0][1].AsInt(), past_length, "ComputePeakMemoryGroupQueryAttention",
      "score row element count");
  const int64_t elements = cpu_compute::CheckedIndexAdd(
      unpack_elements,
      cpu_compute::CheckedIndexAdd(
          rotary_elements,
          cpu_compute::CheckedIndexAdd(
              current_kv_elements,
              cpu_compute::CheckedIndexAdd(past_elements, score_elements,
                                           "ComputePeakMemoryGroupQueryAttention",
                                           "attention row scratch element count"),
              "ComputePeakMemoryGroupQueryAttention", "full KV scratch element count"),
          "ComputePeakMemoryGroupQueryAttention", "attention scratch element count"),
      "ComputePeakMemoryGroupQueryAttention", "total scratch element count");
  return cpu_compute::CheckedIndexMultiply(elements, kConservativeElementBytes,
                                           "ComputePeakMemoryGroupQueryAttention",
                                           "scratch-memory byte size");
}

int64_t ComputePeakMemoryMultiHeadAttention(sym_ns::Device, const std::vector<SymShape> &shapes) {
  if (shapes.size() < 2 || shapes[0].Rank() != 3 ||
      (shapes[1].Rank() != 3 && shapes[1].Rank() != 4))
    return 0;
  const auto &b = shapes[0][0], &s = shapes[0][1];
  const auto &l = shapes[1][shapes[1].Rank() == 4 ? 2 : 1];
  if (!b.IsInt() || !s.IsInt() || !l.IsInt())
    return 0;
  int64_t p = 0;
  if (shapes.size() > 6 && shapes[6].Rank() == 4) {
    if (!shapes[6][2].IsInt())
      return 0;
    p = shapes[6][2].AsInt();
  }
  const auto total =
      cpu_compute::CheckedIndexAdd(p, l.AsInt(), "MultiHeadAttention", "score width");
  const auto rows =
      cpu_compute::CheckedIndexMultiply(b.AsInt(), s.AsInt(), "MultiHeadAttention", "rows");
  // Every query head may own a score row concurrently. The query's hidden
  // dimension is at least the head count, providing a conservative bound.
  if (!shapes[0][2].IsInt())
    return 0;
  return cpu_compute::CheckedIndexMultiply(
      cpu_compute::CheckedIndexMultiply(rows, shapes[0][2].AsInt(), "MultiHeadAttention",
                                        "score rows"),
      cpu_compute::CheckedIndexMultiply(total, int64_t{4}, "MultiHeadAttention", "score bytes"),
      "MultiHeadAttention", "score scratch");
}

int64_t ComputePeakMemoryLinearAttention(sym_ns::Device,
                                         const std::vector<SymShape> &input_shapes) {
  constexpr int64_t kAccumulatorElementBytes = sizeof(float);
  if (input_shapes.size() < 3 || input_shapes[2].Rank() != 3) {
    return 0;
  }

  // ExecuteRanges may process every (batch, KV head) task concurrently. Each
  // range owns one float32 vector of value_head_size elements, so
  // batch * packed_value_width is a safe upper bound independent of kv_num_heads.
  const auto &batch = input_shapes[2][0];
  const auto &packed_value_width = input_shapes[2][2];
  if (!batch.IsInt() || !packed_value_width.IsInt()) {
    return 0;
  }
  if (batch.AsInt() < 0 || packed_value_width.AsInt() < 0) {
    throw std::invalid_argument(
        "ComputePeakMemoryLinearAttention: value dimensions must be non-negative.");
  }
  if (batch.AsInt() != 0 && packed_value_width.AsInt() > std::numeric_limits<int64_t>::max() /
                                                             kAccumulatorElementBytes /
                                                             batch.AsInt()) {
    throw std::overflow_error("ComputePeakMemoryLinearAttention: scratch-memory size overflow.");
  }
  return batch.AsInt() * packed_value_width.AsInt() * kAccumulatorElementBytes;
}

void RegisterMicrosoftShapeAndMemoryFunctions() {
  static std::once_flag once;
  std::call_once(once, [] {
    shapes_ns::RegisterComputeShapeFn(kMicrosoftDomain, "CDist", ComputeShapeCDist);
    shapes_ns::RegisterComputeShapeFn(kMicrosoftDomain, "CausalConvWithState",
                                      ComputeShapeCausalConvWithState);
    shapes_ns::RegisterComputeShapeFn(kMicrosoftDomain, "BiasGelu", ComputeShapeBiasGelu);
    shapes_ns::RegisterComputeShapeFn(kMicrosoftDomain, "SkipLayerNormalization",
                                      ComputeShapeSkipLayerNormalization);
    shapes_ns::RegisterComputeShapeFn(kMicrosoftDomain, "SkipSimplifiedLayerNormalization",
                                      ComputeShapeSkipSimplifiedLayerNormalization);
    shapes_ns::RegisterComputeShapeFn(kMicrosoftDomain, "GroupQueryAttention",
                                      ComputeShapeGroupQueryAttention);
    shapes_ns::RegisterComputeShapeFn(kMicrosoftDomain, "MultiHeadAttention",
                                      ComputeShapeMultiHeadAttention);
    shapes_ns::RegisterComputeShapeFn(kMicrosoftDomain, "LinearAttention",
                                      ComputeShapeLinearAttention);
    shapes_ns::RegisterComputeShapeFn(kMicrosoftDomain, "MatMulNBits", ComputeShapeMatMulNBits,
                                      true);
    shapes_ns::RegisterComputeShapeFn(kMicrosoftDomain, "MoE", ComputeShapeMoE);
    shapes_ns::RegisterComputeShapeFn(kMicrosoftDomain, "QMoE", ComputeShapeQMoE);
    shapes_ns::RegisterComputeShapeFn(kMicrosoftDomain, "RotaryEmbedding",
                                      ComputeShapeRotaryEmbedding);
    shapes_ns::RegisterComputePeakMemoryFn(kMicrosoftDomain, "RotaryEmbedding",
                                           sym_ns::Device::kCPU, ComputePeakMemoryRotaryEmbedding);
    shapes_ns::RegisterComputePeakMemoryFn(kMicrosoftDomain, "CDist", sym_ns::Device::kCPU,
                                           ComputePeakMemoryCDist);
    shapes_ns::RegisterComputePeakMemoryFn(kMicrosoftDomain, "CausalConvWithState",
                                           sym_ns::Device::kCPU,
                                           ComputePeakMemoryCausalConvWithState);
    shapes_ns::RegisterComputePeakMemoryFn(kMicrosoftDomain, "BiasGelu", sym_ns::Device::kCPU,
                                           ComputePeakMemoryBiasGelu);
    shapes_ns::RegisterComputePeakMemoryFn(kMicrosoftDomain, "SkipLayerNormalization",
                                           sym_ns::Device::kCPU,
                                           ComputePeakMemorySkipLayerNormalization);
    shapes_ns::RegisterComputePeakMemoryFn(kMicrosoftDomain, "SkipSimplifiedLayerNormalization",
                                           sym_ns::Device::kCPU,
                                           ComputePeakMemorySkipSimplifiedLayerNormalization);
    shapes_ns::RegisterComputePeakMemoryFn(kMicrosoftDomain, "GroupQueryAttention",
                                           sym_ns::Device::kCPU,
                                           ComputePeakMemoryGroupQueryAttention);
    shapes_ns::RegisterComputePeakMemoryFn(kMicrosoftDomain, "MultiHeadAttention",
                                           sym_ns::Device::kCPU,
                                           ComputePeakMemoryMultiHeadAttention);
    shapes_ns::RegisterComputePeakMemoryFn(kMicrosoftDomain, "LinearAttention",
                                           sym_ns::Device::kCPU, ComputePeakMemoryLinearAttention);
    shapes_ns::RegisterComputePeakMemoryFn(kMicrosoftDomain, "MatMulNBits", sym_ns::Device::kCPU,
                                           ComputePeakMemoryMatMulNBits);
    shapes_ns::RegisterComputePeakMemoryFn(kMicrosoftDomain, "MoE", sym_ns::Device::kCPU,
                                           ComputePeakMemoryMoE);
    shapes_ns::RegisterComputePeakMemoryFn(kMicrosoftDomain, "QMoE", sym_ns::Device::kCPU,
                                           ComputePeakMemoryQMoE);
  });
}

} // namespace onnx_light_cpu
