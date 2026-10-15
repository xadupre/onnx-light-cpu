// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu/impl/checked_arithmetic.h"

#include "onnx_proto/onnx.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace onnx_light_cpu {

inline constexpr std::string_view kMatMulNBitsInt3Format =
    "onnx_light.quantization.v1/matmulnbits_int3";

struct MatMulNBitsInt3Layout {
  std::int64_t k;
  std::int64_t n;
  std::int32_t data_type;
  std::size_t blocks;
  std::size_t weight_bytes;
  std::size_t scale_bytes;
  bool has_zero_points;
};

inline MatMulNBitsInt3Layout
InspectMatMulNBitsInt3(const ONNX_LIGHT_NAMESPACE::EncodedValueProto &encoded,
                       const ONNX_LIGHT_NAMESPACE::StructTypeProto &root) {
  using ONNX_LIGHT_NAMESPACE::TensorProto;
  if (root.name().value() != kMatMulNBitsInt3Format || !root.has_structure() ||
      (root.structure().field().size() != 3 && root.structure().field().size() != 4) ||
      !encoded.has_logical_type() || !encoded.logical_type().has_tensor_type()) {
    throw std::invalid_argument("MatMulNBits INT3: invalid encoded layout.");
  }
  const auto &logical = encoded.logical_type().tensor_type();
  if (!logical.has_shape() || logical.shape().dim().size() != 2 ||
      !logical.shape().dim(0).has_dim_value() || !logical.shape().dim(1).has_dim_value() ||
      logical.shape().dim(0).dim_value() <= 0 || logical.shape().dim(1).dim_value() <= 0 ||
      (logical.elem_type() != TensorProto::FLOAT && logical.elem_type() != TensorProto::FLOAT16 &&
       logical.elem_type() != TensorProto::BFLOAT16)) {
    throw std::invalid_argument("MatMulNBits INT3: invalid logical weight type or shape.");
  }
  const auto &fields = root.structure().field();
  const auto &parameters = fields[0];
  if (parameters.name() != "parameters" || !parameters.has_constant() ||
      parameters.constant().data_type() != TensorProto::INT64 ||
      parameters.constant().dims().size() != 1 || parameters.constant().dims(0) != 2 ||
      parameters.constant().int64_data().size() != 2 || parameters.constant().int64_data(0) != 3 ||
      parameters.constant().int64_data(1) != 32) {
    throw std::invalid_argument("MatMulNBits INT3: expected parameters [3, 32].");
  }
  const auto k = logical.shape().dim(0).dim_value();
  const auto n = logical.shape().dim(1).dim_value();
  const auto blocks = static_cast<std::size_t>(k / 32 + (k % 32 != 0));
  const auto count =
      CheckedMultiply(static_cast<std::size_t>(n), blocks, "MatMulNBits INT3", "block count");
  const auto matches = [n, blocks](const auto &field, std::string_view name, int data_type,
                                   std::int64_t third = -1) {
    if (field.name() != name || !field.has_type() || !field.type().has_tensor_type()) {
      return false;
    }
    const auto &tensor = field.type().tensor_type();
    if (!tensor.has_shape()) {
      return false;
    }
    const auto &dims = tensor.shape().dim();
    return tensor.elem_type() == data_type && dims.size() == (third < 0 ? 2U : 3U) &&
           dims[0].has_dim_value() && dims[0].dim_value() == n && dims[1].has_dim_value() &&
           dims[1].dim_value() == static_cast<std::int64_t>(blocks) &&
           (third < 0 || (dims[2].has_dim_value() && dims[2].dim_value() == third));
  };
  if (!matches(fields[1], "B", TensorProto::UINT8, 12) ||
      !matches(fields[2], "scales", logical.elem_type()) ||
      (fields.size() == 4 && !matches(fields[3], "zero_points", TensorProto::UINT8))) {
    throw std::invalid_argument("MatMulNBits INT3: invalid field types or dimensions.");
  }
  const auto weight_bytes = CheckedMultiply(count, 12, "MatMulNBits INT3", "weight bytes");
  const auto scale_bytes = CheckedMultiply(count, logical.elem_type() == TensorProto::FLOAT ? 4 : 2,
                                           "MatMulNBits INT3", "scale bytes");
  const auto total =
      CheckedAdd(CheckedAdd(weight_bytes, scale_bytes, "MatMulNBits INT3", "payload bytes"),
                 fields.size() == 4 ? count : 0, "MatMulNBits INT3", "payload bytes");
  if (encoded.has_parameter_ref() || encoded.external_data().size() != 0 ||
      (encoded.has_data_location() && encoded.data_location() != TensorProto::DEFAULT) ||
      encoded.raw_data().size() != total) {
    throw std::invalid_argument("MatMulNBits INT3: invalid payload size or location.");
  }
  const auto &raw = encoded.raw_data();
  const auto *data = reinterpret_cast<const std::uint8_t *>(raw.data());
  for (std::size_t column = 0; column < static_cast<std::size_t>(n); ++column) {
    for (std::size_t block = 0; block < blocks; ++block) {
      const std::size_t valid = std::min<std::size_t>(32, static_cast<std::size_t>(k) - block * 32);
      const auto *bytes = data + (column * blocks + block) * 12;
      for (std::size_t bit = valid * 3; bit < 96; ++bit) {
        if ((bytes[bit / 8] >> (bit % 8)) & 1U) {
          throw std::invalid_argument("MatMulNBits INT3: nonzero padding bits.");
        }
      }
    }
  }
  if (fields.size() == 4) {
    for (std::size_t i = 0; i < count; ++i) {
      if (data[weight_bytes + scale_bytes + i] > 7) {
        throw std::invalid_argument("MatMulNBits INT3: zero point must be in [0, 7].");
      }
    }
  }
  return {k, n, logical.elem_type(), blocks, weight_bytes, scale_bytes, fields.size() == 4};
}

} // namespace onnx_light_cpu
