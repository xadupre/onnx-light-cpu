// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu/kernels/com_microsoft/skip_simplified_layer_normalization_kernel.h"
#include "onnx_light_cpu/kernels/math/normalization_helpers.h"

#include "onnx_core/runtime/kernels/node_helpers.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace onnx_light_cpu::skip_simplified_layer_normalization {

namespace runtime = ONNX_LIGHT_NAMESPACE::core::runtime;

struct Dimensions {
  std::size_t rows;
  std::size_t width;
  std::size_t skip_rows;
};

inline void ValidateBuffer(const runtime::Tensor &tensor, const char *role) {
  if (tensor.data_type != runtime::DataType::FLOAT &&
      tensor.data_type != runtime::DataType::FLOAT16 &&
      tensor.data_type != runtime::DataType::BFLOAT16) {
    throw std::invalid_argument(std::string("SkipSimplifiedLayerNormalization: ") + role +
                                " must be FLOAT, FLOAT16, or BFLOAT16.");
  }
  for (const auto dim : tensor.shape) {
    if (dim < 0 || static_cast<std::uint64_t>(dim) > std::numeric_limits<std::size_t>::max()) {
      throw std::invalid_argument("SkipSimplifiedLayerNormalization: invalid tensor dimension.");
    }
  }
  const auto count = normalization::Product(tensor.shape, 0, tensor.shape.size(),
                                            "SkipSimplifiedLayerNormalization");
  const auto width = normalization::ElementSize(tensor.data_type);
  if (count > std::numeric_limits<std::size_t>::max() / width ||
      tensor.size_bytes() != count * width || (count != 0 && tensor.bytes() == nullptr)) {
    throw std::invalid_argument(std::string("SkipSimplifiedLayerNormalization: ") + role +
                                " buffer does not match its shape.");
  }
}

inline Dimensions Validate(const runtime::Tensor &input, const runtime::Tensor &skip,
                           const runtime::Tensor &gamma, const runtime::Tensor *bias,
                           float epsilon) {
  constexpr const char *op = "SkipSimplifiedLayerNormalization";
  if (!std::isfinite(epsilon) || epsilon < 0.0F) {
    throw std::invalid_argument(
        "SkipSimplifiedLayerNormalization: epsilon must be finite and nonnegative.");
  }
  ValidateBuffer(input, "input");
  ValidateBuffer(skip, "skip");
  ValidateBuffer(gamma, "gamma");
  normalization::RequireSameType(input, skip, op, "input and skip");
  normalization::RequireSameType(input, gamma, op, "input and gamma");
  if (input.shape.size() != 2 && input.shape.size() != 3) {
    throw std::invalid_argument("SkipSimplifiedLayerNormalization: input must have rank 2 or 3.");
  }
  if (input.shape.back() <= 0 || input.shape.back() > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(
        "SkipSimplifiedLayerNormalization: hidden size must be positive and fit int.");
  }
  const std::size_t width = static_cast<std::size_t>(input.shape.back());
  normalization::RequireVector(gamma, width, op, "gamma");
  const bool broadcast_skip =
      input.shape.size() == 3 &&
      ((skip.shape.size() == 2 && skip.shape[0] == input.shape[1] &&
        skip.shape[1] == input.shape[2]) ||
       (skip.shape.size() == 3 && skip.shape[0] == 1 && skip.shape[1] == input.shape[1] &&
        skip.shape[2] == input.shape[2]));
  if (skip.shape != input.shape && !broadcast_skip) {
    throw std::invalid_argument("SkipSimplifiedLayerNormalization: skip shape must match input or "
                                "broadcast over its batch dimension.");
  }
  if (bias != nullptr) {
    ValidateBuffer(*bias, "bias");
    normalization::RequireSameType(input, *bias, op, "input and bias");
    normalization::RequireVector(*bias, width, op, "bias");
  }
  const std::size_t rows = normalization::Product(input.shape, 0, input.shape.size() - 1, op);
  const std::size_t skip_rows = normalization::Product(skip.shape, 0, skip.shape.size() - 1, op);
  if (rows > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument("SkipSimplifiedLayerNormalization: row count must fit int64_t.");
  }
  return {rows, width, skip_rows};
}

inline SkipSimplifiedLayerNormalizationResult Allocate(const runtime::Tensor &input,
                                                       bool output_sum, bool output_mean,
                                                       bool output_inv_std_var,
                                                       runtime::RuntimeContext *rt) {
  SkipSimplifiedLayerNormalizationResult result;
  result.output = normalization::AllocateOutput(input.data_type, input.shape, 0, rt);
  if (output_sum) {
    result.input_skip_bias_sum = normalization::AllocateOutput(input.data_type, input.shape, 3, rt);
  }
  if (output_mean || output_inv_std_var) {
    auto shape = input.shape;
    shape.back() = 1;
    if (output_mean) {
      result.mean = normalization::AllocateOutput(runtime::DataType::FLOAT, shape, 1, rt);
    }
    if (output_inv_std_var) {
      result.inv_std_var = normalization::AllocateOutput(runtime::DataType::FLOAT, shape, 2, rt);
    }
  }
  return result;
}

template <runtime::DataType Type>
inline auto Load(const runtime::Tensor &tensor, std::size_t index) {
  normalization::StorageType<Type> value;
  std::memcpy(&value, tensor.bytes() + index * sizeof(value), sizeof(value));
  return normalization::TypeTraits<Type>::Load(&value, 0);
}

template <typename Kernel>
void Run(const Kernel &kernel, const ONNX_LIGHT_NAMESPACE::NodeProto &node,
         runtime::RuntimeContext &rt) {
  if (node.input_size() < 3 || node.input_size() > 4 || node.input(0).empty() ||
      node.input(1).empty() || node.input(2).empty()) {
    throw std::invalid_argument(
        "SkipSimplifiedLayerNormalization: expected input, skip, gamma and optional bias.");
  }
  if (node.output_size() < 1 || node.output_size() > 4 || node.output(0).empty()) {
    throw std::invalid_argument(
        "SkipSimplifiedLayerNormalization: expected output and up to three optional outputs.");
  }
  const bool output_mean = node.output_size() > 1 && !node.output(1).empty();
  const bool output_inv_std_var = node.output_size() > 2 && !node.output(2).empty();
  const bool output_sum = node.output_size() == 4 && !node.output(3).empty();
  const runtime::Tensor *bias = node.input_size() == 4 && !node.input(3).empty()
                                    ? &runtime::GetInput(node, 3, rt.tensors())
                                    : nullptr;
  auto result =
      kernel(runtime::GetInput(node, 0, rt.tensors()), runtime::GetInput(node, 1, rt.tensors()),
             runtime::GetInput(node, 2, rt.tensors()), bias,
             runtime::GetAttributeFloatOrDefault(node, "epsilon", 1.0e-12F), output_sum,
             output_mean, output_inv_std_var, &rt);
  runtime::SetOutput(node, 0, std::move(result.output), rt);
  if (output_mean) {
    runtime::SetOutput(node, 1, std::move(result.mean), rt);
  }
  if (output_inv_std_var) {
    runtime::SetOutput(node, 2, std::move(result.inv_std_var), rt);
  }
  if (output_sum) {
    runtime::SetOutput(node, 3, std::move(result.input_skip_bias_sum), rt);
  }
}

} // namespace onnx_light_cpu::skip_simplified_layer_normalization
