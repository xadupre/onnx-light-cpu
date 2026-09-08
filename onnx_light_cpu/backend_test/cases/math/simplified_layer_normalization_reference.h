// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"
#include "onnx_extensions/kernels/kernels/reduction/include_reduction_kernels.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include <algorithm>
#include <vector>

namespace onnx_light_cpu::backend_test {

/// Independent oracle assembled from concrete onnx-light kernels, never CPU dispatch.
inline ONNX_LIGHT_NAMESPACE::core::runtime::Tensors ReferenceSimplifiedLayerNormalization(
    const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &x,
    const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &scale, std::int64_t axis, float epsilon,
    std::int64_t stash_type, bool output_inverse,
    const ONNX_LIGHT_NAMESPACE::core::runtime::KernelContext &ctx) {
  namespace rt = ONNX_LIGHT_NAMESPACE::core::runtime;
  namespace builtin = ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel;
  const builtin::Cast cast(ctx);
  const builtin::Mul mul(ctx);
  const builtin::Add add(ctx);
  const builtin::ReduceMean mean(ctx);
  const builtin::ReduceSum sum(ctx);
  const builtin::Div div(ctx);
  const builtin::Sqrt sqrt(ctx);
  const builtin::Reciprocal reciprocal(ctx);
  if (axis < 0) {
    axis += static_cast<std::int64_t>(x.shape.size());
  }
  const bool suffix_scale =
      scale.shape.size() == x.shape.size() - static_cast<std::size_t>(axis) &&
      std::equal(scale.shape.begin(), scale.shape.end(), x.shape.begin() + axis);
  const auto work_type =
      static_cast<std::int32_t>(x.data_type == rt::DataType::DOUBLE ||
                                        scale.data_type == rt::DataType::DOUBLE || !suffix_scale
                                    ? rt::DataType::DOUBLE
                                    : rt::DataType::FLOAT);
  std::vector<std::int64_t> axes;
  for (std::int64_t i = axis; i < static_cast<std::int64_t>(x.shape.size()); ++i) {
    axes.push_back(i);
  }
  const auto axes_tensor =
      rt::Tensor::FromInt64("axes", {static_cast<std::int64_t>(axes.size())}, axes);
  const auto input = cast(x, work_type);
  const auto epsilon_tensor = cast(rt::Tensor::FromFloat("epsilon", {}, {epsilon}), work_type);
  const auto squares = mul(input, input);
  // The builtin ReduceMean is FLOAT-only; ReduceSum also implements DOUBLE.
  const auto mean_square =
      work_type == static_cast<std::int32_t>(rt::DataType::DOUBLE)
          ? div(sum(squares, axes_tensor),
                cast(rt::Tensor::FromInt64("width", {},
                                           {x.shape.product(static_cast<std::size_t>(axis),
                                                            x.shape.size(), "reference width")}),
                     work_type))
          : mean(squares, axes_tensor);
  const auto inverse = reciprocal(sqrt(add(mean_square, epsilon_tensor)));
  rt::Tensors outputs{cast(mul(mul(input, inverse), cast(scale, work_type)), scale.data_type)};
  if (output_inverse) {
    outputs.push_back(cast(inverse, static_cast<std::int32_t>(stash_type)));
  }
  return outputs;
}

} // namespace onnx_light_cpu::backend_test
