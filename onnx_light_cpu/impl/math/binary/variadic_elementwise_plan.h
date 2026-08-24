// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu/impl/math/binary/binary_manifest.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace onnx_light_cpu {

enum class VariadicOperator : std::uint8_t {
  kSum,
  kMean,
  kMin,
  kMax,
};

class VariadicElementwisePlan {
public:
  VariadicElementwisePlan(VariadicOperator op, std::span<const BinaryDataType> input_types,
                          std::span<const std::vector<std::int64_t>> input_shapes);

  VariadicOperator op() const noexcept { return op_; }
  BinaryDataType data_type() const noexcept { return data_type_; }
  std::span<const std::int64_t> output_shape() const noexcept { return output_shape_; }
  std::size_t element_count() const noexcept { return element_count_; }
  std::size_t input_count() const noexcept { return input_count_; }
  std::size_t element_size() const noexcept { return element_size_; }
  std::size_t workspace_bytes() const noexcept { return 0; }

  void Execute(std::span<const void *const> inputs, void *output) const;

private:
  VariadicOperator op_;
  BinaryDataType data_type_;
  std::size_t input_count_;
  std::size_t element_size_;
  std::size_t element_count_;
  std::vector<std::int64_t> output_shape_;
  std::vector<std::size_t> input_strides_;
};

const char *ToString(VariadicOperator op) noexcept;

} // namespace onnx_light_cpu
