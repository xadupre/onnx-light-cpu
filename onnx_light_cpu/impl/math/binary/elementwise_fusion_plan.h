// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu/impl/math/binary/binary_manifest.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace onnx_light_cpu {

enum class ElementwiseFusionTemplate : std::uint8_t {
  kSwiGLUGate,
  kScaledMaskedScores,
};

struct ElementwiseFusionGuards {
  std::span<const std::size_t> intermediate_consumer_counts;
  bool has_unsafe_alias = false;
  bool numerical_contract_matches = true;
  bool has_static_shapes = true;
};

class ElementwiseFusionPlan {
public:
  static std::optional<ElementwiseFusionPlan> TryCreateSwiGLUGate(
      DataType data_type, std::span<const std::int64_t> gate_shape,
      std::span<const std::int64_t> sigmoid_shape, std::span<const std::int64_t> inner_mul_shape,
      std::span<const std::int64_t> up_shape, std::span<const std::int64_t> output_shape,
      const ElementwiseFusionGuards &guards);

  static std::optional<ElementwiseFusionPlan> TryCreateScaledMaskedScores(
      DataType data_type, std::span<const std::int64_t> scores_shape,
      std::span<const std::int64_t> scale_shape, std::span<const std::int64_t> mask_shape,
      std::span<const std::int64_t> mul_shape, std::span<const std::int64_t> output_shape,
      const ElementwiseFusionGuards &guards);

  ElementwiseFusionTemplate fusion_template() const noexcept { return fusion_template_; }
  DataType data_type() const noexcept { return data_type_; }
  std::span<const std::int64_t> output_shape() const noexcept { return output_shape_; }
  std::size_t element_count() const noexcept { return element_count_; }
  std::size_t workspace_bytes() const noexcept { return 0; }

  void Execute(const void *first, const void *second, const void *third, void *output) const;

private:
  ElementwiseFusionPlan(ElementwiseFusionTemplate fusion_template, DataType data_type,
                        std::span<const std::int64_t> output_shape);

  ElementwiseFusionTemplate fusion_template_;
  DataType data_type_;
  std::vector<std::int64_t> output_shape_;
  std::size_t element_count_;
  std::size_t inner_elements_;
};

const char *ToString(ElementwiseFusionTemplate fusion_template) noexcept;

} // namespace onnx_light_cpu
