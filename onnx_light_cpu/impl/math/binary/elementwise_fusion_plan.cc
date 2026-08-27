// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/binary/elementwise_fusion_plan.h"

#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/impl/math/math_kernels.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>

namespace onnx_light_cpu {
namespace {

constexpr std::size_t kBlockSize = 256;
constexpr ExecutionSchedule kFusionSchedule{1 << 14, 1 << 12, 4};

bool HasExclusiveIntermediates(const ElementwiseFusionGuards &guards, std::size_t count) {
  return guards.has_static_shapes && guards.numerical_contract_matches &&
         !guards.has_unsafe_alias && guards.intermediate_consumer_counts.size() == count &&
         std::all_of(guards.intermediate_consumer_counts.begin(),
                     guards.intermediate_consumer_counts.end(),
                     [](std::size_t consumers) { return consumers == 1; });
}

bool IsShape(std::span<const std::int64_t> shape, std::initializer_list<std::int64_t> expected) {
  return std::equal(shape.begin(), shape.end(), expected.begin(), expected.end());
}

bool SameShape(std::span<const std::int64_t> left, std::span<const std::int64_t> right) {
  return left.size() == right.size() && std::equal(left.begin(), left.end(), right.begin());
}

bool SupportedSwiGLUShape(std::span<const std::int64_t> shape) {
  if (shape.size() != 3 || shape[0] != 1) {
    return false;
  }
  return (shape[1] == 1 || shape[1] == 32 || shape[1] == 512) &&
         (shape[2] == 3072 || shape[2] == 9728);
}

bool SupportedScoresShape(std::span<const std::int64_t> shape) {
  if (shape.size() != 4 || shape[0] != 1 || (shape[1] != 16 && shape[1] != 32)) {
    return false;
  }
  return (shape[2] == 1 && shape[3] == 128) || (shape[2] == 32 && shape[3] == 1024) ||
         (shape[2] == 128 && shape[3] == 4096);
}

void CheckPointers(std::span<const void *const> inputs, void *output) {
  if (output == nullptr || std::any_of(inputs.begin(), inputs.end(),
                                       [](const void *input) { return input == nullptr; })) {
    throw std::invalid_argument(
        "onnx_light_cpu::ElementwiseFusionPlan: non-empty tensors require data.");
  }
  if (std::any_of(inputs.begin(), inputs.end(),
                  [output](const void *input) { return input == output; })) {
    throw std::invalid_argument(
        "onnx_light_cpu::ElementwiseFusionPlan: fused output must not alias an input.");
  }
}

void ExecuteSwiGLUFloat(const float *gate, const float *up, float *output, std::size_t begin,
                        std::size_t end) {
  alignas(64) std::array<float, kBlockSize> exponent;
  for (std::size_t offset = begin; offset < end; offset += kBlockSize) {
    const std::size_t count = std::min(kBlockSize, end - offset);
    for (std::size_t i = 0; i < count; ++i) {
      exponent[i] = -gate[offset + i];
    }
    ExpFloat32(exponent.data(), exponent.data(), count);
    for (std::size_t i = 0; i < count; ++i) {
      const float sigmoid = 1.0f / (1.0f + exponent[i]);
      const float inner = gate[offset + i] * sigmoid;
      output[offset + i] = inner * up[offset + i];
    }
  }
}

void ExecuteSwiGLUBFloat16(const std::uint16_t *gate, const std::uint16_t *up,
                           std::uint16_t *output, std::size_t begin, std::size_t end) {
  alignas(64) std::array<float, kBlockSize> gate_float;
  alignas(64) std::array<float, kBlockSize> exponent;
  for (std::size_t offset = begin; offset < end; offset += kBlockSize) {
    const std::size_t count = std::min(kBlockSize, end - offset);
    detail::ConvertBFloat16ToFloat32(gate + offset, gate_float.data(), count);
    for (std::size_t i = 0; i < count; ++i) {
      exponent[i] = -gate_float[i];
    }
    ExpFloat32(exponent.data(), exponent.data(), count);
    for (std::size_t i = 0; i < count; ++i) {
      const std::uint16_t sigmoid_bits = detail::FloatToBFloat16Bits(1.0f / (1.0f + exponent[i]));
      const float sigmoid = detail::Bfloat16BitsToFloat(sigmoid_bits);
      const std::uint16_t inner_bits = detail::FloatToBFloat16Bits(gate_float[i] * sigmoid);
      const float inner = detail::Bfloat16BitsToFloat(inner_bits);
      const float up_value = detail::Bfloat16BitsToFloat(up[offset + i]);
      output[offset + i] = detail::FloatToBFloat16Bits(inner * up_value);
    }
  }
}

} // namespace

ElementwiseFusionPlan::ElementwiseFusionPlan(ElementwiseFusionTemplate fusion_template,
                                             DataType data_type,
                                             std::span<const std::int64_t> output_shape)
    : fusion_template_(fusion_template), data_type_(data_type),
      output_shape_(output_shape.begin(), output_shape.end()), element_count_(1),
      inner_elements_(1) {
  for (std::int64_t dimension : output_shape_) {
    element_count_ *= static_cast<std::size_t>(dimension);
  }
  if (fusion_template_ == ElementwiseFusionTemplate::kScaledMaskedScores) {
    inner_elements_ = static_cast<std::size_t>(output_shape_[2] * output_shape_[3]);
  }
}

std::optional<ElementwiseFusionPlan> ElementwiseFusionPlan::TryCreateSwiGLUGate(
    DataType data_type, std::span<const std::int64_t> gate_shape,
    std::span<const std::int64_t> sigmoid_shape, std::span<const std::int64_t> inner_mul_shape,
    std::span<const std::int64_t> up_shape, std::span<const std::int64_t> output_shape,
    const ElementwiseFusionGuards &guards) {
  if ((data_type != DataType::FLOAT && data_type != DataType::BFLOAT16) ||
      !HasExclusiveIntermediates(guards, 2) || !SupportedSwiGLUShape(gate_shape) ||
      !SameShape(gate_shape, sigmoid_shape) || !SameShape(gate_shape, inner_mul_shape) ||
      !SameShape(gate_shape, up_shape) || !SameShape(gate_shape, output_shape)) {
    return std::nullopt;
  }
  return ElementwiseFusionPlan(ElementwiseFusionTemplate::kSwiGLUGate, data_type, output_shape);
}

std::optional<ElementwiseFusionPlan> ElementwiseFusionPlan::TryCreateScaledMaskedScores(
    DataType data_type, std::span<const std::int64_t> scores_shape,
    std::span<const std::int64_t> scale_shape, std::span<const std::int64_t> mask_shape,
    std::span<const std::int64_t> mul_shape, std::span<const std::int64_t> output_shape,
    const ElementwiseFusionGuards &guards) {
  if (data_type != DataType::FLOAT || !HasExclusiveIntermediates(guards, 1) ||
      !SupportedScoresShape(scores_shape) || !scale_shape.empty() ||
      !SameShape(scores_shape, mul_shape) || !SameShape(scores_shape, output_shape) ||
      !IsShape(mask_shape, {1, 1, scores_shape[2], scores_shape[3]})) {
    return std::nullopt;
  }
  return ElementwiseFusionPlan(ElementwiseFusionTemplate::kScaledMaskedScores, data_type,
                               output_shape);
}

void ElementwiseFusionPlan::Execute(const void *first, const void *second, const void *third,
                                    void *output) const {
  if (fusion_template_ == ElementwiseFusionTemplate::kSwiGLUGate) {
    const std::array<const void *, 2> inputs{first, second};
    CheckPointers(inputs, output);
    ExecuteRanges(
        static_cast<std::int64_t>(element_count_), kFusionSchedule, ExecutionSimdLanes<float>(),
        [&](std::int64_t begin, std::int64_t end) {
          if (data_type_ == DataType::FLOAT) {
            ExecuteSwiGLUFloat(static_cast<const float *>(first),
                               static_cast<const float *>(second), static_cast<float *>(output),
                               static_cast<std::size_t>(begin), static_cast<std::size_t>(end));
          } else {
            ExecuteSwiGLUBFloat16(static_cast<const std::uint16_t *>(first),
                                  static_cast<const std::uint16_t *>(second),
                                  static_cast<std::uint16_t *>(output),
                                  static_cast<std::size_t>(begin), static_cast<std::size_t>(end));
          }
        });
    return;
  }

  const std::array<const void *, 3> inputs{first, second, third};
  CheckPointers(inputs, output);
  const auto *scores = static_cast<const float *>(first);
  const float scale = *static_cast<const float *>(second);
  const auto *mask = static_cast<const float *>(third);
  auto *out = static_cast<float *>(output);
  ExecuteRanges(static_cast<std::int64_t>(element_count_), kFusionSchedule,
                static_cast<std::int64_t>(inner_elements_),
                [&](std::int64_t begin, std::int64_t end) {
                  for (std::size_t offset = static_cast<std::size_t>(begin);
                       offset < static_cast<std::size_t>(end); offset += inner_elements_) {
                    for (std::size_t i = 0; i < inner_elements_; ++i) {
                      const float scaled = scores[offset + i] * scale;
                      out[offset + i] = scaled + mask[i];
                    }
                  }
                });
}

const char *ToString(ElementwiseFusionTemplate fusion_template) noexcept {
  switch (fusion_template) {
  case ElementwiseFusionTemplate::kSwiGLUGate:
    return "swiglu_gate";
  case ElementwiseFusionTemplate::kScaledMaskedScores:
    return "scaled_masked_scores";
  }
  return "elementwise_fusion";
}

} // namespace onnx_light_cpu
