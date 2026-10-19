// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu_compute/com_microsoft/qmoe.h"

#include "onnx_light_cpu_compute/checked_arithmetic.h"
#include "onnx_light_cpu_compute/com_microsoft/moe_matvec.h"
#include "onnx_light_cpu_compute/execution.h"
#include "onnx_light_cpu_compute/math/half_conversion.h"
#include "onnx_light_cpu_compute/simd_level.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace onnx_light_cpu {
#if defined(ONNX_LIGHT_CPU_HAVE_AVX512)
void DequantizeQMoEAvx512(QMoEWeight, float *, std::size_t, std::size_t, std::size_t,
                          const QMoEAttributes &, std::size_t, std::size_t);
#endif

#if defined(ONNX_LIGHT_CPU_HAVE_AVX2)
void DequantizeQMoEAvx2(QMoEWeight, float *, std::size_t, std::size_t, std::size_t,
                        const QMoEAttributes &, std::size_t, std::size_t);
#endif

namespace {

using QMoEDequantizeRange = void (*)(QMoEWeight, float *, std::size_t, std::size_t, std::size_t,
                                     const QMoEAttributes &, std::size_t, std::size_t);

QMoEDequantizeRange SelectQMoEDequantizeRange() {
#if defined(ONNX_LIGHT_CPU_HAVE_AVX512)
  if (DetectSimdLevel() >= SimdLevel::kAVX512) {
    return DequantizeQMoEAvx512;
  }
#endif
#if defined(ONNX_LIGHT_CPU_HAVE_AVX2)
  if (DetectSimdLevel() >= SimdLevel::kAVX2) {
    return DequantizeQMoEAvx2;
  }
#endif
  return nullptr;
}

void ValidateType(MoEDataType type) {
  if (type != MoEDataType::kFloat32 && type != MoEDataType::kFloat16 &&
      type != MoEDataType::kBFloat16) {
    throw std::invalid_argument("QMoE: unsupported floating-point type.");
  }
}

float Load(const void *data, std::size_t index, MoEDataType type) noexcept {
  if (type == MoEDataType::kFloat32) {
    return static_cast<const float *>(data)[index];
  }
  const auto bits = static_cast<const std::uint16_t *>(data)[index];
  return type == MoEDataType::kFloat16 ? detail::Float16BitsToFloat(bits)
                                       : detail::Bfloat16BitsToFloat(bits);
}

void Validate(std::size_t tokens, std::size_t hidden, std::size_t experts, std::size_t intermediate,
              const QMoEAttributes &attributes, bool has_fc3) {
  if ((attributes.expert_weight_bits != 4 && attributes.expert_weight_bits != 8) || hidden == 0 ||
      experts == 0 || intermediate == 0 || attributes.k == 0 || attributes.k > experts ||
      (has_fc3 && attributes.activation == "swiglu")) {
    throw std::invalid_argument("QMoE: invalid dimensions, expert_weight_bits, k, or FC3 fusion.");
  }
  if (attributes.block_size != 0 &&
      (attributes.block_size < 16 || (attributes.block_size & (attributes.block_size - 1)) != 0)) {
    throw std::invalid_argument("QMoE: block_size must be zero or a power of two at least 16.");
  }
  if (attributes.activation != "relu" && attributes.activation != "gelu" &&
      attributes.activation != "silu" && attributes.activation != "identity" &&
      attributes.activation != "swiglu") {
    throw std::invalid_argument("QMoE: unsupported activation_type.");
  }
  if (!std::isfinite(attributes.alpha) || !std::isfinite(attributes.beta) ||
      std::isnan(attributes.limit) || attributes.limit < 0) {
    throw std::invalid_argument("QMoE: invalid activation parameters.");
  }
  if (tokens > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument("QMoE: token count overflows int64_t.");
  }
}

float Activation(float value, const QMoEAttributes &attributes) noexcept {
  if (attributes.activation == "relu") {
    return std::max(value, 0.0f);
  }
  if (attributes.activation == "gelu") {
    return 0.5f * value *
           (1.0f + std::tanh(0.7978845608f * (value + 0.044715f * value * value * value)));
  }
  if (attributes.activation == "silu") {
    return value / (1.0f + std::exp(-value));
  }
  return value;
}

} // namespace

QMoEScratchSize ComputeQMoEScratchSize(std::size_t tokens, std::size_t hidden, std::size_t experts,
                                       std::size_t intermediate, const QMoEAttributes &attributes,
                                       bool has_fc3) {
  Validate(tokens, hidden, experts, intermediate, attributes, has_fc3);
  if (tokens == 0) {
    return {};
  }
  const auto width =
      CheckedMultiply(intermediate, attributes.activation == "swiglu" ? 2 : 1, "QMoE", "FC1 width");
  QMoEScratchSize result;
  const auto add = [&](std::size_t count) {
    result.value_count = CheckedAdd(result.value_count, count, "QMoE", "scratch values");
  };
  add(CheckedProduct({tokens, hidden, 2}, "QMoE", "input and output"));
  add(CheckedMultiply(tokens, experts, "QMoE", "router"));
  add(CheckedProduct({experts, width, hidden}, "QMoE", "FC1"));
  add(CheckedProduct({experts, hidden, intermediate}, "QMoE", "FC2"));
  add(CheckedMultiply(experts, width, "QMoE", "FC1 bias"));
  add(CheckedMultiply(experts, hidden, "QMoE", "FC2 bias"));
  if (has_fc3) {
    add(CheckedProduct({experts, intermediate, hidden}, "QMoE", "FC3"));
    add(CheckedMultiply(experts, intermediate, "QMoE", "FC3 bias"));
    const auto workspace = CheckedAdd(CheckedMultiply(intermediate, 3, "QMoE", "FC3 workspace"),
                                      hidden, "QMoE", "FC3 workspace");
    const auto blocks = static_cast<std::size_t>(
        ExecutionBlockCount(static_cast<std::int64_t>(tokens), ExecutionSchedule{1, 1, 256, 0}));
    result.moe = {CheckedMultiply(blocks, experts, "QMoE", "routes"),
                  CheckedMultiply(blocks, workspace, "QMoE", "FC3 workspace"), blocks};
  } else {
    result.moe = ComputeMoEScratchSize(tokens, hidden, experts, intermediate, MoEDataType::kFloat32,
                                       attributes);
  }
  CheckedByteSize(result.value_count, sizeof(float), "QMoE", "scratch bytes");
  CheckedByteSize(result.moe.value_count, sizeof(float), "QMoE", "MoE scratch bytes");
  CheckedByteSize(result.moe.route_count, sizeof(MoERoute), "QMoE", "routing bytes");
  return result;
}

void DequantizeQMoEWeights(QMoEWeight weight, float *output, std::size_t experts,
                           std::size_t inputs, std::size_t outputs,
                           const QMoEAttributes &attributes) {
  ValidateType(weight.scale_type);
  if ((attributes.expert_weight_bits != 4 && attributes.expert_weight_bits != 8) ||
      weight.data == nullptr || weight.scales == nullptr || output == nullptr) {
    throw std::invalid_argument("QMoE: invalid quantized weights or expert_weight_bits.");
  }
  if (attributes.block_size != 0 &&
      (attributes.block_size < 16 || (attributes.block_size & (attributes.block_size - 1)) != 0)) {
    throw std::invalid_argument("QMoE: block_size must be zero or a power of two at least 16.");
  }
  const auto rows = weight.output_major ? outputs : inputs;
  const auto columns = weight.output_major ? inputs : outputs;
  const auto packed = columns / (8 / attributes.expert_weight_bits) +
                      (columns % (8 / attributes.expert_weight_bits) != 0);
  const auto blocks = attributes.block_size == 0 ? 0
                                                 : columns / attributes.block_size +
                                                       (columns % attributes.block_size != 0);
  const auto count = CheckedProduct({experts, inputs, outputs}, "QMoE", "dequantized weights");
  CheckedByteSize(count, sizeof(float), "QMoE", "dequantized bytes");
  CheckedProduct({experts, rows, packed}, "QMoE", "packed weights");
  CheckedProduct({experts, rows, blocks}, "QMoE", "block scales");
  CheckedMultiply(experts, columns, "QMoE", "tensor scales");
  if (count > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument("QMoE: weight count overflows int64_t.");
  }
  if (attributes.block_size != 0 || weight.rowwise_scales) {
    static const auto range = SelectQMoEDequantizeRange();
    if (range != nullptr) {
      const auto rows_per_expert = weight.output_major ? outputs : inputs;
      const auto columns = weight.output_major ? inputs : outputs;
      const auto row_count = CheckedMultiply(experts, rows_per_expert, "QMoE", "dequantized rows");
      ExecuteRanges(static_cast<std::int64_t>(row_count), 8.0 * static_cast<double>(columns),
                    [&](std::int64_t begin, std::int64_t end) {
                      range(weight, output, experts, inputs, outputs, attributes,
                            static_cast<std::size_t>(begin), static_cast<std::size_t>(end));
                    });
      return;
    }
  }
  ExecuteRanges(static_cast<std::int64_t>(count), 8.0, [&](std::int64_t begin, std::int64_t end) {
    for (auto index = begin; index < end; ++index) {
      const auto i = static_cast<std::size_t>(index);
      const auto expert = i / (inputs * outputs);
      const auto out = i / inputs % outputs;
      const auto in = i % inputs;
      const auto row = weight.output_major ? out : in;
      const auto column = weight.output_major ? in : out;
      const auto byte =
          weight
              .data[(expert * rows + row) * packed + column / (8 / attributes.expert_weight_bits)];
      const int quantized = attributes.expert_weight_bits == 4
                                ? static_cast<int>((byte >> (4 * (column % 2))) & 15) - 8
                            : weight.unsigned_int8 ? static_cast<int>(byte) - 128
                            : byte < 128           ? static_cast<int>(byte)
                                                   : static_cast<int>(byte) - 256;
      const auto scale =
          blocks == 0 ? (weight.rowwise_scales ? expert * rows + row : expert * columns + column)
                      : (expert * rows + row) * blocks + column / attributes.block_size;
      output[i] = static_cast<float>(quantized) * Load(weight.scales, scale, weight.scale_type);
    }
  });
}

void ComputeQMoE(const void *input, const void *router, QMoEWeight fc1, const void *bias1,
                 QMoEWeight fc2, const void *bias2, const QMoEWeight *fc3, const void *bias3,
                 void *output, std::size_t tokens, std::size_t hidden, std::size_t experts,
                 std::size_t intermediate, MoEDataType data_type, const QMoEAttributes &attributes,
                 float *values, std::size_t value_count, MoEScratch scratch) {
  ValidateType(data_type);
  if (fc3 == nullptr && bias3 != nullptr) {
    throw std::invalid_argument("QMoE: FC3 bias requires FC3 weights.");
  }
  const auto required =
      ComputeQMoEScratchSize(tokens, hidden, experts, intermediate, attributes, fc3 != nullptr);
  if (tokens == 0) {
    return;
  }
  if (value_count < required.value_count || values == nullptr ||
      scratch.route_count < required.moe.route_count ||
      scratch.value_count < required.moe.value_count ||
      scratch.token_blocks != required.moe.token_blocks ||
      (required.moe.route_count && scratch.routes == nullptr) ||
      (required.moe.value_count && scratch.values == nullptr)) {
    throw std::invalid_argument("QMoE: insufficient caller-provided scratch.");
  }
  if (input == nullptr || router == nullptr || output == nullptr) {
    throw std::invalid_argument("QMoE: null input, router, or output.");
  }
  const QMoEWeight *weights[] = {&fc1, &fc2, fc3};
  for (const auto *weight : weights) {
    if (weight) {
      ValidateType(weight->scale_type);
      if (weight->data == nullptr || weight->scales == nullptr) {
        throw std::invalid_argument("QMoE: null expert weights or scales.");
      }
    }
  }
  const auto width = intermediate * (attributes.activation == "swiglu" ? 2 : 1);
  float *cursor = values;
  const auto take = [&](std::size_t count) {
    float *result = cursor;
    cursor += count;
    return result;
  };
  float *x = take(tokens * hidden);
  float *r = take(tokens * experts);
  float *y = take(tokens * hidden);
  float *w1 = take(experts * width * hidden);
  float *w2 = take(experts * hidden * intermediate);
  float *b1 = take(experts * width);
  float *b2 = take(experts * hidden);
  float *w3 = fc3 ? take(experts * intermediate * hidden) : nullptr;
  float *b3 = fc3 ? take(experts * intermediate) : nullptr;
  const auto convert = [&](const void *source, float *destination, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
      destination[i] = source ? Load(source, i, data_type) : 0.0f;
    }
  };
  convert(input, x, tokens * hidden);
  convert(router, r, tokens * experts);
  convert(bias1, b1, experts * width);
  convert(bias2, b2, experts * hidden);
  DequantizeQMoEWeights(fc1, w1, experts, hidden, width, attributes);
  DequantizeQMoEWeights(fc2, w2, experts, intermediate, hidden, attributes);
  if (fc3 == nullptr) {
    ComputeMoE(x, r, w1, bias1 ? b1 : nullptr, w2, bias2 ? b2 : nullptr, y, tokens, hidden, experts,
               intermediate, MoEDataType::kFloat32, attributes, scratch);
  } else {
    DequantizeQMoEWeights(*fc3, w3, experts, hidden, intermediate, attributes);
    convert(bias3, b3, experts * intermediate);
    const auto workspace = 3 * intermediate + hidden;
    const auto blocks = required.moe.token_blocks;
    const auto grain = (tokens - 1) / blocks + 1;
    ExecuteRanges(
        static_cast<std::int64_t>(blocks), ExecutionSchedule{1, 1, 256, 0},
        [&](std::int64_t begin, std::int64_t end) {
          for (auto block = begin; block < end; ++block) {
            auto *routes = scratch.routes + block * experts;
            auto *first = scratch.values + block * workspace;
            auto *third = first + intermediate;
            auto *activated = third + intermediate;
            auto *second = activated + intermediate;
            const auto token_begin = std::min(tokens, static_cast<std::size_t>(block) * grain);
            const auto token_end = token_begin + std::min(grain, tokens - token_begin);
            for (auto token = token_begin; token < token_end; ++token) {
              const float *logits = r + token * experts;
              const auto maximum = *std::max_element(logits, logits + experts);
              float sum = 0;
              for (std::size_t e = 0; e < experts; ++e) {
                routes[e] = {std::exp(logits[e] - maximum), e};
                sum += routes[e].weight;
              }
              std::partial_sort(routes, routes + attributes.k, routes + experts,
                                [](const MoERoute &a, const MoERoute &b) {
                                  return a.weight > b.weight ||
                                         (a.weight == b.weight && a.expert > b.expert);
                                });
              if (attributes.normalize) {
                sum = 0;
                for (std::size_t slot = 0; slot < attributes.k; ++slot) {
                  sum += routes[slot].weight;
                }
              }
              float *result = y + token * hidden;
              std::fill_n(result, hidden, 0.0f);
              for (std::size_t slot = 0; slot < attributes.k; ++slot) {
                const auto expert = routes[slot].expert;
                const float weight = routes[slot].weight / sum;
                if (weight == 0.0f) {
                  continue;
                }
                detail::MoEMatVecFloat32(x + token * hidden, w1 + expert * intermediate * hidden,
                                         first, intermediate, hidden);
                detail::MoEMatVecFloat32(x + token * hidden, w3 + expert * intermediate * hidden,
                                         third, intermediate, hidden);
                for (std::size_t j = 0; j < intermediate; ++j) {
                  activated[j] = Activation(first[j] + b1[expert * intermediate + j], attributes) *
                                 (third[j] + b3[expert * intermediate + j]);
                }
                detail::MoEMatVecFloat32(activated, w2 + expert * hidden * intermediate, second,
                                         hidden, intermediate);
                for (std::size_t j = 0; j < hidden; ++j) {
                  result[j] += weight * (second[j] + b2[expert * hidden + j]);
                }
              }
            }
          }
        });
  }
  for (std::size_t i = 0; i < tokens * hidden; ++i) {
    if (data_type == MoEDataType::kFloat32) {
      static_cast<float *>(output)[i] = y[i];
    } else {
      static_cast<std::uint16_t *>(output)[i] = data_type == MoEDataType::kFloat16
                                                    ? detail::FloatToFloat16Bits(y[i])
                                                    : detail::FloatToBFloat16Bits(y[i]);
    }
  }
}

} // namespace onnx_light_cpu
