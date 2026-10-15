// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/com_microsoft/moe.h"
#include "onnx_light_cpu/impl/com_microsoft/moe_matvec.h"

#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace onnx_light_cpu {
namespace {

template <typename T> float Read(const T *data, std::size_t index) {
  if constexpr (sizeof(T) == sizeof(float)) {
    return data[index];
  } else {
    return detail::Float16BitsToFloat(data[index]);
  }
}

template <typename T> T Narrow(float value) {
  if constexpr (sizeof(T) == sizeof(float)) {
    return value;
  } else {
    return detail::FloatToFloat16Bits(value);
  }
}

struct Float16Codec {
  static float Load(std::uint16_t value) noexcept { return detail::Float16BitsToFloat(value); }
  static void Convert(const std::uint16_t *source, float *destination, std::size_t count) {
    detail::ConvertFloat16ToFloat32(source, destination, count);
  }
  static void Store(const float *source, std::uint16_t *destination, std::size_t count) {
    detail::ConvertFloat32ToFloat16(source, destination, count);
  }
};

struct BFloat16Codec {
  static float Load(std::uint16_t value) noexcept { return detail::Bfloat16BitsToFloat(value); }
  static void Convert(const std::uint16_t *source, float *destination, std::size_t count) {
    detail::ConvertBFloat16ToFloat32(source, destination, count);
  }
  static void Store(const float *source, std::uint16_t *destination, std::size_t count) {
    detail::ConvertFloat32ToBFloat16(source, destination, count);
  }
};

template <typename Codec>
void LowPrecisionMatVec(const std::uint16_t *input, const std::uint16_t *weights, float *output,
                        std::size_t rows, std::size_t columns) {
  if constexpr (std::is_same_v<Codec, Float16Codec>) {
    detail::MoEMatVecFloat16(input, weights, output, rows, columns);
  } else {
    detail::MoEMatVecBFloat16(input, weights, output, rows, columns);
  }
}

template <typename Codec>
void MixedPrecisionMatVec(const float *input, const std::uint16_t *weights, float *output,
                          std::size_t rows, std::size_t columns) {
  if constexpr (std::is_same_v<Codec, Float16Codec>) {
    detail::MoEMatVecFloat32Float16(input, weights, output, rows, columns);
  } else {
    detail::MoEMatVecFloat32BFloat16(input, weights, output, rows, columns);
  }
}

float Activate(float value, const MoEAttributes &attributes) {
  if (attributes.activation == "relu") {
    return std::max(0.0f, value);
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

template <typename T>
void ComputeRange(const T *input, const T *router, const T *fc1, const T *bias1, const T *fc2,
                  const T *bias2, T *output, std::size_t begin, std::size_t end, std::size_t hidden,
                  std::size_t experts, std::size_t intermediate, const MoEAttributes &attributes) {
  const bool swiglu = attributes.activation == "swiglu";
  const std::size_t fc1_width = intermediate * (swiglu ? 2 : 1);
  std::vector<std::pair<float, std::size_t>> routes(experts);
  std::vector<float> result(hidden);
  for (std::size_t token = begin; token < end; ++token) {
    float max_logit = Read(router, token * experts);
    for (std::size_t expert = 1; expert < experts; ++expert) {
      max_logit = std::max(max_logit, Read(router, token * experts + expert));
    }
    float denominator = 0.0f;
    for (std::size_t expert = 0; expert < experts; ++expert) {
      const float weight = std::exp(Read(router, token * experts + expert) - max_logit);
      routes[expert] = {weight, expert};
      denominator += weight;
    }
    for (auto &route : routes) {
      route.first /= denominator;
    }
    std::partial_sort(routes.begin(), routes.begin() + attributes.k, routes.end(),
                      std::greater<>());
    float top_sum = 0.0f;
    for (std::size_t slot = 0; slot < attributes.k; ++slot) {
      top_sum += routes[slot].first;
    }
    std::fill(result.begin(), result.end(), 0.0f);
    const bool parallel_experts =
        attributes.k > 1 && fc1_width * hidden >= 512 * 1024 && !ExecutionInParallelRegion();
    std::vector<float> expert_results(parallel_experts ? attributes.k * hidden : 0);
    const auto compute_slots = [&](std::size_t slot_begin, std::size_t slot_end,
                                   float *disjoint_results) {
      std::vector<float> first(fc1_width), activated(intermediate), second(hidden);
      for (std::size_t slot = slot_begin; slot < slot_end; ++slot) {
        const auto [weight, expert] = routes[slot];
        if (weight == 0.0f) {
          continue;
        }
        detail::MoEMatVecFloat32(input + token * hidden, fc1 + expert * fc1_width * hidden,
                                 first.data(), fc1_width, hidden);
        if (bias1 != nullptr) {
          for (std::size_t row = 0; row < fc1_width; ++row) {
            first[row] += bias1[expert * fc1_width + row];
          }
        }
        for (std::size_t row = 0; row < intermediate; ++row) {
          float value;
          if (swiglu) {
            const float gate = std::min(first[2 * row], attributes.limit);
            const float linear =
                std::clamp(first[2 * row + 1], -attributes.limit, attributes.limit);
            const float arg = attributes.alpha * gate;
            const float sigmoid =
                arg > 0 ? 1.0f / (1.0f + std::exp(-arg)) : std::exp(arg) / (1.0f + std::exp(arg));
            value = gate * sigmoid * (linear + attributes.beta);
          } else {
            value = Activate(first[row], attributes);
          }
          activated[row] = value;
        }
        detail::MoEMatVecFloat32(activated.data(), fc2 + expert * hidden * intermediate,
                                 second.data(), hidden, intermediate);
        const float route_weight = attributes.normalize ? weight / top_sum : weight;
        for (std::size_t row = 0; row < hidden; ++row) {
          const float value =
              second[row] + (bias2 != nullptr ? bias2[expert * hidden + row] : 0.0f);
          if (disjoint_results != nullptr) {
            disjoint_results[slot * hidden + row] = route_weight * value;
          } else {
            result[row] += route_weight * value;
          }
        }
      }
    };
    if (parallel_experts) {
      const double expert_cost =
          static_cast<double>(fc1_width) * hidden + static_cast<double>(hidden) * intermediate;
      const ExecutionWorkCost cost{sizeof(T) * expert_cost,
                                   static_cast<double>(sizeof(float) * hidden), expert_cost};
      const ExecutionSchedule schedule{2, 1, std::numeric_limits<std::int64_t>::max(), 0};
      ExecuteCostedRanges(static_cast<std::int64_t>(attributes.k), cost, schedule, std::int64_t{1},
                          [&](std::int64_t slot_begin, std::int64_t slot_end) {
                            compute_slots(static_cast<std::size_t>(slot_begin),
                                          static_cast<std::size_t>(slot_end),
                                          expert_results.data());
                          });
      for (std::size_t slot = 0; slot < attributes.k; ++slot) {
        for (std::size_t row = 0; row < hidden; ++row) {
          result[row] += expert_results[slot * hidden + row];
        }
      }
    } else {
      compute_slots(0, attributes.k, nullptr);
    }
    for (std::size_t row = 0; row < hidden; ++row) {
      output[token * hidden + row] = Narrow<T>(result[row]);
    }
  }
}

template <typename Codec>
void ComputeLowPrecisionRange(const std::uint16_t *input, const std::uint16_t *router,
                              const std::uint16_t *fc1, const std::uint16_t *bias1,
                              const std::uint16_t *fc2, const std::uint16_t *bias2,
                              std::uint16_t *output, std::size_t begin, std::size_t end,
                              std::size_t hidden, std::size_t experts, std::size_t intermediate,
                              const MoEAttributes &attributes) {
  const bool swiglu = attributes.activation == "swiglu";
  const std::size_t fc1_width = intermediate * (swiglu ? 2 : 1);
  std::vector<std::pair<float, std::size_t>> routes(experts);
  std::vector<float> router_values(experts), result(hidden);
  for (std::size_t token = begin; token < end; ++token) {
    Codec::Convert(router + token * experts, router_values.data(), experts);
    const float max_logit = *std::max_element(router_values.begin(), router_values.end());
    float denominator = 0.0f;
    for (std::size_t expert = 0; expert < experts; ++expert) {
      const float weight = std::exp(router_values[expert] - max_logit);
      routes[expert] = {weight, expert};
      denominator += weight;
    }
    for (auto &route : routes) {
      route.first /= denominator;
    }
    std::partial_sort(routes.begin(), routes.begin() + attributes.k, routes.end(),
                      std::greater<>());
    float top_sum = 0.0f;
    for (std::size_t slot = 0; slot < attributes.k; ++slot) {
      top_sum += routes[slot].first;
    }
    std::fill(result.begin(), result.end(), 0.0f);
    const bool parallel_experts =
        attributes.k > 1 && fc1_width * hidden >= 512 * 1024 && !ExecutionInParallelRegion();
    std::vector<float> expert_results(parallel_experts ? attributes.k * hidden : 0);
    const auto compute_slots = [&](std::size_t slot_begin, std::size_t slot_end,
                                   float *disjoint_results) {
      std::vector<float> first(fc1_width), activated(intermediate), second(hidden);
      for (std::size_t slot = slot_begin; slot < slot_end; ++slot) {
        const auto [weight, expert] = routes[slot];
        if (weight == 0.0f) {
          continue;
        }
        LowPrecisionMatVec<Codec>(input + token * hidden, fc1 + expert * fc1_width * hidden,
                                  first.data(), fc1_width, hidden);
        if (bias1 != nullptr) {
          for (std::size_t row = 0; row < fc1_width; ++row) {
            first[row] += Codec::Load(bias1[expert * fc1_width + row]);
          }
        }
        for (std::size_t row = 0; row < intermediate; ++row) {
          if (swiglu) {
            const float gate = std::min(first[2 * row], attributes.limit);
            const float linear =
                std::clamp(first[2 * row + 1], -attributes.limit, attributes.limit);
            const float arg = attributes.alpha * gate;
            const float sigmoid =
                arg > 0 ? 1.0f / (1.0f + std::exp(-arg)) : std::exp(arg) / (1.0f + std::exp(arg));
            activated[row] = gate * sigmoid * (linear + attributes.beta);
          } else {
            activated[row] = Activate(first[row], attributes);
          }
        }
        MixedPrecisionMatVec<Codec>(activated.data(), fc2 + expert * hidden * intermediate,
                                    second.data(), hidden, intermediate);
        const float route_weight = attributes.normalize ? weight / top_sum : weight;
        for (std::size_t row = 0; row < hidden; ++row) {
          const float value =
              second[row] + (bias2 != nullptr ? Codec::Load(bias2[expert * hidden + row]) : 0.0f);
          if (disjoint_results != nullptr) {
            disjoint_results[slot * hidden + row] = route_weight * value;
          } else {
            result[row] += route_weight * value;
          }
        }
      }
    };
    if (parallel_experts) {
      const double expert_cost =
          static_cast<double>(fc1_width) * hidden + static_cast<double>(hidden) * intermediate;
      const ExecutionWorkCost cost{2.0 * expert_cost, static_cast<double>(sizeof(float) * hidden),
                                   expert_cost};
      const ExecutionSchedule schedule{2, 1, std::numeric_limits<std::int64_t>::max(), 0};
      ExecuteCostedRanges(static_cast<std::int64_t>(attributes.k), cost, schedule, std::int64_t{1},
                          [&](std::int64_t slot_begin, std::int64_t slot_end) {
                            compute_slots(static_cast<std::size_t>(slot_begin),
                                          static_cast<std::size_t>(slot_end),
                                          expert_results.data());
                          });
      for (std::size_t slot = 0; slot < attributes.k; ++slot) {
        for (std::size_t row = 0; row < hidden; ++row) {
          result[row] += expert_results[slot * hidden + row];
        }
      }
    } else {
      compute_slots(0, attributes.k, nullptr);
    }
    Codec::Store(result.data(), output + token * hidden, hidden);
  }
}

template <typename T>
void Compute(const T *input, const T *router, const T *fc1, const T *bias1, const T *fc2,
             const T *bias2, T *output, std::size_t tokens, std::size_t hidden, std::size_t experts,
             std::size_t intermediate, const MoEAttributes &attributes) {
  if (tokens == 0) {
    return;
  }
  if (tokens == 1) {
    ComputeRange(input, router, fc1, bias1, fc2, bias2, output, 0, 1, hidden, experts, intermediate,
                 attributes);
    return;
  }
  const std::size_t fc1_width =
      intermediate * (attributes.activation == "swiglu" ? std::size_t{2} : std::size_t{1});
  const double expert_multiply_adds =
      static_cast<double>(fc1_width) * static_cast<double>(hidden) +
      static_cast<double>(hidden) * static_cast<double>(intermediate);
  const double selected_experts = static_cast<double>(attributes.k);
  const double element_bytes = static_cast<double>(sizeof(T));
  const ExecutionWorkCost cost{
      element_bytes * (static_cast<double>(experts) + selected_experts * expert_multiply_adds),
      element_bytes * static_cast<double>(hidden), selected_experts * expert_multiply_adds};
  const ExecutionSchedule schedule{2, 1, std::numeric_limits<std::int64_t>::max(), 0};
  ExecuteCostedRanges(static_cast<std::int64_t>(tokens), cost, schedule, std::int64_t{1},
                      [=](std::int64_t begin, std::int64_t end) {
                        ComputeRange(input, router, fc1, bias1, fc2, bias2, output,
                                     static_cast<std::size_t>(begin), static_cast<std::size_t>(end),
                                     hidden, experts, intermediate, attributes);
                      });
}

template <typename Codec>
void ComputeLowPrecision(const std::uint16_t *input, const std::uint16_t *router,
                         const std::uint16_t *fc1, const std::uint16_t *bias1,
                         const std::uint16_t *fc2, const std::uint16_t *bias2,
                         std::uint16_t *output, std::size_t tokens, std::size_t hidden,
                         std::size_t experts, std::size_t intermediate,
                         const MoEAttributes &attributes) {
  if (tokens == 0) {
    return;
  }
  if (tokens == 1) {
    ComputeLowPrecisionRange<Codec>(input, router, fc1, bias1, fc2, bias2, output, 0, 1, hidden,
                                    experts, intermediate, attributes);
    return;
  }
  const std::size_t fc1_width =
      intermediate * (attributes.activation == "swiglu" ? std::size_t{2} : std::size_t{1});
  const double expert_multiply_adds =
      static_cast<double>(fc1_width) * static_cast<double>(hidden) +
      static_cast<double>(hidden) * static_cast<double>(intermediate);
  const double selected_experts = static_cast<double>(attributes.k);
  const ExecutionWorkCost cost{
      2.0 * (static_cast<double>(experts) + selected_experts * expert_multiply_adds),
      2.0 * static_cast<double>(hidden), selected_experts * expert_multiply_adds};
  const ExecutionSchedule schedule{2, 1, std::numeric_limits<std::int64_t>::max(), 0};
  ExecuteCostedRanges(static_cast<std::int64_t>(tokens), cost, schedule, std::int64_t{1},
                      [=](std::int64_t begin, std::int64_t end) {
                        ComputeLowPrecisionRange<Codec>(input, router, fc1, bias1, fc2, bias2,
                                                        output, static_cast<std::size_t>(begin),
                                                        static_cast<std::size_t>(end), hidden,
                                                        experts, intermediate, attributes);
                      });
}

} // namespace

void ComputeMoE(const void *input, const void *router, const void *fc1, const void *bias1,
                const void *fc2, const void *bias2, void *output, std::size_t tokens,
                std::size_t hidden, std::size_t experts, std::size_t intermediate,
                MoEDataType data_type, const MoEAttributes &attributes) {
  if (data_type == MoEDataType::kFloat32) {
    Compute(static_cast<const float *>(input), static_cast<const float *>(router),
            static_cast<const float *>(fc1), static_cast<const float *>(bias1),
            static_cast<const float *>(fc2), static_cast<const float *>(bias2),
            static_cast<float *>(output), tokens, hidden, experts, intermediate, attributes);
    return;
  }
  const auto *input_bits = static_cast<const std::uint16_t *>(input);
  const auto *router_bits = static_cast<const std::uint16_t *>(router);
  const auto *fc1_bits = static_cast<const std::uint16_t *>(fc1);
  const auto *bias1_bits = static_cast<const std::uint16_t *>(bias1);
  const auto *fc2_bits = static_cast<const std::uint16_t *>(fc2);
  const auto *bias2_bits = static_cast<const std::uint16_t *>(bias2);
  auto *output_bits = static_cast<std::uint16_t *>(output);
  if (data_type == MoEDataType::kFloat16) {
    ComputeLowPrecision<Float16Codec>(input_bits, router_bits, fc1_bits, bias1_bits, fc2_bits,
                                      bias2_bits, output_bits, tokens, hidden, experts,
                                      intermediate, attributes);
  } else {
    ComputeLowPrecision<BFloat16Codec>(input_bits, router_bits, fc1_bits, bias1_bits, fc2_bits,
                                       bias2_bits, output_bits, tokens, hidden, experts,
                                       intermediate, attributes);
  }
}

} // namespace onnx_light_cpu
