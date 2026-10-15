// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/com_microsoft/moe.h"

#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
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
  std::vector<T> first(fc1_width), activated(intermediate), second(hidden);
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
    for (std::size_t slot = 0; slot < attributes.k; ++slot) {
      const auto [weight, expert] = routes[slot];
      if (weight == 0.0f) {
        continue;
      }
      for (std::size_t row = 0; row < fc1_width; ++row) {
        float value = 0.0f;
        for (std::size_t col = 0; col < hidden; ++col) {
          value += Read(input, token * hidden + col) *
                   Read(fc1, (expert * fc1_width + row) * hidden + col);
        }
        first[row] = Narrow<T>(value);
        if (bias1 != nullptr) {
          first[row] = Narrow<T>(Read(first.data(), row) + Read(bias1, expert * fc1_width + row));
        }
      }
      for (std::size_t row = 0; row < intermediate; ++row) {
        float value;
        if (swiglu) {
          const float gate = std::min(Read(first.data(), 2 * row), attributes.limit);
          const float linear =
              std::clamp(Read(first.data(), 2 * row + 1), -attributes.limit, attributes.limit);
          const float arg = attributes.alpha * gate;
          const float sigmoid =
              arg > 0 ? 1.0f / (1.0f + std::exp(-arg)) : std::exp(arg) / (1.0f + std::exp(arg));
          value = gate * sigmoid * (linear + attributes.beta);
        } else {
          value = Activate(Read(first.data(), row), attributes);
        }
        activated[row] = Narrow<T>(value);
      }
      for (std::size_t row = 0; row < hidden; ++row) {
        float value = 0.0f;
        for (std::size_t col = 0; col < intermediate; ++col) {
          value +=
              Read(activated.data(), col) * Read(fc2, (expert * hidden + row) * intermediate + col);
        }
        second[row] = Narrow<T>(value);
        if (bias2 != nullptr) {
          second[row] = Narrow<T>(Read(second.data(), row) + Read(bias2, expert * hidden + row));
        }
        result[row] +=
            (attributes.normalize ? weight / top_sum : weight) * Read(second.data(), row);
      }
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
  std::vector<float> input_values(hidden), router_values(experts), first(fc1_width),
      activated(intermediate), result(hidden), weight_row(std::max(hidden, intermediate));
  for (std::size_t token = begin; token < end; ++token) {
    Codec::Convert(input + token * hidden, input_values.data(), hidden);
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
    for (std::size_t slot = 0; slot < attributes.k; ++slot) {
      const auto [weight, expert] = routes[slot];
      if (weight == 0.0f) {
        continue;
      }
      for (std::size_t row = 0; row < fc1_width; ++row) {
        Codec::Convert(fc1 + (expert * fc1_width + row) * hidden, weight_row.data(), hidden);
        first[row] =
            std::inner_product(input_values.begin(), input_values.end(), weight_row.begin(), 0.0f);
        if (bias1 != nullptr) {
          first[row] += Codec::Load(bias1[expert * fc1_width + row]);
        }
      }
      for (std::size_t row = 0; row < intermediate; ++row) {
        if (swiglu) {
          const float gate = std::min(first[2 * row], attributes.limit);
          const float linear = std::clamp(first[2 * row + 1], -attributes.limit, attributes.limit);
          const float arg = attributes.alpha * gate;
          const float sigmoid =
              arg > 0 ? 1.0f / (1.0f + std::exp(-arg)) : std::exp(arg) / (1.0f + std::exp(arg));
          activated[row] = gate * sigmoid * (linear + attributes.beta);
        } else {
          activated[row] = Activate(first[row], attributes);
        }
      }
      for (std::size_t row = 0; row < hidden; ++row) {
        Codec::Convert(fc2 + (expert * hidden + row) * intermediate, weight_row.data(),
                       intermediate);
        float value =
            std::inner_product(activated.begin(), activated.end(), weight_row.begin(), 0.0f);
        if (bias2 != nullptr) {
          value += Codec::Load(bias2[expert * hidden + row]);
        }
        result[row] += (attributes.normalize ? weight / top_sum : weight) * value;
      }
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
