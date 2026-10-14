// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/com_microsoft/moe.h"

#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
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
void Compute(const T *input, const T *router, const T *fc1, const T *bias1, const T *fc2,
             const T *bias2, T *output, std::size_t tokens, std::size_t hidden, std::size_t experts,
             std::size_t intermediate, const MoEAttributes &attributes) {
  const bool swiglu = attributes.activation == "swiglu";
  const std::size_t fc1_width = intermediate * (swiglu ? 2 : 1);
  std::vector<std::pair<float, std::size_t>> routes(experts);
  std::vector<T> first(fc1_width), activated(intermediate), second(hidden);
  std::vector<float> result(hidden);
  for (std::size_t token = 0; token < tokens; ++token) {
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

} // namespace

void ComputeMoE(const void *input, const void *router, const void *fc1, const void *bias1,
                const void *fc2, const void *bias2, void *output, std::size_t tokens,
                std::size_t hidden, std::size_t experts, std::size_t intermediate, bool float16,
                const MoEAttributes &attributes) {
  if (float16) {
    Compute(static_cast<const std::uint16_t *>(input), static_cast<const std::uint16_t *>(router),
            static_cast<const std::uint16_t *>(fc1), static_cast<const std::uint16_t *>(bias1),
            static_cast<const std::uint16_t *>(fc2), static_cast<const std::uint16_t *>(bias2),
            static_cast<std::uint16_t *>(output), tokens, hidden, experts, intermediate,
            attributes);
  } else {
    Compute(static_cast<const float *>(input), static_cast<const float *>(router),
            static_cast<const float *>(fc1), static_cast<const float *>(bias1),
            static_cast<const float *>(fc2), static_cast<const float *>(bias2),
            static_cast<float *>(output), tokens, hidden, experts, intermediate, attributes);
  }
}

} // namespace onnx_light_cpu
