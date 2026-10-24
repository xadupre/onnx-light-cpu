// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "ml_cpu_compute/com_microsoft/causal_conv_with_state.h"

#include "ml_cpu_compute/execution.h"
#include "ml_cpu_compute/math/half_conversion.h"

#include <algorithm>
#include <utility>

namespace ml_cpu_compute {

void CausalConvWidth3Float16(const std::uint16_t *input, const std::uint16_t *weight,
                             const std::uint16_t *bias, const std::uint16_t *past,
                             std::uint16_t *output, std::uint16_t *present, std::size_t batch,
                             std::size_t channels, std::size_t length, bool channels_last) {
  constexpr std::size_t time_tile = 64;
  constexpr std::size_t channel_tile = 16;
  const std::size_t groups =
      channels_last ? channels / channel_tile + (channels % channel_tile != 0) : channels;
  ExecuteRanges(
      static_cast<std::int64_t>(batch * groups),
      static_cast<double>(length) * (channels_last ? channel_tile : 1),
      [&](std::int64_t begin, std::int64_t end) {
        for (auto task = begin; task < end; ++task) {
          const std::size_t row = static_cast<std::size_t>(task) / groups;
          const std::size_t group = static_cast<std::size_t>(task) % groups;
          if (!channels_last) {
            const std::size_t base = (row * channels + group) * length;
            const std::size_t state_base = (row * channels + group) * 2;
            float weights[3];
            detail::ConvertFloat16ToFloat32(weight + group * 3, weights, 3);
            const float initial = bias ? detail::Float16BitsToFloat(bias[group]) : 0.0f;
            float values[time_tile + 2]{};
            float result[time_tile];
            if (past) {
              detail::ConvertFloat16ToFloat32(past + state_base, values, 2);
            }
            for (std::size_t t = 0; t < length;) {
              const std::size_t count = std::min(time_tile, length - t);
              detail::ConvertFloat16ToFloat32(input + base + t, values + 2, count);
              for (std::size_t i = 0; i < count; ++i) {
                float sum = initial;
                sum += weights[0] * values[i];
                sum += weights[1] * values[i + 1];
                sum += weights[2] * values[i + 2];
                result[i] = sum;
              }
              detail::ConvertFloat32ToFloat16(result, output + base + t, count);
              values[0] = values[count];
              values[1] = values[count + 1];
              t += count;
            }
            std::copy_n(input + base + length - 2, 2, present + state_base);
          } else {
            const std::size_t channel = group * channel_tile;
            const std::size_t count = std::min(channel_tile, channels - channel);
            const std::size_t base = row * length * channels + channel;
            const std::size_t state_base = row * 2 * channels + channel;
            float packed[3 * channel_tile];
            float weights[3][channel_tile];
            float initial[channel_tile]{};
            float history[3][channel_tile]{};
            float result[channel_tile];
            detail::ConvertFloat16ToFloat32(weight + channel * 3, packed, count * 3);
            for (std::size_t i = 0; i < count; ++i) {
              for (std::size_t k = 0; k < 3; ++k) {
                weights[k][i] = packed[i * 3 + k];
              }
            }
            if (bias) {
              detail::ConvertFloat16ToFloat32(bias + channel, initial, count);
            }
            if (past) {
              detail::ConvertFloat16ToFloat32(past + state_base, history[0], count);
              detail::ConvertFloat16ToFloat32(past + state_base + channels, history[1], count);
            }
            float *oldest = history[0];
            float *previous = history[1];
            float *current = history[2];
            for (std::size_t t = 0; t < length; ++t) {
              detail::ConvertFloat16ToFloat32(input + base + t * channels, current, count);
              for (std::size_t i = 0; i < count; ++i) {
                float sum = initial[i];
                sum += weights[0][i] * oldest[i];
                sum += weights[1][i] * previous[i];
                sum += weights[2][i] * current[i];
                result[i] = sum;
              }
              detail::ConvertFloat32ToFloat16(result, output + base + t * channels, count);
              std::swap(oldest, previous);
              std::swap(previous, current);
            }
            for (std::size_t t = 0; t < 2; ++t) {
              std::copy_n(input + base + (length - 2 + t) * channels, count,
                          present + state_base + t * channels);
            }
          }
        }
      });
}

} // namespace ml_cpu_compute
