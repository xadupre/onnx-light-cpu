// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace onnx_light_cpu {

struct MatMulNBitsInt8Tile {
  const std::uint8_t *activations;
  const double *activation_scales;
  const std::uint8_t *weights;
  const std::int32_t *weight_sums;
  const float *scales;
  const float *bias;
  float *output;
  std::size_t rows;
  std::size_t k;
  std::size_t n;
  std::size_t columns;
};

template <auto Dot, bool FullTile>
inline void ComputeMatMulNBitsInt8TileImpl(const MatMulNBitsInt8Tile &tile) {
  const std::size_t blocks = tile.k / 32;
  const std::size_t rows = FullTile ? 4 : tile.rows;
  float sums[4][8];
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t lane = 0; lane < 8; ++lane) {
      sums[row][lane] = tile.bias != nullptr && lane < tile.columns ? tile.bias[lane] : 0.0f;
    }
  }
  for (std::size_t block = 0; block < blocks; ++block) {
    for (std::size_t row = 0; row < rows; ++row) {
      const auto *a = tile.activations + row * tile.k + block * 32;
      std::int32_t dots[8];
      Dot(a, tile.weights + block * 256, dots);
      for (std::size_t lane = 0; lane < 8; ++lane) {
        const std::int32_t corrected = dots[lane] - 128 * tile.weight_sums[block * 8 + lane];
        const double factor =
            tile.activation_scales[row * blocks + block] * tile.scales[block * 8 + lane];
        sums[row][lane] =
            static_cast<float>(static_cast<double>(sums[row][lane]) + corrected * factor);
      }
    }
  }
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t lane = 0; lane < tile.columns; ++lane) {
      tile.output[row * tile.n + lane] = sums[row][lane];
    }
  }
}

template <auto Dot> inline void ComputeMatMulNBitsInt8Tile(const MatMulNBitsInt8Tile &tile) {
  if (tile.rows == 4) {
    ComputeMatMulNBitsInt8TileImpl<Dot, true>(tile);
  } else {
    ComputeMatMulNBitsInt8TileImpl<Dot, false>(tile);
  }
}

} // namespace onnx_light_cpu
