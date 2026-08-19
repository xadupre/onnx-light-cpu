// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/gemm/amx/gemm_amx_int8.h"

#include "onnx_light_cpu/impl/math/gemm/amx/gemm_amx_tile.h"
#include "onnx_light_cpu/impl/math/gemm/vnni/integer_gemm_vnni.h"

#include <bit>
#include <cstring>
#include <vector>

#include <algorithm>

#include <immintrin.h>

namespace onnx_light_cpu {

namespace {

#define ONNX_LIGHT_CPU_AMX_INT8_TILE_C 0
#define ONNX_LIGHT_CPU_AMX_INT8_TILE_A 1
#define ONNX_LIGHT_CPU_AMX_INT8_TILE_B 2

constexpr std::size_t kTileRows = 16;
constexpr std::size_t kTileCols = 16;
constexpr std::size_t kTileK = 64;
constexpr int kTileStride = 64;

} // namespace

void GemmMatMulIntegerAmxInt8(const std::uint8_t *a, bool a_signed, const std::uint8_t *b,
                              bool b_signed, std::int32_t *c, std::size_t m, std::size_t n,
                              std::size_t k, const std::int32_t *a_zero_points,
                              std::size_t a_zero_point_count, const std::int32_t *b_zero_points,
                              std::size_t b_zero_point_count) {
  AmxTileConfig config;
  AmxTileConfigSetTile(config, ONNX_LIGHT_CPU_AMX_INT8_TILE_C, kTileRows,
                       kTileCols * sizeof(std::uint32_t));
  AmxTileConfigSetTile(config, ONNX_LIGHT_CPU_AMX_INT8_TILE_A, kTileRows, kTileK);
  AmxTileConfigSetTile(config, ONNX_LIGHT_CPU_AMX_INT8_TILE_B, kTileRows, kTileK);
  AmxTileScope scope(config);
  if (!scope.configured()) {
    detail::IntegerMatMul2DWithDot(&detail::IntegerDotU8S8Scalar, a, a_signed, b, b_signed, c,
                                   static_cast<std::int64_t>(m), static_cast<std::int64_t>(n),
                                   static_cast<std::int64_t>(k), a_zero_points,
                                   static_cast<std::int64_t>(a_zero_point_count), b_zero_points,
                                   static_cast<std::int64_t>(b_zero_point_count));
    return;
  }

  const std::uint8_t a_flip = a_signed ? 0x80 : 0x00;
  const std::uint8_t b_flip = b_signed ? 0x80 : 0x00;
  const std::int32_t a_bias = a_signed ? 128 : 0;
  const std::int32_t b_bias = b_signed ? 128 : 0;
  std::vector<std::uint32_t> a_sum(m, 0);
  std::vector<std::uint32_t> b_sum(n, 0);

  for (std::size_t row = 0; row < m; ++row) {
    for (std::size_t depth = 0; depth < k; ++depth) {
      a_sum[row] += static_cast<std::uint8_t>(a[row * k + depth] ^ a_flip);
    }
  }
  for (std::size_t col = 0; col < n; ++col) {
    for (std::size_t depth = 0; depth < k; ++depth) {
      b_sum[col] += static_cast<std::uint8_t>(b[depth * n + col] ^ b_flip);
    }
  }

  alignas(64) std::uint8_t a_tile[kTileRows * kTileK];
  alignas(64) std::uint8_t b_tile[kTileRows * kTileK];
  alignas(64) std::uint32_t c_tile[kTileRows * kTileCols];
  const std::uint32_t k_wrapped = static_cast<std::uint32_t>(k);

  for (std::size_t row0 = 0; row0 < m; row0 += kTileRows) {
    const std::size_t rows = std::min(kTileRows, m - row0);
    for (std::size_t col0 = 0; col0 < n; col0 += kTileCols) {
      const std::size_t cols = std::min(kTileCols, n - col0);
      _tile_zero(ONNX_LIGHT_CPU_AMX_INT8_TILE_C);
      for (std::size_t depth0 = 0; depth0 < k; depth0 += kTileK) {
        const std::size_t depth_count = std::min(kTileK, k - depth0);
        std::memset(a_tile, 0, sizeof(a_tile));
        std::memset(b_tile, 0, sizeof(b_tile));
        for (std::size_t row = 0; row < rows; ++row) {
          for (std::size_t depth = 0; depth < depth_count; ++depth) {
            a_tile[row * kTileK + depth] =
                static_cast<std::uint8_t>(a[(row0 + row) * k + depth0 + depth] ^ a_flip);
          }
        }
        for (std::size_t depth = 0; depth < depth_count; ++depth) {
          const std::size_t packed_row = depth / 4;
          const std::size_t packed_offset = depth % 4;
          for (std::size_t col = 0; col < cols; ++col) {
            b_tile[packed_row * kTileK + col * 4 + packed_offset] =
                static_cast<std::uint8_t>(b[(depth0 + depth) * n + col0 + col] ^ b_flip);
          }
        }
        _tile_loadd(ONNX_LIGHT_CPU_AMX_INT8_TILE_A, a_tile, kTileStride);
        _tile_loadd(ONNX_LIGHT_CPU_AMX_INT8_TILE_B, b_tile, kTileStride);
        _tile_dpbuud(ONNX_LIGHT_CPU_AMX_INT8_TILE_C, ONNX_LIGHT_CPU_AMX_INT8_TILE_A,
                     ONNX_LIGHT_CPU_AMX_INT8_TILE_B);
      }
      _tile_stored(ONNX_LIGHT_CPU_AMX_INT8_TILE_C, c_tile, kTileStride);
      for (std::size_t row = 0; row < rows; ++row) {
        const std::uint32_t azu = static_cast<std::uint32_t>(
            (a_zero_point_count == 1 ? a_zero_points[0] : a_zero_points[row0 + row]) + a_bias);
        for (std::size_t col = 0; col < cols; ++col) {
          const std::uint32_t bzu = static_cast<std::uint32_t>(
              (b_zero_point_count == 1 ? b_zero_points[0] : b_zero_points[col0 + col]) + b_bias);
          std::uint32_t value = c_tile[row * kTileCols + col];
          value -= azu * b_sum[col0 + col];
          value -= bzu * a_sum[row0 + row];
          value += k_wrapped * azu * bzu;
          c[(row0 + row) * n + col0 + col] = std::bit_cast<std::int32_t>(value);
        }
      }
    }
  }
}

} // namespace onnx_light_cpu
