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

#define ONNX_LIGHT_CPU_AMX_INT8_TILE_C00 0
#define ONNX_LIGHT_CPU_AMX_INT8_TILE_C01 1
#define ONNX_LIGHT_CPU_AMX_INT8_TILE_C10 2
#define ONNX_LIGHT_CPU_AMX_INT8_TILE_C11 3
#define ONNX_LIGHT_CPU_AMX_INT8_TILE_A0 4
#define ONNX_LIGHT_CPU_AMX_INT8_TILE_A1 5
#define ONNX_LIGHT_CPU_AMX_INT8_TILE_B0 6
#define ONNX_LIGHT_CPU_AMX_INT8_TILE_B1 7

constexpr std::size_t kTileRows = 16;
constexpr std::size_t kTileCols = 16;
constexpr std::size_t kTileK = 64;
constexpr int kTileStride = 64;
constexpr int kInt32TileStride = kTileCols * sizeof(std::uint32_t);

constexpr std::size_t RoundUp(std::size_t value, std::size_t multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

inline void PackB4x16(const std::uint8_t *source, std::size_t stride, std::uint8_t flip,
                      std::uint8_t *destination, __m512i &column_sums, bool signed_values) {
  const __m128i mask = _mm_set1_epi8(static_cast<char>(flip));
  const __m128i row0 =
      _mm_xor_si128(_mm_loadu_si128(reinterpret_cast<const __m128i *>(source)), mask);
  const __m128i row1 =
      _mm_xor_si128(_mm_loadu_si128(reinterpret_cast<const __m128i *>(source + stride)), mask);
  const __m128i row2 =
      _mm_xor_si128(_mm_loadu_si128(reinterpret_cast<const __m128i *>(source + 2 * stride)), mask);
  const __m128i row3 =
      _mm_xor_si128(_mm_loadu_si128(reinterpret_cast<const __m128i *>(source + 3 * stride)), mask);
  const __m128i rows01_low = _mm_unpacklo_epi8(row0, row1);
  const __m128i rows01_high = _mm_unpackhi_epi8(row0, row1);
  const __m128i rows23_low = _mm_unpacklo_epi8(row2, row3);
  const __m128i rows23_high = _mm_unpackhi_epi8(row2, row3);
  __m512i packed = _mm512_castsi128_si512(_mm_unpacklo_epi16(rows01_low, rows23_low));
  packed = _mm512_inserti32x4(packed, _mm_unpackhi_epi16(rows01_low, rows23_low), 1);
  packed = _mm512_inserti32x4(packed, _mm_unpacklo_epi16(rows01_high, rows23_high), 2);
  packed = _mm512_inserti32x4(packed, _mm_unpackhi_epi16(rows01_high, rows23_high), 3);
  _mm512_storeu_si512(destination, packed);

  const __m512i ones = _mm512_set1_epi8(1);
  column_sums = signed_values ? _mm512_dpbusd_epi32(column_sums, ones, packed)
                              : _mm512_dpbusd_epi32(column_sums, packed, ones);
}

void GemvU8S8(const std::uint8_t *a, const std::uint8_t *b, std::int32_t *c, std::size_t n,
              std::size_t k, std::int32_t a_zero_point, const std::int32_t *b_zero_points,
              std::size_t b_zero_point_count) {
  std::uint32_t a_sum = 0;
  for (std::size_t depth = 0; depth < k; ++depth) {
    a_sum += a[depth];
  }
  for (std::size_t col0 = 0; col0 < n; col0 += kTileCols) {
    const std::size_t cols = std::min(kTileCols, n - col0);
    __m512i accumulator = _mm512_setzero_si512();
    __m512i b_sum = _mm512_setzero_si512();
    std::size_t depth = 0;
    if (cols == kTileCols) {
      for (; depth + 4 <= k; depth += 4) {
        const __m128i row0 =
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + depth * n + col0));
        const __m128i row1 =
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + (depth + 1) * n + col0));
        const __m128i row2 =
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + (depth + 2) * n + col0));
        const __m128i row3 =
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + (depth + 3) * n + col0));
        const __m128i rows01_low = _mm_unpacklo_epi8(row0, row1);
        const __m128i rows01_high = _mm_unpackhi_epi8(row0, row1);
        const __m128i rows23_low = _mm_unpacklo_epi8(row2, row3);
        const __m128i rows23_high = _mm_unpackhi_epi8(row2, row3);
        __m512i packed_b = _mm512_castsi128_si512(_mm_unpacklo_epi16(rows01_low, rows23_low));
        packed_b = _mm512_inserti32x4(packed_b, _mm_unpackhi_epi16(rows01_low, rows23_low), 1);
        packed_b = _mm512_inserti32x4(packed_b, _mm_unpacklo_epi16(rows01_high, rows23_high), 2);
        packed_b = _mm512_inserti32x4(packed_b, _mm_unpackhi_epi16(rows01_high, rows23_high), 3);
        std::uint32_t packed_a;
        std::memcpy(&packed_a, a + depth, sizeof(packed_a));
        accumulator = _mm512_dpbusd_epi32(accumulator,
                                          _mm512_set1_epi32(static_cast<int>(packed_a)), packed_b);
        b_sum = _mm512_add_epi32(
            b_sum, _mm512_add_epi32(
                       _mm512_add_epi32(_mm512_cvtepi8_epi32(row0), _mm512_cvtepi8_epi32(row1)),
                       _mm512_add_epi32(_mm512_cvtepi8_epi32(row2), _mm512_cvtepi8_epi32(row3))));
      }
    }
    alignas(64) std::int32_t values[kTileCols] = {};
    alignas(64) std::int32_t b_sums[kTileCols] = {};
    _mm512_store_si512(values, accumulator);
    _mm512_store_si512(b_sums, b_sum);
    for (; depth < k; ++depth) {
      for (std::size_t col = 0; col < cols; ++col) {
        const std::int32_t b_value =
            static_cast<std::int32_t>(static_cast<std::int8_t>(b[depth * n + col0 + col]));
        values[col] += static_cast<std::int32_t>(a[depth]) * b_value;
        b_sums[col] += b_value;
      }
    }
    for (std::size_t col = 0; col < cols; ++col) {
      const std::int32_t b_zero_point =
          b_zero_point_count == 1 ? b_zero_points[0] : b_zero_points[col0 + col];
      std::uint32_t value = static_cast<std::uint32_t>(values[col]);
      value -= static_cast<std::uint32_t>(a_zero_point * b_sums[col]);
      value -= static_cast<std::uint32_t>(b_zero_point) * a_sum;
      value += static_cast<std::uint32_t>(k) * static_cast<std::uint32_t>(a_zero_point) *
               static_cast<std::uint32_t>(b_zero_point);
      c[col0 + col] = std::bit_cast<std::int32_t>(value);
    }
  }
}

void GemvU8S8Transposed(const std::uint8_t *a, const std::uint8_t *b, std::int32_t *c,
                        std::size_t m, std::size_t k, const std::int32_t *a_zero_points,
                        std::size_t a_zero_point_count, std::int32_t b_zero_point) {
  std::int32_t b_sum = 0;
  for (std::size_t depth = 0; depth < k; ++depth) {
    b_sum += static_cast<std::int8_t>(b[depth]);
  }
  for (std::size_t row = 0; row < m; ++row) {
    __m512i accumulator = _mm512_setzero_si512();
    __m512i a_sum_vector = _mm512_setzero_si512();
    std::size_t depth = 0;
    for (; depth + 64 <= k; depth += 64) {
      const __m512i a_values =
          _mm512_loadu_si512(reinterpret_cast<const void *>(a + row * k + depth));
      const __m512i b_values = _mm512_loadu_si512(reinterpret_cast<const void *>(b + depth));
      accumulator = _mm512_dpbusd_epi32(accumulator, a_values, b_values);
      a_sum_vector =
          _mm512_add_epi64(a_sum_vector, _mm512_sad_epu8(a_values, _mm512_setzero_si512()));
    }
    std::uint32_t value = static_cast<std::uint32_t>(_mm512_reduce_add_epi32(accumulator));
    std::uint32_t a_sum = static_cast<std::uint32_t>(_mm512_reduce_add_epi64(a_sum_vector));
    for (; depth < k; ++depth) {
      value += static_cast<std::uint32_t>(static_cast<std::int32_t>(a[row * k + depth]) *
                                          static_cast<std::int8_t>(b[depth]));
      a_sum += a[row * k + depth];
    }
    const std::int32_t a_zero_point =
        a_zero_point_count == 1 ? a_zero_points[0] : a_zero_points[row];
    value -= static_cast<std::uint32_t>(a_zero_point * b_sum);
    value -= static_cast<std::uint32_t>(b_zero_point) * a_sum;
    value += static_cast<std::uint32_t>(k) * static_cast<std::uint32_t>(a_zero_point) *
             static_cast<std::uint32_t>(b_zero_point);
    c[row] = std::bit_cast<std::int32_t>(value);
  }
}

} // namespace

void GemmMatMulIntegerAmxInt8(const std::uint8_t *a, bool a_signed, const std::uint8_t *b,
                              bool b_signed, std::int32_t *c, std::size_t m, std::size_t n,
                              std::size_t k, const std::int32_t *a_zero_points,
                              std::size_t a_zero_point_count, const std::int32_t *b_zero_points,
                              std::size_t b_zero_point_count) {
  if (!a_signed && b_signed && m == 1) {
    GemvU8S8(a, b, c, n, k, a_zero_points[0], b_zero_points, b_zero_point_count);
    return;
  }
  if (!a_signed && b_signed && n == 1 && b_zero_point_count == 1) {
    GemvU8S8Transposed(a, b, c, m, k, a_zero_points, a_zero_point_count, b_zero_points[0]);
    return;
  }
  AmxTileConfig config;
  for (int tile = ONNX_LIGHT_CPU_AMX_INT8_TILE_C00; tile <= ONNX_LIGHT_CPU_AMX_INT8_TILE_C11;
       ++tile) {
    AmxTileConfigSetTile(config, tile, kTileRows, kTileCols * sizeof(std::uint32_t));
  }
  for (int tile = ONNX_LIGHT_CPU_AMX_INT8_TILE_A0; tile <= ONNX_LIGHT_CPU_AMX_INT8_TILE_B1;
       ++tile) {
    AmxTileConfigSetTile(config, tile, kTileRows, kTileK);
  }
  AmxTileScope scope(config);
  if (!scope.configured()) {
    detail::IntegerMatMul2DWithDot(&detail::IntegerDotU8S8Scalar, a, a_signed, b, b_signed, c,
                                   static_cast<std::int64_t>(m), static_cast<std::int64_t>(n),
                                   static_cast<std::int64_t>(k), a_zero_points,
                                   static_cast<std::int64_t>(a_zero_point_count), b_zero_points,
                                   static_cast<std::int64_t>(b_zero_point_count));
    return;
  }

  const bool use_u8s8 = !a_signed && b_signed;
  const std::uint8_t a_flip = a_signed ? 0x80 : 0x00;
  const std::uint8_t b_flip = b_signed && !use_u8s8 ? 0x80 : 0x00;
  const std::int32_t a_bias = a_signed ? 128 : 0;
  const std::int32_t b_bias = b_signed && !use_u8s8 ? 128 : 0;
  const std::size_t padded_m = RoundUp(m, kTileRows);
  const std::size_t padded_n = RoundUp(n, kTileCols);
  const std::size_t padded_k = RoundUp(k, kTileK);
  const std::size_t column_block_count = padded_n / kTileCols;
  const std::size_t depth_block_count = padded_k / kTileK;
  const std::size_t packed_a_size = padded_m * padded_k;
  const std::size_t packed_b_size = column_block_count * depth_block_count * kTileRows * kTileK;
  thread_local std::vector<std::uint32_t> a_sum;
  thread_local std::vector<std::uint32_t> b_sum;
  thread_local std::vector<std::uint8_t> packed_storage;
  a_sum.resize(m);
  b_sum.resize(n);
  std::fill(a_sum.begin(), a_sum.end(), 0);
  std::fill(b_sum.begin(), b_sum.end(), 0);
  packed_storage.resize(packed_a_size + packed_b_size);
  std::uint8_t *packed_a = packed_storage.data();
  std::uint8_t *packed_b = packed_a + packed_a_size;
  if (padded_m != m || padded_k != k) {
    std::fill_n(packed_a, packed_a_size, 0);
  }
  if (padded_n != n || padded_k != k) {
    std::fill_n(packed_b, packed_b_size, 0);
  }

  for (std::size_t row = 0; row < m; ++row) {
    std::uint8_t *packed_row = packed_a + row * padded_k;
    const __m512i mask512 = _mm512_set1_epi8(static_cast<char>(a_flip));
    __m512i sum512 = _mm512_setzero_si512();
    std::size_t depth = 0;
    for (; depth + 64 <= k; depth += 64) {
      const __m512i values = _mm512_xor_si512(
          _mm512_loadu_si512(reinterpret_cast<const void *>(a + row * k + depth)), mask512);
      _mm512_storeu_si512(reinterpret_cast<void *>(packed_row + depth), values);
      sum512 = _mm512_add_epi64(sum512, _mm512_sad_epu8(values, _mm512_setzero_si512()));
    }
    a_sum[row] += static_cast<std::uint32_t>(_mm512_reduce_add_epi64(sum512));
    const __m128i mask128 = _mm_set1_epi8(static_cast<char>(a_flip));
    __m128i sum128 = _mm_setzero_si128();
    for (; depth + 16 <= k; depth += 16) {
      const __m128i values = _mm_xor_si128(
          _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + row * k + depth)), mask128);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(packed_row + depth), values);
      sum128 = _mm_add_epi64(sum128, _mm_sad_epu8(values, _mm_setzero_si128()));
    }
    alignas(16) std::uint64_t partial_sums128[2];
    _mm_store_si128(reinterpret_cast<__m128i *>(partial_sums128), sum128);
    a_sum[row] += static_cast<std::uint32_t>(partial_sums128[0] + partial_sums128[1]);
    for (; depth < k; ++depth) {
      const std::uint8_t value = static_cast<std::uint8_t>(a[row * k + depth] ^ a_flip);
      packed_row[depth] = value;
      a_sum[row] += value;
    }
  }
  constexpr std::size_t kPackColumnBlocks = 8;
  for (std::size_t column_block0 = 0; column_block0 < column_block_count;
       column_block0 += kPackColumnBlocks) {
    const std::size_t packed_column_blocks =
        std::min(kPackColumnBlocks, column_block_count - column_block0);
    __m512i column_sums[kPackColumnBlocks];
    for (std::size_t block = 0; block < packed_column_blocks; ++block) {
      column_sums[block] = _mm512_setzero_si512();
    }
    for (std::size_t depth0 = 0; depth0 < k; depth0 += kTileK) {
      const std::size_t depth_block = depth0 / kTileK;
      const std::size_t depth_count = std::min(kTileK, k - depth0);
      std::size_t depth = 0;
      for (; depth + 4 <= depth_count; depth += 4) {
        for (std::size_t block = 0; block < packed_column_blocks; ++block) {
          const std::size_t column_block = column_block0 + block;
          const std::size_t col0 = column_block * kTileCols;
          if (col0 + kTileCols > n) {
            break;
          }
          std::uint8_t *packed_tile =
              packed_b + (column_block * depth_block_count + depth_block) * kTileRows * kTileK;
          PackB4x16(b + (depth0 + depth) * n + col0, n, b_flip, packed_tile + (depth / 4) * kTileK,
                    column_sums[block], use_u8s8);
        }
      }
      for (std::size_t block = 0; block < packed_column_blocks; ++block) {
        const std::size_t column_block = column_block0 + block;
        const std::size_t col0 = column_block * kTileCols;
        const std::size_t cols = std::min(kTileCols, n - col0);
        if (cols == kTileCols && depth == depth_count) {
          continue;
        }
        std::uint8_t *packed_tile =
            packed_b + (column_block * depth_block_count + depth_block) * kTileRows * kTileK;
        const std::size_t tail_start = cols == kTileCols ? depth : 0;
        for (std::size_t tail_depth = tail_start; tail_depth < depth_count; ++tail_depth) {
          for (std::size_t col = 0; col < cols; ++col) {
            const std::uint8_t value =
                static_cast<std::uint8_t>(b[(depth0 + tail_depth) * n + col0 + col] ^ b_flip);
            packed_tile[(tail_depth / 4) * kTileK + col * 4 + tail_depth % 4] = value;
            b_sum[col0 + col] += use_u8s8 ? static_cast<std::uint32_t>(static_cast<std::int32_t>(
                                                static_cast<std::int8_t>(value)))
                                          : static_cast<std::uint32_t>(value);
          }
        }
      }
    }
    for (std::size_t block = 0; block < packed_column_blocks; ++block) {
      const std::size_t col0 = (column_block0 + block) * kTileCols;
      if (col0 + kTileCols > n) {
        continue;
      }
      const __m512i tail_sums =
          _mm512_loadu_si512(reinterpret_cast<const void *>(b_sum.data() + col0));
      _mm512_storeu_si512(reinterpret_cast<void *>(b_sum.data() + col0),
                          _mm512_add_epi32(column_sums[block], tail_sums));
    }
  }

  alignas(64) std::uint32_t c_tile[kTileRows * kTileCols];
  const std::uint32_t k_wrapped = static_cast<std::uint32_t>(k);

  auto correct_row = [&](const std::uint32_t *raw, std::int32_t *output, std::size_t row,
                         std::size_t col0, std::size_t cols) {
    const std::uint32_t azu = static_cast<std::uint32_t>(
        (a_zero_point_count == 1 ? a_zero_points[0] : a_zero_points[row]) + a_bias);
    if (cols == kTileCols) {
      const __m512i azu_vector = _mm512_set1_epi32(static_cast<std::int32_t>(azu));
      __m512i bzu_vector;
      if (b_zero_point_count == 1) {
        bzu_vector = _mm512_set1_epi32(b_zero_points[0] + b_bias);
      } else {
        bzu_vector = _mm512_add_epi32(
            _mm512_loadu_si512(reinterpret_cast<const void *>(b_zero_points + col0)),
            _mm512_set1_epi32(b_bias));
      }
      __m512i values = _mm512_loadu_si512(reinterpret_cast<const void *>(raw));
      values = _mm512_sub_epi32(
          values, _mm512_mullo_epi32(azu_vector, _mm512_loadu_si512(reinterpret_cast<const void *>(
                                                     b_sum.data() + col0))));
      values = _mm512_sub_epi32(
          values,
          _mm512_mullo_epi32(bzu_vector, _mm512_set1_epi32(static_cast<std::int32_t>(a_sum[row]))));
      values = _mm512_add_epi32(
          values, _mm512_mullo_epi32(
                      _mm512_set1_epi32(std::bit_cast<std::int32_t>(k_wrapped * azu)), bzu_vector));
      _mm512_storeu_si512(reinterpret_cast<void *>(output), values);
      return;
    }
    for (std::size_t col = 0; col < cols; ++col) {
      const std::uint32_t bzu = static_cast<std::uint32_t>(
          (b_zero_point_count == 1 ? b_zero_points[0] : b_zero_points[col0 + col]) + b_bias);
      std::uint32_t value = raw[col];
      value -= azu * b_sum[col0 + col];
      value -= bzu * a_sum[row];
      value += k_wrapped * azu * bzu;
      output[col] = std::bit_cast<std::int32_t>(value);
    }
  };

  auto compute_16x16 = [&](std::size_t row0, std::size_t col0) {
    const std::size_t rows = std::min(kTileRows, m - row0);
    const std::size_t cols = std::min(kTileCols, n - col0);
    _tile_zero(ONNX_LIGHT_CPU_AMX_INT8_TILE_C00);
    for (std::size_t depth0 = 0; depth0 < k; depth0 += kTileK) {
      const std::size_t depth_block = depth0 / kTileK;
      const std::uint8_t *packed_a_tile = packed_a + row0 * padded_k + depth0;
      const std::uint8_t *packed_b_tile =
          packed_b + ((col0 / kTileCols) * depth_block_count + depth_block) * kTileRows * kTileK;
      _tile_loadd(ONNX_LIGHT_CPU_AMX_INT8_TILE_A0, packed_a_tile, static_cast<int>(padded_k));
      _tile_loadd(ONNX_LIGHT_CPU_AMX_INT8_TILE_B0, packed_b_tile, kTileStride);
      if (use_u8s8) {
        _tile_dpbusd(ONNX_LIGHT_CPU_AMX_INT8_TILE_C00, ONNX_LIGHT_CPU_AMX_INT8_TILE_A0,
                     ONNX_LIGHT_CPU_AMX_INT8_TILE_B0);
      } else {
        _tile_dpbuud(ONNX_LIGHT_CPU_AMX_INT8_TILE_C00, ONNX_LIGHT_CPU_AMX_INT8_TILE_A0,
                     ONNX_LIGHT_CPU_AMX_INT8_TILE_B0);
      }
    }
    _tile_stored(ONNX_LIGHT_CPU_AMX_INT8_TILE_C00, c_tile, kInt32TileStride);
    for (std::size_t row = 0; row < rows; ++row) {
      correct_row(c_tile + row * kTileCols, c + (row0 + row) * n + col0, row0 + row, col0, cols);
    }
  };

  for (std::size_t row0 = 0; row0 < m;) {
    const bool full_rows = row0 + 2 * kTileRows <= m;
    for (std::size_t col0 = 0; col0 < n;) {
      if (!full_rows || col0 + 2 * kTileCols > n) {
        compute_16x16(row0, col0);
        if (full_rows) {
          compute_16x16(row0 + kTileRows, col0);
        }
        col0 += kTileCols;
        continue;
      }

      _tile_zero(ONNX_LIGHT_CPU_AMX_INT8_TILE_C00);
      _tile_zero(ONNX_LIGHT_CPU_AMX_INT8_TILE_C01);
      _tile_zero(ONNX_LIGHT_CPU_AMX_INT8_TILE_C10);
      _tile_zero(ONNX_LIGHT_CPU_AMX_INT8_TILE_C11);
      for (std::size_t depth0 = 0; depth0 < k; depth0 += kTileK) {
        const std::size_t depth_block = depth0 / kTileK;
        const std::uint8_t *packed_a0 = packed_a + row0 * padded_k + depth0;
        const std::uint8_t *packed_a1 = packed_a0 + kTileRows * padded_k;
        const std::uint8_t *packed_b0 =
            packed_b + ((col0 / kTileCols) * depth_block_count + depth_block) * kTileRows * kTileK;
        const std::uint8_t *packed_b1 = packed_b0 + depth_block_count * kTileRows * kTileK;
        _tile_loadd(ONNX_LIGHT_CPU_AMX_INT8_TILE_A0, packed_a0, static_cast<int>(padded_k));
        _tile_loadd(ONNX_LIGHT_CPU_AMX_INT8_TILE_A1, packed_a1, static_cast<int>(padded_k));
        _tile_loadd(ONNX_LIGHT_CPU_AMX_INT8_TILE_B0, packed_b0, kTileStride);
        _tile_loadd(ONNX_LIGHT_CPU_AMX_INT8_TILE_B1, packed_b1, kTileStride);
        if (use_u8s8) {
          _tile_dpbusd(ONNX_LIGHT_CPU_AMX_INT8_TILE_C00, ONNX_LIGHT_CPU_AMX_INT8_TILE_A0,
                       ONNX_LIGHT_CPU_AMX_INT8_TILE_B0);
          _tile_dpbusd(ONNX_LIGHT_CPU_AMX_INT8_TILE_C01, ONNX_LIGHT_CPU_AMX_INT8_TILE_A0,
                       ONNX_LIGHT_CPU_AMX_INT8_TILE_B1);
          _tile_dpbusd(ONNX_LIGHT_CPU_AMX_INT8_TILE_C10, ONNX_LIGHT_CPU_AMX_INT8_TILE_A1,
                       ONNX_LIGHT_CPU_AMX_INT8_TILE_B0);
          _tile_dpbusd(ONNX_LIGHT_CPU_AMX_INT8_TILE_C11, ONNX_LIGHT_CPU_AMX_INT8_TILE_A1,
                       ONNX_LIGHT_CPU_AMX_INT8_TILE_B1);
        } else {
          _tile_dpbuud(ONNX_LIGHT_CPU_AMX_INT8_TILE_C00, ONNX_LIGHT_CPU_AMX_INT8_TILE_A0,
                       ONNX_LIGHT_CPU_AMX_INT8_TILE_B0);
          _tile_dpbuud(ONNX_LIGHT_CPU_AMX_INT8_TILE_C01, ONNX_LIGHT_CPU_AMX_INT8_TILE_A0,
                       ONNX_LIGHT_CPU_AMX_INT8_TILE_B1);
          _tile_dpbuud(ONNX_LIGHT_CPU_AMX_INT8_TILE_C10, ONNX_LIGHT_CPU_AMX_INT8_TILE_A1,
                       ONNX_LIGHT_CPU_AMX_INT8_TILE_B0);
          _tile_dpbuud(ONNX_LIGHT_CPU_AMX_INT8_TILE_C11, ONNX_LIGHT_CPU_AMX_INT8_TILE_A1,
                       ONNX_LIGHT_CPU_AMX_INT8_TILE_B1);
        }
      }
      std::int32_t *c00 = c + row0 * n + col0;
      std::int32_t *c01 = c00 + kTileCols;
      std::int32_t *c10 = c00 + kTileRows * n;
      std::int32_t *c11 = c10 + kTileCols;
      const int output_stride = static_cast<int>(n * sizeof(std::int32_t));
      _tile_stored(ONNX_LIGHT_CPU_AMX_INT8_TILE_C00, c00, output_stride);
      _tile_stored(ONNX_LIGHT_CPU_AMX_INT8_TILE_C01, c01, output_stride);
      _tile_stored(ONNX_LIGHT_CPU_AMX_INT8_TILE_C10, c10, output_stride);
      _tile_stored(ONNX_LIGHT_CPU_AMX_INT8_TILE_C11, c11, output_stride);
      for (std::size_t row = 0; row < 2 * kTileRows; ++row) {
        std::int32_t *output = c + (row0 + row) * n + col0;
        correct_row(reinterpret_cast<const std::uint32_t *>(output), output, row0 + row, col0,
                    kTileCols);
        correct_row(reinterpret_cast<const std::uint32_t *>(output + kTileCols), output + kTileCols,
                    row0 + row, col0 + kTileCols, kTileCols);
      }
      col0 += 2 * kTileCols;
    }
    row0 += full_rows ? 2 * kTileRows : kTileRows;
  }
}

} // namespace onnx_light_cpu
