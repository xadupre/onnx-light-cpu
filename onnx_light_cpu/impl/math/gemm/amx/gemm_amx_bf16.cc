// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Native AMX-BF16 Gemm micro-kernel (Roadmap PR07.6). This translation unit is
// compiled with the extra ``-mamx-tile -mamx-bf16`` (see the per-file
// COMPILE_OPTIONS override in CMakeLists.txt) even though the rest of
// onnx_light_cpu keeps the project's baseline ONNX_LIGHT_CPU_SIMD_FLAGS, so a
// single binary can carry this native BFLOAT16 kernel and still run correctly on
// CPUs that lack the ISA: the BFLOAT16 GEMM path in gemm_kernel.cc only
// dispatches here when ``CpuSupportsAmxBf16()`` reports the instruction set and
// ``AmxTileStateAvailable()`` confirms OS-enabled tile state at runtime, and
// otherwise falls back to the AVX-512BF16 kernel or the convert-while-packing
// float32 path.
//
// The kernel reuses the PR07.5 tile-state lifecycle (``AmxTileConfig`` /
// ``AmxTileScope``) and reduces the ``K`` dimension with the AMX ``tdpbf16ps``
// (``_tile_dpbf16ps``) tile dot-product, which multiplies a 16x32 BFLOAT16 ``A``
// tile by a 16-pair x 16-column BFLOAT16 ``B`` tile laid out in the required
// VNNI pair order and accumulates a 16x16 float32 ``C`` tile. Every partial
// ``mr``/``K``/column block is zero-padded into the fixed 16x16 tiles so a
// single tile configuration handles all shapes, and the float32 result matches
// the widen-then-float32 reference within float32 tolerance while halving the
// ``B`` memory footprint.

#include "onnx_light_cpu/impl/math/gemm/amx/gemm_amx_bf16.h"

#include "onnx_light_cpu/impl/math/gemm/amx/gemm_amx_tile.h"

#include <cassert>
#include <cstring>

#include <algorithm>

#include <immintrin.h>

namespace onnx_light_cpu {

namespace {

// Fixed AMX tile assignment. All three tiles are configured to their 16x64-byte
// maximum so a single LDTILECFG covers every (row block, column, K) tail; the
// unused rows/columns are zero-padded in the staging buffers. These are
// preprocessor literals because the ``_tile_*`` intrinsics token-paste the tile
// number into the register name and therefore require an integer literal.
#define ONNX_LIGHT_CPU_AMX_TILE_C 0 // 16 rows x 16 float32 accumulators.
#define ONNX_LIGHT_CPU_AMX_TILE_A 1 // 16 rows x 32 BFLOAT16 (``A``).
#define ONNX_LIGHT_CPU_AMX_TILE_B 2 // 16 pairs x 32 BFLOAT16 (``B``, VNNI order).

// A single AMX-BF16 step reduces up to 32 ``k`` values (16 VNNI pairs) across a
// 16x16 output tile.
constexpr std::size_t kAmxTileRows = 16;
constexpr std::size_t kAmxTileCols = 16;
constexpr std::size_t kAmxTileK = 32;

// Row stride, in bytes, of every 16x64-byte staging buffer / tile.
constexpr int kAmxTileStride = 64;

} // namespace

void GemmMicroKernel_AMXBF16(std::size_t mr, std::size_t nb, std::size_t K, float alpha, float beta,
                             const std::uint16_t *Bmat, std::size_t N, const float *Crow_base,
                             std::size_t Cstride, float *Yrow_base, std::size_t Ystride,
                             std::size_t n0, GemmAccumMode mode, const std::uint16_t *Apack) {
  assert(mr <= kAmxTileRows && "mr exceeds the 16-row AMX tile budget");

  // Configure the three tiles once for this call and load the config for the
  // current worker thread. If AMX tile state is unavailable (or the scope could
  // not configure), fall back to the portable scalar BFLOAT16 kernel so the
  // result is still correct.
  AmxTileConfig config;
  AmxTileConfigSetTile(config, ONNX_LIGHT_CPU_AMX_TILE_C, kAmxTileRows,
                       kAmxTileCols * sizeof(float));
  AmxTileConfigSetTile(config, ONNX_LIGHT_CPU_AMX_TILE_A, kAmxTileRows,
                       kAmxTileK * sizeof(std::uint16_t));
  AmxTileConfigSetTile(config, ONNX_LIGHT_CPU_AMX_TILE_B, kAmxTileRows,
                       kAmxTileK * sizeof(std::uint16_t));
  AmxTileScope scope(config);
  if (!scope.configured()) {
    GemmMicroKernel_ScalarBf16(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                               Ystride, n0, mode, Apack);
    return;
  }

  alignas(64) std::uint16_t Abuf[kAmxTileRows * kAmxTileK];
  alignas(64) std::uint16_t Bbuf[kAmxTileRows * kAmxTileK];
  alignas(64) float Cbuf[kAmxTileRows * kAmxTileCols];

  const bool alpha_is_one = alpha == 1.0f;
  const bool beta_is_one = beta == 1.0f;

  for (std::size_t n = 0; n < nb; n += kAmxTileCols) {
    const std::size_t ncols = std::min(kAmxTileCols, nb - n);
    _tile_zero(ONNX_LIGHT_CPU_AMX_TILE_C);
    for (std::size_t k = 0; k < K; k += kAmxTileK) {
      const std::size_t krem = std::min(kAmxTileK, K - k);

      // Stage the ``A`` tile: rows beyond ``mr`` and columns beyond ``krem`` are
      // zero so they contribute nothing to the dot-product.
      std::memset(Abuf, 0, sizeof(Abuf));
      for (std::size_t r = 0; r < mr; ++r) {
        std::memcpy(Abuf + r * kAmxTileK, Apack + r * K + k, krem * sizeof(std::uint16_t));
      }

      // Stage the ``B`` tile in VNNI pair order: entry [p][2 * col + s] holds
      // ``B[k + 2 * p + s][n0 + n + col]`` so ``tdpbf16ps`` reduces the two
      // consecutive ``k`` values of each pair. Padding stays zero.
      std::memset(Bbuf, 0, sizeof(Bbuf));
      for (std::size_t kk = 0; kk < krem; ++kk) {
        const std::size_t p = kk / 2;
        const std::size_t s = kk % 2;
        const std::uint16_t *Brow = Bmat + (k + kk) * N + n0 + n;
        std::uint16_t *dst = Bbuf + p * kAmxTileK + s;
        for (std::size_t col = 0; col < ncols; ++col) {
          dst[col * 2] = Brow[col];
        }
      }

      _tile_loadd(ONNX_LIGHT_CPU_AMX_TILE_A, Abuf, kAmxTileStride);
      _tile_loadd(ONNX_LIGHT_CPU_AMX_TILE_B, Bbuf, kAmxTileStride);
      _tile_dpbf16ps(ONNX_LIGHT_CPU_AMX_TILE_C, ONNX_LIGHT_CPU_AMX_TILE_A,
                     ONNX_LIGHT_CPU_AMX_TILE_B);
    }

    _tile_stored(ONNX_LIGHT_CPU_AMX_TILE_C, Cbuf, kAmxTileStride);

    // Combine ``alpha * acc`` with ``Y`` according to ``mode`` and write float32.
    for (std::size_t r = 0; r < mr; ++r) {
      float *Yrow = Yrow_base + r * Ystride + n0 + n;
      const float *Cacc = Cbuf + r * kAmxTileCols;
      for (std::size_t col = 0; col < ncols; ++col) {
        float res = alpha_is_one ? Cacc[col] : alpha * Cacc[col];
        if (mode == GemmAccumMode::kInitBias) {
          const float c = Crow_base[r * Cstride + n0 + n + col];
          res += beta_is_one ? c : beta * c;
        } else if (mode == GemmAccumMode::kAccumulate) {
          res += Yrow[col];
        }
        Yrow[col] = res;
      }
    }
  }
}

} // namespace onnx_light_cpu
