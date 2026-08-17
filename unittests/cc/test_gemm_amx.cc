// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Focused lifecycle tests for the AMX tile-state module (Roadmap PR07.5). No
// GEMM kernel is exercised here: the tests cover the hardware-defined TILECFG
// layout, the tile-config builder validation, OS-enabled tile-state detection,
// and the safe fallback of the per-worker RAII scope. They run on any CPU --
// when AMX tile state is unavailable every entry point degrades to a no-op that
// reports "not configured", which is exactly what CI hardware without AMX
// verifies.

#include "onnx_light_cpu/impl/math/gemm/amx/gemm_amx_tile.h"
#include "onnx_light_cpu/impl/simd_level.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <thread>

namespace {

using onnx_light_cpu::AmxTileConfig;
using onnx_light_cpu::AmxTileConfigSetTile;
using onnx_light_cpu::AmxTileScope;
using onnx_light_cpu::AmxTileStateAvailable;
using onnx_light_cpu::CpuSupportsAmxBf16;
using onnx_light_cpu::CpuSupportsAmxInt8;
using onnx_light_cpu::CpuSupportsAmxTile;
using onnx_light_cpu::kAmxMaxColBytes;
using onnx_light_cpu::kAmxMaxRows;
using onnx_light_cpu::kAmxMaxTiles;

// The TILECFG structure is consumed directly by LDTILECFG, so its size and the
// offsets of every field are fixed by the architecture.
TEST(GemmAmxTile, TileConfigMatchesHardwareLayout) {
  EXPECT_EQ(sizeof(AmxTileConfig), 64u);
  EXPECT_EQ(alignof(AmxTileConfig), 64u);
  EXPECT_EQ(offsetof(AmxTileConfig, palette_id), 0u);
  EXPECT_EQ(offsetof(AmxTileConfig, start_row), 1u);
  EXPECT_EQ(offsetof(AmxTileConfig, reserved), 2u);
  EXPECT_EQ(offsetof(AmxTileConfig, colsb), 16u);
  EXPECT_EQ(offsetof(AmxTileConfig, rows), 48u);
}

TEST(GemmAmxTile, TileConfigDefaultsAreZeroed) {
  AmxTileConfig config;
  EXPECT_EQ(config.palette_id, 0u);
  EXPECT_EQ(config.start_row, 0u);
  for (std::size_t i = 0; i < kAmxMaxTiles; ++i) {
    EXPECT_EQ(config.rows[i], 0u);
    EXPECT_EQ(config.colsb[i], 0u);
  }
}

TEST(GemmAmxTile, SetTileProgramsPaletteRowsAndColsb) {
  AmxTileConfig config;
  ASSERT_TRUE(AmxTileConfigSetTile(config, 0, kAmxMaxRows, kAmxMaxColBytes));
  EXPECT_EQ(config.palette_id, 1u);
  EXPECT_EQ(static_cast<std::size_t>(config.rows[0]), kAmxMaxRows);
  EXPECT_EQ(static_cast<std::size_t>(config.colsb[0]), kAmxMaxColBytes);

  ASSERT_TRUE(AmxTileConfigSetTile(config, kAmxMaxTiles - 1, 8, 32));
  EXPECT_EQ(config.rows[kAmxMaxTiles - 1], 8u);
  EXPECT_EQ(config.colsb[kAmxMaxTiles - 1], 32u);
  // Earlier tiles remain untouched.
  EXPECT_EQ(static_cast<std::size_t>(config.rows[0]), kAmxMaxRows);
}

TEST(GemmAmxTile, SetTileRejectsOutOfRangeRequests) {
  AmxTileConfig config;
  // Out-of-range tile index.
  EXPECT_FALSE(AmxTileConfigSetTile(config, kAmxMaxTiles, 4, 4));
  // Row count out of [1, kAmxMaxRows].
  EXPECT_FALSE(AmxTileConfigSetTile(config, 0, 0, 4));
  EXPECT_FALSE(AmxTileConfigSetTile(config, 0, kAmxMaxRows + 1, 4));
  // Column bytes out of [1, kAmxMaxColBytes] or not a multiple of four.
  EXPECT_FALSE(AmxTileConfigSetTile(config, 0, 4, 0));
  EXPECT_FALSE(AmxTileConfigSetTile(config, 0, 4, kAmxMaxColBytes + 4));
  EXPECT_FALSE(AmxTileConfigSetTile(config, 0, 4, 6));

  // A rejected request leaves the configuration untouched.
  EXPECT_EQ(config.palette_id, 0u);
  EXPECT_EQ(config.rows[0], 0u);
  EXPECT_EQ(config.colsb[0], 0u);
}

// Detection must be self-consistent and never claim a narrower AMX subset
// without the shared AMX-TILE state.
TEST(GemmAmxTile, DetectionIsConsistent) {
  const bool tile = CpuSupportsAmxTile();
  EXPECT_EQ(AmxTileStateAvailable(), tile);
  if (CpuSupportsAmxBf16())
    EXPECT_TRUE(tile);
  if (CpuSupportsAmxInt8())
    EXPECT_TRUE(tile);
}

// AmxTileStateAvailable caches its result (the OS permission request runs at
// most once) so repeated calls must agree.
TEST(GemmAmxTile, AvailabilityIsIdempotent) {
  const bool first = AmxTileStateAvailable();
  EXPECT_EQ(AmxTileStateAvailable(), first);
}

// The per-worker RAII scope must be safe on every CPU: when AMX tile state is
// unavailable it configures nothing and reports it, and the destructor is safe
// to run regardless. When AMX is available the tiles load and release without
// faulting.
TEST(GemmAmxTile, TileScopeSafeFallbackAndConfiguration) {
  AmxTileConfig config;
  ASSERT_TRUE(AmxTileConfigSetTile(config, 0, kAmxMaxRows, kAmxMaxColBytes));
  AmxTileScope scope(config);
  EXPECT_EQ(scope.configured(), AmxTileStateAvailable());
}

// Tile configuration is thread-local hardware state, so each worker owns its
// scope. Exercising the scope from a separate thread must behave identically.
TEST(GemmAmxTile, TileScopeIsPerWorker) {
  bool configured = false;
  bool available = false;
  std::thread worker([&] {
    AmxTileConfig config;
    AmxTileConfigSetTile(config, 0, kAmxMaxRows, kAmxMaxColBytes);
    AmxTileScope scope(config);
    configured = scope.configured();
    available = AmxTileStateAvailable();
  });
  worker.join();
  EXPECT_EQ(configured, available);
}

} // namespace
