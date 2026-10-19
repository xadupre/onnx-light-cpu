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

#include "cpu_compute/math/gemm/amx/gemm_amx_tile.h"
#include "cpu_compute/simd_level.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <thread>

namespace {

using cpu_compute::AmxTileConfig;
using cpu_compute::AmxTileConfigSetTile;
using cpu_compute::AmxTileScope;
using cpu_compute::AmxTileStateAvailable;
using cpu_compute::CpuSupportsAmxBf16;
using cpu_compute::CpuSupportsAmxInt8;
using cpu_compute::CpuSupportsAmxTile;
using cpu_compute::kAmxMaxColBytes;
using cpu_compute::kAmxMaxRows;
using cpu_compute::kAmxMaxTiles;

// The TILECFG structure is consumed directly by LDTILECFG, so its size and the
// offsets of every field are fixed by the architecture.
TEST(GemmAmxTile, TileConfigMatchesHardwareLayout) {
  EXPECT_EQ(sizeof(cpu_compute::AmxTileConfig), 64u);
  EXPECT_EQ(alignof(cpu_compute::AmxTileConfig), 64u);
  EXPECT_EQ(offsetof(cpu_compute::AmxTileConfig, palette_id), 0u);
  EXPECT_EQ(offsetof(cpu_compute::AmxTileConfig, start_row), 1u);
  EXPECT_EQ(offsetof(cpu_compute::AmxTileConfig, reserved), 2u);
  EXPECT_EQ(offsetof(cpu_compute::AmxTileConfig, colsb), 16u);
  EXPECT_EQ(offsetof(cpu_compute::AmxTileConfig, rows), 48u);
}

TEST(GemmAmxTile, TileConfigDefaultsAreZeroed) {
  cpu_compute::AmxTileConfig config;
  EXPECT_EQ(config.palette_id, 0u);
  EXPECT_EQ(config.start_row, 0u);
  for (std::size_t i = 0; i < cpu_compute::kAmxMaxTiles; ++i) {
    EXPECT_EQ(config.rows[i], 0u);
    EXPECT_EQ(config.colsb[i], 0u);
  }
}

TEST(GemmAmxTile, SetTileProgramsPaletteRowsAndColsb) {
  cpu_compute::AmxTileConfig config;
  ASSERT_TRUE(cpu_compute::AmxTileConfigSetTile(config, 0, cpu_compute::kAmxMaxRows,
                                                cpu_compute::kAmxMaxColBytes));
  EXPECT_EQ(config.palette_id, 1u);
  EXPECT_EQ(static_cast<std::size_t>(config.rows[0]), cpu_compute::kAmxMaxRows);
  EXPECT_EQ(static_cast<std::size_t>(config.colsb[0]), cpu_compute::kAmxMaxColBytes);

  ASSERT_TRUE(cpu_compute::AmxTileConfigSetTile(config, cpu_compute::kAmxMaxTiles - 1, 8, 32));
  EXPECT_EQ(config.rows[cpu_compute::kAmxMaxTiles - 1], 8u);
  EXPECT_EQ(config.colsb[cpu_compute::kAmxMaxTiles - 1], 32u);
  // Earlier tiles remain untouched.
  EXPECT_EQ(static_cast<std::size_t>(config.rows[0]), cpu_compute::kAmxMaxRows);
}

TEST(GemmAmxTile, SetTileRejectsOutOfRangeRequests) {
  cpu_compute::AmxTileConfig config;
  // Out-of-range tile index.
  EXPECT_FALSE(cpu_compute::AmxTileConfigSetTile(config, cpu_compute::kAmxMaxTiles, 4, 4));
  // Row count out of [1, kAmxMaxRows].
  EXPECT_FALSE(cpu_compute::AmxTileConfigSetTile(config, 0, 0, 4));
  EXPECT_FALSE(cpu_compute::AmxTileConfigSetTile(config, 0, cpu_compute::kAmxMaxRows + 1, 4));
  // Column bytes out of [1, kAmxMaxColBytes] or not a multiple of four.
  EXPECT_FALSE(cpu_compute::AmxTileConfigSetTile(config, 0, 4, 0));
  EXPECT_FALSE(cpu_compute::AmxTileConfigSetTile(config, 0, 4, cpu_compute::kAmxMaxColBytes + 4));
  EXPECT_FALSE(cpu_compute::AmxTileConfigSetTile(config, 0, 4, 6));

  // A rejected request leaves the configuration untouched.
  EXPECT_EQ(config.palette_id, 0u);
  EXPECT_EQ(config.rows[0], 0u);
  EXPECT_EQ(config.colsb[0], 0u);
}

// Detection must be self-consistent and never claim a narrower AMX subset
// without the shared AMX-TILE state. AmxTileStateAvailable() is queried first so
// its one-time OS permission request runs before CpuSupportsAmxTile() observes
// XCR0 on an AMX-capable Linux host.
TEST(GemmAmxTile, DetectionIsConsistent) {
  const bool available = cpu_compute::AmxTileStateAvailable();
  const bool tile = cpu_compute::CpuSupportsAmxTile();
  EXPECT_EQ(available, tile);
  if (cpu_compute::CpuSupportsAmxBf16())
    EXPECT_TRUE(tile);
  if (cpu_compute::CpuSupportsAmxInt8())
    EXPECT_TRUE(tile);
}

// AmxTileStateAvailable caches its result (the OS permission request runs at
// most once) so repeated calls must agree.
TEST(GemmAmxTile, AvailabilityIsIdempotent) {
  const bool first = cpu_compute::AmxTileStateAvailable();
  EXPECT_EQ(cpu_compute::AmxTileStateAvailable(), first);
}

// The per-worker RAII scope must be safe on every CPU: when AMX tile state is
// unavailable (or the intrinsics were not compiled in) it configures nothing
// and reports it, and the destructor is safe to run regardless. Whenever it
// does configure the tiles, AMX tile state must be available -- the invariant
// that holds no matter whether the binary was built with ``-mamx-tile``.
TEST(GemmAmxTile, TileScopeSafeFallbackAndConfiguration) {
  cpu_compute::AmxTileConfig config;
  ASSERT_TRUE(cpu_compute::AmxTileConfigSetTile(config, 0, cpu_compute::kAmxMaxRows,
                                                cpu_compute::kAmxMaxColBytes));
  cpu_compute::AmxTileScope scope(config);
  if (scope.configured())
    EXPECT_TRUE(cpu_compute::AmxTileStateAvailable());
}

// Tile configuration is thread-local hardware state, so each worker owns its
// scope. Exercising the scope from a separate thread must behave identically:
// configuring the tiles only when AMX tile state is available.
TEST(GemmAmxTile, TileScopeIsPerWorker) {
  bool configured = false;
  bool available = false;
  std::thread worker([&] {
    cpu_compute::AmxTileConfig config;
    cpu_compute::AmxTileConfigSetTile(config, 0, cpu_compute::kAmxMaxRows,
                                      cpu_compute::kAmxMaxColBytes);
    cpu_compute::AmxTileScope scope(config);
    configured = scope.configured();
    available = cpu_compute::AmxTileStateAvailable();
  });
  worker.join();
  if (configured)
    EXPECT_TRUE(available);
}

} // namespace
