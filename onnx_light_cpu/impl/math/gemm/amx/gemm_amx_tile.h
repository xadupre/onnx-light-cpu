// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// AMX tile-state lifecycle (Roadmap PR07.5). This module owns the *lifecycle*
// pieces that any future AMX GEMM kernel (AMX-BF16 in PR07.6, AMX-INT8 in
// PR09.4) must share: OS-enabled tile-state detection, per-worker tile
// configuration, and a safe fallback when AMX is unavailable. It deliberately
// contains no GEMM kernel.
//
// AMX has non-trivial per-thread costs (requesting XTILEDATA permission and
// running ``LDTILECFG``/``TILERELEASE``), so it must stay optional and every
// entry point degrades to a no-op that reports "not configured" when the CPU or
// OS does not support AMX tile state.

#pragma once

#include <cstddef>
#include <cstdint>

namespace onnx_light_cpu {

/// Maximum number of AMX tile registers (``TMM0``..``TMM7``).
inline constexpr std::size_t kAmxMaxTiles = 8;
/// Maximum number of rows a single AMX tile can hold.
inline constexpr std::size_t kAmxMaxRows = 16;
/// Maximum number of bytes per tile row (a tile is at most 16 rows x 64 bytes).
inline constexpr std::size_t kAmxMaxColBytes = 64;

/// Hardware-defined 64-byte ``TILECFG`` structure consumed by ``LDTILECFG``.
///
/// The layout is fixed by the architecture: a palette selector, a start-row
/// used by interruptible tile ops, reserved padding, then the per-tile column
/// byte counts (``colsb``) and row counts (``rows``). It is populated with
/// :cpp:func:`AmxTileConfigSetTile` and loaded through :cpp:class:`AmxTileScope`.
struct alignas(64) AmxTileConfig {
  std::uint8_t palette_id = 0;
  std::uint8_t start_row = 0;
  std::uint8_t reserved[14] = {};
  std::uint16_t colsb[16] = {};
  std::uint8_t rows[16] = {};
};

static_assert(sizeof(AmxTileConfig) == 64, "TILECFG must be exactly 64 bytes");

/// Programs tile ``index`` of ``config`` with ``rows`` rows and ``colsb`` bytes
/// per row, selecting palette 1 (the only palette defined today).
///
/// Returns ``false`` without modifying the tile when the request exceeds the
/// architectural limits (``index`` >= :cpp:var:`kAmxMaxTiles`, ``rows`` in
/// ``[1, kAmxMaxRows]``, ``colsb`` in ``[1, kAmxMaxColBytes]`` and a multiple of
/// four bytes). Palette 1 is set on success.
bool AmxTileConfigSetTile(AmxTileConfig &config, std::size_t index, std::size_t rows,
                          std::size_t colsb);

/// Returns whether AMX tile state can be used from the current process.
///
/// On the first call this requests the OS permission needed to use the tile
/// data XSAVE component (``ARCH_REQ_XCOMP_PERM`` on Linux) and then re-checks
/// :cpp:func:`CpuSupportsAmxTile`. The result is cached for subsequent calls, so
/// the permission request happens at most once. Returns ``false`` on any
/// platform without AMX or without OS-enabled tile state.
bool AmxTileStateAvailable();

/// RAII guard that configures the AMX tiles for the current (worker) thread.
///
/// Tile configuration is thread-local hardware state, so each worker that runs
/// AMX tile operations constructs its own scope. When AMX tile state is
/// available the constructor runs ``LDTILECFG`` with ``config`` and the
/// destructor runs ``TILERELEASE``; otherwise it is a no-op and
/// :cpp:func:`configured` reports ``false`` so callers can take the fallback
/// path. The guard is neither copyable nor movable.
class AmxTileScope {
public:
  explicit AmxTileScope(const AmxTileConfig &config);
  ~AmxTileScope();

  AmxTileScope(const AmxTileScope &) = delete;
  AmxTileScope &operator=(const AmxTileScope &) = delete;
  AmxTileScope(AmxTileScope &&) = delete;
  AmxTileScope &operator=(AmxTileScope &&) = delete;

  /// Whether the tiles were configured (AMX available and loaded).
  bool configured() const noexcept { return configured_; }

private:
  bool configured_;
};

} // namespace onnx_light_cpu
