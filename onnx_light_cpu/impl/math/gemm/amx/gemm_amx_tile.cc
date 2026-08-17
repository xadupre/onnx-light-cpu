// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// AMX tile-state lifecycle implementation (Roadmap PR07.5). See
// gemm_amx_tile.h for the contract. The native ``LDTILECFG``/``TILERELEASE``
// path is only compiled when the toolchain accepts ``-mamx-tile`` (guarded by
// ONNX_LIGHT_CPU_HAVE_AMX_TILE from CMake); every other build keeps the safe
// fallback so the module still links and reports "not available".

#include "onnx_light_cpu/impl/math/gemm/amx/gemm_amx_tile.h"

#include "onnx_light_cpu/impl/simd_level.h"

#if defined(ONNX_LIGHT_CPU_HAVE_AMX_TILE)
#include <immintrin.h>
#endif

// ``arch_prctl`` (used to request the AMX tile-data XSAVE permission) is an
// x86-only syscall, so the permission path is limited to x86 Linux builds.
#if defined(__linux__) && (defined(__x86_64__) || defined(__i386__))
#define ONNX_LIGHT_CPU_AMX_REQUEST_PERM 1
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace onnx_light_cpu {

namespace {

#if defined(ONNX_LIGHT_CPU_AMX_REQUEST_PERM)
// arch_prctl request codes and XSAVE feature indices for AMX tile data. These
// are stable ABI constants but may be missing from older <asm/prctl.h>, so they
// are defined locally to avoid a hard header dependency.
constexpr unsigned long kArchReqXCompPerm = 0x1023;
constexpr unsigned long kXfeatureXTileData = 18;

// Requests OS permission to use the AMX tile-data XSAVE component. The syscall
// is harmless when AMX is absent (it simply returns an error), so the result is
// ignored and callers re-check ``CpuSupportsAmxTile`` afterwards.
void RequestAmxTileDataPermission() {
  syscall(SYS_arch_prctl, kArchReqXCompPerm, kXfeatureXTileData);
}
#endif

} // namespace

bool AmxTileConfigSetTile(AmxTileConfig &config, std::size_t index, std::size_t rows,
                          std::size_t colsb) {
  if (index >= kAmxMaxTiles)
    return false;
  if (rows == 0 || rows > kAmxMaxRows)
    return false;
  // A tile row holds up to 64 bytes and every supported element type is a
  // multiple of four bytes wide, matching the AMX ``colsb`` granularity.
  if (colsb == 0 || colsb > kAmxMaxColBytes || (colsb % 4) != 0)
    return false;

  config.palette_id = 1;
  config.rows[index] = static_cast<std::uint8_t>(rows);
  config.colsb[index] = static_cast<std::uint16_t>(colsb);
  return true;
}

bool AmxTileStateAvailable() {
  static const bool available = [] {
#if defined(ONNX_LIGHT_CPU_AMX_REQUEST_PERM)
    // The XTILEDATA component is disabled by default on Linux and must be
    // requested once per process before ``CpuSupportsAmxTile`` can observe it
    // in XCR0. The request is a no-op on CPUs without AMX.
    RequestAmxTileDataPermission();
#endif
    return CpuSupportsAmxTile();
  }();
  return available;
}

AmxTileScope::AmxTileScope(const AmxTileConfig &config) : configured_(false) {
#if defined(ONNX_LIGHT_CPU_HAVE_AMX_TILE)
  if (AmxTileStateAvailable()) {
    _tile_loadconfig(&config);
    configured_ = true;
  }
#else
  (void)config;
#endif
}

AmxTileScope::~AmxTileScope() {
#if defined(ONNX_LIGHT_CPU_HAVE_AMX_TILE)
  if (configured_) {
    _tile_release();
  }
#endif
}

} // namespace onnx_light_cpu
