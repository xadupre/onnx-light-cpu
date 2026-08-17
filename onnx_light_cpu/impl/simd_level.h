// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace onnx_light_cpu {

/// Supported SIMD instruction sets for CPU kernel dispatch.
///
/// Shared by every kernel family (``math``, ``logical``, ...): the runtime
/// dispatch of each kernel selects its best code path based on the level
/// returned by ``DetectSimdLevel``.
enum class SimdLevel : int {
  kNone = 0,   ///< Scalar fallback (no SIMD).
  kSSE2 = 1,   ///< SSE2 (128-bit).
  kAVX = 2,    ///< AVX (256-bit).
  kAVX2 = 3,   ///< AVX2 (256-bit with FMA, integer ops).
  kAVX512 = 4, ///< AVX-512F (512-bit).
};

/// Detects the highest SIMD level supported by the current CPU at runtime.
SimdLevel DetectSimdLevel();

/// Returns whether AVX-512BW byte/word instructions are available and enabled
/// by the operating system. AVX-512F alone is not sufficient for byte kernels.
bool CpuSupportsAvx512BW();

/// Returns whether FMA instructions are available and AVX state is enabled by
/// the operating system.
bool CpuSupportsFma();

/// Returns whether F16C half-precision conversion instructions are available
/// and AVX state is enabled by the operating system.
bool CpuSupportsF16C();

/// Returns whether the AVX-512FP16 instruction set (native half-precision
/// arithmetic and conversion) is available and AVX-512 state is enabled by the
/// operating system.
bool CpuSupportsAvx512Fp16();

/// Returns whether the AVX-512BF16 instruction set (native ``vdpbf16ps``
/// bfloat16 dot-product) is available and AVX-512 state is enabled by the
/// operating system.
bool CpuSupportsAvx512Bf16();

/// Returns whether the AMX tile architecture (``AMX-TILE``) is present and the
/// operating system has enabled AMX tile state (the ``XTILECFG`` and
/// ``XTILEDATA`` components of ``XCR0``). On Linux the tile-data component is
/// only enabled after a successful ``ARCH_REQ_XCOMP_PERM`` request; use the AMX
/// tile-state lifecycle helper ``AmxTileStateAvailable`` (which performs that
/// request once) rather than calling this directly when deciding to dispatch an
/// AMX kernel.
bool CpuSupportsAmxTile();

/// Returns whether the AMX-BF16 tile multiply instructions are available and
/// AMX tile state is OS-enabled (see :cpp:func:`CpuSupportsAmxTile`).
bool CpuSupportsAmxBf16();

/// Returns whether the AMX-INT8 tile multiply instructions are available and
/// AMX tile state is OS-enabled (see :cpp:func:`CpuSupportsAmxTile`).
bool CpuSupportsAmxInt8();

} // namespace onnx_light_cpu
