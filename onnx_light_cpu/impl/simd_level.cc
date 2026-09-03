// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/simd_level.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define ONNX_LIGHT_CPU_X86 1
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#else
#define ONNX_LIGHT_CPU_X86 0
#endif

namespace onnx_light_cpu {

// ---------------------------------------------------------------------------
// CPUID-based runtime SIMD detection (x86 only)
// ---------------------------------------------------------------------------

#if ONNX_LIGHT_CPU_X86

namespace {

#ifndef ONNX_LIGHT_CPU_MAX_SIMD_LEVEL
#define ONNX_LIGHT_CPU_MAX_SIMD_LEVEL 4
#endif

static_assert(ONNX_LIGHT_CPU_MAX_SIMD_LEVEL >= static_cast<int>(SimdLevel::kAVX2) &&
              ONNX_LIGHT_CPU_MAX_SIMD_LEVEL <= static_cast<int>(SimdLevel::kAVX512));
constexpr SimdLevel kMaximumSimdLevel = static_cast<SimdLevel>(ONNX_LIGHT_CPU_MAX_SIMD_LEVEL);

struct CpuidResult {
  unsigned int eax, ebx, ecx, edx;
};

CpuidResult Cpuid(unsigned int leaf, unsigned int subleaf = 0) {
  CpuidResult r{};
#if defined(_MSC_VER)
  int regs[4];
  __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
  r.eax = static_cast<unsigned int>(regs[0]);
  r.ebx = static_cast<unsigned int>(regs[1]);
  r.ecx = static_cast<unsigned int>(regs[2]);
  r.edx = static_cast<unsigned int>(regs[3]);
#else
  __cpuid_count(leaf, subleaf, r.eax, r.ebx, r.ecx, r.edx);
#endif
  return r;
}

// Check if the OS has enabled AVX state saving (XGETBV).
bool OsSupportsAvx() {
  // OSXSAVE bit in CPUID.01H:ECX[27]
  auto info1 = Cpuid(1);
  if (!(info1.ecx & (1u << 27)))
    return false;
  // XCR0[2:1] = 0b11 means OS saves YMM state.
  unsigned long long xcr0;
#if defined(_MSC_VER)
  xcr0 = _xgetbv(0);
#else
  unsigned int lo, hi;
  __asm__ __volatile__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
  xcr0 = (static_cast<unsigned long long>(hi) << 32) | lo;
#endif
  return (xcr0 & 0x6) == 0x6;
}

bool OsSupportsAvx512() {
  if (!OsSupportsAvx())
    return false;
  unsigned long long xcr0;
#if defined(_MSC_VER)
  xcr0 = _xgetbv(0);
#else
  unsigned int lo, hi;
  __asm__ __volatile__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
  xcr0 = (static_cast<unsigned long long>(hi) << 32) | lo;
#endif
  // opmask (bit 5), ZMM_Hi256 (bit 6), Hi16_ZMM (bit 7)
  return (xcr0 & 0xE6) == 0xE6;
}

// Check if the OS has enabled AMX tile state (XTILECFG bit 17 and XTILEDATA
// bit 18 of XCR0). AMX does not require AVX-512 state, so this is independent
// of ``OsSupportsAvx512``.
bool OsSupportsAmxTileState() {
  // OSXSAVE bit in CPUID.01H:ECX[27] gates XGETBV.
  auto info1 = Cpuid(1);
  if (!(info1.ecx & (1u << 27)))
    return false;
  unsigned long long xcr0;
#if defined(_MSC_VER)
  xcr0 = _xgetbv(0);
#else
  unsigned int lo, hi;
  __asm__ __volatile__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
  xcr0 = (static_cast<unsigned long long>(hi) << 32) | lo;
#endif
  // XTILECFG (bit 17) and XTILEDATA (bit 18).
  return (xcr0 & 0x60000) == 0x60000;
}

} // namespace

SimdLevel DetectSimdLevel() {
  auto info1 = Cpuid(1);
  bool has_sse2 = (info1.edx & (1u << 26)) != 0;
  bool has_avx = (info1.ecx & (1u << 28)) != 0;

  auto info7 = Cpuid(7);
  bool has_avx2 = (info7.ebx & (1u << 5)) != 0;
  bool has_avx512f = (info7.ebx & (1u << 16)) != 0;

  if (kMaximumSimdLevel >= SimdLevel::kAVX512 && has_avx512f && OsSupportsAvx512())
    return SimdLevel::kAVX512;
  if (has_avx2 && OsSupportsAvx())
    return SimdLevel::kAVX2;
  if (has_avx && OsSupportsAvx())
    return SimdLevel::kAVX;
  if (has_sse2)
    return SimdLevel::kSSE2;
  return SimdLevel::kNone;
}

bool CpuSupportsAvx512BW() {
  if constexpr (kMaximumSimdLevel < SimdLevel::kAVX512)
    return false;
  const auto info7 = Cpuid(7);
  const bool has_avx512bw = (info7.ebx & (1u << 30)) != 0;
  return has_avx512bw && OsSupportsAvx512();
}

bool CpuSupportsFma() {
  const auto info1 = Cpuid(1);
  const bool has_fma = (info1.ecx & (1u << 12)) != 0;
  return has_fma && OsSupportsAvx();
}

bool CpuSupportsF16C() {
  const auto info1 = Cpuid(1);
  const bool has_f16c = (info1.ecx & (1u << 29)) != 0;
  return has_f16c && OsSupportsAvx();
}

bool CpuSupportsAvx512Fp16() {
  if constexpr (kMaximumSimdLevel < SimdLevel::kAVX512)
    return false;
  // AVX512_FP16 is reported by CPUID.(EAX=7,ECX=0):EDX[23]. Native FP16
  // instructions load their operands into ZMM registers, so the OS must also
  // save AVX-512 state (checked by ``OsSupportsAvx512``).
  const auto info7 = Cpuid(7);
  const bool has_avx512fp16 = (info7.edx & (1u << 23)) != 0;
  return has_avx512fp16 && OsSupportsAvx512();
}

bool CpuSupportsAvx512Bf16() {
  if constexpr (kMaximumSimdLevel < SimdLevel::kAVX512)
    return false;
  // AVX512_BF16 is reported by CPUID.(EAX=7,ECX=1):EAX[5]. The native
  // ``vdpbf16ps`` dot-product reads its operands from ZMM registers, so the OS
  // must also save AVX-512 state (checked by ``OsSupportsAvx512``).
  const auto info7 = Cpuid(7, 1);
  const bool has_avx512bf16 = (info7.eax & (1u << 5)) != 0;
  return has_avx512bf16 && OsSupportsAvx512();
}

bool CpuSupportsAvx512Vnni() {
  if constexpr (kMaximumSimdLevel < SimdLevel::kAVX512)
    return false;
  // AVX512_VNNI is reported by CPUID.(EAX=7,ECX=0):ECX[11]. The native
  // ``vpdpbusd`` dot-product reads its operands from ZMM registers, so the OS
  // must also save AVX-512 state (checked by ``OsSupportsAvx512``).
  const auto info7 = Cpuid(7);
  const bool has_avx512vnni = (info7.ecx & (1u << 11)) != 0;
  return has_avx512vnni && OsSupportsAvx512();
}

bool CpuSupportsAmxTile() {
  if constexpr (kMaximumSimdLevel < SimdLevel::kAVX512)
    return false;
  // AMX-TILE is reported by CPUID.(EAX=7,ECX=0):EDX[24]. Tile registers live in
  // the AMX XSAVE components, so the OS must also enable tile state (checked by
  // ``OsSupportsAmxTileState``).
  const auto info7 = Cpuid(7);
  const bool has_amx_tile = (info7.edx & (1u << 24)) != 0;
  return has_amx_tile && OsSupportsAmxTileState();
}

bool CpuSupportsAmxBf16() {
  if constexpr (kMaximumSimdLevel < SimdLevel::kAVX512)
    return false;
  // AMX-BF16 (EDX[22]) relies on the same OS-enabled tile state as AMX-TILE
  // (EDX[24]). Both bits and the OS check are read from a single CPUID leaf.
  const auto info7 = Cpuid(7);
  const bool has_amx_bf16 = (info7.edx & (1u << 22)) != 0;
  const bool has_amx_tile = (info7.edx & (1u << 24)) != 0;
  return has_amx_bf16 && has_amx_tile && OsSupportsAmxTileState();
}

bool CpuSupportsAmxInt8() {
  if constexpr (kMaximumSimdLevel < SimdLevel::kAVX512)
    return false;
  // AMX-INT8 (EDX[25]) relies on the same OS-enabled tile state as AMX-TILE
  // (EDX[24]). Both bits and the OS check are read from a single CPUID leaf.
  const auto info7 = Cpuid(7);
  const bool has_amx_int8 = (info7.edx & (1u << 25)) != 0;
  const bool has_amx_tile = (info7.edx & (1u << 24)) != 0;
  return has_amx_int8 && has_amx_tile && OsSupportsAmxTileState();
}

#else // Non-x86

SimdLevel DetectSimdLevel() { return SimdLevel::kNone; }

bool CpuSupportsAvx512BW() { return false; }

bool CpuSupportsFma() { return false; }

bool CpuSupportsF16C() { return false; }

bool CpuSupportsAvx512Fp16() { return false; }

bool CpuSupportsAvx512Bf16() { return false; }

bool CpuSupportsAvx512Vnni() { return false; }

bool CpuSupportsAmxTile() { return false; }

bool CpuSupportsAmxBf16() { return false; }

bool CpuSupportsAmxInt8() { return false; }

#endif // ONNX_LIGHT_CPU_X86

} // namespace onnx_light_cpu
