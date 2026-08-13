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

} // namespace

SimdLevel DetectSimdLevel() {
  auto info1 = Cpuid(1);
  bool has_sse2 = (info1.edx & (1u << 26)) != 0;
  bool has_avx = (info1.ecx & (1u << 28)) != 0;

  auto info7 = Cpuid(7);
  bool has_avx2 = (info7.ebx & (1u << 5)) != 0;
  bool has_avx512f = (info7.ebx & (1u << 16)) != 0;

  if (has_avx512f && OsSupportsAvx512())
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
  const auto info7 = Cpuid(7);
  const bool has_avx512bw = (info7.ebx & (1u << 30)) != 0;
  return has_avx512bw && OsSupportsAvx512();
}

#else // Non-x86

SimdLevel DetectSimdLevel() { return SimdLevel::kNone; }

bool CpuSupportsAvx512BW() { return false; }

#endif // ONNX_LIGHT_CPU_X86

} // namespace onnx_light_cpu
