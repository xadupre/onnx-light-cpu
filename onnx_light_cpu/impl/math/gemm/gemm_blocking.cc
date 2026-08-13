// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/gemm/gemm_common.h"

#include <algorithm>
#include <cstddef>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define ONNX_LIGHT_CPU_X86 1
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#else
#define ONNX_LIGHT_CPU_X86 0
#endif

namespace onnx_light_cpu::detail {

namespace {

struct CacheSizes {
  std::size_t l1 = 32 * 1024;
  std::size_t l2 = 256 * 1024;
  std::size_t l3 = 0;
};

#if ONNX_LIGHT_CPU_X86

struct CpuidResult {
  unsigned int eax = 0;
  unsigned int ebx = 0;
  unsigned int ecx = 0;
  unsigned int edx = 0;
};

CpuidResult Cpuid(unsigned int leaf, unsigned int subleaf = 0) {
  CpuidResult result;
#if defined(_MSC_VER)
  int registers[4];
  __cpuidex(registers, static_cast<int>(leaf), static_cast<int>(subleaf));
  result.eax = static_cast<unsigned int>(registers[0]);
  result.ebx = static_cast<unsigned int>(registers[1]);
  result.ecx = static_cast<unsigned int>(registers[2]);
  result.edx = static_cast<unsigned int>(registers[3]);
#else
  __cpuid_count(leaf, subleaf, result.eax, result.ebx, result.ecx, result.edx);
#endif
  return result;
}

bool ReadDeterministicCaches(unsigned int leaf, CacheSizes &sizes) {
  bool found = false;
  for (unsigned int index = 0;; ++index) {
    const CpuidResult cache = Cpuid(leaf, index);
    const unsigned int type = cache.eax & 0x1fu;
    if (type == 0) {
      break;
    }
    if (type == 2) {
      continue;
    }
    found = true;
    const unsigned int level = (cache.eax >> 5) & 0x7u;
    const std::size_t line_size = (cache.ebx & 0xfffu) + 1;
    const std::size_t partitions = ((cache.ebx >> 12) & 0x3ffu) + 1;
    const std::size_t ways = ((cache.ebx >> 22) & 0x3ffu) + 1;
    const std::size_t sets = static_cast<std::size_t>(cache.ecx) + 1;
    const std::size_t size = line_size * partitions * ways * sets;
    if (level == 1) {
      sizes.l1 = std::max(sizes.l1, size);
    } else if (level == 2) {
      sizes.l2 = std::max(sizes.l2, size);
    } else if (level == 3) {
      sizes.l3 = std::max(sizes.l3, size);
    }
  }
  return found;
}

#endif

CacheSizes DetectCacheSizes() {
  CacheSizes sizes;
#if ONNX_LIGHT_CPU_X86
  const unsigned int max_basic = Cpuid(0).eax;
  const bool found = max_basic >= 4 && ReadDeterministicCaches(4, sizes);
  const unsigned int max_extended = Cpuid(0x80000000u).eax;
  if (!found && max_extended >= 0x8000001du) {
    ReadDeterministicCaches(0x8000001du, sizes);
  }
#endif
  return sizes;
}

std::size_t AlignDown(std::size_t value, std::size_t alignment) {
  return value / alignment * alignment;
}

std::size_t BoundedAligned(std::size_t value, std::size_t minimum, std::size_t maximum,
                           std::size_t alignment) {
  value = std::clamp(value, minimum, maximum);
  return std::max(minimum, AlignDown(value, alignment));
}

} // namespace

GemmBlocking SelectGemmBlocking(std::size_t element_size, std::size_t vector_lanes) {
  const CacheSizes caches = DetectCacheSizes();
  const std::size_t mr = kGemmMR;
  const std::size_t nr = std::max<std::size_t>(2, 2 * vector_lanes);

  // Keep one A micro-panel and one B micro-panel within roughly 75% of L1.
  const std::size_t kc_capacity = (caches.l1 * 3 / 4) / (element_size * (mr + nr));
  const std::size_t kc = BoundedAligned(kc_capacity, 64, 512, 16);

  // Reserve half of L2 for the packed A panel and competing working data.
  const std::size_t mc_capacity = (caches.l2 / 2) / (element_size * kc);
  const std::size_t mc = BoundedAligned(mc_capacity, mr, 256, mr);

  std::size_t nc = kGemmTileN;
  if (caches.l3 != 0) {
    // Bound a shared B panel to half of L3 and avoid oversized packing tasks.
    const std::size_t nc_capacity = (caches.l3 / 2) / (element_size * kc);
    nc = BoundedAligned(nc_capacity, nr, 1024, nr);
  }
  return GemmBlocking{mc, nc, kc, mr, nr};
}

} // namespace onnx_light_cpu::detail
