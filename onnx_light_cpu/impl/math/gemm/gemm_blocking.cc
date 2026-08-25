// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/cpu_cache_topology.h"
#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/gemm/gemm_common.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

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

constexpr std::size_t kTargetFmasPerSchedulingTask = 8 * 1024 * 1024;
constexpr std::size_t kKcL1CapacityMultiplier = 3;

// Portable defaults used whenever the reusable cache topology (see
// cpu_cache_topology.h) reports no detected data or unified cache at a
// level. These match the historical hard-coded values so existing blocking
// behavior is unchanged when detection is unavailable.
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

GemmMicroarchitecture DetectMicroarchitecture() {
  const CpuidResult vendor_leaf = Cpuid(0);
  const bool intel = vendor_leaf.ebx == 0x756e6547u && vendor_leaf.edx == 0x49656e69u &&
                     vendor_leaf.ecx == 0x6c65746eu;
  const bool amd = vendor_leaf.ebx == 0x68747541u && vendor_leaf.edx == 0x69746e65u &&
                   vendor_leaf.ecx == 0x444d4163u;
  const CpuidResult version = Cpuid(1);
  const unsigned int base_family = (version.eax >> 8) & 0xfu;
  const unsigned int base_model = (version.eax >> 4) & 0xfu;
  const unsigned int extended_family = (version.eax >> 20) & 0xffu;
  const unsigned int extended_model = (version.eax >> 16) & 0xfu;
  const unsigned int family = base_family == 0xfu ? base_family + extended_family : base_family;
  const unsigned int model = base_model | (extended_model << 4);
  const bool modern_intel_core =
      model == 0x4eu || model == 0x55u || model == 0x5eu || model == 0x66u || model == 0x6au ||
      model == 0x6cu || model == 0x7du || model == 0x7eu || model == 0x8cu || model == 0x8du ||
      model == 0x8eu || model == 0x8fu || model == 0x97u || model == 0x9au || model == 0x9eu ||
      model == 0xa7u || model == 0xb7u || model == 0xbau || model == 0xbfu;
  if (intel && family == 6 && modern_intel_core) {
    return GemmMicroarchitecture::kIntelCore;
  }
  if (amd && family >= 0x17u) {
    return GemmMicroarchitecture::kAmdZen;
  }
  return GemmMicroarchitecture::kGeneric;
}

#endif

// Reads the reusable, cross-platform cache descriptor (see
// cpu_cache_topology.h) once and falls back to the historical portable
// defaults for any level it did not detect.
CacheSizes DetectCacheSizes() {
  CacheSizes sizes;
  const CpuCacheTopology &topology = GetCpuCacheTopology();
  sizes.l1 = CpuCacheSizeBytesOrFallback(topology, 1, sizes.l1);
  sizes.l2 = CpuCacheSizeBytesOrFallback(topology, 2, sizes.l2);
  sizes.l3 = CpuCacheSizeBytesOrFallback(topology, 3, sizes.l3);
  return sizes;
}

const CacheSizes &CachedCacheSizes() {
  static const CacheSizes sizes = DetectCacheSizes();
  return sizes;
}

std::size_t AlignDown(std::size_t value, std::size_t alignment) {
  return value / alignment * alignment;
}

std::size_t AlignUp(std::size_t value, std::size_t alignment) {
  const std::size_t remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  const std::size_t increment = alignment - remainder;
  return value > std::numeric_limits<std::size_t>::max() - increment ? AlignDown(value, alignment)
                                                                     : value + increment;
}

std::size_t CeilDiv(std::size_t value, std::size_t divisor) {
  return value / divisor + static_cast<std::size_t>(value % divisor != 0);
}

std::size_t BoundedAligned(std::size_t value, std::size_t minimum, std::size_t maximum,
                           std::size_t alignment) {
  value = std::clamp(value, minimum, maximum);
  return std::max(minimum, AlignDown(value, alignment));
}

} // namespace

GemmBlocking SelectGemmBlocking(std::size_t element_size, std::size_t vector_lanes,
                                std::size_t register_rows) {
  const CacheSizes caches = CachedCacheSizes();
  const std::size_t mr = register_rows;
  const std::size_t nr = std::max<std::size_t>(2, 2 * vector_lanes);

  // A larger KC amortizes packing, workspace setup, and executor barriers
  // across fewer K chunks. The micro-kernel still walks cache-sized column
  // slices, so measured transformer and square workloads tolerate several
  // times the strict one-A-plus-one-B-micro-panel L1 capacity.
  const std::size_t kc_capacity = (caches.l1 * 3 / 4) / (element_size * (mr + nr));
  const std::size_t kc = BoundedAligned(kc_capacity * kKcL1CapacityMultiplier, 64, 512, 16);

  // Reserve half of L2 for the packed A panel and competing working data.
  const std::size_t mc_capacity = (caches.l2 / 2) / (element_size * kc);
  const std::size_t mc = BoundedAligned(mc_capacity, mr, 256, mr);

  // Packing rounds each B row to the active vector width. Keep NC aligned to
  // NR even without an L3-derived profile so scalable, non-power-of-two SVE
  // widths cannot make the packed row stride exceed its allocated panel.
  std::size_t nc = std::max(nr, AlignDown(kGemmTileN, nr));
  if (caches.l3 != 0) {
    // Bound a shared B panel to half of L3 and avoid oversized packing tasks.
    const std::size_t nc_capacity = (caches.l3 / 2) / (element_size * kc);
    nc = BoundedAligned(nc_capacity, nr, 1024, nr);
  }
  return GemmBlocking{mc, nc, kc, mr, nr};
}

GemmBlocking ConstrainGemmBlockingForTasks(GemmBlocking blocking, std::size_t m, std::size_t n,
                                           std::size_t k, std::size_t thread_count,
                                           std::size_t element_size,
                                           std::size_t target_fmas_per_participant) {
  if (m == 0 || n == 0 || k == 0 || thread_count <= 1) {
    return blocking;
  }

  thread_count = SelectGemmParticipantCount(m, n, k, thread_count, target_fmas_per_participant);

  const std::size_t max_row_tasks = CeilDiv(m, blocking.mr);
  const std::size_t scheduling_column_block = SelectGemmColumnBlock(blocking, element_size);
  const std::size_t column_tasks = CeilDiv(n, scheduling_column_block);
  const std::size_t max_tasks =
      max_row_tasks > thread_count / column_tasks ? thread_count : max_row_tasks * column_tasks;
  const std::size_t target_tasks = std::min(thread_count, max_tasks);

  const std::size_t row_tasks = CeilDiv(m, blocking.mc);
  if (row_tasks * column_tasks >= target_tasks) {
    return blocking;
  }

  const std::size_t desired_row_tasks =
      std::min(max_row_tasks, CeilDiv(target_tasks, column_tasks));
  blocking.mc = std::max(blocking.mr, AlignUp(CeilDiv(m, desired_row_tasks), blocking.mr));

  return blocking;
}

std::size_t SelectGemmParticipantCount(std::size_t m, std::size_t n, std::size_t k,
                                       std::size_t available_threads,
                                       std::size_t target_fmas_per_participant) {
  if (m == 0 || n == 0 || available_threads <= 1) {
    return 1;
  }
  const long double total_work =
      k == 0
          ? static_cast<long double>(m) * static_cast<long double>(n)
          : static_cast<long double>(m) * static_cast<long double>(n) * static_cast<long double>(k);
  const std::size_t target_work =
      k == 0 ? static_cast<std::size_t>(kExecutionGrainSize)
             : (target_fmas_per_participant == 0 ? kTargetFmasPerSchedulingTask
                                                 : target_fmas_per_participant);
  const long double work_tasks = std::ceil(total_work / static_cast<long double>(target_work));
  const std::size_t work_limited_threads =
      work_tasks >= static_cast<long double>(std::numeric_limits<std::size_t>::max())
          ? available_threads
          : std::max<std::size_t>(1, static_cast<std::size_t>(work_tasks));
  return std::min(available_threads, work_limited_threads);
}

std::size_t SelectGemmColumnBlock(const GemmBlocking &blocking, std::size_t element_size) {
  const std::size_t bytes = std::max<std::size_t>(1, element_size);
  const std::size_t nr = std::max<std::size_t>(1, blocking.nr);
  const std::size_t kc = std::max<std::size_t>(1, blocking.kc);
  // Measured width of the contiguous column micro-panel the tile loops walk:
  // a few cache lines per k row keep the packed B slice cheap to prefetch and
  // small enough to be reused from cache by every row tile of the packed A
  // panel.
  std::size_t block = std::max(nr, AlignDown(kGemmColumnPanelBytes / bytes, nr));
  // Never let the kc x block slice crowd the packed A panel out of L2:
  // SelectGemmBlocking already reserves half of L2 for that panel, so the B
  // micro-panel gets a quarter and the rest absorbs the output tile and the
  // competing working data.
  const std::size_t l2_capacity = (CachedCacheSizes().l2 / 4) / (bytes * kc);
  block = std::max(nr, std::min(block, AlignDown(l2_capacity, nr)));
  return blocking.nc == 0 ? block : std::min(block, blocking.nc);
}

GemmMicroarchitecture DetectGemmMicroarchitecture() {
#if ONNX_LIGHT_CPU_X86
  static const GemmMicroarchitecture microarchitecture = DetectMicroarchitecture();
  return microarchitecture;
#else
  return GemmMicroarchitecture::kGeneric;
#endif
}

std::size_t SelectGemmRegisterRowsForMicroarchitecture(SimdLevel level, bool has_fma,
                                                       GemmMicroarchitecture microarchitecture) {
  if (level >= SimdLevel::kAVX512) {
    return 6;
  }
  if (level >= SimdLevel::kAVX2 && has_fma) {
    // AMD Zen's two FMA pipelines need a wider register tile than the
    // conservative default to keep enough independent accumulator chains in
    // flight; modern Intel Core parts prefer a five-row tile. Unknown
    // (generic) x86 parts keep the safe four-row default.
    switch (microarchitecture) {
    case GemmMicroarchitecture::kIntelCore:
      return kGemmIntelAVX2MR;
    case GemmMicroarchitecture::kAmdZen:
      return kGemmZenAVX2MR;
    case GemmMicroarchitecture::kGeneric:
      break;
    }
    return kGemmMR;
  }
  return kGemmMR;
}

std::size_t SelectGemmRegisterRows(SimdLevel level, bool has_fma) {
  return SelectGemmRegisterRowsForMicroarchitecture(level, has_fma, DetectGemmMicroarchitecture());
}

} // namespace onnx_light_cpu::detail
