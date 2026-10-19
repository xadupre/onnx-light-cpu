// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <vector>

namespace onnx_light_cpu {

/// Functional kind reported for one cache descriptor.
enum class CpuCacheKind {
  kUnknown,
  kData,
  kInstruction,
  kUnified,
};

/// How confidently a cache descriptor's fields were established.
enum class CpuCacheConfidence {
  /// Read directly from a platform topology or CPUID interface.
  kDetected,
  /// Derived from a partial platform read (for example a size without a
  /// matching line size or sharing count).
  kInferred,
  /// No platform information was available; the value is a portable default.
  kFallback,
};

/// One reusable cache level descriptor. Multiple descriptors may share the
/// same ``level`` when the topology is heterogeneous (for example distinct
/// private caches for performance and efficiency core clusters).
struct CpuCacheDescriptor {
  unsigned int level = 0;
  CpuCacheKind kind = CpuCacheKind::kUnknown;
  std::size_t size_bytes = 0;
  std::size_t line_size_bytes = 0;
  /// Number of logical threads sharing one instance of this cache, or 0 when
  /// unknown.
  std::size_t sharing_thread_count = 0;
  CpuCacheConfidence confidence = CpuCacheConfidence::kFallback;
};

/// Reusable, process-visible cache topology.
struct CpuCacheTopology {
  std::vector<CpuCacheDescriptor> caches;
  /// True when a platform-specific detector (CPUID, sysfs, Windows API,
  /// sysctl) supplied at least one descriptor. False means every consumer
  /// must rely on the explicit generic/partial fallback.
  bool platform_detected = false;
};

/// Returns the process-visible cache topology, detected once.
const CpuCacheTopology &GetCpuCacheTopology();

/// Returns the first descriptor at ``level`` matching ``kind`` (or any data
/// or unified cache at that level when ``kind`` is ``kUnknown``), or nullptr
/// when the topology reports no such cache.
const CpuCacheDescriptor *FindCpuCacheDescriptor(const CpuCacheTopology &topology,
                                                 unsigned int level,
                                                 CpuCacheKind kind = CpuCacheKind::kUnknown);

/// Returns the largest data/unified cache size at ``level``, or
/// ``fallback_bytes`` when the topology reports no such cache. Used by
/// callers (for example GEMM blocking) that need one deterministic byte
/// count per level regardless of detection confidence.
std::size_t CpuCacheSizeBytesOrFallback(const CpuCacheTopology &topology, unsigned int level,
                                        std::size_t fallback_bytes);

namespace detail {

/// One platform-neutral cache observation used to build a
/// :cpp:struct:`CpuCacheTopology`. Test code injects these directly so
/// topology construction can be validated without depending on the host
/// machine's actual cache hierarchy.
struct CpuCacheRecord {
  unsigned int level = 0;
  CpuCacheKind kind = CpuCacheKind::kUnknown;
  std::size_t size_bytes = 0;
  std::size_t line_size_bytes = 0;
  /// 0 means unknown/unreported.
  std::size_t sharing_thread_count = 0;
  CpuCacheConfidence confidence = CpuCacheConfidence::kDetected;
  /// False marks a malformed record (for example an out-of-range level or a
  /// zero size) that must be discarded instead of reported.
  bool valid = true;
};

/// Builds a deterministic topology from raw records, discarding invalid
/// entries. ``platform_supported`` records whether the caller's detector
/// understood the current platform at all (even if it found nothing),
/// which becomes :cpp:member:`CpuCacheTopology::platform_detected` together
/// with whether any valid record survived.
CpuCacheTopology BuildCpuCacheTopology(const std::vector<CpuCacheRecord> &records,
                                       bool platform_supported);

/// Platform-specific raw cache record collection, exposed for reuse and
/// testing. Tries, in order, an OS topology interface (Linux sysfs, Windows
/// ``GetLogicalProcessorInformationEx``, macOS ``sysctlbyname``) and, when
/// unavailable, x86 CPUID deterministic cache parameters. Returns an empty
/// vector when no platform-specific source is available.
std::vector<CpuCacheRecord> DetectRawCpuCacheRecords();

} // namespace detail

} // namespace onnx_light_cpu
