// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "onnx_light_cpu/impl/cpu_cache_topology.h"

namespace onnx_light_cpu {

/// Memory level targeted by one working-set selection or measurement.
enum class MemoryProfileLevel {
  kL1,
  kL2,
  kL3,
  kRam,
};

/// Traffic pattern exercised by a bandwidth kernel.
enum class MemoryTrafficMode {
  kRead,
  kWrite,
  kCopy,
  kReadModifyWrite,
};

/// Number and placement of participants used by one measurement.
enum class MemoryParticipantPolicy {
  kSingle,
  kPhysical,
};

/// Explicit reason a measurement could not produce a truthful result.
enum class MemoryProfileUnavailableReason {
  kNone,
  /// The process-visible topology has no usable descriptor for the level.
  kLevelUnavailable,
  /// The configured memory budget cannot satisfy the level's working-set
  /// contract (for example RAM must exceed twice the last-level cache).
  kMemoryBudgetExceeded,
  /// The requested repeat count, minimum duration, or memory budget is
  /// invalid; the measurement fails before allocating or timing anything.
  kInvalidOptions,
};

/// Options shared by every memory bandwidth and latency measurement.
struct MemoryProfileOptions {
  /// Number of recorded samples after warmup.
  std::size_t repeats = 7;
  /// Minimum wall-clock duration, per recorded sample, used to size the
  /// number of internal passes so timer resolution stays negligible.
  double minimum_duration_ms = 20.0;
  /// Total memory budget available across every participant.
  std::size_t memory_budget_bytes = 512 * 1024 * 1024;
};

/// Deterministic, pure working-set selection for one memory level. Exposed
/// directly so tests can validate selection against injected cache
/// descriptors without allocating or timing anything.
struct MemoryWorkingSet {
  bool available = false;
  std::size_t working_set_bytes = 0;
  MemoryProfileUnavailableReason unavailable_reason = MemoryProfileUnavailableReason::kNone;
};

/// Chooses a safe per-participant working-set size for ``level`` from
/// ``topology``, honoring the memory budget split across ``participant_count``
/// participants (at least one). Never returns an RAM-labelled working set
/// that fits within twice the detected last-level cache.
MemoryWorkingSet SelectMemoryWorkingSet(const CpuCacheTopology &topology, MemoryProfileLevel level,
                                        std::size_t participant_count,
                                        std::size_t memory_budget_bytes);

/// Pure, timing-independent byte accounting for one traffic mode. Every
/// stream touched by the kernel (source, destination, or the same address
/// read then written) is counted explicitly so the convention is auditable.
struct MemoryTrafficAccounting {
  /// Number of ``element_bytes``-sized elements touched per stream.
  std::size_t element_count = 0;
  /// Useful bytes moved by one full pass over the working set.
  std::uint64_t useful_bytes_per_pass = 0;
};

/// Computes the exact useful-byte accounting for ``mode`` over a working set
/// of ``working_set_bytes`` using ``element_bytes``-sized elements. Copy
/// splits the working set into equal source and destination streams;
/// read/write use one stream; read-modify-write touches one stream twice
/// (one load, one store) per element.
MemoryTrafficAccounting ComputeMemoryTrafficAccounting(MemoryTrafficMode mode,
                                                       std::size_t working_set_bytes,
                                                       std::size_t element_bytes);

/// Wall-clock timer identity used by every measurement in this file.
const char *MemoryProfileTimerName();

/// Result of one bandwidth measurement.
struct MemoryBandwidthResult {
  bool available = false;
  MemoryProfileUnavailableReason unavailable_reason = MemoryProfileUnavailableReason::kNone;
  std::string diagnostic;

  MemoryProfileLevel level = MemoryProfileLevel::kL1;
  MemoryTrafficMode mode = MemoryTrafficMode::kRead;
  MemoryParticipantPolicy policy = MemoryParticipantPolicy::kSingle;

  std::size_t working_set_bytes = 0;
  std::size_t participant_count = 0;
  bool affinity_pinned = false;
  std::string timer_name;

  /// Useful bytes accounted for one pass over one participant's working set.
  std::uint64_t useful_bytes_per_pass_per_participant = 0;

  /// One aggregate-throughput sample (summed across participants) per
  /// recorded repeat, in useful gigabytes per second (10^9 bytes/second).
  std::vector<double> raw_gbps_samples;
  double median_gbps = 0.0;
  /// Interquartile range of ``raw_gbps_samples``.
  double dispersion_gbps = 0.0;
};

/// Measures bandwidth for ``mode`` at ``level`` using ``policy`` participants.
/// Allocation, prefaulting, participant creation, and checksum consumption
/// happen outside every timed sample.
MemoryBandwidthResult MeasureMemoryBandwidth(MemoryProfileLevel level, MemoryTrafficMode mode,
                                             MemoryParticipantPolicy policy,
                                             const MemoryProfileOptions &options = {});

/// Result of one dependent pointer-chase latency measurement.
struct MemoryLatencyResult {
  bool available = false;
  MemoryProfileUnavailableReason unavailable_reason = MemoryProfileUnavailableReason::kNone;
  std::string diagnostic;

  MemoryProfileLevel level = MemoryProfileLevel::kL1;
  MemoryParticipantPolicy policy = MemoryParticipantPolicy::kSingle;

  std::size_t working_set_bytes = 0;
  std::size_t participant_count = 0;
  bool affinity_pinned = false;
  std::string timer_name;

  /// One nanoseconds-per-dependent-load sample per recorded repeat, averaged
  /// across participants for the physical policy.
  std::vector<double> raw_ns_per_load_samples;
  double median_ns_per_load = 0.0;
  double dispersion_ns_per_load = 0.0;
};

/// Measures dependent-load latency at ``level`` using ``policy`` participants.
/// The traversal permutation is built and validated outside every timed
/// sample.
MemoryLatencyResult MeasureMemoryLatency(MemoryProfileLevel level, MemoryParticipantPolicy policy,
                                         const MemoryProfileOptions &options = {});

namespace detail {

/// Builds a randomized single-cycle permutation over ``count`` indices
/// (Sattolo's algorithm), so a dependent pointer chase starting from any
/// index visits every element exactly once before repeating. Exposed for
/// direct testing. ``count`` must be at least 1.
std::vector<std::uint32_t> BuildPointerChasePermutation(std::size_t count, std::uint64_t seed);

} // namespace detail

} // namespace onnx_light_cpu
