// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Aggregate processor performance profile (Processor Profile PR04, see
// docs/next_steps/2026/2026_08_processor_performance_profile.rst). This is
// the versioned public result assembled from the memory bandwidth/latency
// engine (memory_traffic_profile.h) and the register-resident compute
// engine (compute_arithmetic_profile.h). It performs no measurement of its
// own beyond calling those two engines and deriving Roofline crossovers from
// their results.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "onnx_light_cpu/impl/compute_arithmetic_profile.h"
#include "onnx_light_cpu/impl/cpu_cache_topology.h"
#include "onnx_light_cpu/impl/memory_traffic_profile.h"
#include "onnx_light_cpu/impl/thread_topology.h"

namespace onnx_light_cpu {

/// Schema version of the ``ProcessorPerformanceProfile`` shape returned by
/// ``BenchmarkProcessorPerformance``. Bump whenever a field is added,
/// removed, renamed, or reinterpreted so downstream serialization consumers
/// can detect incompatible changes.
constexpr int kProcessorPerformanceProfileSchemaVersion = 1;

/// Participant policy requested for one profile section. Kept distinct from
/// ``MemoryParticipantPolicy``/``ComputeParticipantPolicy`` so the public
/// aggregate API does not depend on either engine's internal enum.
enum class ProcessorThreadPolicy {
  kSingle,
  kPhysical,
};

/// Parses the stable lowercase spelling of a policy (``"single"`` or
/// ``"physical"``). Returns false, leaving ``policy`` unchanged, when
/// ``name`` does not match a known policy.
bool ParseProcessorThreadPolicy(const std::string &name, ProcessorThreadPolicy &policy);

/// Returns the stable lowercase spelling of ``policy`` (``"single"`` or
/// ``"physical"``).
const char *ProcessorThreadPolicyName(ProcessorThreadPolicy policy);

/// Options accepted by ``BenchmarkProcessorPerformance``. Every field is
/// validated by ``ValidateProcessorProfileOptions`` before any allocation or
/// timing happens.
struct ProcessorProfileOptions {
  /// Thread policies to measure for every memory and compute section. Must
  /// be non-empty.
  std::vector<ProcessorThreadPolicy> thread_policies{ProcessorThreadPolicy::kSingle,
                                                     ProcessorThreadPolicy::kPhysical};
  /// Number of recorded samples after warmup, forwarded to both engines.
  std::size_t repeats = 7;
  /// Minimum wall-clock duration, per recorded sample, forwarded to both
  /// engines.
  double minimum_duration_ms = 20.0;
  /// Total memory budget available across every participant, forwarded to
  /// the memory engine.
  std::size_t memory_budget_bytes = 512 * 1024 * 1024;
  /// Whether to measure dependent-load latency in addition to bandwidth.
  bool include_latency = true;
  /// Explicit logical-processor affinity used to pin the ``kSingle`` policy's
  /// lone participant. Empty (the default) leaves the calling thread
  /// unpinned. When set, it must reference a logical processor present in
  /// ``GetCpuTopology()``.
  std::optional<CpuAffinity> explicit_single_affinity;
};

/// Validates ``options`` without allocating or timing anything. Returns an
/// empty string when ``options`` is valid, or a human-readable explanation of
/// the first problem found otherwise.
std::string ValidateProcessorProfileOptions(const ProcessorProfileOptions &options);

/// Metadata describing one profile run: schema version, wall-clock
/// timestamp, platform/compiler identification, the resolved options, the
/// shared timer identity, and free-form diagnostics unrelated to a specific
/// measurement.
struct ProcessorProfileMetadata {
  int schema_version = kProcessorPerformanceProfileSchemaVersion;
  /// Nanoseconds since the Unix epoch, sampled once at the start of the run.
  std::int64_t unix_timestamp_ns = 0;
  std::string platform;
  std::string compiler;
  std::string timer_name;
  ProcessorProfileOptions options;
  std::vector<std::string> diagnostics;
};

/// Process-visible topology and cache descriptors reused verbatim from
/// ``thread_topology.h``/``cpu_cache_topology.h`` so the profile is
/// self-contained.
struct ProcessorProfileTopology {
  std::size_t logical_thread_count = 1;
  std::size_t physical_core_count = 1;
  std::size_t performance_core_count = 0;
  std::size_t efficiency_core_count = 0;
  std::vector<CpuCacheDescriptor> caches;
  bool cache_topology_detected = false;
};

/// One memory-level, one-policy entry. Each bandwidth/latency field is only
/// present (``has_value()``) when the underlying engine reported it as
/// available; unavailable measurements are omitted here and explained in
/// ``ProcessorPerformanceProfile::warnings`` instead of being represented by
/// zero or fabricated values.
struct ProcessorProfileMemoryEntry {
  MemoryProfileLevel level = MemoryProfileLevel::kL1;
  ProcessorThreadPolicy policy = ProcessorThreadPolicy::kSingle;

  std::optional<MemoryBandwidthResult> read;
  std::optional<MemoryBandwidthResult> write;
  std::optional<MemoryBandwidthResult> copy;
  std::optional<MemoryBandwidthResult> read_modify_write;
  std::optional<MemoryLatencyResult> latency;
};

/// One element-type, one-policy compute entry, present only when the
/// underlying engine reported the measurement as available.
struct ProcessorProfileComputeEntry {
  ComputeElementType element_type = ComputeElementType::kFloat32;
  ProcessorThreadPolicy policy = ProcessorThreadPolicy::kSingle;
  ComputeThroughputResult result;
};

/// One derived Roofline crossover point for one element type, policy, and
/// memory level, built from an available compute entry and that policy's
/// available ``read`` bandwidth at that level. The crossover is the
/// arithmetic intensity (operations per useful byte) at which the compute
/// ceiling and the memory bandwidth ceiling are equal.
struct ProcessorProfileRooflineEntry {
  ComputeElementType element_type = ComputeElementType::kFloat32;
  ProcessorThreadPolicy policy = ProcessorThreadPolicy::kSingle;
  MemoryProfileLevel level = MemoryProfileLevel::kL1;

  /// Source compute throughput (billions of operations per second) this
  /// entry was derived from.
  double compute_gops = 0.0;
  /// Source memory read bandwidth (gigabytes per second) this entry was
  /// derived from.
  double memory_read_gbps = 0.0;
  /// ``compute_gops / memory_read_gbps``: operations per useful byte at the
  /// crossover between compute-bound and memory-bound execution.
  double arithmetic_intensity_crossover = 0.0;
};

/// Immutable, versioned, serializable processor performance profile. See the
/// module comment and
/// ``docs/next_steps/2026/2026_08_processor_performance_profile.rst`` for the
/// measurement contract every entry honors.
struct ProcessorPerformanceProfile {
  ProcessorProfileMetadata metadata;
  ProcessorProfileTopology topology;
  std::vector<ProcessorProfileMemoryEntry> memory;
  std::vector<ProcessorProfileComputeEntry> compute;
  std::vector<ProcessorProfileRooflineEntry> roofline;
  /// Explicit unavailable, inferred, noisy, unpinned, or memory-budget
  /// limited conditions encountered while assembling this profile. A missing
  /// memory level or compute element type is always accompanied by one entry
  /// here explaining why it is absent.
  std::vector<std::string> warnings;
};

/// Runs every configured memory and compute measurement and assembles the
/// resulting ``ProcessorPerformanceProfile``. Throws ``std::invalid_argument``
/// (with the message from ``ValidateProcessorProfileOptions``) before
/// allocating or timing anything when ``options`` is invalid.
ProcessorPerformanceProfile
BenchmarkProcessorPerformance(const ProcessorProfileOptions &options = {});

} // namespace onnx_light_cpu
