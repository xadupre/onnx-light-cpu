// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Python binding for the versioned processor performance profile
// (Processor Profile PR04). Converts the pure C++
// ``onnx_light_cpu::ProcessorPerformanceProfile`` returned by
// ``BenchmarkProcessorPerformance`` into plain tuples/lists/optionals so
// nanobind can convert them without a dependency on onnx-light. The Python
// package wraps this raw shape into immutable, documented result classes
// (see ``onnx_light_cpu/_processor_profile.py``).
//
// This module never runs a benchmark unless ``benchmark_processor_performance``
// is called explicitly.

#include "onnx_light_cpu/impl/processor_performance_profile.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "onnx_light_cpu/onnx_py/_cpupy_kernels.h"

namespace nb = nanobind;

namespace onnx_light_cpu {

namespace {

// ``MemoryLevelName`` and ``ComputeElementTypeName`` are declared in
// ``processor_performance_profile.h`` and defined once in
// ``processor_performance_profile.cc`` so this binding and the C++
// aggregator always report the same names.

const char *CpuCacheKindName(CpuCacheKind kind) {
  switch (kind) {
  case CpuCacheKind::kUnknown:
    return "unknown";
  case CpuCacheKind::kData:
    return "data";
  case CpuCacheKind::kInstruction:
    return "instruction";
  case CpuCacheKind::kUnified:
    return "unified";
  }
  return "unknown";
}

const char *CpuCacheConfidenceName(CpuCacheConfidence confidence) {
  switch (confidence) {
  case CpuCacheConfidence::kDetected:
    return "detected";
  case CpuCacheConfidence::kInferred:
    return "inferred";
  case CpuCacheConfidence::kFallback:
    return "fallback";
  }
  return "unknown";
}

using CacheTuple =
    std::tuple<int, std::string, std::uint64_t, std::uint64_t, std::uint64_t, std::string>;
using AffinityTuple = std::tuple<std::uint16_t, std::uint16_t>;
using OptionsEchoTuple = std::tuple<std::vector<std::string>, std::uint64_t, double, std::uint64_t,
                                    bool, std::optional<AffinityTuple>>;
using MetadataTuple = std::tuple<int, std::int64_t, std::string, std::string, std::string,
                                 OptionsEchoTuple, std::vector<std::string>>;
using TopologyTuple = std::tuple<std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t,
                                 std::vector<CacheTuple>, bool>;
using BandwidthTuple = std::tuple<std::uint64_t, std::uint64_t, bool, std::string, std::uint64_t,
                                  std::vector<double>, double, double>;
using LatencyTuple = std::tuple<std::uint64_t, std::uint64_t, bool, std::string,
                                std::vector<double>, double, double>;
using MemoryEntryTuple = std::tuple<std::string, std::string, std::optional<BandwidthTuple>,
                                    std::optional<BandwidthTuple>, std::optional<BandwidthTuple>,
                                    std::optional<BandwidthTuple>, std::optional<LatencyTuple>,
                                    std::vector<BandwidthTuple>>;
using ComputeEntryTuple =
    std::tuple<std::string, std::string, std::string, std::uint64_t, bool, std::string,
               std::uint64_t, std::size_t, std::vector<double>, double, double>;
using RooflineEntryTuple =
    std::tuple<std::string, std::string, std::string, double, double, double>;
using ProfileTuple = std::tuple<MetadataTuple, TopologyTuple, std::vector<MemoryEntryTuple>,
                                std::vector<ComputeEntryTuple>, std::vector<RooflineEntryTuple>,
                                std::vector<std::string>>;

BandwidthTuple ToBandwidthTuple(const MemoryBandwidthResult &result) {
  return {result.working_set_bytes,
          result.participant_count,
          result.affinity_pinned,
          result.timer_name,
          result.useful_bytes_per_pass_per_participant,
          result.raw_gbps_samples,
          result.median_gbps,
          result.dispersion_gbps};
}

LatencyTuple ToLatencyTuple(const MemoryLatencyResult &result) {
  return {result.working_set_bytes,       result.participant_count,
          result.affinity_pinned,         result.timer_name,
          result.raw_ns_per_load_samples, result.median_ns_per_load,
          result.dispersion_ns_per_load};
}

std::optional<BandwidthTuple>
ToOptionalBandwidthTuple(const std::optional<MemoryBandwidthResult> &result) {
  if (!result.has_value()) {
    return std::nullopt;
  }
  return ToBandwidthTuple(*result);
}

ProfileTuple ToProfileTuple(const ProcessorPerformanceProfile &profile) {
  std::vector<std::string> policy_names;
  policy_names.reserve(profile.metadata.options.thread_policies.size());
  for (const ProcessorThreadPolicy policy : profile.metadata.options.thread_policies) {
    policy_names.emplace_back(ProcessorThreadPolicyName(policy));
  }
  std::optional<AffinityTuple> options_affinity;
  if (profile.metadata.options.explicit_single_affinity.has_value()) {
    options_affinity = AffinityTuple{profile.metadata.options.explicit_single_affinity->group,
                                     profile.metadata.options.explicit_single_affinity->index};
  }
  const OptionsEchoTuple options_echo{policy_names,
                                      profile.metadata.options.repeats,
                                      profile.metadata.options.minimum_duration_ms,
                                      profile.metadata.options.memory_budget_bytes,
                                      profile.metadata.options.include_latency,
                                      options_affinity};
  const MetadataTuple metadata{profile.metadata.schema_version, profile.metadata.unix_timestamp_ns,
                               profile.metadata.platform,       profile.metadata.compiler,
                               profile.metadata.timer_name,     options_echo,
                               profile.metadata.diagnostics};

  std::vector<CacheTuple> caches;
  caches.reserve(profile.topology.caches.size());
  for (const CpuCacheDescriptor &cache : profile.topology.caches) {
    caches.emplace_back(static_cast<int>(cache.level), CpuCacheKindName(cache.kind),
                        cache.size_bytes, cache.line_size_bytes, cache.sharing_thread_count,
                        CpuCacheConfidenceName(cache.confidence));
  }
  const TopologyTuple topology{profile.topology.logical_thread_count,
                               profile.topology.physical_core_count,
                               profile.topology.performance_core_count,
                               profile.topology.efficiency_core_count,
                               caches,
                               profile.topology.cache_topology_detected};

  std::vector<MemoryEntryTuple> memory;
  memory.reserve(profile.memory.size());
  for (const ProcessorProfileMemoryEntry &entry : profile.memory) {
    std::vector<BandwidthTuple> read_scaling;
    read_scaling.reserve(entry.read_scaling.size());
    for (const MemoryBandwidthResult &point : entry.read_scaling) {
      read_scaling.push_back(ToBandwidthTuple(point));
    }
    memory.emplace_back(
        MemoryLevelName(entry.level), ProcessorThreadPolicyName(entry.policy),
        ToOptionalBandwidthTuple(entry.read), ToOptionalBandwidthTuple(entry.write),
        ToOptionalBandwidthTuple(entry.copy), ToOptionalBandwidthTuple(entry.read_modify_write),
        entry.latency.has_value() ? std::optional<LatencyTuple>(ToLatencyTuple(*entry.latency))
                                  : std::nullopt,
        std::move(read_scaling));
  }

  std::vector<ComputeEntryTuple> compute;
  compute.reserve(profile.compute.size());
  for (const ProcessorProfileComputeEntry &entry : profile.compute) {
    compute.emplace_back(ComputeElementTypeName(entry.element_type),
                         ProcessorThreadPolicyName(entry.policy), entry.result.implementation_name,
                         entry.result.participant_count, entry.result.affinity_pinned,
                         entry.result.timer_name, entry.result.operations_per_pass_per_participant,
                         entry.result.dot_product_length, entry.result.raw_gops_samples,
                         entry.result.median_gops, entry.result.dispersion_gops);
  }

  std::vector<RooflineEntryTuple> roofline;
  roofline.reserve(profile.roofline.size());
  for (const ProcessorProfileRooflineEntry &entry : profile.roofline) {
    roofline.emplace_back(ComputeElementTypeName(entry.element_type),
                          ProcessorThreadPolicyName(entry.policy), MemoryLevelName(entry.level),
                          entry.compute_gops, entry.memory_read_gbps,
                          entry.arithmetic_intensity_crossover);
  }

  return {metadata, topology, memory, compute, roofline, profile.warnings};
}

ProfileTuple BenchmarkProcessorPerformanceForPython(
    const std::vector<std::string> &thread_policies, std::size_t repeats,
    double minimum_duration_ms, std::size_t memory_budget_bytes, bool include_latency,
    const std::optional<AffinityTuple> &explicit_single_affinity) {
  ProcessorProfileOptions options;
  options.thread_policies.clear();
  for (const std::string &name : thread_policies) {
    ProcessorThreadPolicy policy = ProcessorThreadPolicy::kSingle;
    if (!ParseProcessorThreadPolicy(name, policy)) {
      throw std::invalid_argument("unknown thread policy: " + name);
    }
    options.thread_policies.push_back(policy);
  }
  options.repeats = repeats;
  options.minimum_duration_ms = minimum_duration_ms;
  options.memory_budget_bytes = memory_budget_bytes;
  options.include_latency = include_latency;
  if (explicit_single_affinity.has_value()) {
    options.explicit_single_affinity =
        CpuAffinity{std::get<0>(*explicit_single_affinity), std::get<1>(*explicit_single_affinity)};
  }

  return ToProfileTuple(BenchmarkProcessorPerformance(options));
}

} // namespace

void RegisterProcessorPerformanceProfile(nb::module_ &m) {
  m.def("benchmark_processor_performance_raw", &BenchmarkProcessorPerformanceForPython,
        nb::arg("thread_policies"), nb::arg("repeats"), nb::arg("minimum_duration_ms"),
        nb::arg("memory_budget_bytes"), nb::arg("include_latency"),
        nb::arg("explicit_single_affinity") = std::nullopt,
        "Runs the versioned processor performance profile (memory bandwidth/latency plus "
        "register-resident compute throughput) and returns it as a plain nested-tuple "
        "structure. Raises ValueError before allocating or timing anything when an option "
        "is invalid. Wrapped by onnx_light_cpu.benchmark_processor_performance into "
        "immutable, documented result objects.");
}

} // namespace onnx_light_cpu
