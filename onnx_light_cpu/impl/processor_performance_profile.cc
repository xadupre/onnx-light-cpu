// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/processor_performance_profile.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <stdexcept>

namespace onnx_light_cpu {

namespace {

const char *PlatformName() {
#if defined(_WIN32)
  return "windows";
#elif defined(__APPLE__)
  return "macos";
#elif defined(__linux__)
  return "linux";
#else
  return "unknown";
#endif
}

const char *CompilerName() {
#if defined(__clang__)
  return "clang";
#elif defined(__GNUC__)
  return "gcc";
#elif defined(_MSC_VER)
  return "msvc";
#else
  return "unknown";
#endif
}

} // namespace

const char *MemoryLevelName(MemoryProfileLevel level) {
  switch (level) {
  case MemoryProfileLevel::kL1:
    return "L1";
  case MemoryProfileLevel::kL2:
    return "L2";
  case MemoryProfileLevel::kL3:
    return "L3";
  case MemoryProfileLevel::kRam:
    return "RAM";
  }
  return "unknown";
}

const char *ComputeElementTypeName(DataType element_type) {
  switch (element_type) {
  case DataType::FLOAT:
    return "float32";
  case DataType::DOUBLE:
    return "float64";
  case DataType::FLOAT16:
    return "float16";
  case DataType::BFLOAT16:
    return "bfloat16";
  case DataType::INT8:
    return "int8";
  }
  return "unknown";
}

namespace {

MemoryParticipantPolicy ToMemoryPolicy(ProcessorThreadPolicy policy) {
  return policy == ProcessorThreadPolicy::kPhysical ? MemoryParticipantPolicy::kPhysical
                                                    : MemoryParticipantPolicy::kSingle;
}

ComputeParticipantPolicy ToComputePolicy(ProcessorThreadPolicy policy) {
  return policy == ProcessorThreadPolicy::kPhysical ? ComputeParticipantPolicy::kPhysical
                                                    : ComputeParticipantPolicy::kSingle;
}

bool HasLogicalProcessor(const CpuTopology &topology, const CpuAffinity &affinity) {
  return std::any_of(
      topology.threads.begin(), topology.threads.end(), [&affinity](const CpuThread &thread) {
        return thread.affinity.group == affinity.group && thread.affinity.index == affinity.index;
      });
}

constexpr std::array<MemoryProfileLevel, 4> kMemoryLevels{
    MemoryProfileLevel::kL1, MemoryProfileLevel::kL2, MemoryProfileLevel::kL3,
    MemoryProfileLevel::kRam};

constexpr std::array<MemoryTrafficMode, 4> kMemoryModes{
    MemoryTrafficMode::kRead, MemoryTrafficMode::kWrite, MemoryTrafficMode::kCopy,
    MemoryTrafficMode::kReadModifyWrite};

constexpr std::array<DataType, 5> kComputeElementTypes{
    DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16, DataType::INT8};

std::optional<MemoryBandwidthResult> &SelectBandwidthSlot(ProcessorProfileMemoryEntry &entry,
                                                          MemoryTrafficMode mode) {
  switch (mode) {
  case MemoryTrafficMode::kRead:
    return entry.read;
  case MemoryTrafficMode::kWrite:
    return entry.write;
  case MemoryTrafficMode::kCopy:
    return entry.copy;
  case MemoryTrafficMode::kReadModifyWrite:
    return entry.read_modify_write;
  }
  return entry.read;
}

const char *MemoryTrafficModeName(MemoryTrafficMode mode) {
  switch (mode) {
  case MemoryTrafficMode::kRead:
    return "read";
  case MemoryTrafficMode::kWrite:
    return "write";
  case MemoryTrafficMode::kCopy:
    return "copy";
  case MemoryTrafficMode::kReadModifyWrite:
    return "read_modify_write";
  }
  return "unknown";
}

} // namespace

std::vector<ProcessorProfileRooflineEntry>
DeriveRooflineEntries(const std::vector<ProcessorProfileComputeEntry> &compute,
                      const std::vector<ProcessorProfileMemoryEntry> &memory) {
  std::vector<ProcessorProfileRooflineEntry> roofline;
  for (const ProcessorProfileComputeEntry &compute_entry : compute) {
    for (const ProcessorProfileMemoryEntry &memory_entry : memory) {
      if (memory_entry.policy != compute_entry.policy || !memory_entry.read.has_value()) {
        continue;
      }
      const double compute_gops = compute_entry.result.median_gops;
      const double memory_read_gbps = memory_entry.read->median_gbps;
      if (!(memory_read_gbps > 0.0)) {
        continue;
      }
      ProcessorProfileRooflineEntry roofline_entry;
      roofline_entry.element_type = compute_entry.element_type;
      roofline_entry.policy = compute_entry.policy;
      roofline_entry.level = memory_entry.level;
      roofline_entry.compute_gops = compute_gops;
      roofline_entry.memory_read_gbps = memory_read_gbps;
      roofline_entry.arithmetic_intensity_crossover = compute_gops / memory_read_gbps;
      roofline.push_back(roofline_entry);
    }
  }
  return roofline;
}

bool ParseProcessorThreadPolicy(const std::string &name, ProcessorThreadPolicy &policy) {
  if (name == "single") {
    policy = ProcessorThreadPolicy::kSingle;
    return true;
  }
  if (name == "physical") {
    policy = ProcessorThreadPolicy::kPhysical;
    return true;
  }
  return false;
}

const char *ProcessorThreadPolicyName(ProcessorThreadPolicy policy) {
  return policy == ProcessorThreadPolicy::kPhysical ? "physical" : "single";
}

std::string ValidateProcessorProfileOptions(const ProcessorProfileOptions &options) {
  if (options.thread_policies.empty()) {
    return "thread_policies must not be empty";
  }
  if (options.repeats == 0) {
    return "repeats must be at least 1";
  }
  if (!(options.minimum_duration_ms > 0.0)) {
    return "minimum_duration_ms must be positive";
  }
  if (options.memory_budget_bytes == 0) {
    return "memory_budget_bytes must be positive";
  }
  if (options.explicit_single_affinity.has_value() &&
      !HasLogicalProcessor(GetCpuTopology(), *options.explicit_single_affinity)) {
    return "explicit_single_affinity does not reference a logical processor visible to this "
           "process";
  }
  return {};
}

ProcessorPerformanceProfile BenchmarkProcessorPerformance(const ProcessorProfileOptions &options) {
  const std::string validation_error = ValidateProcessorProfileOptions(options);
  if (!validation_error.empty()) {
    throw std::invalid_argument(validation_error);
  }

  ProcessorPerformanceProfile profile;

  profile.metadata.unix_timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::system_clock::now().time_since_epoch())
                                           .count();
  profile.metadata.platform = PlatformName();
  profile.metadata.compiler = CompilerName();
  profile.metadata.timer_name = MemoryProfileTimerName();
  profile.metadata.options = options;

  const CpuTopology &cpu_topology = GetCpuTopology();
  const CpuCacheTopology &cache_topology = GetCpuCacheTopology();
  profile.topology.logical_thread_count = cpu_topology.logical_thread_count;
  profile.topology.physical_core_count = cpu_topology.physical_core_count;
  profile.topology.performance_core_count = cpu_topology.performance_core_count;
  profile.topology.efficiency_core_count = cpu_topology.efficiency_core_count;
  profile.topology.caches = cache_topology.caches;
  profile.topology.cache_topology_detected = cache_topology.platform_detected;

  MemoryProfileOptions memory_options;
  memory_options.repeats = options.repeats;
  memory_options.minimum_duration_ms = options.minimum_duration_ms;
  memory_options.memory_budget_bytes = options.memory_budget_bytes;
  memory_options.explicit_single_affinity = options.explicit_single_affinity;

  ComputeProfileOptions compute_options;
  compute_options.repeats = options.repeats;
  compute_options.minimum_duration_ms = options.minimum_duration_ms;
  compute_options.explicit_single_affinity = options.explicit_single_affinity;

  // Memory bandwidth and (optionally) latency, one entry per policy/level.
  for (const ProcessorThreadPolicy policy : options.thread_policies) {
    const MemoryParticipantPolicy memory_policy = ToMemoryPolicy(policy);
    for (const MemoryProfileLevel level : kMemoryLevels) {
      ProcessorProfileMemoryEntry entry;
      entry.level = level;
      entry.policy = policy;
      bool any_available = false;

      for (const MemoryTrafficMode mode : kMemoryModes) {
        MemoryBandwidthResult result;
        if (policy == ProcessorThreadPolicy::kPhysical && mode == MemoryTrafficMode::kRead &&
            level != MemoryProfileLevel::kRam) {
          const std::vector<MemoryBandwidthResult> scaling =
              MeasureMemoryBandwidthScaling(level, mode, memory_options);
          if (!scaling.empty()) {
            result = scaling.back();
          }
          for (const MemoryBandwidthResult &point : scaling) {
            if (point.available) {
              entry.read_scaling.push_back(point);
            } else {
              profile.warnings.push_back(
                  std::string("memory ") + MemoryLevelName(level) + " read scaling (" +
                  std::to_string(point.participant_count) + " participants): " + point.diagnostic);
            }
          }
        } else {
          result = MeasureMemoryBandwidth(level, mode, memory_policy, memory_options);
        }
        if (result.available) {
          any_available = true;
          SelectBandwidthSlot(entry, mode) = result;
        } else {
          profile.warnings.push_back(std::string("memory ") + MemoryLevelName(level) + " " +
                                     MemoryTrafficModeName(mode) + " (" +
                                     ProcessorThreadPolicyName(policy) + "): " + result.diagnostic);
        }
      }

      if (options.include_latency) {
        const MemoryLatencyResult latency_result =
            MeasureMemoryLatency(level, memory_policy, memory_options);
        if (latency_result.available) {
          any_available = true;
          entry.latency = latency_result;
        } else {
          profile.warnings.push_back(std::string("memory ") + MemoryLevelName(level) +
                                     " latency (" + ProcessorThreadPolicyName(policy) +
                                     "): " + latency_result.diagnostic);
        }
      }

      if (any_available) {
        profile.memory.push_back(std::move(entry));
      }
    }
  }

  // Register-resident compute throughput, one entry per policy/element type.
  for (const ProcessorThreadPolicy policy : options.thread_policies) {
    const ComputeParticipantPolicy compute_policy = ToComputePolicy(policy);
    for (const DataType element_type : kComputeElementTypes) {
      const ComputeThroughputResult result =
          MeasureComputeArithmeticThroughput(element_type, compute_policy, compute_options);
      if (result.available) {
        ProcessorProfileComputeEntry entry;
        entry.element_type = element_type;
        entry.policy = policy;
        entry.result = result;
        profile.compute.push_back(std::move(entry));
      } else {
        profile.warnings.push_back(std::string("compute ") + ComputeElementTypeName(element_type) +
                                   " (" + ProcessorThreadPolicyName(policy) +
                                   "): " + result.diagnostic);
      }
    }
  }

  // Roofline: derive crossover points from the compute/memory entries just
  // assembled (pure function, independently tested against synthetic
  // profiles in test_processor_performance_profile.cc).
  profile.roofline = DeriveRooflineEntries(profile.compute, profile.memory);

  return profile;
}

} // namespace onnx_light_cpu
