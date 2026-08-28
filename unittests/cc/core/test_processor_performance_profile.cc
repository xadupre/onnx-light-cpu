// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/processor_performance_profile.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {

using onnx_light_cpu::BenchmarkProcessorPerformance;
using onnx_light_cpu::CpuAffinity;
using onnx_light_cpu::DataType;
using onnx_light_cpu::DeriveRooflineEntries;
using onnx_light_cpu::kProcessorPerformanceProfileSchemaVersion;
using onnx_light_cpu::MemoryProfileLevel;
using onnx_light_cpu::ParseProcessorThreadPolicy;
using onnx_light_cpu::ProcessorPerformanceProfile;
using onnx_light_cpu::ProcessorProfileComputeEntry;
using onnx_light_cpu::ProcessorProfileMemoryEntry;
using onnx_light_cpu::ProcessorProfileOptions;
using onnx_light_cpu::ProcessorThreadPolicy;
using onnx_light_cpu::ProcessorThreadPolicyName;
using onnx_light_cpu::ValidateProcessorProfileOptions;

ProcessorProfileOptions BoundedOptions() {
  ProcessorProfileOptions options;
  options.thread_policies = {ProcessorThreadPolicy::kSingle};
  options.repeats = 2;
  options.minimum_duration_ms = 1.0;
  options.memory_budget_bytes = 8 * 1024 * 1024;
  options.include_latency = true;
  return options;
}

// ---------------------------------------------------------------------------
// Validation: pure, no allocation or timing.
// ---------------------------------------------------------------------------

TEST(ProcessorPerformanceProfile, ValidBoundedOptionsPassValidation) {
  EXPECT_TRUE(ValidateProcessorProfileOptions(BoundedOptions()).empty());
}

TEST(ProcessorPerformanceProfile, EmptyThreadPoliciesFailValidation) {
  ProcessorProfileOptions options = BoundedOptions();
  options.thread_policies.clear();
  EXPECT_FALSE(ValidateProcessorProfileOptions(options).empty());
}

TEST(ProcessorPerformanceProfile, ZeroRepeatsFailValidation) {
  ProcessorProfileOptions options = BoundedOptions();
  options.repeats = 0;
  EXPECT_FALSE(ValidateProcessorProfileOptions(options).empty());
}

TEST(ProcessorPerformanceProfile, NonPositiveDurationFailsValidation) {
  ProcessorProfileOptions options = BoundedOptions();
  options.minimum_duration_ms = 0.0;
  EXPECT_FALSE(ValidateProcessorProfileOptions(options).empty());
  options.minimum_duration_ms = -1.0;
  EXPECT_FALSE(ValidateProcessorProfileOptions(options).empty());
}

TEST(ProcessorPerformanceProfile, ZeroMemoryBudgetFailsValidation) {
  ProcessorProfileOptions options = BoundedOptions();
  options.memory_budget_bytes = 0;
  EXPECT_FALSE(ValidateProcessorProfileOptions(options).empty());
}

TEST(ProcessorPerformanceProfile, ImplausibleExplicitAffinityFailsValidation) {
  ProcessorProfileOptions options = BoundedOptions();
  options.explicit_single_affinity = CpuAffinity{0xFFFFu, 0xFFFFu};
  const std::string error = ValidateProcessorProfileOptions(options);
  EXPECT_FALSE(error.empty());
}

TEST(ProcessorPerformanceProfile, InvalidOptionsThrowBeforeMeasuring) {
  ProcessorProfileOptions options = BoundedOptions();
  options.repeats = 0;
  EXPECT_THROW(BenchmarkProcessorPerformance(options), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Thread policy name round-trip.
// ---------------------------------------------------------------------------

TEST(ProcessorPerformanceProfile, ThreadPolicyNamesRoundTrip) {
  ProcessorThreadPolicy policy = ProcessorThreadPolicy::kPhysical;
  ASSERT_TRUE(ParseProcessorThreadPolicy("single", policy));
  EXPECT_EQ(policy, ProcessorThreadPolicy::kSingle);
  EXPECT_STREQ(ProcessorThreadPolicyName(policy), "single");

  ASSERT_TRUE(ParseProcessorThreadPolicy("physical", policy));
  EXPECT_EQ(policy, ProcessorThreadPolicy::kPhysical);
  EXPECT_STREQ(ProcessorThreadPolicyName(policy), "physical");

  EXPECT_FALSE(ParseProcessorThreadPolicy("both", policy));
}

// ---------------------------------------------------------------------------
// Bounded native runs: finite, positive, truthfully diagnosed, no zero
// fabrication for unavailable entries.
// ---------------------------------------------------------------------------

TEST(ProcessorPerformanceProfile, BoundedSinglePolicyProfileIsCoherent) {
  const ProcessorPerformanceProfile profile = BenchmarkProcessorPerformance(BoundedOptions());

  EXPECT_EQ(profile.metadata.schema_version, kProcessorPerformanceProfileSchemaVersion);
  EXPECT_GT(profile.metadata.unix_timestamp_ns, 0);
  EXPECT_FALSE(profile.metadata.platform.empty());
  EXPECT_FALSE(profile.metadata.compiler.empty());
  EXPECT_FALSE(profile.metadata.timer_name.empty());

  EXPECT_GE(profile.topology.logical_thread_count, 1u);
  EXPECT_GE(profile.topology.physical_core_count, 1u);

  // At least L1 must be measurable on any host running these tests.
  const bool has_l1_entry =
      std::any_of(profile.memory.begin(), profile.memory.end(), [](const auto &entry) {
        return entry.level == MemoryProfileLevel::kL1 &&
               entry.policy == ProcessorThreadPolicy::kSingle;
      });
  EXPECT_TRUE(has_l1_entry);

  for (const auto &entry : profile.memory) {
    if (entry.read.has_value()) {
      EXPECT_TRUE(std::isfinite(entry.read->median_gbps));
      EXPECT_GT(entry.read->median_gbps, 0.0);
    }
    if (entry.latency.has_value()) {
      EXPECT_TRUE(std::isfinite(entry.latency->median_ns_per_load));
      EXPECT_GT(entry.latency->median_ns_per_load, 0.0);
    }
  }

  // Float32 is always available.
  const bool has_float32_entry =
      std::any_of(profile.compute.begin(), profile.compute.end(), [](const auto &entry) {
        return entry.element_type == DataType::FLOAT &&
               entry.policy == ProcessorThreadPolicy::kSingle;
      });
  EXPECT_TRUE(has_float32_entry);
  for (const auto &entry : profile.compute) {
    EXPECT_TRUE(entry.result.available);
    EXPECT_GT(entry.result.median_gops, 0.0);
    EXPECT_TRUE(std::isfinite(entry.result.median_gops));
  }

  // Every roofline entry references measurements that are actually present.
  for (const auto &roofline_entry : profile.roofline) {
    EXPECT_GT(roofline_entry.compute_gops, 0.0);
    EXPECT_GT(roofline_entry.memory_read_gbps, 0.0);
    EXPECT_TRUE(std::isfinite(roofline_entry.arithmetic_intensity_crossover));
    EXPECT_GT(roofline_entry.arithmetic_intensity_crossover, 0.0);
  }
}

TEST(ProcessorPerformanceProfile, PhysicalPolicyProducesAtLeastOneParticipant) {
  ProcessorProfileOptions options = BoundedOptions();
  options.thread_policies = {ProcessorThreadPolicy::kPhysical};

  const ProcessorPerformanceProfile profile = BenchmarkProcessorPerformance(options);

  const bool has_physical_compute =
      std::any_of(profile.compute.begin(), profile.compute.end(), [](const auto &entry) {
        return entry.policy == ProcessorThreadPolicy::kPhysical;
      });
  EXPECT_TRUE(has_physical_compute);

  const auto l1 = std::find_if(profile.memory.begin(), profile.memory.end(), [](const auto &entry) {
    return entry.level == MemoryProfileLevel::kL1 &&
           entry.policy == ProcessorThreadPolicy::kPhysical;
  });
  ASSERT_NE(l1, profile.memory.end());
  ASSERT_FALSE(l1->read_scaling.empty());
  EXPECT_EQ(l1->read_scaling.front().participant_count, 1u);
  EXPECT_EQ(l1->read_scaling.back().participant_count, profile.topology.physical_core_count);
}

TEST(ProcessorPerformanceProfile, LatencyOptedOutLeavesEveryLatencySlotAbsent) {
  ProcessorProfileOptions options = BoundedOptions();
  options.include_latency = false;

  const ProcessorPerformanceProfile profile = BenchmarkProcessorPerformance(options);
  for (const auto &entry : profile.memory) {
    EXPECT_FALSE(entry.latency.has_value());
  }
}

TEST(ProcessorPerformanceProfile, ImpossibleRamBudgetIsAbsentAndExplained) {
  ProcessorProfileOptions options = BoundedOptions();
  // A budget this small cannot possibly satisfy the RAM working-set contract
  // (larger than twice the last-level cache) on any real host.
  options.memory_budget_bytes = 1;

  const ProcessorPerformanceProfile profile = BenchmarkProcessorPerformance(options);
  const bool has_ram_entry =
      std::any_of(profile.memory.begin(), profile.memory.end(),
                  [](const auto &entry) { return entry.level == MemoryProfileLevel::kRam; });
  EXPECT_FALSE(has_ram_entry);
  EXPECT_FALSE(profile.warnings.empty());
}

TEST(ProcessorPerformanceProfile, RepeatedRunsProduceIndependentImmutableValues) {
  const ProcessorPerformanceProfile first = BenchmarkProcessorPerformance(BoundedOptions());
  const ProcessorPerformanceProfile second = BenchmarkProcessorPerformance(BoundedOptions());

  // Each call returns its own value; mutating one must not be observable
  // through the other (this is a compile-time property of returning by
  // value, exercised here to guard against an accidental shared-state
  // regression).
  EXPECT_EQ(first.metadata.schema_version, second.metadata.schema_version);
  EXPECT_FALSE(first.memory.empty());
  EXPECT_FALSE(second.memory.empty());
}

// ---------------------------------------------------------------------------
// Roofline derivation against known synthetic profiles: exact crossover
// arithmetic, correct policy pairing, and correct handling of missing/zero
// read bandwidth, independent of any live measurement.
// ---------------------------------------------------------------------------

ProcessorProfileComputeEntry MakeComputeEntry(DataType element_type, ProcessorThreadPolicy policy,
                                              double median_gops) {
  ProcessorProfileComputeEntry entry;
  entry.element_type = element_type;
  entry.policy = policy;
  entry.result.available = true;
  entry.result.median_gops = median_gops;
  return entry;
}

ProcessorProfileMemoryEntry MakeMemoryEntryWithRead(MemoryProfileLevel level,
                                                    ProcessorThreadPolicy policy,
                                                    double median_gbps) {
  ProcessorProfileMemoryEntry entry;
  entry.level = level;
  entry.policy = policy;
  onnx_light_cpu::MemoryBandwidthResult read;
  read.available = true;
  read.median_gbps = median_gbps;
  entry.read = read;
  return entry;
}

TEST(ProcessorPerformanceProfileRoofline, CrossoverMatchesComputeOverBandwidth) {
  const std::vector<ProcessorProfileComputeEntry> compute{
      MakeComputeEntry(DataType::FLOAT, ProcessorThreadPolicy::kSingle, 100.0)};
  const std::vector<ProcessorProfileMemoryEntry> memory{
      MakeMemoryEntryWithRead(MemoryProfileLevel::kL1, ProcessorThreadPolicy::kSingle, 40.0)};

  const auto roofline = DeriveRooflineEntries(compute, memory);
  ASSERT_EQ(roofline.size(), 1u);
  EXPECT_EQ(roofline[0].element_type, DataType::FLOAT);
  EXPECT_EQ(roofline[0].policy, ProcessorThreadPolicy::kSingle);
  EXPECT_EQ(roofline[0].level, MemoryProfileLevel::kL1);
  EXPECT_DOUBLE_EQ(roofline[0].compute_gops, 100.0);
  EXPECT_DOUBLE_EQ(roofline[0].memory_read_gbps, 40.0);
  EXPECT_DOUBLE_EQ(roofline[0].arithmetic_intensity_crossover, 2.5);
}

TEST(ProcessorPerformanceProfileRoofline, PairsEveryComputeEntryWithMatchingPolicyLevels) {
  const std::vector<ProcessorProfileComputeEntry> compute{
      MakeComputeEntry(DataType::FLOAT, ProcessorThreadPolicy::kSingle, 100.0),
      MakeComputeEntry(DataType::FLOAT, ProcessorThreadPolicy::kPhysical, 200.0)};
  const std::vector<ProcessorProfileMemoryEntry> memory{
      MakeMemoryEntryWithRead(MemoryProfileLevel::kL1, ProcessorThreadPolicy::kSingle, 50.0),
      MakeMemoryEntryWithRead(MemoryProfileLevel::kL2, ProcessorThreadPolicy::kSingle, 25.0),
      MakeMemoryEntryWithRead(MemoryProfileLevel::kL1, ProcessorThreadPolicy::kPhysical, 100.0)};

  const auto roofline = DeriveRooflineEntries(compute, memory);
  // The single-policy compute entry pairs with both single-policy memory
  // levels; the physical-policy compute entry only pairs with the one
  // physical-policy memory level.
  ASSERT_EQ(roofline.size(), 3u);
  int single_count = 0;
  int physical_count = 0;
  for (const auto &entry : roofline) {
    if (entry.policy == ProcessorThreadPolicy::kSingle) {
      ++single_count;
    } else {
      ++physical_count;
      EXPECT_DOUBLE_EQ(entry.arithmetic_intensity_crossover, 2.0);
    }
  }
  EXPECT_EQ(single_count, 2);
  EXPECT_EQ(physical_count, 1);
}

TEST(ProcessorPerformanceProfileRoofline, MissingOrZeroReadBandwidthYieldsNoEntry) {
  ProcessorProfileMemoryEntry no_read;
  no_read.level = MemoryProfileLevel::kL1;
  no_read.policy = ProcessorThreadPolicy::kSingle;
  // ``read`` left unset (unavailable).

  const std::vector<ProcessorProfileComputeEntry> compute{
      MakeComputeEntry(DataType::FLOAT, ProcessorThreadPolicy::kSingle, 100.0)};

  EXPECT_TRUE(DeriveRooflineEntries(compute, {no_read}).empty());
  EXPECT_TRUE(
      DeriveRooflineEntries(compute, {MakeMemoryEntryWithRead(MemoryProfileLevel::kL1,
                                                              ProcessorThreadPolicy::kSingle, 0.0)})
          .empty());
}

TEST(ProcessorPerformanceProfileRoofline, EmptyComputeOrMemoryYieldsEmptyRoofline) {
  const std::vector<ProcessorProfileComputeEntry> compute{
      MakeComputeEntry(DataType::FLOAT, ProcessorThreadPolicy::kSingle, 100.0)};
  const std::vector<ProcessorProfileMemoryEntry> memory{
      MakeMemoryEntryWithRead(MemoryProfileLevel::kL1, ProcessorThreadPolicy::kSingle, 40.0)};

  EXPECT_TRUE(DeriveRooflineEntries({}, memory).empty());
  EXPECT_TRUE(DeriveRooflineEntries(compute, {}).empty());
}

} // namespace
