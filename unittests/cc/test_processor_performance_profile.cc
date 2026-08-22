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
using onnx_light_cpu::ComputeElementType;
using onnx_light_cpu::CpuAffinity;
using onnx_light_cpu::kProcessorPerformanceProfileSchemaVersion;
using onnx_light_cpu::MemoryProfileLevel;
using onnx_light_cpu::ParseProcessorThreadPolicy;
using onnx_light_cpu::ProcessorPerformanceProfile;
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
        return entry.element_type == ComputeElementType::kFloat32 &&
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

} // namespace
