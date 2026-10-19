// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "cpu_compute/memory_traffic_profile.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

using cpu_compute::ComputeMemoryTrafficAccounting;
using cpu_compute::CpuCacheConfidence;
using cpu_compute::CpuCacheKind;
using cpu_compute::CpuCacheTopology;
using cpu_compute::MeasureMemoryBandwidth;
using cpu_compute::MeasureMemoryBandwidthScaling;
using cpu_compute::MeasureMemoryLatency;
using cpu_compute::MemoryBandwidthResult;
using cpu_compute::MemoryBandwidthScalingParticipantCounts;
using cpu_compute::MemoryLatencyResult;
using cpu_compute::MemoryParticipantPolicy;
using cpu_compute::MemoryProfileLevel;
using cpu_compute::MemoryProfileOptions;
using cpu_compute::MemoryProfileUnavailableReason;
using cpu_compute::MemoryTrafficMode;
using cpu_compute::MemoryWorkingSet;
using cpu_compute::SelectMemoryWorkingSet;
using cpu_compute::detail::BuildCpuCacheTopology;
using cpu_compute::detail::BuildPointerChasePermutation;
using cpu_compute::detail::CpuCacheRecord;

cpu_compute::CpuCacheTopology MakeHomogeneousTopology() {
  const std::vector<CpuCacheRecord> records = {
      {1, cpu_compute::CpuCacheKind::kData, 32 * 1024, 64, 1,
       cpu_compute::CpuCacheConfidence::kDetected, true},
      {2, cpu_compute::CpuCacheKind::kUnified, 1024 * 1024, 64, 2,
       cpu_compute::CpuCacheConfidence::kDetected, true},
      {3, cpu_compute::CpuCacheKind::kUnified, 16 * 1024 * 1024, 64, 16,
       cpu_compute::CpuCacheConfidence::kDetected, true},
  };
  return BuildCpuCacheTopology(records, true);
}

// ---------------------------------------------------------------------------
// Working-set selection: pure, deterministic, no timing.
// ---------------------------------------------------------------------------

TEST(MemoryTrafficProfile, WorkingSetSelectsHalfOfEachCacheLevel) {
  const cpu_compute::CpuCacheTopology topology = MakeHomogeneousTopology();

  const cpu_compute::MemoryWorkingSet l1 = cpu_compute::SelectMemoryWorkingSet(
      topology, cpu_compute::MemoryProfileLevel::kL1, 1, 512 * 1024 * 1024);
  ASSERT_TRUE(l1.available);
  EXPECT_EQ(l1.working_set_bytes, 16u * 1024u);

  const cpu_compute::MemoryWorkingSet l2 = cpu_compute::SelectMemoryWorkingSet(
      topology, cpu_compute::MemoryProfileLevel::kL2, 1, 512 * 1024 * 1024);
  ASSERT_TRUE(l2.available);
  EXPECT_EQ(l2.working_set_bytes, 512u * 1024u);
  EXPECT_GT(l2.working_set_bytes, l1.working_set_bytes);

  const cpu_compute::MemoryWorkingSet l3 = cpu_compute::SelectMemoryWorkingSet(
      topology, cpu_compute::MemoryProfileLevel::kL3, 1, 512 * 1024 * 1024);
  ASSERT_TRUE(l3.available);
  EXPECT_EQ(l3.working_set_bytes, 8u * 1024u * 1024u);
}

TEST(MemoryTrafficProfile, RamWorkingSetExceedsTwiceLastLevelCache) {
  const cpu_compute::CpuCacheTopology topology = MakeHomogeneousTopology();

  const cpu_compute::MemoryWorkingSet ram = cpu_compute::SelectMemoryWorkingSet(
      topology, cpu_compute::MemoryProfileLevel::kRam, 1, 512 * 1024 * 1024);
  ASSERT_TRUE(ram.available);
  EXPECT_GT(ram.working_set_bytes, 2u * 16u * 1024u * 1024u);
}

TEST(MemoryTrafficProfile, RamNeverLabelsACacheSizedAllocation) {
  const cpu_compute::CpuCacheTopology topology = MakeHomogeneousTopology();

  // A budget too small to exceed twice the last-level cache must be reported
  // unavailable, never truncated to a cache-sized "RAM" allocation.
  const cpu_compute::MemoryWorkingSet ram = cpu_compute::SelectMemoryWorkingSet(
      topology, cpu_compute::MemoryProfileLevel::kRam, 1, 16 * 1024 * 1024);
  EXPECT_FALSE(ram.available);
  EXPECT_EQ(ram.unavailable_reason,
            cpu_compute::MemoryProfileUnavailableReason::kMemoryBudgetExceeded);
}

TEST(MemoryTrafficProfile, WorkingSetSplitsBudgetAcrossParticipants) {
  const cpu_compute::CpuCacheTopology topology = MakeHomogeneousTopology();

  const cpu_compute::MemoryWorkingSet single = cpu_compute::SelectMemoryWorkingSet(
      topology, cpu_compute::MemoryProfileLevel::kL1, 1, 64 * 1024);
  ASSERT_TRUE(single.available);

  // The same total budget split across many participants can no longer
  // afford one participant's full L1 share; it must be explicitly
  // unavailable rather than silently shrinking below the cache contract.
  const cpu_compute::MemoryWorkingSet many = cpu_compute::SelectMemoryWorkingSet(
      topology, cpu_compute::MemoryProfileLevel::kL1, 8, 64 * 1024);
  EXPECT_FALSE(many.available);
  EXPECT_EQ(many.unavailable_reason,
            cpu_compute::MemoryProfileUnavailableReason::kMemoryBudgetExceeded);
}

TEST(MemoryTrafficProfile, MissingLevelIsExplicitlyUnavailable) {
  const std::vector<CpuCacheRecord> records = {
      {1, cpu_compute::CpuCacheKind::kData, 32 * 1024, 64, 1,
       cpu_compute::CpuCacheConfidence::kDetected, true},
  };
  const cpu_compute::CpuCacheTopology topology = BuildCpuCacheTopology(records, true);

  const cpu_compute::MemoryWorkingSet l2 = cpu_compute::SelectMemoryWorkingSet(
      topology, cpu_compute::MemoryProfileLevel::kL2, 1, 512 * 1024 * 1024);
  EXPECT_FALSE(l2.available);
  EXPECT_EQ(l2.unavailable_reason, cpu_compute::MemoryProfileUnavailableReason::kLevelUnavailable);

  const cpu_compute::MemoryWorkingSet l3 = cpu_compute::SelectMemoryWorkingSet(
      topology, cpu_compute::MemoryProfileLevel::kL3, 1, 512 * 1024 * 1024);
  EXPECT_FALSE(l3.available);
  EXPECT_EQ(l3.unavailable_reason, cpu_compute::MemoryProfileUnavailableReason::kLevelUnavailable);
}

TEST(MemoryTrafficProfile, GenericPartialTopologyMarksEveryCacheLevelUnavailable) {
  // A generic/partial platform (no detector at all) reports an empty
  // topology; every cache level must be explicitly unavailable rather than
  // falling back to a plausible-looking guess. RAM alone can still run,
  // using the full budget, since it does not require cache descriptors.
  const cpu_compute::CpuCacheTopology topology = BuildCpuCacheTopology({}, false);

  EXPECT_FALSE(cpu_compute::SelectMemoryWorkingSet(topology, cpu_compute::MemoryProfileLevel::kL1,
                                                   1, 512 * 1024 * 1024)
                   .available);
  EXPECT_FALSE(cpu_compute::SelectMemoryWorkingSet(topology, cpu_compute::MemoryProfileLevel::kL2,
                                                   1, 512 * 1024 * 1024)
                   .available);
  EXPECT_FALSE(cpu_compute::SelectMemoryWorkingSet(topology, cpu_compute::MemoryProfileLevel::kL3,
                                                   1, 512 * 1024 * 1024)
                   .available);

  const cpu_compute::MemoryWorkingSet ram = cpu_compute::SelectMemoryWorkingSet(
      topology, cpu_compute::MemoryProfileLevel::kRam, 1, 512 * 1024 * 1024);
  EXPECT_TRUE(ram.available);
  EXPECT_EQ(ram.working_set_bytes, 512u * 1024u * 1024u);
}

TEST(MemoryTrafficProfile, HeterogeneousClusterUsesLargestPrivateCacheAtEachLevel) {
  // Distinct performance/efficiency private L2 sizes at the same level:
  // selection uses the largest reported size (the same deterministic,
  // insertion-order-independent convention GEMM blocking already relies on).
  const std::vector<CpuCacheRecord> records = {
      {1, cpu_compute::CpuCacheKind::kData, 48 * 1024, 64, 1,
       cpu_compute::CpuCacheConfidence::kDetected, true},
      {2, cpu_compute::CpuCacheKind::kUnified, 2 * 1024 * 1024, 64, 1,
       cpu_compute::CpuCacheConfidence::kDetected, true},
      {2, cpu_compute::CpuCacheKind::kUnified, 4 * 1024 * 1024, 64, 4,
       cpu_compute::CpuCacheConfidence::kDetected, true},
  };
  const cpu_compute::CpuCacheTopology topology = BuildCpuCacheTopology(records, true);

  const cpu_compute::MemoryWorkingSet l2 = cpu_compute::SelectMemoryWorkingSet(
      topology, cpu_compute::MemoryProfileLevel::kL2, 1, 512 * 1024 * 1024);
  ASSERT_TRUE(l2.available);
  EXPECT_EQ(l2.working_set_bytes, 2u * 1024u * 1024u);
}

TEST(MemoryTrafficProfile, ZeroMemoryBudgetIsUnavailable) {
  const cpu_compute::CpuCacheTopology topology = MakeHomogeneousTopology();
  const cpu_compute::MemoryWorkingSet l1 =
      cpu_compute::SelectMemoryWorkingSet(topology, cpu_compute::MemoryProfileLevel::kL1, 1, 0);
  EXPECT_FALSE(l1.available);
  EXPECT_EQ(l1.unavailable_reason,
            cpu_compute::MemoryProfileUnavailableReason::kMemoryBudgetExceeded);
}

// ---------------------------------------------------------------------------
// Byte accounting: pure arithmetic, independent of elapsed time.
// ---------------------------------------------------------------------------

TEST(MemoryTrafficProfile, ReadAndWriteAccountForOneStream) {
  const auto read =
      cpu_compute::ComputeMemoryTrafficAccounting(cpu_compute::MemoryTrafficMode::kRead, 4096, 8);
  EXPECT_EQ(read.element_count, 512u);
  EXPECT_EQ(read.useful_bytes_per_pass, 4096u);

  const auto write =
      cpu_compute::ComputeMemoryTrafficAccounting(cpu_compute::MemoryTrafficMode::kWrite, 4096, 8);
  EXPECT_EQ(write.element_count, 512u);
  EXPECT_EQ(write.useful_bytes_per_pass, 4096u);
}

TEST(MemoryTrafficProfile, CopyAccountsForBothStreamsWithinTheSameFootprint) {
  const auto copy =
      cpu_compute::ComputeMemoryTrafficAccounting(cpu_compute::MemoryTrafficMode::kCopy, 4096, 8);
  // Each stream (source, destination) gets half the working set...
  EXPECT_EQ(copy.element_count, 256u);
  // ...but the two streams combined still move the full working-set footprint
  // once per pass.
  EXPECT_EQ(copy.useful_bytes_per_pass, 4096u);
}

TEST(MemoryTrafficProfile, ReadModifyWriteTouchesTheSameFootprintTwice) {
  const auto rmw = cpu_compute::ComputeMemoryTrafficAccounting(
      cpu_compute::MemoryTrafficMode::kReadModifyWrite, 4096, 8);
  EXPECT_EQ(rmw.element_count, 512u);
  // One load and one store per element: double the read/write traffic for
  // the same footprint.
  EXPECT_EQ(rmw.useful_bytes_per_pass, 8192u);
}

TEST(MemoryTrafficProfile, ZeroElementBytesAccountsForNothing) {
  const auto accounting =
      cpu_compute::ComputeMemoryTrafficAccounting(cpu_compute::MemoryTrafficMode::kRead, 4096, 0);
  EXPECT_EQ(accounting.element_count, 0u);
  EXPECT_EQ(accounting.useful_bytes_per_pass, 0u);
}

TEST(MemoryTrafficProfile, ScalingCountsUsePowersOfTwoAndIncludeMaximum) {
  EXPECT_EQ(cpu_compute::MemoryBandwidthScalingParticipantCounts(1), (std::vector<std::size_t>{1}));
  EXPECT_EQ(cpu_compute::MemoryBandwidthScalingParticipantCounts(8),
            (std::vector<std::size_t>{1, 2, 4, 8}));
  EXPECT_EQ(cpu_compute::MemoryBandwidthScalingParticipantCounts(6),
            (std::vector<std::size_t>{1, 2, 4, 6}));
}

// ---------------------------------------------------------------------------
// Pointer-chase permutation: deterministic and exhaustive, no timing.
// ---------------------------------------------------------------------------

TEST(MemoryTrafficProfile, PointerChasePermutationVisitsEveryElementExactlyOnce) {
  const std::vector<std::uint32_t> permutation = BuildPointerChasePermutation(1024, 42);
  ASSERT_EQ(permutation.size(), 1024u);

  std::vector<bool> visited(1024, false);
  std::uint32_t index = 0;
  for (std::size_t step = 0; step < permutation.size(); ++step) {
    ASSERT_FALSE(visited[index]) << "index " << index << " revisited before a full cycle";
    visited[index] = true;
    index = permutation[index];
  }
  EXPECT_EQ(index, 0u) << "the cycle must return to its start after visiting every element";
  EXPECT_TRUE(std::all_of(visited.begin(), visited.end(), [](bool value) { return value; }));
}

TEST(MemoryTrafficProfile, PointerChasePermutationIsDeterministicForTheSameSeed) {
  const std::vector<std::uint32_t> first = BuildPointerChasePermutation(256, 7);
  const std::vector<std::uint32_t> second = BuildPointerChasePermutation(256, 7);
  EXPECT_EQ(first, second);
}

TEST(MemoryTrafficProfile, SingleElementPermutationIsTrivial) {
  const std::vector<std::uint32_t> permutation = BuildPointerChasePermutation(1, 1);
  ASSERT_EQ(permutation.size(), 1u);
  EXPECT_EQ(permutation[0], 0u);
}

// ---------------------------------------------------------------------------
// Bounded native measurements: finite, positive, and truthfully diagnosed.
// ---------------------------------------------------------------------------

cpu_compute::MemoryProfileOptions BoundedOptions() {
  cpu_compute::MemoryProfileOptions options;
  options.repeats = 3;
  options.minimum_duration_ms = 1.0;
  options.memory_budget_bytes = 8 * 1024 * 1024;
  return options;
}

TEST(MemoryTrafficProfile, SingleParticipantReadBandwidthIsFiniteAndPositive) {
  const cpu_compute::MemoryBandwidthResult result = cpu_compute::MeasureMemoryBandwidth(
      cpu_compute::MemoryProfileLevel::kL1, cpu_compute::MemoryTrafficMode::kRead,
      cpu_compute::MemoryParticipantPolicy::kSingle, BoundedOptions());

  ASSERT_TRUE(result.available) << result.diagnostic;
  EXPECT_EQ(result.participant_count, 1u);
  EXPECT_GT(result.working_set_bytes, 0u);
  EXPECT_EQ(result.raw_gbps_samples.size(), 3u);
  for (double sample : result.raw_gbps_samples) {
    EXPECT_TRUE(std::isfinite(sample));
    EXPECT_GT(sample, 0.0);
  }
  EXPECT_TRUE(std::isfinite(result.median_gbps));
  EXPECT_GT(result.median_gbps, 0.0);
  EXPECT_GE(result.dispersion_gbps, 0.0);
  EXPECT_EQ(result.timer_name, std::string("std::chrono::steady_clock"));
}

TEST(MemoryTrafficProfile, EveryTrafficModeProducesAFiniteBandwidth) {
  for (cpu_compute::MemoryTrafficMode mode :
       {cpu_compute::MemoryTrafficMode::kRead, cpu_compute::MemoryTrafficMode::kWrite,
        cpu_compute::MemoryTrafficMode::kCopy, cpu_compute::MemoryTrafficMode::kReadModifyWrite}) {
    const cpu_compute::MemoryBandwidthResult result = cpu_compute::MeasureMemoryBandwidth(
        cpu_compute::MemoryProfileLevel::kL1, mode, cpu_compute::MemoryParticipantPolicy::kSingle,
        BoundedOptions());
    ASSERT_TRUE(result.available) << result.diagnostic;
    EXPECT_GT(result.useful_bytes_per_pass_per_participant, 0u);
    EXPECT_GT(result.median_gbps, 0.0);
    EXPECT_TRUE(std::isfinite(result.median_gbps));
  }
}

TEST(MemoryTrafficProfile, PhysicalParticipantPolicyReportsAtLeastOneParticipant) {
  const cpu_compute::MemoryBandwidthResult result = cpu_compute::MeasureMemoryBandwidth(
      cpu_compute::MemoryProfileLevel::kL1, cpu_compute::MemoryTrafficMode::kRead,
      cpu_compute::MemoryParticipantPolicy::kPhysical, BoundedOptions());

  ASSERT_TRUE(result.available) << result.diagnostic;
  EXPECT_GE(result.participant_count, 1u);
  EXPECT_GT(result.median_gbps, 0.0);
  EXPECT_TRUE(std::isfinite(result.median_gbps));
}

TEST(MemoryTrafficProfile, BandwidthScalingReportsIncreasingParticipantCounts) {
  const std::vector<cpu_compute::MemoryBandwidthResult> curve =
      cpu_compute::MeasureMemoryBandwidthScaling(cpu_compute::MemoryProfileLevel::kL1,
                                                 cpu_compute::MemoryTrafficMode::kRead,
                                                 BoundedOptions());

  ASSERT_FALSE(curve.empty());
  EXPECT_EQ(curve.front().participant_count, 1u);
  for (std::size_t index = 0; index < curve.size(); ++index) {
    ASSERT_TRUE(curve[index].available) << curve[index].diagnostic;
    EXPECT_EQ(curve[index].participant_count,
              cpu_compute::MemoryBandwidthScalingParticipantCounts(
                  cpu_compute::GetCpuTopology().physical_core_count)[index]);
    EXPECT_GT(curve[index].median_gbps, 0.0);
  }
}

TEST(MemoryTrafficProfile, RamUnavailableWhenBudgetCannotExceedTwiceLastLevelCache) {
  cpu_compute::MemoryProfileOptions options = BoundedOptions();
  // Force a budget the host's real last-level cache cannot possibly clear by
  // twice its size while still being nonzero.
  options.memory_budget_bytes = 1;

  const cpu_compute::MemoryBandwidthResult result = cpu_compute::MeasureMemoryBandwidth(
      cpu_compute::MemoryProfileLevel::kRam, cpu_compute::MemoryTrafficMode::kRead,
      cpu_compute::MemoryParticipantPolicy::kSingle, options);

  EXPECT_FALSE(result.available);
  EXPECT_NE(result.unavailable_reason, cpu_compute::MemoryProfileUnavailableReason::kNone);
  EXPECT_FALSE(result.diagnostic.empty());
}

TEST(MemoryTrafficProfile, InvalidOptionsFailBeforeAllocatingOrTiming) {
  cpu_compute::MemoryProfileOptions options = BoundedOptions();
  options.repeats = 0;

  const cpu_compute::MemoryBandwidthResult result = cpu_compute::MeasureMemoryBandwidth(
      cpu_compute::MemoryProfileLevel::kL1, cpu_compute::MemoryTrafficMode::kRead,
      cpu_compute::MemoryParticipantPolicy::kSingle, options);
  EXPECT_FALSE(result.available);
  EXPECT_EQ(result.unavailable_reason,
            cpu_compute::MemoryProfileUnavailableReason::kInvalidOptions);
  EXPECT_FALSE(result.diagnostic.empty());
}

TEST(MemoryTrafficProfile, SingleParticipantLatencyIsFiniteAndPositive) {
  const cpu_compute::MemoryLatencyResult result = cpu_compute::MeasureMemoryLatency(
      cpu_compute::MemoryProfileLevel::kL1, cpu_compute::MemoryParticipantPolicy::kSingle,
      BoundedOptions());

  ASSERT_TRUE(result.available) << result.diagnostic;
  EXPECT_EQ(result.participant_count, 1u);
  EXPECT_EQ(result.raw_ns_per_load_samples.size(), 3u);
  for (double sample : result.raw_ns_per_load_samples) {
    EXPECT_TRUE(std::isfinite(sample));
    EXPECT_GT(sample, 0.0);
  }
  EXPECT_TRUE(std::isfinite(result.median_ns_per_load));
  EXPECT_GT(result.median_ns_per_load, 0.0);
  EXPECT_GE(result.dispersion_ns_per_load, 0.0);
}

TEST(MemoryTrafficProfile, PhysicalParticipantLatencyReportsAtLeastOneParticipant) {
  const cpu_compute::MemoryLatencyResult result = cpu_compute::MeasureMemoryLatency(
      cpu_compute::MemoryProfileLevel::kL1, cpu_compute::MemoryParticipantPolicy::kPhysical,
      BoundedOptions());

  ASSERT_TRUE(result.available) << result.diagnostic;
  EXPECT_GE(result.participant_count, 1u);
  EXPECT_GT(result.median_ns_per_load, 0.0);
}

TEST(MemoryTrafficProfile, MissingLevelMakesLatencyExplicitlyUnavailable) {
  // A generic host with no cache descriptors marks even L1 latency
  // unavailable rather than fabricating a working set; this test only
  // asserts the observable path is exercised (level unavailability itself is
  // covered directly against injected topologies above).
  cpu_compute::MemoryProfileOptions options = BoundedOptions();
  options.memory_budget_bytes = 1;

  const cpu_compute::MemoryLatencyResult result =
      cpu_compute::MeasureMemoryLatency(cpu_compute::MemoryProfileLevel::kRam,
                                        cpu_compute::MemoryParticipantPolicy::kSingle, options);
  EXPECT_FALSE(result.available);
  EXPECT_NE(result.unavailable_reason, cpu_compute::MemoryProfileUnavailableReason::kNone);
}

} // namespace
