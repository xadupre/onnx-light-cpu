// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/memory_traffic_profile.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

using onnx_light_cpu::ComputeMemoryTrafficAccounting;
using onnx_light_cpu::CpuCacheConfidence;
using onnx_light_cpu::CpuCacheKind;
using onnx_light_cpu::CpuCacheTopology;
using onnx_light_cpu::MeasureMemoryBandwidth;
using onnx_light_cpu::MeasureMemoryBandwidthScaling;
using onnx_light_cpu::MeasureMemoryLatency;
using onnx_light_cpu::MemoryBandwidthResult;
using onnx_light_cpu::MemoryBandwidthScalingParticipantCounts;
using onnx_light_cpu::MemoryLatencyResult;
using onnx_light_cpu::MemoryParticipantPolicy;
using onnx_light_cpu::MemoryProfileLevel;
using onnx_light_cpu::MemoryProfileOptions;
using onnx_light_cpu::MemoryProfileUnavailableReason;
using onnx_light_cpu::MemoryTrafficMode;
using onnx_light_cpu::MemoryWorkingSet;
using onnx_light_cpu::SelectMemoryWorkingSet;
using onnx_light_cpu::detail::BuildCpuCacheTopology;
using onnx_light_cpu::detail::BuildPointerChasePermutation;
using onnx_light_cpu::detail::CpuCacheRecord;

CpuCacheTopology MakeHomogeneousTopology() {
  const std::vector<CpuCacheRecord> records = {
      {1, CpuCacheKind::kData, 32 * 1024, 64, 1, CpuCacheConfidence::kDetected, true},
      {2, CpuCacheKind::kUnified, 1024 * 1024, 64, 2, CpuCacheConfidence::kDetected, true},
      {3, CpuCacheKind::kUnified, 16 * 1024 * 1024, 64, 16, CpuCacheConfidence::kDetected, true},
  };
  return BuildCpuCacheTopology(records, true);
}

// ---------------------------------------------------------------------------
// Working-set selection: pure, deterministic, no timing.
// ---------------------------------------------------------------------------

TEST(MemoryTrafficProfile, WorkingSetSelectsHalfOfEachCacheLevel) {
  const CpuCacheTopology topology = MakeHomogeneousTopology();

  const MemoryWorkingSet l1 =
      SelectMemoryWorkingSet(topology, MemoryProfileLevel::kL1, 1, 512 * 1024 * 1024);
  ASSERT_TRUE(l1.available);
  EXPECT_EQ(l1.working_set_bytes, 16u * 1024u);

  const MemoryWorkingSet l2 =
      SelectMemoryWorkingSet(topology, MemoryProfileLevel::kL2, 1, 512 * 1024 * 1024);
  ASSERT_TRUE(l2.available);
  EXPECT_EQ(l2.working_set_bytes, 512u * 1024u);
  EXPECT_GT(l2.working_set_bytes, l1.working_set_bytes);

  const MemoryWorkingSet l3 =
      SelectMemoryWorkingSet(topology, MemoryProfileLevel::kL3, 1, 512 * 1024 * 1024);
  ASSERT_TRUE(l3.available);
  EXPECT_EQ(l3.working_set_bytes, 8u * 1024u * 1024u);
}

TEST(MemoryTrafficProfile, RamWorkingSetExceedsTwiceLastLevelCache) {
  const CpuCacheTopology topology = MakeHomogeneousTopology();

  const MemoryWorkingSet ram =
      SelectMemoryWorkingSet(topology, MemoryProfileLevel::kRam, 1, 512 * 1024 * 1024);
  ASSERT_TRUE(ram.available);
  EXPECT_GT(ram.working_set_bytes, 2u * 16u * 1024u * 1024u);
}

TEST(MemoryTrafficProfile, RamNeverLabelsACacheSizedAllocation) {
  const CpuCacheTopology topology = MakeHomogeneousTopology();

  // A budget too small to exceed twice the last-level cache must be reported
  // unavailable, never truncated to a cache-sized "RAM" allocation.
  const MemoryWorkingSet ram =
      SelectMemoryWorkingSet(topology, MemoryProfileLevel::kRam, 1, 16 * 1024 * 1024);
  EXPECT_FALSE(ram.available);
  EXPECT_EQ(ram.unavailable_reason, MemoryProfileUnavailableReason::kMemoryBudgetExceeded);
}

TEST(MemoryTrafficProfile, WorkingSetSplitsBudgetAcrossParticipants) {
  const CpuCacheTopology topology = MakeHomogeneousTopology();

  const MemoryWorkingSet single =
      SelectMemoryWorkingSet(topology, MemoryProfileLevel::kL1, 1, 64 * 1024);
  ASSERT_TRUE(single.available);

  // The same total budget split across many participants can no longer
  // afford one participant's full L1 share; it must be explicitly
  // unavailable rather than silently shrinking below the cache contract.
  const MemoryWorkingSet many =
      SelectMemoryWorkingSet(topology, MemoryProfileLevel::kL1, 8, 64 * 1024);
  EXPECT_FALSE(many.available);
  EXPECT_EQ(many.unavailable_reason, MemoryProfileUnavailableReason::kMemoryBudgetExceeded);
}

TEST(MemoryTrafficProfile, MissingLevelIsExplicitlyUnavailable) {
  const std::vector<CpuCacheRecord> records = {
      {1, CpuCacheKind::kData, 32 * 1024, 64, 1, CpuCacheConfidence::kDetected, true},
  };
  const CpuCacheTopology topology = BuildCpuCacheTopology(records, true);

  const MemoryWorkingSet l2 =
      SelectMemoryWorkingSet(topology, MemoryProfileLevel::kL2, 1, 512 * 1024 * 1024);
  EXPECT_FALSE(l2.available);
  EXPECT_EQ(l2.unavailable_reason, MemoryProfileUnavailableReason::kLevelUnavailable);

  const MemoryWorkingSet l3 =
      SelectMemoryWorkingSet(topology, MemoryProfileLevel::kL3, 1, 512 * 1024 * 1024);
  EXPECT_FALSE(l3.available);
  EXPECT_EQ(l3.unavailable_reason, MemoryProfileUnavailableReason::kLevelUnavailable);
}

TEST(MemoryTrafficProfile, GenericPartialTopologyMarksEveryCacheLevelUnavailable) {
  // A generic/partial platform (no detector at all) reports an empty
  // topology; every cache level must be explicitly unavailable rather than
  // falling back to a plausible-looking guess. RAM alone can still run,
  // using the full budget, since it does not require cache descriptors.
  const CpuCacheTopology topology = BuildCpuCacheTopology({}, false);

  EXPECT_FALSE(
      SelectMemoryWorkingSet(topology, MemoryProfileLevel::kL1, 1, 512 * 1024 * 1024).available);
  EXPECT_FALSE(
      SelectMemoryWorkingSet(topology, MemoryProfileLevel::kL2, 1, 512 * 1024 * 1024).available);
  EXPECT_FALSE(
      SelectMemoryWorkingSet(topology, MemoryProfileLevel::kL3, 1, 512 * 1024 * 1024).available);

  const MemoryWorkingSet ram =
      SelectMemoryWorkingSet(topology, MemoryProfileLevel::kRam, 1, 512 * 1024 * 1024);
  EXPECT_TRUE(ram.available);
  EXPECT_EQ(ram.working_set_bytes, 512u * 1024u * 1024u);
}

TEST(MemoryTrafficProfile, HeterogeneousClusterUsesLargestPrivateCacheAtEachLevel) {
  // Distinct performance/efficiency private L2 sizes at the same level:
  // selection uses the largest reported size (the same deterministic,
  // insertion-order-independent convention GEMM blocking already relies on).
  const std::vector<CpuCacheRecord> records = {
      {1, CpuCacheKind::kData, 48 * 1024, 64, 1, CpuCacheConfidence::kDetected, true},
      {2, CpuCacheKind::kUnified, 2 * 1024 * 1024, 64, 1, CpuCacheConfidence::kDetected, true},
      {2, CpuCacheKind::kUnified, 4 * 1024 * 1024, 64, 4, CpuCacheConfidence::kDetected, true},
  };
  const CpuCacheTopology topology = BuildCpuCacheTopology(records, true);

  const MemoryWorkingSet l2 =
      SelectMemoryWorkingSet(topology, MemoryProfileLevel::kL2, 1, 512 * 1024 * 1024);
  ASSERT_TRUE(l2.available);
  EXPECT_EQ(l2.working_set_bytes, 2u * 1024u * 1024u);
}

TEST(MemoryTrafficProfile, ZeroMemoryBudgetIsUnavailable) {
  const CpuCacheTopology topology = MakeHomogeneousTopology();
  const MemoryWorkingSet l1 = SelectMemoryWorkingSet(topology, MemoryProfileLevel::kL1, 1, 0);
  EXPECT_FALSE(l1.available);
  EXPECT_EQ(l1.unavailable_reason, MemoryProfileUnavailableReason::kMemoryBudgetExceeded);
}

// ---------------------------------------------------------------------------
// Byte accounting: pure arithmetic, independent of elapsed time.
// ---------------------------------------------------------------------------

TEST(MemoryTrafficProfile, ReadAndWriteAccountForOneStream) {
  const auto read = ComputeMemoryTrafficAccounting(MemoryTrafficMode::kRead, 4096, 8);
  EXPECT_EQ(read.element_count, 512u);
  EXPECT_EQ(read.useful_bytes_per_pass, 4096u);

  const auto write = ComputeMemoryTrafficAccounting(MemoryTrafficMode::kWrite, 4096, 8);
  EXPECT_EQ(write.element_count, 512u);
  EXPECT_EQ(write.useful_bytes_per_pass, 4096u);
}

TEST(MemoryTrafficProfile, CopyAccountsForBothStreamsWithinTheSameFootprint) {
  const auto copy = ComputeMemoryTrafficAccounting(MemoryTrafficMode::kCopy, 4096, 8);
  // Each stream (source, destination) gets half the working set...
  EXPECT_EQ(copy.element_count, 256u);
  // ...but the two streams combined still move the full working-set footprint
  // once per pass.
  EXPECT_EQ(copy.useful_bytes_per_pass, 4096u);
}

TEST(MemoryTrafficProfile, ReadModifyWriteTouchesTheSameFootprintTwice) {
  const auto rmw = ComputeMemoryTrafficAccounting(MemoryTrafficMode::kReadModifyWrite, 4096, 8);
  EXPECT_EQ(rmw.element_count, 512u);
  // One load and one store per element: double the read/write traffic for
  // the same footprint.
  EXPECT_EQ(rmw.useful_bytes_per_pass, 8192u);
}

TEST(MemoryTrafficProfile, ZeroElementBytesAccountsForNothing) {
  const auto accounting = ComputeMemoryTrafficAccounting(MemoryTrafficMode::kRead, 4096, 0);
  EXPECT_EQ(accounting.element_count, 0u);
  EXPECT_EQ(accounting.useful_bytes_per_pass, 0u);
}

TEST(MemoryTrafficProfile, ScalingCountsUsePowersOfTwoAndIncludeMaximum) {
  EXPECT_EQ(MemoryBandwidthScalingParticipantCounts(1), (std::vector<std::size_t>{1}));
  EXPECT_EQ(MemoryBandwidthScalingParticipantCounts(8), (std::vector<std::size_t>{1, 2, 4, 8}));
  EXPECT_EQ(MemoryBandwidthScalingParticipantCounts(6), (std::vector<std::size_t>{1, 2, 4, 6}));
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

MemoryProfileOptions BoundedOptions() {
  MemoryProfileOptions options;
  options.repeats = 3;
  options.minimum_duration_ms = 1.0;
  options.memory_budget_bytes = 8 * 1024 * 1024;
  return options;
}

TEST(MemoryTrafficProfile, SingleParticipantReadBandwidthIsFiniteAndPositive) {
  const MemoryBandwidthResult result =
      MeasureMemoryBandwidth(MemoryProfileLevel::kL1, MemoryTrafficMode::kRead,
                             MemoryParticipantPolicy::kSingle, BoundedOptions());

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
  for (MemoryTrafficMode mode : {MemoryTrafficMode::kRead, MemoryTrafficMode::kWrite,
                                 MemoryTrafficMode::kCopy, MemoryTrafficMode::kReadModifyWrite}) {
    const MemoryBandwidthResult result = MeasureMemoryBandwidth(
        MemoryProfileLevel::kL1, mode, MemoryParticipantPolicy::kSingle, BoundedOptions());
    ASSERT_TRUE(result.available) << result.diagnostic;
    EXPECT_GT(result.useful_bytes_per_pass_per_participant, 0u);
    EXPECT_GT(result.median_gbps, 0.0);
    EXPECT_TRUE(std::isfinite(result.median_gbps));
  }
}

TEST(MemoryTrafficProfile, PhysicalParticipantPolicyReportsAtLeastOneParticipant) {
  const MemoryBandwidthResult result =
      MeasureMemoryBandwidth(MemoryProfileLevel::kL1, MemoryTrafficMode::kRead,
                             MemoryParticipantPolicy::kPhysical, BoundedOptions());

  ASSERT_TRUE(result.available) << result.diagnostic;
  EXPECT_GE(result.participant_count, 1u);
  EXPECT_GT(result.median_gbps, 0.0);
  EXPECT_TRUE(std::isfinite(result.median_gbps));
}

TEST(MemoryTrafficProfile, BandwidthScalingReportsIncreasingParticipantCounts) {
  const std::vector<MemoryBandwidthResult> curve = MeasureMemoryBandwidthScaling(
      MemoryProfileLevel::kL1, MemoryTrafficMode::kRead, BoundedOptions());

  ASSERT_FALSE(curve.empty());
  EXPECT_EQ(curve.front().participant_count, 1u);
  for (std::size_t index = 0; index < curve.size(); ++index) {
    ASSERT_TRUE(curve[index].available) << curve[index].diagnostic;
    EXPECT_EQ(curve[index].participant_count,
              MemoryBandwidthScalingParticipantCounts(
                  onnx_light_cpu::GetCpuTopology().physical_core_count)[index]);
    EXPECT_GT(curve[index].median_gbps, 0.0);
  }
}

TEST(MemoryTrafficProfile, RamUnavailableWhenBudgetCannotExceedTwiceLastLevelCache) {
  MemoryProfileOptions options = BoundedOptions();
  // Force a budget the host's real last-level cache cannot possibly clear by
  // twice its size while still being nonzero.
  options.memory_budget_bytes = 1;

  const MemoryBandwidthResult result =
      MeasureMemoryBandwidth(MemoryProfileLevel::kRam, MemoryTrafficMode::kRead,
                             MemoryParticipantPolicy::kSingle, options);

  EXPECT_FALSE(result.available);
  EXPECT_NE(result.unavailable_reason, MemoryProfileUnavailableReason::kNone);
  EXPECT_FALSE(result.diagnostic.empty());
}

TEST(MemoryTrafficProfile, InvalidOptionsFailBeforeAllocatingOrTiming) {
  MemoryProfileOptions options = BoundedOptions();
  options.repeats = 0;

  const MemoryBandwidthResult result = MeasureMemoryBandwidth(
      MemoryProfileLevel::kL1, MemoryTrafficMode::kRead, MemoryParticipantPolicy::kSingle, options);
  EXPECT_FALSE(result.available);
  EXPECT_EQ(result.unavailable_reason, MemoryProfileUnavailableReason::kInvalidOptions);
  EXPECT_FALSE(result.diagnostic.empty());
}

TEST(MemoryTrafficProfile, SingleParticipantLatencyIsFiniteAndPositive) {
  const MemoryLatencyResult result = MeasureMemoryLatency(
      MemoryProfileLevel::kL1, MemoryParticipantPolicy::kSingle, BoundedOptions());

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
  const MemoryLatencyResult result = MeasureMemoryLatency(
      MemoryProfileLevel::kL1, MemoryParticipantPolicy::kPhysical, BoundedOptions());

  ASSERT_TRUE(result.available) << result.diagnostic;
  EXPECT_GE(result.participant_count, 1u);
  EXPECT_GT(result.median_ns_per_load, 0.0);
}

TEST(MemoryTrafficProfile, MissingLevelMakesLatencyExplicitlyUnavailable) {
  // A generic host with no cache descriptors marks even L1 latency
  // unavailable rather than fabricating a working set; this test only
  // asserts the observable path is exercised (level unavailability itself is
  // covered directly against injected topologies above).
  MemoryProfileOptions options = BoundedOptions();
  options.memory_budget_bytes = 1;

  const MemoryLatencyResult result =
      MeasureMemoryLatency(MemoryProfileLevel::kRam, MemoryParticipantPolicy::kSingle, options);
  EXPECT_FALSE(result.available);
  EXPECT_NE(result.unavailable_reason, MemoryProfileUnavailableReason::kNone);
}

} // namespace
