// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "cpu_compute/cpu_cache_topology.h"

#include <gtest/gtest.h>

#include <vector>

namespace {

using cpu_compute::CpuCacheConfidence;
using cpu_compute::CpuCacheDescriptor;
using cpu_compute::CpuCacheKind;
using cpu_compute::CpuCacheSizeBytesOrFallback;
using cpu_compute::CpuCacheTopology;
using cpu_compute::FindCpuCacheDescriptor;
using cpu_compute::detail::BuildCpuCacheTopology;
using cpu_compute::detail::CpuCacheRecord;

TEST(CpuCacheTopology, PrivateAndSharedCachesReportSharingCounts) {
  const std::vector<CpuCacheRecord> records = {
      // Private per-core L1 data cache.
      {1, cpu_compute::CpuCacheKind::kData, 32 * 1024, 64, 1,
       cpu_compute::CpuCacheConfidence::kDetected, true},
      // L2 shared by a two-thread SMT pair.
      {2, cpu_compute::CpuCacheKind::kUnified, 1024 * 1024, 64, 2,
       cpu_compute::CpuCacheConfidence::kDetected, true},
      // L3 shared by every logical thread.
      {3, cpu_compute::CpuCacheKind::kUnified, 16 * 1024 * 1024, 64, 16,
       cpu_compute::CpuCacheConfidence::kDetected, true},
  };

  const cpu_compute::CpuCacheTopology topology = BuildCpuCacheTopology(records, true);

  ASSERT_EQ(topology.caches.size(), 3u);
  EXPECT_TRUE(topology.platform_detected);

  const cpu_compute::CpuCacheDescriptor *l1 = cpu_compute::FindCpuCacheDescriptor(topology, 1);
  ASSERT_NE(l1, nullptr);
  EXPECT_EQ(l1->sharing_thread_count, 1u);

  const cpu_compute::CpuCacheDescriptor *l2 = cpu_compute::FindCpuCacheDescriptor(topology, 2);
  ASSERT_NE(l2, nullptr);
  EXPECT_EQ(l2->sharing_thread_count, 2u);

  const cpu_compute::CpuCacheDescriptor *l3 = cpu_compute::FindCpuCacheDescriptor(topology, 3);
  ASSERT_NE(l3, nullptr);
  EXPECT_EQ(l3->sharing_thread_count, 16u);
}

TEST(CpuCacheTopology, AbsentLevelIsExplicitlyUnavailable) {
  const std::vector<CpuCacheRecord> records = {
      {1, cpu_compute::CpuCacheKind::kData, 32 * 1024, 64, 1,
       cpu_compute::CpuCacheConfidence::kDetected, true},
      {2, cpu_compute::CpuCacheKind::kUnified, 512 * 1024, 64, 2,
       cpu_compute::CpuCacheConfidence::kDetected, true},
  };

  const cpu_compute::CpuCacheTopology topology = BuildCpuCacheTopology(records, true);

  EXPECT_EQ(cpu_compute::FindCpuCacheDescriptor(topology, 3), nullptr);
  EXPECT_EQ(cpu_compute::CpuCacheSizeBytesOrFallback(topology, 3, 4 * 1024 * 1024),
            4u * 1024u * 1024u);
}

TEST(CpuCacheTopology, HeterogeneousCoreClustersReportDistinctPrivateCaches) {
  // A hybrid processor may expose distinct private L2 sizes for its
  // performance and efficiency core clusters at the same level.
  const std::vector<CpuCacheRecord> records = {
      {1, cpu_compute::CpuCacheKind::kData, 48 * 1024, 64, 1,
       cpu_compute::CpuCacheConfidence::kDetected, true},
      {2, cpu_compute::CpuCacheKind::kUnified, 2 * 1024 * 1024, 64, 1,
       cpu_compute::CpuCacheConfidence::kDetected, true},
      {2, cpu_compute::CpuCacheKind::kUnified, 4 * 1024 * 1024, 64, 4,
       cpu_compute::CpuCacheConfidence::kDetected, true},
  };

  const cpu_compute::CpuCacheTopology topology = BuildCpuCacheTopology(records, true);

  ASSERT_EQ(topology.caches.size(), 3u);
  EXPECT_EQ(cpu_compute::CpuCacheSizeBytesOrFallback(topology, 2, 0), 4u * 1024u * 1024u);
}

TEST(CpuCacheTopology, InvalidRecordsAreDiscarded) {
  const std::vector<CpuCacheRecord> records = {
      // Valid L1 data cache.
      {1, cpu_compute::CpuCacheKind::kData, 32 * 1024, 64, 1,
       cpu_compute::CpuCacheConfidence::kDetected, true},
      // Explicitly marked invalid (for example a malformed platform read).
      {2, cpu_compute::CpuCacheKind::kUnified, 256 * 1024, 64, 1,
       cpu_compute::CpuCacheConfidence::kInferred, false},
      // Zero size is nonsensical and must not be reported even if not
      // explicitly flagged invalid by the caller.
      {3, cpu_compute::CpuCacheKind::kUnified, 0, 64, 1, cpu_compute::CpuCacheConfidence::kDetected,
       false},
  };

  const cpu_compute::CpuCacheTopology topology = BuildCpuCacheTopology(records, true);

  ASSERT_EQ(topology.caches.size(), 1u);
  EXPECT_EQ(topology.caches[0].level, 1u);
  EXPECT_EQ(cpu_compute::FindCpuCacheDescriptor(topology, 2), nullptr);
  EXPECT_EQ(cpu_compute::FindCpuCacheDescriptor(topology, 3), nullptr);
}

TEST(CpuCacheTopology, GenericFallbackReportsNoPlatformDetection) {
  const cpu_compute::CpuCacheTopology topology = BuildCpuCacheTopology({}, false);

  EXPECT_TRUE(topology.caches.empty());
  EXPECT_FALSE(topology.platform_detected);
  EXPECT_EQ(cpu_compute::CpuCacheSizeBytesOrFallback(topology, 1, 32 * 1024), 32u * 1024u);
}

TEST(CpuCacheTopology, PlatformSupportedWithNoValidRecordsIsStillFallback) {
  // A platform-specific detector may run without error yet find nothing
  // usable (for example every record failed validation); the topology must
  // still report the generic/partial fallback state rather than claiming
  // detection succeeded.
  const std::vector<CpuCacheRecord> records = {
      {0, cpu_compute::CpuCacheKind::kUnknown, 0, 0, 0, cpu_compute::CpuCacheConfidence::kDetected,
       false},
  };

  const cpu_compute::CpuCacheTopology topology = BuildCpuCacheTopology(records, true);

  EXPECT_TRUE(topology.caches.empty());
  EXPECT_FALSE(topology.platform_detected);
}

TEST(CpuCacheTopology, RealTopologyIsDeterministicAcrossCalls) {
  const cpu_compute::CpuCacheTopology &first = cpu_compute::GetCpuCacheTopology();
  const cpu_compute::CpuCacheTopology &second = cpu_compute::GetCpuCacheTopology();

  EXPECT_EQ(&first, &second);
  EXPECT_EQ(first.caches.size(), second.caches.size());
  for (std::size_t index = 0; index < first.caches.size(); ++index) {
    EXPECT_EQ(first.caches[index].level, second.caches[index].level);
    EXPECT_EQ(first.caches[index].size_bytes, second.caches[index].size_bytes);
  }
}

} // namespace
