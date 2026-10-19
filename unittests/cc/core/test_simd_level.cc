// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "cpu_compute/simd_level.h"

#include <cstdlib>
#include <cstring>

#include <gtest/gtest.h>

#ifndef CPU_COMPUTE_MAX_SIMD_LEVEL
#define CPU_COMPUTE_MAX_SIMD_LEVEL 4
#endif

namespace cpu_compute {
namespace {

TEST(SimdLevel, DoesNotExceedConfiguredMaximum) {
  EXPECT_LE(static_cast<int>(cpu_compute::DetectSimdLevel()), CPU_COMPUTE_MAX_SIMD_LEVEL);
}

TEST(SimdLevel, FeatureMaskedHost) {
  const char *expected = std::getenv("ONNX_LIGHT_CPU_TEST_EXPECT_SIMD_LEVEL");
  if (expected == nullptr) {
    GTEST_SKIP() << "Set ONNX_LIGHT_CPU_TEST_EXPECT_SIMD_LEVEL for feature-masked CPU tests.";
  }
  ASSERT_TRUE(std::strcmp(expected, "SSE2") == 0 || std::strcmp(expected, "AVX") == 0);
  const cpu_compute::SimdLevel level = std::strcmp(expected, "SSE2") == 0
                                           ? cpu_compute::SimdLevel::kSSE2
                                           : cpu_compute::SimdLevel::kAVX;
  EXPECT_EQ(cpu_compute::DetectSimdLevel(), level);
  EXPECT_FALSE(cpu_compute::CpuSupportsAvx512BW());
  EXPECT_FALSE(cpu_compute::CpuSupportsAvx512DQ());
  EXPECT_FALSE(cpu_compute::CpuSupportsAvx512Fp16());
  EXPECT_FALSE(cpu_compute::CpuSupportsAvx512Bf16());
  EXPECT_FALSE(cpu_compute::CpuSupportsAvx512Vnni());
  EXPECT_FALSE(cpu_compute::CpuSupportsAmxTile());
  EXPECT_FALSE(cpu_compute::CpuSupportsAmxBf16());
  EXPECT_FALSE(cpu_compute::CpuSupportsAmxInt8());
  EXPECT_FALSE(cpu_compute::CpuSupportsFma());
  if (level == cpu_compute::SimdLevel::kSSE2) {
    EXPECT_FALSE(cpu_compute::CpuSupportsF16C());
  }
}

TEST(SimdLevel, HigherInstructionSetsRespectAvx2Ceiling) {
  if constexpr (CPU_COMPUTE_MAX_SIMD_LEVEL <= static_cast<int>(cpu_compute::SimdLevel::kAVX2)) {
    EXPECT_FALSE(cpu_compute::CpuSupportsAvx512BW());
    EXPECT_FALSE(cpu_compute::CpuSupportsAvx512DQ());
    EXPECT_FALSE(cpu_compute::CpuSupportsAvx512Fp16());
    EXPECT_FALSE(cpu_compute::CpuSupportsAvx512Bf16());
    EXPECT_FALSE(cpu_compute::CpuSupportsAvx512Vnni());
    EXPECT_FALSE(cpu_compute::CpuSupportsAmxTile());
    EXPECT_FALSE(cpu_compute::CpuSupportsAmxBf16());
    EXPECT_FALSE(cpu_compute::CpuSupportsAmxInt8());
  }
}

} // namespace
} // namespace cpu_compute
