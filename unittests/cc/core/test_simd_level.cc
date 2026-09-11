// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/simd_level.h"

#include <cstdlib>
#include <cstring>

#include <gtest/gtest.h>

#ifndef ONNX_LIGHT_CPU_MAX_SIMD_LEVEL
#define ONNX_LIGHT_CPU_MAX_SIMD_LEVEL 4
#endif

namespace onnx_light_cpu {
namespace {

TEST(SimdLevel, DoesNotExceedConfiguredMaximum) {
  EXPECT_LE(static_cast<int>(DetectSimdLevel()), ONNX_LIGHT_CPU_MAX_SIMD_LEVEL);
}

TEST(SimdLevel, FeatureMaskedHost) {
  const char *expected = std::getenv("ONNX_LIGHT_CPU_TEST_EXPECT_SIMD_LEVEL");
  if (expected == nullptr) {
    GTEST_SKIP() << "Set ONNX_LIGHT_CPU_TEST_EXPECT_SIMD_LEVEL for feature-masked CPU tests.";
  }
  ASSERT_TRUE(std::strcmp(expected, "SSE2") == 0 || std::strcmp(expected, "AVX") == 0);
  const SimdLevel level = std::strcmp(expected, "SSE2") == 0 ? SimdLevel::kSSE2 : SimdLevel::kAVX;
  EXPECT_EQ(DetectSimdLevel(), level);
  EXPECT_FALSE(CpuSupportsAvx512BW());
  EXPECT_FALSE(CpuSupportsAvx512Fp16());
  EXPECT_FALSE(CpuSupportsAvx512Bf16());
  EXPECT_FALSE(CpuSupportsAvx512Vnni());
  EXPECT_FALSE(CpuSupportsAmxTile());
  EXPECT_FALSE(CpuSupportsAmxBf16());
  EXPECT_FALSE(CpuSupportsAmxInt8());
  EXPECT_FALSE(CpuSupportsFma());
  if (level == SimdLevel::kSSE2) {
    EXPECT_FALSE(CpuSupportsF16C());
  }
}

TEST(SimdLevel, HigherInstructionSetsRespectAvx2Ceiling) {
  if constexpr (ONNX_LIGHT_CPU_MAX_SIMD_LEVEL <= static_cast<int>(SimdLevel::kAVX2)) {
    EXPECT_FALSE(CpuSupportsAvx512BW());
    EXPECT_FALSE(CpuSupportsAvx512Fp16());
    EXPECT_FALSE(CpuSupportsAvx512Bf16());
    EXPECT_FALSE(CpuSupportsAvx512Vnni());
    EXPECT_FALSE(CpuSupportsAmxTile());
    EXPECT_FALSE(CpuSupportsAmxBf16());
    EXPECT_FALSE(CpuSupportsAmxInt8());
  }
}

} // namespace
} // namespace onnx_light_cpu
