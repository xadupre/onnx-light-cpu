// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/simd_level.h"

#include <gtest/gtest.h>

#ifndef ONNX_LIGHT_CPU_MAX_SIMD_LEVEL
#define ONNX_LIGHT_CPU_MAX_SIMD_LEVEL 4
#endif

namespace onnx_light_cpu {
namespace {

TEST(SimdLevel, DoesNotExceedConfiguredMaximum) {
  EXPECT_LE(static_cast<int>(DetectSimdLevel()), ONNX_LIGHT_CPU_MAX_SIMD_LEVEL);
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
