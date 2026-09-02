// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/math_kernels.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace {

std::size_t SelectedAbsFloat32Lanes() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  const auto level = onnx_light_cpu::DetectSimdLevel();
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
  if (level >= onnx_light_cpu::SimdLevel::kAVX512) {
    return 16;
  }
#endif
  if (level >= onnx_light_cpu::SimdLevel::kAVX) {
    return 8;
  }
  if (level >= onnx_light_cpu::SimdLevel::kSSE2) {
    return 4;
  }
#endif
  return 1;
}

double SelectedAbsFloat32ComputeCycles() {
  return std::max(1.0, 8.0 / static_cast<double>(SelectedAbsFloat32Lanes()));
}

// ---------------------------------------------------------------------------
// DetectSimdLevel
// ---------------------------------------------------------------------------

TEST(SimdDetection, DetectsAtLeastScalar) {
  auto level = onnx_light_cpu::DetectSimdLevel();
  EXPECT_GE(static_cast<int>(level), 0);
}

TEST(SimdDetection, FmaRequiresAvx2OrAvx) {
  if (onnx_light_cpu::CpuSupportsFma()) {
    EXPECT_GE(onnx_light_cpu::DetectSimdLevel(), onnx_light_cpu::SimdLevel::kAVX);
  }
}

// ---------------------------------------------------------------------------
// AbsFloat32
// ---------------------------------------------------------------------------

TEST(AbsFloat32, EmptyInput) {
  float dummy = 1.0f;
  onnx_light_cpu::AbsFloat32(&dummy, &dummy, 0);
}

TEST(AbsFloat32, CostModelUsesDispatchComputeCost) {
  struct PlanningExecutor {
    onnx_light_cpu::ExecutionWorkCost observed_cost{};

    static onnx_light_cpu::ExecutionParallelPlan Plan(void *context, std::int64_t,
                                                      const onnx_light_cpu::ExecutionWorkCost &cost,
                                                      std::int64_t, std::int64_t) {
      static_cast<PlanningExecutor *>(context)->observed_cost = cost;
      return {1, 1};
    }

    static void Run(void *, std::int64_t, void *, onnx_light_cpu::ExecutionBlockFn) {}
  };

  std::vector<float> input(16, -1.0f);
  std::vector<float> output(input.size());
  auto tuning = onnx_light_cpu::kDefaultAbsFloat32ExecutionTuning;
  PlanningExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 4, &PlanningExecutor::Run,
                                             &PlanningExecutor::Plan};
  onnx_light_cpu::ExecutionExecutorScope scope(&view);
  onnx_light_cpu::AbsFloat32WithTuning(input.data(), output.data(), input.size(), tuning);
  EXPECT_DOUBLE_EQ(executor.observed_cost.compute_cycles, SelectedAbsFloat32ComputeCycles());
}

TEST(AbsInt32, CostModelUsesMemoryBoundCostAndParticipantCap) {
  struct PlanningExecutor {
    onnx_light_cpu::ExecutionWorkCost observed_cost{};
    std::int64_t observed_maximum = 0;

    static onnx_light_cpu::ExecutionParallelPlan Plan(void *context, std::int64_t,
                                                      const onnx_light_cpu::ExecutionWorkCost &cost,
                                                      std::int64_t maximum_participants,
                                                      std::int64_t) {
      auto &self = *static_cast<PlanningExecutor *>(context);
      self.observed_cost = cost;
      self.observed_maximum = maximum_participants;
      return {1, 1};
    }

    static void Run(void *, std::int64_t, void *, onnx_light_cpu::ExecutionBlockFn) {}
  };

  std::vector<std::int32_t> input(16, -1);
  std::vector<std::int32_t> output(input.size());
  PlanningExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 64, &PlanningExecutor::Run,
                                             &PlanningExecutor::Plan};
  onnx_light_cpu::ExecutionExecutorScope scope(&view);
  onnx_light_cpu::AbsInt32(input.data(), output.data(), input.size());
  EXPECT_DOUBLE_EQ(executor.observed_cost.bytes_read, sizeof(std::int32_t));
  EXPECT_DOUBLE_EQ(executor.observed_cost.bytes_written, sizeof(std::int32_t));
  EXPECT_DOUBLE_EQ(executor.observed_cost.compute_cycles, 1.0);
  EXPECT_EQ(executor.observed_maximum, 32);
}

TEST(AbsFloat32, SingleElement) {
  float in = -3.14f;
  float out = 0.0f;
  onnx_light_cpu::AbsFloat32(&in, &out, 1);
  EXPECT_FLOAT_EQ(out, 3.14f);
}

TEST(AbsFloat32, PositiveValues) {
  std::vector<float> in = {1.0f, 2.5f, 100.0f};
  std::vector<float> out(3, 0.0f);
  onnx_light_cpu::AbsFloat32(in.data(), out.data(), in.size());
  for (std::size_t i = 0; i < in.size(); ++i) {
    EXPECT_FLOAT_EQ(out[i], in[i]);
  }
}

TEST(AbsFloat32, NegativeValues) {
  std::vector<float> in = {-1.0f, -2.5f, -100.0f};
  std::vector<float> out(3, 0.0f);
  onnx_light_cpu::AbsFloat32(in.data(), out.data(), in.size());
  EXPECT_FLOAT_EQ(out[0], 1.0f);
  EXPECT_FLOAT_EQ(out[1], 2.5f);
  EXPECT_FLOAT_EQ(out[2], 100.0f);
}

TEST(AbsFloat32, MixedValues) {
  std::vector<float> in = {-1.0f, 0.0f, 3.0f, -7.5f, 0.001f};
  std::vector<float> out(in.size(), 0.0f);
  onnx_light_cpu::AbsFloat32(in.data(), out.data(), in.size());
  for (std::size_t i = 0; i < in.size(); ++i) {
    EXPECT_FLOAT_EQ(out[i], std::fabs(in[i]));
  }
}

TEST(AbsFloat32, LargeArray) {
  // Test with sizes that exercise SIMD loop and remainder.
  for (std::size_t size : {7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 1000, 1023, 1024, 1025}) {
    std::vector<float> in(size);
    for (std::size_t i = 0; i < size; ++i) {
      in[i] = static_cast<float>(i) - static_cast<float>(size / 2);
    }
    std::vector<float> out(size, -1.0f);
    onnx_light_cpu::AbsFloat32(in.data(), out.data(), size);
    for (std::size_t i = 0; i < size; ++i) {
      EXPECT_FLOAT_EQ(out[i], std::fabs(in[i])) << "at index " << i << " size=" << size;
    }
  }
}

TEST(AbsFloat32, NegativeZero) {
  float in = -0.0f;
  float out = -1.0f;
  onnx_light_cpu::AbsFloat32(&in, &out, 1);
  EXPECT_FLOAT_EQ(out, 0.0f);
  // Ensure the sign bit is cleared.
  EXPECT_FALSE(std::signbit(out));
}

TEST(AbsFloat32, InPlace) {
  std::vector<float> data = {-1.0f, -2.0f, 3.0f, -4.0f};
  onnx_light_cpu::AbsFloat32(data.data(), data.data(), data.size());
  EXPECT_FLOAT_EQ(data[0], 1.0f);
  EXPECT_FLOAT_EQ(data[1], 2.0f);
  EXPECT_FLOAT_EQ(data[2], 3.0f);
  EXPECT_FLOAT_EQ(data[3], 4.0f);
}

TEST(AbsFloat32, StreamingStoreTuningHandlesMisalignedOutputAndTail) {
  constexpr std::size_t size = 1025;
  std::vector<float> input(size);
  std::iota(input.begin(), input.end(), -512.0f);
  std::vector<float> storage(size + 1, -1.0f);
  auto tuning = onnx_light_cpu::kDefaultAbsFloat32ExecutionTuning;
  tuning.streaming_store_threshold_bytes = 1;

  onnx_light_cpu::AbsFloat32WithTuning(input.data(), storage.data() + 1, size, tuning);

  for (std::size_t i = 0; i < size; ++i) {
    EXPECT_FLOAT_EQ(storage[i + 1], std::fabs(input[i]));
  }
}

// ---------------------------------------------------------------------------
// AbsFloat64
// ---------------------------------------------------------------------------

TEST(AbsFloat64, MixedValues) {
  std::vector<double> in = {-1.0, 0.0, 3.0, -7.5, 0.001};
  std::vector<double> out(in.size(), 0.0);
  onnx_light_cpu::AbsFloat64(in.data(), out.data(), in.size());
  for (std::size_t i = 0; i < in.size(); ++i) {
    EXPECT_DOUBLE_EQ(out[i], std::fabs(in[i]));
  }
}

TEST(AbsFloat64, LargeArray) {
  for (std::size_t size : {3, 4, 5, 7, 8, 9, 100, 255, 256, 257}) {
    std::vector<double> in(size);
    for (std::size_t i = 0; i < size; ++i) {
      in[i] = static_cast<double>(i) - static_cast<double>(size / 2);
    }
    std::vector<double> out(size, -1.0);
    onnx_light_cpu::AbsFloat64(in.data(), out.data(), size);
    for (std::size_t i = 0; i < size; ++i) {
      EXPECT_DOUBLE_EQ(out[i], std::fabs(in[i])) << "at index " << i << " size=" << size;
    }
  }
}

// ---------------------------------------------------------------------------
// AbsFloat16 (raw IEEE-754 half bit patterns; abs clears the sign bit)
// ---------------------------------------------------------------------------

TEST(AbsFloat16, ClearsSignBit) {
  // Representative half-precision bit patterns: +0, -0, 1.0, -1.0, +inf, -inf,
  // a NaN and a subnormal; abs must equal the input with the sign bit cleared.
  std::vector<std::uint16_t> in = {0x0000, 0x8000, 0x3C00, 0xBC00, 0x7C00, 0xFC00, 0xFE00, 0x8001};
  std::vector<std::uint16_t> out(in.size(), 0xFFFF);
  onnx_light_cpu::AbsFloat16(in.data(), out.data(), in.size());
  for (std::size_t i = 0; i < in.size(); ++i) {
    EXPECT_EQ(out[i], static_cast<std::uint16_t>(in[i] & 0x7FFF)) << "at index " << i;
  }
}

TEST(AbsFloat16, LargeArray) {
  for (std::size_t size : {7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 1000, 1023, 1024, 1025}) {
    std::vector<std::uint16_t> in(size);
    for (std::size_t i = 0; i < size; ++i) {
      in[i] = static_cast<std::uint16_t>((i * 2654435761u) & 0xFFFFu);
    }

    std::vector<std::uint16_t> out(size, 0);
    onnx_light_cpu::AbsFloat16(in.data(), out.data(), size);
    for (std::size_t i = 0; i < size; ++i) {
      EXPECT_EQ(out[i], static_cast<std::uint16_t>(in[i] & 0x7FFF))
          << "at index " << i << " size=" << size;
    }
  }
}

TEST(AbsInt16, MixedValues) {
  const std::vector<std::int16_t> input = {
      std::numeric_limits<std::int16_t>::min(), -123, -1, 0, 1, 123,
      std::numeric_limits<std::int16_t>::max()};
  std::vector<std::int16_t> output(input.size());
  onnx_light_cpu::AbsInt16(input.data(), output.data(), input.size());
  EXPECT_EQ(output, (std::vector<std::int16_t>{std::numeric_limits<std::int16_t>::min(), 123, 1, 0,
                                               1, 123, std::numeric_limits<std::int16_t>::max()}));
}

// ---------------------------------------------------------------------------
// AbsInt8
// ---------------------------------------------------------------------------

TEST(AbsInt8, MixedValues) {
  // Note: |-128| overflows int8 and wraps to -128, matching numpy.abs.
  std::vector<int8_t> in = {-1, 0, 3, -7, 100, -128, 127};
  std::vector<int8_t> out(in.size(), 0);
  onnx_light_cpu::AbsInt8(in.data(), out.data(), in.size());
  EXPECT_EQ(out[0], 1);
  EXPECT_EQ(out[1], 0);
  EXPECT_EQ(out[2], 3);
  EXPECT_EQ(out[3], 7);
  EXPECT_EQ(out[4], 100);
  EXPECT_EQ(out[5], -128);
  EXPECT_EQ(out[6], 127);
}

TEST(AbsInt8, LargeArray) {
  for (std::size_t size : {7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 255, 256, 257}) {
    std::vector<int8_t> in(size);
    for (std::size_t i = 0; i < size; ++i) {
      in[i] = static_cast<int8_t>(static_cast<int>(i) - static_cast<int>(size / 2));
    }
    std::vector<int8_t> out(size, 0);
    onnx_light_cpu::AbsInt8(in.data(), out.data(), size);
    for (std::size_t i = 0; i < size; ++i) {
      const int v = static_cast<int>(in[i]);
      EXPECT_EQ(out[i], static_cast<int8_t>(v < 0 ? -v : v))
          << "at index " << i << " size=" << size;
    }
  }
}

// ---------------------------------------------------------------------------
// AbsInt32
// ---------------------------------------------------------------------------

TEST(AbsInt32, MixedValues) {
  std::vector<int32_t> in = {-1, 0, 3, -7, 100, -2147483647};
  std::vector<int32_t> out(in.size(), 0);
  onnx_light_cpu::AbsInt32(in.data(), out.data(), in.size());
  EXPECT_EQ(out[0], 1);
  EXPECT_EQ(out[1], 0);
  EXPECT_EQ(out[2], 3);
  EXPECT_EQ(out[3], 7);
  EXPECT_EQ(out[4], 100);
  EXPECT_EQ(out[5], 2147483647);
}

TEST(AbsInt32, LargeArray) {
  for (std::size_t size : {7, 8, 9, 15, 16, 17, 100, 255, 256, 257}) {
    std::vector<int32_t> in(size);
    for (std::size_t i = 0; i < size; ++i) {
      in[i] = static_cast<int32_t>(i) - static_cast<int32_t>(size / 2);
    }
    std::vector<int32_t> out(size, -1);
    onnx_light_cpu::AbsInt32(in.data(), out.data(), size);
    for (std::size_t i = 0; i < size; ++i) {
      EXPECT_EQ(out[i], std::abs(in[i])) << "at index " << i << " size=" << size;
    }
  }
}

// ---------------------------------------------------------------------------
// AbsInt64
// ---------------------------------------------------------------------------

TEST(AbsInt64, MixedValues) {
  std::vector<int64_t> in = {-1, 0, 3, -7, 100, -9223372036854775807LL};
  std::vector<int64_t> out(in.size(), 0);
  onnx_light_cpu::AbsInt64(in.data(), out.data(), in.size());
  EXPECT_EQ(out[0], 1);
  EXPECT_EQ(out[1], 0);
  EXPECT_EQ(out[2], 3);
  EXPECT_EQ(out[3], 7);
  EXPECT_EQ(out[4], 100);
  EXPECT_EQ(out[5], 9223372036854775807LL);
}

TEST(AbsInt64, LargeArray) {
  for (std::size_t size : {1, 2, 3, 4, 5, 7, 8, 9, 100, 255, 256, 257}) {
    std::vector<int64_t> in(size);
    for (std::size_t i = 0; i < size; ++i) {
      in[i] = static_cast<int64_t>(i) - static_cast<int64_t>(size / 2);
    }
    std::vector<int64_t> out(size, -1);
    onnx_light_cpu::AbsInt64(in.data(), out.data(), size);
    for (std::size_t i = 0; i < size; ++i) {
      EXPECT_EQ(out[i], std::abs(in[i])) << "at index " << i << " size=" << size;
    }
  }
}

TEST(AbsParallel, LargeArrayMatchesScalar) {
  // Abs discounts each SIMD element to 1/2 of a generic work unit, so a size
  // above two grains exercises the multi-block path.
  const std::size_t size = 2000000;
  std::vector<float> in(size);
  std::vector<float> out(size, -1.0f);
  for (std::size_t i = 0; i < size; ++i) {
    in[i] = static_cast<float>(static_cast<long long>(i) - static_cast<long long>(size / 2)) * 0.5f;
  }
  onnx_light_cpu::AbsFloat32(in.data(), out.data(), size);
  for (std::size_t i = 0; i < size; ++i) {
    EXPECT_FLOAT_EQ(out[i], std::fabs(in[i])) << "at index " << i;
  }

  std::vector<std::int32_t> in32(size);
  std::vector<std::int32_t> out32(size, -1);
  for (std::size_t i = 0; i < size; ++i) {
    in32[i] = static_cast<std::int32_t>(i) - static_cast<std::int32_t>(size / 2);
  }
  onnx_light_cpu::AbsInt32(in32.data(), out32.data(), size);
  for (std::size_t i = 0; i < size; ++i) {
    EXPECT_EQ(out32[i], in32[i] < 0 ? -in32[i] : in32[i]) << "at index " << i;
  }
}

} // namespace
