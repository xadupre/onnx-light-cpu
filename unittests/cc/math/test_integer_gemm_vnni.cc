// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Differential correctness gate for byte and packed-4 integer matrix
// multiplication (Roadmap PR10.2). Every case checks the public dispatcher and
// portable scalar path, then directly checks each native implementation
// available on the running CPU against an independent reference.

#include "onnx_light_cpu/impl/math/gemm/vnni/integer_gemm_vnni.h"

#include "onnx_light_cpu/impl/execution.h"

#if defined(ONNX_LIGHT_CPU_HAVE_AVX2_INTEGER) || defined(ONNX_LIGHT_CPU_HAVE_AVX512VNNI)
#include "onnx_light_cpu/impl/simd_level.h"
#endif

#ifdef ONNX_LIGHT_CPU_HAVE_AMX_INT8
#include "onnx_light_cpu/impl/math/gemm/amx/gemm_amx_int8.h"
#include "onnx_light_cpu/impl/math/gemm/amx/gemm_amx_tile.h"
#include "onnx_light_cpu/impl/simd_level.h"
#endif

#ifdef ONNX_LIGHT_CPU_HAVE_NEON_DOTPROD
#include "onnx_light_cpu/impl/arm_simd_level.h"
#include "onnx_light_cpu/impl/math/gemm/arm/gemm_kernel_arm.h"
#endif

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <random>
#include <vector>

namespace {

// Naive reference matching the portable PR09.1 accumulation exactly.
std::vector<std::int32_t> Reference(const std::vector<std::uint8_t> &a, bool a_signed,
                                    const std::vector<std::uint8_t> &b, bool b_signed,
                                    std::int64_t rows, std::int64_t cols, std::int64_t depth,
                                    const std::vector<std::int32_t> &azp,
                                    const std::vector<std::int32_t> &bzp) {
  std::vector<std::int32_t> c(static_cast<std::size_t>(rows * cols), 0);
  for (std::int64_t i = 0; i < rows; ++i) {
    const std::int32_t az = azp.size() == 1 ? azp[0] : azp[static_cast<std::size_t>(i)];
    for (std::int64_t j = 0; j < cols; ++j) {
      const std::int32_t bz = bzp.size() == 1 ? bzp[0] : bzp[static_cast<std::size_t>(j)];
      std::uint32_t accumulator = 0;
      for (std::int64_t d = 0; d < depth; ++d) {
        const std::int32_t av =
            a_signed ? static_cast<std::int8_t>(a[static_cast<std::size_t>(i * depth + d)])
                     : static_cast<std::int32_t>(a[static_cast<std::size_t>(i * depth + d)]);
        const std::int32_t bv =
            b_signed ? static_cast<std::int8_t>(b[static_cast<std::size_t>(d * cols + j)])
                     : static_cast<std::int32_t>(b[static_cast<std::size_t>(d * cols + j)]);
        accumulator += static_cast<std::uint32_t>((av - az) * (bv - bz));
      }
      c[static_cast<std::size_t>(i * cols + j)] = std::bit_cast<std::int32_t>(accumulator);
    }
  }
  return c;
}

void CheckAllPaths(const std::vector<std::uint8_t> &a, bool a_signed,
                   const std::vector<std::uint8_t> &b, bool b_signed, std::int64_t rows,
                   std::int64_t cols, std::int64_t depth, const std::vector<std::int32_t> &azp,
                   const std::vector<std::int32_t> &bzp) {
  const std::vector<std::int32_t> expected =
      Reference(a, a_signed, b, b_signed, rows, cols, depth, azp, bzp);

  std::vector<std::int32_t> dispatched(expected.size(), 0);
  onnx_light_cpu::IntegerMatMul2D(a.data(), a_signed, b.data(), b_signed, dispatched.data(), rows,
                                  cols, depth, azp.data(), static_cast<std::int64_t>(azp.size()),
                                  bzp.data(), static_cast<std::int64_t>(bzp.size()));
  EXPECT_EQ(dispatched, expected);

  std::vector<std::int32_t> scalar(expected.size(), 0);
  onnx_light_cpu::detail::IntegerMatMul2DWithDot(
      &onnx_light_cpu::detail::IntegerDotU8S8Scalar, a.data(), a_signed, b.data(), b_signed,
      scalar.data(), rows, cols, depth, azp.data(), static_cast<std::int64_t>(azp.size()),
      bzp.data(), static_cast<std::int64_t>(bzp.size()));
  EXPECT_EQ(scalar, expected);

#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_INTEGER
  if (onnx_light_cpu::DetectSimdLevel() >= onnx_light_cpu::SimdLevel::kAVX2) {
    std::vector<std::int32_t> native(expected.size(), 0);
    onnx_light_cpu::detail::IntegerMatMul2DWithDot(
        &onnx_light_cpu::detail::IntegerDotU8S8Avx2, a.data(), a_signed, b.data(), b_signed,
        native.data(), rows, cols, depth, azp.data(), static_cast<std::int64_t>(azp.size()),
        bzp.data(), static_cast<std::int64_t>(bzp.size()));
    EXPECT_EQ(native, expected);
    if (rows == 1) {
      std::fill(native.begin(), native.end(), 0);
      onnx_light_cpu::detail::IntegerMatMulSkinnyMAvx2(
          a.data(), a_signed, b.data(), b_signed, native.data(), cols, depth, azp[0], bzp.data(),
          static_cast<std::int64_t>(bzp.size()));
      EXPECT_EQ(native, expected);
    }
  }
#endif

#ifdef ONNX_LIGHT_CPU_HAVE_AVX512VNNI
  if (onnx_light_cpu::IntegerMatMul2DUsesVnni()) {
    std::vector<std::int32_t> native(expected.size(), 0);
    onnx_light_cpu::detail::IntegerMatMul2DWithDot(
        &onnx_light_cpu::detail::IntegerDotU8S8Avx512Vnni, a.data(), a_signed, b.data(), b_signed,
        native.data(), rows, cols, depth, azp.data(), static_cast<std::int64_t>(azp.size()),
        bzp.data(), static_cast<std::int64_t>(bzp.size()));
    EXPECT_EQ(native, expected);
  }
#endif

#if defined(ONNX_LIGHT_CPU_HAVE_AVX512VNNI) && defined(ONNX_LIGHT_CPU_HAVE_AVX512BW)
  if (rows == 1 && onnx_light_cpu::CpuSupportsAvx512Vnni() &&
      onnx_light_cpu::CpuSupportsAvx512BW()) {
    std::vector<std::int32_t> native(expected.size(), 0);
    onnx_light_cpu::detail::IntegerMatMulSkinnyMAvx512(
        a.data(), a_signed, b.data(), b_signed, native.data(), cols, depth, azp[0], bzp.data(),
        static_cast<std::int64_t>(bzp.size()));
    EXPECT_EQ(native, expected);
  }
#endif

#ifdef ONNX_LIGHT_CPU_HAVE_NEON_DOTPROD
  if (onnx_light_cpu::CpuSupportsNeonDotProd()) {
    std::vector<std::int32_t> native(expected.size(), 0);
    onnx_light_cpu::GemmMatMulIntegerNeonDotProd(
        a.data(), a_signed, b.data(), b_signed, native.data(), static_cast<std::size_t>(rows),
        static_cast<std::size_t>(cols), static_cast<std::size_t>(depth), azp.data(), azp.size(),
        bzp.data(), bzp.size());
    EXPECT_EQ(native, expected);
  }
#endif

#ifdef ONNX_LIGHT_CPU_HAVE_AMX_INT8
  if (onnx_light_cpu::AmxTileStateAvailable() && onnx_light_cpu::CpuSupportsAmxInt8()) {
    std::vector<std::int32_t> native(expected.size(), 0);
    onnx_light_cpu::GemmMatMulIntegerAmxInt8(
        a.data(), a_signed, b.data(), b_signed, native.data(), static_cast<std::size_t>(rows),
        static_cast<std::size_t>(cols), static_cast<std::size_t>(depth), azp.data(), azp.size(),
        bzp.data(), bzp.size());
    EXPECT_EQ(native, expected);
  }
#endif
}

std::vector<std::uint8_t> Pack4Bit(const std::vector<std::int32_t> &values) {
  std::vector<std::uint8_t> packed((values.size() + 1) / 2, 0xa0);
  for (std::size_t index = 0; index < values.size(); ++index) {
    const std::uint8_t nibble = static_cast<std::uint8_t>(values[index]) & 0x0f;
    if ((index & 1) == 0) {
      packed[index / 2] = static_cast<std::uint8_t>((packed[index / 2] & 0xf0) | nibble);
    } else {
      packed[index / 2] = static_cast<std::uint8_t>((packed[index / 2] & 0x0f) | (nibble << 4));
    }
  }
  return packed;
}

std::vector<std::int32_t> Reference4Bit(const std::vector<std::int32_t> &a,
                                        const std::vector<std::int32_t> &b, std::int64_t rows,
                                        std::int64_t cols, std::int64_t depth,
                                        const std::vector<std::int32_t> &azp,
                                        const std::vector<std::int32_t> &bzp) {
  std::vector<std::int32_t> c(static_cast<std::size_t>(rows * cols), 0);
  for (std::int64_t row = 0; row < rows; ++row) {
    const std::int32_t az = azp.size() == 1 ? azp[0] : azp[static_cast<std::size_t>(row)];
    for (std::int64_t column = 0; column < cols; ++column) {
      const std::int32_t bz = bzp.size() == 1 ? bzp[0] : bzp[static_cast<std::size_t>(column)];
      std::uint32_t accumulator = 0;
      for (std::int64_t inner = 0; inner < depth; ++inner) {
        const std::int64_t av = a[static_cast<std::size_t>(row * depth + inner)];
        const std::int64_t bv = b[static_cast<std::size_t>(inner * cols + column)];
        accumulator += static_cast<std::uint32_t>((av - az) * (bv - bz));
      }
      c[static_cast<std::size_t>(row * cols + column)] = std::bit_cast<std::int32_t>(accumulator);
    }
  }
  return c;
}

void Check4BitAllPaths(bool a_signed, bool b_signed, std::int64_t rows, std::int64_t cols,
                       std::int64_t depth, bool per_axis) {
  std::vector<std::int32_t> a(static_cast<std::size_t>(rows * depth));
  std::vector<std::int32_t> b(static_cast<std::size_t>(depth * cols));
  for (std::size_t index = 0; index < a.size(); ++index) {
    const std::int32_t nibble = static_cast<std::int32_t>((index * 5 + 3) & 0x0f);
    a[index] = a_signed && nibble >= 8 ? nibble - 16 : nibble;
  }
  for (std::size_t index = 0; index < b.size(); ++index) {
    const std::int32_t nibble = static_cast<std::int32_t>((index * 7 + 1) & 0x0f);
    b[index] = b_signed && nibble >= 8 ? nibble - 16 : nibble;
  }

  std::vector<std::int32_t> azp(per_axis ? static_cast<std::size_t>(rows) : 1);
  std::vector<std::int32_t> bzp(per_axis ? static_cast<std::size_t>(cols) : 1);
  for (std::size_t index = 0; index < azp.size(); ++index) {
    azp[index] = a_signed ? static_cast<std::int32_t>(index % 16) - 8
                          : static_cast<std::int32_t>(index % 16);
  }
  for (std::size_t index = 0; index < bzp.size(); ++index) {
    bzp[index] = b_signed ? 7 - static_cast<std::int32_t>(index % 16)
                          : 15 - static_cast<std::int32_t>(index % 16);
  }

  const std::vector<std::uint8_t> packed_a = Pack4Bit(a);
  const std::vector<std::uint8_t> packed_b = Pack4Bit(b);
  const std::vector<std::int32_t> expected = Reference4Bit(a, b, rows, cols, depth, azp, bzp);

  std::vector<std::int32_t> dispatched(expected.size(), 0);
  onnx_light_cpu::IntegerMatMul4Bit2D(packed_a.data(), a_signed, packed_b.data(), b_signed,
                                      dispatched.data(), rows, cols, depth, azp.data(),
                                      static_cast<std::int64_t>(azp.size()), bzp.data(),
                                      static_cast<std::int64_t>(bzp.size()));
  EXPECT_EQ(dispatched, expected);

  std::vector<std::int32_t> scalar(expected.size(), 0);
  onnx_light_cpu::detail::IntegerMatMul4Bit2DWithDot(
      &onnx_light_cpu::detail::IntegerDotU8S8Scalar, packed_a.data(), a_signed, packed_b.data(),
      b_signed, scalar.data(), rows, cols, depth, azp.data(), static_cast<std::int64_t>(azp.size()),
      bzp.data(), static_cast<std::int64_t>(bzp.size()));
  EXPECT_EQ(scalar, expected);

#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_INTEGER
  if (onnx_light_cpu::DetectSimdLevel() >= onnx_light_cpu::SimdLevel::kAVX2) {
    std::vector<std::int32_t> native(expected.size(), 0);
    onnx_light_cpu::detail::IntegerMatMul4Bit2DWithDot(
        &onnx_light_cpu::detail::IntegerDotU8S8Avx2, packed_a.data(), a_signed, packed_b.data(),
        b_signed, native.data(), rows, cols, depth, azp.data(),
        static_cast<std::int64_t>(azp.size()), bzp.data(), static_cast<std::int64_t>(bzp.size()));
    EXPECT_EQ(native, expected);
  }
#endif

#ifdef ONNX_LIGHT_CPU_HAVE_AVX512VNNI
  if (onnx_light_cpu::IntegerMatMul2DUsesVnni()) {
    std::vector<std::int32_t> native(expected.size(), 0);
    onnx_light_cpu::detail::IntegerMatMul4Bit2DWithDot(
        &onnx_light_cpu::detail::IntegerDotU8S8Avx512Vnni, packed_a.data(), a_signed,
        packed_b.data(), b_signed, native.data(), rows, cols, depth, azp.data(),
        static_cast<std::int64_t>(azp.size()), bzp.data(), static_cast<std::int64_t>(bzp.size()));
    EXPECT_EQ(native, expected);
  }
#endif

#ifdef ONNX_LIGHT_CPU_HAVE_NEON_DOTPROD
  if (onnx_light_cpu::CpuSupportsNeonDotProd()) {
    std::vector<std::int32_t> native(expected.size(), 0);
    onnx_light_cpu::detail::IntegerMatMul4Bit2DWithDot(
        &onnx_light_cpu::detail::IntegerDot4BitU8S8NeonDotProd, packed_a.data(), a_signed,
        packed_b.data(), b_signed, native.data(), rows, cols, depth, azp.data(),
        static_cast<std::int64_t>(azp.size()), bzp.data(), static_cast<std::int64_t>(bzp.size()));
    EXPECT_EQ(native, expected);
  }
#endif
}

TEST(IntegerVnniKernel, MatchesReferenceAcrossSignednessAndZeroPoints) {
  std::mt19937 rng(1234);
  std::uniform_int_distribution<int> byte(0, 255);
  std::uniform_int_distribution<int> zp(-5, 5);

  // Include depths that are not a multiple of the 64-byte VNNI vector so the
  // scalar tail is exercised, plus tiny and larger matrices.
  const std::vector<std::array<std::int64_t, 3>> shapes = {
      {1, 1, 1},   {2, 3, 5},   {3, 2, 64},  {4, 4, 65},  {5, 7, 100}, {8, 6, 128},
      {2, 9, 131}, {6, 3, 200}, {1, 16, 68}, {1, 19, 67}, {17, 1, 68}, {33, 35, 70}};

  for (const auto &shape : shapes) {
    const std::int64_t rows = shape[0];
    const std::int64_t cols = shape[1];
    const std::int64_t depth = shape[2];
    for (bool a_signed : {false, true}) {
      for (bool b_signed : {false, true}) {
        std::vector<std::uint8_t> a(static_cast<std::size_t>(rows * depth));
        std::vector<std::uint8_t> b(static_cast<std::size_t>(depth * cols));
        for (auto &value : a) {
          value = static_cast<std::uint8_t>(byte(rng));
        }
        for (auto &value : b) {
          value = static_cast<std::uint8_t>(byte(rng));
        }

        // Scalar zero points.
        CheckAllPaths(a, a_signed, b, b_signed, rows, cols, depth, {zp(rng)}, {zp(rng)});

        // Per-row / per-column zero points.
        std::vector<std::int32_t> azp(static_cast<std::size_t>(rows));
        std::vector<std::int32_t> bzp(static_cast<std::size_t>(cols));
        for (auto &value : azp) {
          value = zp(rng);
        }
        for (auto &value : bzp) {
          value = zp(rng);
        }
        CheckAllPaths(a, a_signed, b, b_signed, rows, cols, depth, azp, bzp);
      }
    }
  }
}

TEST(IntegerVnniKernel, AccumulationWrapsModuloInt32) {
  constexpr std::int64_t depth = 40000;
  const std::vector<std::uint8_t> a(static_cast<std::size_t>(depth), 255);
  const std::vector<std::uint8_t> b(static_cast<std::size_t>(depth), 255);
  const std::vector<std::int32_t> zero = {0};

  std::vector<std::int32_t> out(1, 0);
  onnx_light_cpu::IntegerMatMul2D(a.data(), /*a_signed=*/false, b.data(), /*b_signed=*/false,
                                  out.data(), 1, 1, depth, zero.data(), 1, zero.data(), 1);

  const std::uint32_t wrapped = static_cast<std::uint32_t>(65025ULL * depth);
  EXPECT_EQ(std::bit_cast<std::uint32_t>(out[0]), wrapped);
}

TEST(IntegerVnniKernel, SkinnyPlannerPreservesPackedBoundary) {
  using namespace onnx_light_cpu;
  EXPECT_EQ(SelectIntegerMatMulPlan(3, 4096, 4096), IntegerMatMulPlan::kPacked);
  EXPECT_EQ(SelectIntegerMatMulPlan(2, 31, 128), IntegerMatMulPlan::kPacked);
  EXPECT_EQ(SelectIntegerMatMulPlan(2, 32, 3), IntegerMatMulPlan::kPacked);
  EXPECT_EQ(SelectIntegerMatMulPlan(0, 32, 4), IntegerMatMulPlan::kPacked);
  EXPECT_EQ(SelectIntegerMatMulPlan(2, 0, 4), IntegerMatMulPlan::kPacked);
  auto expected = IntegerMatMulPlan::kPacked;
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_INTEGER
  if (DetectSimdLevel() >= SimdLevel::kAVX2) {
    expected = IntegerMatMulPlan::kNoPackAvx2;
  }
#endif
#if defined(ONNX_LIGHT_CPU_HAVE_AVX512VNNI) && defined(ONNX_LIGHT_CPU_HAVE_AVX512BW)
  if (CpuSupportsAvx512Vnni() && CpuSupportsAvx512BW()) {
    expected = IntegerMatMulPlan::kNoPackVnni;
  }
#endif
  EXPECT_EQ(SelectIntegerMatMulPlan(1, 4096, 4096), expected);
  EXPECT_EQ(SelectIntegerMatMulPlan(2, 32, 4), expected);
  EXPECT_EQ(SelectIntegerMatMulPlan(2, 4096, 4096), expected);
  EXPECT_STREQ(IntegerMatMulPlanName(IntegerMatMulPlan::kPacked), "packed");
  EXPECT_STREQ(IntegerMatMulPlanName(IntegerMatMulPlan::kNoPackAvx2), "no-pack-avx2");
  EXPECT_STREQ(IntegerMatMulPlanName(IntegerMatMulPlan::kNoPackVnni), "no-pack-vnni");
}

TEST(IntegerVnniKernel, SkinnySignednessZeroPointsAndEveryColumnTail) {
  std::mt19937 rng(725);
  for (const std::int64_t rows : {1, 2, 3}) {
    for (const std::int64_t depth : {0, 1, 3, 4, 5, 127, 128, 129, 131, 257}) {
      for (std::int64_t cols = 31; cols <= 97; ++cols) {
        SCOPED_TRACE(::testing::PrintToString(std::array<std::int64_t, 3>{rows, cols, depth}));
        std::vector<std::uint8_t> a(rows * depth), b(depth * cols);
        std::generate(a.begin(), a.end(), [&] { return static_cast<std::uint8_t>(rng()); });
        std::generate(b.begin(), b.end(), [&] { return static_cast<std::uint8_t>(rng()); });
        for (const bool a_signed : {false, true}) {
          for (const bool b_signed : {false, true}) {
            CheckAllPaths(a, a_signed, b, b_signed, rows, cols, depth, {0}, {0});
            CheckAllPaths(a, a_signed, b, b_signed, rows, cols, depth, {a_signed ? 0 : 128}, {0});
            const std::int32_t az = a_signed ? -128 : 255;
            const std::int32_t bz = b_signed ? 127 : 255;
            CheckAllPaths(a, a_signed, b, b_signed, rows, cols, depth, {az}, {bz});
            std::vector<std::int32_t> azp(rows, az), bzp(cols, bz);
            for (std::int64_t i = 0; i < rows; ++i) {
              azp[i] = a_signed ? static_cast<std::int32_t>(i) - 1 : static_cast<std::int32_t>(i);
            }
            for (std::int64_t j = 0; j < cols; ++j) {
              bzp[j] = b_signed ? static_cast<std::int32_t>(j) - 32 : static_cast<std::int32_t>(j);
            }
            CheckAllPaths(a, a_signed, b, b_signed, rows, cols, depth, azp, bzp);
          }
        }
      }
    }
  }
}

TEST(IntegerVnniKernel, SkinnyVectorAccumulatorsWrap) {
  constexpr std::int64_t rows = 2, cols = 65, depth = 40003;
  const std::vector<std::uint8_t> a(rows * depth, 255), b(cols * depth, 255);
  CheckAllPaths(a, false, b, false, rows, cols, depth, {0, 255}, {0});
  CheckAllPaths(a, true, b, true, rows, cols, depth, {-128, 127}, {-128});
}

// Splits every ``ExecuteRanges`` call in the packing, bulk AVX2 microkernel,
// and correction loops across several inline blocks (run synchronously, on
// the calling thread, in reverse block order) so a real physical-core policy
// exercises the same disjoint-range writes without needing a live thread
// pool in this differential test binary.
struct InlineBlockExecutor {
  static void Run(void *, std::int64_t num_blocks, void *task_context,
                  onnx_light_cpu::ExecutionBlockFn task) {
    for (std::int64_t block = num_blocks; block > 0; --block) {
      task(task_context, block - 1);
    }
  }
};

struct SkinnyBlockExecutor {
  std::int64_t calls = 0;
  std::int64_t blocks = 0;

  static void Run(void *context, std::int64_t num_blocks, void *task_context,
                  onnx_light_cpu::ExecutionBlockFn task) {
    auto &self = *static_cast<SkinnyBlockExecutor *>(context);
    ++self.calls;
    self.blocks += num_blocks;
    EXPECT_GT(num_blocks, 1);
    for (std::int64_t block = num_blocks; block > 0; --block) {
      task(task_context, block - 1);
    }
  }
};

TEST(IntegerVnniKernel, SkinnySchedulingIsDisjointAndDoesNotNest) {
  using namespace onnx_light_cpu;
  if (SelectIntegerMatMulPlan(2, 1025, 1025) == IntegerMatMulPlan::kPacked) {
    GTEST_SKIP() << "No streaming ISA compiled or available";
  }
  SkinnyBlockExecutor executor;
  ExecutionExecutorView view{&executor, 8, &SkinnyBlockExecutor::Run, nullptr};
  ExecutionExecutorScope scope(&view);
  const std::vector<std::uint8_t> a(2 * 1025, 253), b(1025 * 1025, 129);
  std::vector<std::int32_t> c(2 * 1025, 0);
  const std::int32_t az[] = {0, 255}, bz = -128;
  const auto run = [&](std::int64_t n, std::int64_t k) {
    IntegerMatMul2D(a.data(), false, b.data(), true, c.data(), 2, n, k, az, 2, &bz, 1);
  };
  run(32, 4);
  EXPECT_EQ(executor.calls, 0);
  run(1025, 1025);
  EXPECT_GT(executor.calls, 0);
  EXPECT_GT(executor.blocks, executor.calls);
  EXPECT_EQ(c, Reference(a, false, b, true, 2, 1025, 1025, {0, 255}, {-128}));
  const auto calls = executor.calls;
  {
    detail::ExecutionRegionScope nested;
    run(1025, 1025);
  }
  EXPECT_EQ(executor.calls, calls);
}

TEST(IntegerVnniKernel, ParallelExecutionMatchesSerialReferenceAcrossShapes) {
  onnx_light_cpu::ExecutionExecutorView view{nullptr, 8, &InlineBlockExecutor::Run, nullptr};
  onnx_light_cpu::ExecutionExecutorScope scope(&view);

  std::mt19937 rng(4242);
  std::uniform_int_distribution<int> byte(0, 255);
  // Large enough on every axis (row-packing, column-packing, and the bulk
  // depth>=32 microkernel) that ``ExecuteRanges`` splits into more than one
  // block with the 8 declared participants above.
  const std::vector<std::array<std::int64_t, 3>> shapes = {
      {1, 4096, 4096}, {4096, 1, 4096}, {64, 64, 512}, {65, 63, 97}};
  for (const auto &shape : shapes) {
    const std::int64_t rows = shape[0];
    const std::int64_t cols = shape[1];
    const std::int64_t depth = shape[2];
    std::vector<std::uint8_t> a(static_cast<std::size_t>(rows * depth));
    std::vector<std::uint8_t> b(static_cast<std::size_t>(depth * cols));
    for (auto &value : a) {
      value = static_cast<std::uint8_t>(byte(rng));
    }
    for (auto &value : b) {
      value = static_cast<std::uint8_t>(byte(rng));
    }
    CheckAllPaths(a, /*a_signed=*/true, b, /*b_signed=*/false, rows, cols, depth, {-3}, {2});
    Check4BitAllPaths(/*a_signed=*/true, /*b_signed=*/false, rows, cols, depth, /*per_axis=*/true);
  }
}

TEST(IntegerVnniKernel, HandlesZeroDepth) {
  const std::vector<std::int32_t> zero = {0};
  std::int32_t output = 1;
  onnx_light_cpu::IntegerMatMul2D(nullptr, true, nullptr, true, &output, 1, 1, 0, zero.data(), 1,
                                  zero.data(), 1);
  EXPECT_EQ(output, 0);
}

#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_INTEGER
TEST(IntegerVnniKernel, Avx2DotAvoidsPairwiseSaturation) {
  if (onnx_light_cpu::DetectSimdLevel() < onnx_light_cpu::SimdLevel::kAVX2) {
    GTEST_SKIP() << "AVX2 is not available on this CPU";
  }
  // 15 and below never reach the 128-bit remainder step; 16 through 31 cover
  // the SSSE3 remainder path added to ``IntegerDotU8S8Avx2ShortTail`` to
  // close the short-K "direct" GEMM gap, both below and above its own
  // 16-byte boundary. Every depth is checked against both dot variants: the
  // plain ``IntegerDotU8S8Avx2`` (selected for depths that are an exact
  // multiple of 32) and ``IntegerDotU8S8Avx2ShortTail`` (selected otherwise).
  for (std::int64_t depth : {1, 15, 16, 17, 20, 31, 32, 33, 255, 256, 257}) {
    std::vector<std::uint8_t> a(static_cast<std::size_t>(depth));
    std::vector<std::int8_t> b(static_cast<std::size_t>(depth));
    for (std::int64_t index = 0; index < depth; ++index) {
      a[static_cast<std::size_t>(index)] = index % 3 == 0 ? std::uint8_t{255} : std::uint8_t{128};
      b[static_cast<std::size_t>(index)] = index % 2 == 0 ? std::int8_t{127} : std::int8_t{-128};
    }
    const std::int32_t expected =
        onnx_light_cpu::detail::IntegerDotU8S8Scalar(a.data(), b.data(), depth);
    EXPECT_EQ(onnx_light_cpu::detail::IntegerDotU8S8Avx2(a.data(), b.data(), depth), expected);
    EXPECT_EQ(onnx_light_cpu::detail::IntegerDotU8S8Avx2ShortTail(a.data(), b.data(), depth),
              expected);
  }
}

TEST(IntegerVnniKernel, Avx2BlockedKernelHandlesOutputAndDepthTails) {
  if (onnx_light_cpu::DetectSimdLevel() < onnx_light_cpu::SimdLevel::kAVX2) {
    GTEST_SKIP() << "AVX2 is not available on this CPU";
  }
  constexpr std::int64_t rows = 3;
  constexpr std::int64_t cols = 5;
  for (const std::int64_t depth : {32, 65}) {
    std::vector<std::uint8_t> a(static_cast<std::size_t>(rows * depth));
    std::vector<std::uint8_t> b(static_cast<std::size_t>(depth * cols));
    for (std::size_t index = 0; index < a.size(); ++index) {
      a[index] = index % 2 == 0 ? std::uint8_t{255} : std::uint8_t{128};
    }
    for (std::size_t index = 0; index < b.size(); ++index) {
      b[index] = index % 2 == 0 ? std::uint8_t{127} : std::uint8_t{128};
    }
    CheckAllPaths(a, false, b, true, rows, cols, depth, {0}, {0});
  }
}
#endif

TEST(IntegerPacked4Bit, MatchesReferenceAcrossFormatsAndOddTails) {
  for (bool a_signed : {false, true}) {
    for (bool b_signed : {false, true}) {
      Check4BitAllPaths(a_signed, b_signed, 1, 1, 1, false);
      Check4BitAllPaths(a_signed, b_signed, 5, 7, 17, false);
      Check4BitAllPaths(a_signed, b_signed, 4, 6, 64, true);
      Check4BitAllPaths(a_signed, b_signed, 3, 5, 65, true);
    }
  }
}

TEST(IntegerPacked4Bit, IgnoresUnusedFinalHighNibble) {
  const std::vector<std::int32_t> zero = {0};
  std::vector<std::uint8_t> a = Pack4Bit({2, 3, 4});
  std::vector<std::uint8_t> b = Pack4Bit({5, 6, 7});
  a.back() = static_cast<std::uint8_t>((a.back() & 0x0f) | 0xf0);
  b.back() = static_cast<std::uint8_t>((b.back() & 0x0f) | 0x80);
  std::int32_t output = 0;
  onnx_light_cpu::IntegerMatMul4Bit2D(a.data(), false, b.data(), false, &output, 1, 1, 3,
                                      zero.data(), 1, zero.data(), 1);
  EXPECT_EQ(output, 2 * 5 + 3 * 6 + 4 * 7);
}

TEST(IntegerPacked4Bit, HandlesZeroDepthAndModuloInt32Correction) {
  const std::vector<std::int32_t> zero = {0};
  std::int32_t output = 1;
  onnx_light_cpu::IntegerMatMul4Bit2D(nullptr, true, nullptr, true, &output, 1, 1, 0, zero.data(),
                                      1, zero.data(), 1);
  EXPECT_EQ(output, 0);

  const std::vector<std::uint8_t> packed = Pack4Bit({0});
  const std::vector<std::int32_t> large_negative_zero_point = {-2000000000};
  const std::uint32_t expected = static_cast<std::uint32_t>(2000000000LL * 2000000000LL);
  onnx_light_cpu::IntegerMatMul4Bit2D(packed.data(), false, packed.data(), false, &output, 1, 1, 1,
                                      large_negative_zero_point.data(), 1,
                                      large_negative_zero_point.data(), 1);
  EXPECT_EQ(std::bit_cast<std::uint32_t>(output), expected);
}

} // namespace
