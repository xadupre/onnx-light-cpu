// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Differential tests for the x86 VNNI INT8 matrix multiplication (Roadmap
// PR09.2). Every case checks the public dispatcher, the portable scalar path,
// and -- when the CPU exposes AVX-512 VNNI -- the native vpdpbusd path against
// an independent naive reference that mirrors the portable PR09.1 fallback.

#include "onnx_light_cpu/impl/math/gemm/vnni/integer_gemm_vnni.h"

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
  const std::vector<std::array<std::int64_t, 3>> shapes = {{1, 1, 1},   {2, 3, 5},   {3, 2, 64},
                                                           {4, 4, 65},  {5, 7, 100}, {8, 6, 128},
                                                           {2, 9, 131}, {6, 3, 200}};

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
