// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "ml_cpu_compute/math/binary/binary_comparison.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <vector>

namespace {

using namespace ml_cpu_compute;

constexpr std::array kOperators{
    ml_cpu_compute::BinaryOperator::kEqual, ml_cpu_compute::BinaryOperator::kGreater,
    ml_cpu_compute::BinaryOperator::kGreaterOrEqual, ml_cpu_compute::BinaryOperator::kLess,
    ml_cpu_compute::BinaryOperator::kLessOrEqual};
constexpr std::array<std::uint64_t, 12> kBoundaryBits{0,
                                                      1,
                                                      2,
                                                      0x7fffffff,
                                                      0x80000000,
                                                      0xffffffff,
                                                      0x100000000,
                                                      0x7ffffffffffffffe,
                                                      0x7fffffffffffffff,
                                                      0x8000000000000000,
                                                      0x8000000000000001,
                                                      0xffffffffffffffff};

template <typename T> bool Expected(T a, T b, ml_cpu_compute::BinaryOperator op) {
  switch (op) {
  case ml_cpu_compute::BinaryOperator::kEqual:
    return a == b;
  case ml_cpu_compute::BinaryOperator::kGreater:
    return a > b;
  case ml_cpu_compute::BinaryOperator::kGreaterOrEqual:
    return a >= b;
  case ml_cpu_compute::BinaryOperator::kLess:
    return a < b;
  case ml_cpu_compute::BinaryOperator::kLessOrEqual:
    return a <= b;
  default:
    ADD_FAILURE() << "Unexpected comparison";
    return false;
  }
}

template <typename T>
using Kernel = void (*)(const T *, const T *, std::uint8_t *, std::size_t,
                        ml_cpu_compute::BinaryOperator, bool, bool);

template <typename T> void Check(Kernel<T> kernel) {
  for (std::size_t width : {4, 8}) {
    for (unsigned mask = 0; mask < (1U << width); ++mask) {
      std::vector<T> left(width, 1), right(width);
      std::vector<std::uint8_t> output(width + 2, 0xa5);
      for (std::size_t lane = 0; lane < width; ++lane) {
        right[lane] = (mask >> lane) & 1U;
      }
      kernel(left.data(), right.data(), output.data() + 1, width,
             ml_cpu_compute::BinaryOperator::kEqual, false, false);
      EXPECT_EQ(output.front(), 0xa5);
      EXPECT_EQ(output.back(), 0xa5);
      for (std::size_t lane = 0; lane < width; ++lane) {
        EXPECT_EQ(output[lane + 1], (mask >> lane) & 1U) << "mask=" << mask << " lane=" << lane;
      }
    }
  }
  for (auto op : kOperators) {
    for (bool left_scalar : {false, true}) {
      for (bool right_scalar : {false, true}) {
        kernel(nullptr, nullptr, nullptr, 0, op, left_scalar, right_scalar);
        for (std::size_t count = 1; count <= 33; ++count) {
          for (std::size_t shift = 0; shift < kBoundaryBits.size(); ++shift) {
            SCOPED_TRACE(::testing::Message() << "count=" << count << " shift=" << shift
                                              << " op=" << static_cast<int>(op) << " left_scalar="
                                              << left_scalar << " right_scalar=" << right_scalar);
            // Offset the inputs and output, and allocate no input padding after the range.
            std::vector<T> left((left_scalar ? 1 : count) + 1);
            std::vector<T> right((right_scalar ? 1 : count) + 1);
            for (std::size_t i = 1; i < left.size(); ++i) {
              left[i] = std::bit_cast<T>(kBoundaryBits[(i - 1 + shift) % kBoundaryBits.size()]);
            }
            for (std::size_t i = 1; i < right.size(); ++i) {
              right[i] =
                  std::bit_cast<T>(kBoundaryBits[(i - 1 + 2 * shift) % kBoundaryBits.size()]);
            }
            std::vector<std::uint8_t> output(count + 2, 0xa5);
            kernel(left.data() + 1, right.data() + 1, output.data() + 1, count, op, left_scalar,
                   right_scalar);
            EXPECT_EQ(output.front(), 0xa5);
            EXPECT_EQ(output.back(), 0xa5);
            for (std::size_t i = 0; i < count; ++i) {
              EXPECT_EQ(output[i + 1], Expected(left[left_scalar ? 1 : i + 1],
                                                right[right_scalar ? 1 : i + 1], op));
            }
          }
        }
        for (std::size_t count : {257, 4099}) {
          std::vector<T> left(left_scalar ? 1 : count), right(right_scalar ? 1 : count);
          std::uint64_t state = 0x9e3779b97f4a7c15;
          auto fill = [&](std::vector<T> &values) {
            for (auto &value : values) {
              state ^= state << 13;
              state ^= state >> 7;
              state ^= state << 17;
              value = std::bit_cast<T>(state);
            }
          };
          fill(left);
          fill(right);
          if (!left_scalar && !right_scalar) {
            for (std::size_t i = 0; i < count; i += 3) {
              right[i] = left[i];
            }
          }
          std::vector<std::uint8_t> output(count + 2, 0xa5);
          kernel(left.data(), right.data(), output.data() + 1, count, op, left_scalar,
                 right_scalar);
          EXPECT_EQ(output.front(), 0xa5);
          EXPECT_EQ(output.back(), 0xa5);
          for (std::size_t i = 0; i < count; ++i) {
            EXPECT_EQ(output[i + 1],
                      Expected(left[left_scalar ? 0 : i], right[right_scalar ? 0 : i], op));
          }
        }
      }
    }
  }
}

TEST(BinaryComparisonInt64, DispatchSigned) { Check<std::int64_t>(BinaryCompareInt64); }

TEST(BinaryComparisonInt64, DispatchUnsigned) { Check<std::uint64_t>(BinaryCompareUInt64); }

TEST(BinaryComparisonInt64, ReportsActualVectorPath) {
  const char *(*preferred_implementation)() = &ml_cpu_compute::BinaryCompareInt64Implementation;
  EXPECT_STREQ(preferred_implementation(), ml_cpu_compute::BinaryCompareInt64Implementation(8));
  EXPECT_EQ(preferred_implementation(), preferred_implementation());
  for (std::size_t count = 0; count < 4; ++count) {
    EXPECT_EQ(ml_cpu_compute::BinaryComparisonInt64SimdLevel(count),
              ml_cpu_compute::SimdLevel::kNone);
    EXPECT_STREQ(ml_cpu_compute::BinaryCompareInt64Implementation(count), "scalar");
  }
  const auto short_level = ml_cpu_compute::BinaryComparisonInt64SimdLevel(4);
  EXPECT_TRUE(short_level == ml_cpu_compute::SimdLevel::kNone ||
              short_level == ml_cpu_compute::SimdLevel::kAVX2);
  EXPECT_EQ(ml_cpu_compute::BinaryComparisonInt64SimdLevel(7), short_level);
  const auto long_level = ml_cpu_compute::BinaryComparisonInt64SimdLevel(8);
  EXPECT_GE(long_level, short_level);
  EXPECT_EQ(ml_cpu_compute::BinaryComparisonInt64SimdLevel(4099), long_level);
  for (std::size_t count : {4, 8, 4099}) {
    const auto level = ml_cpu_compute::BinaryComparisonInt64SimdLevel(count);
    EXPECT_LE(level, ml_cpu_compute::DetectSimdLevel());
    EXPECT_TRUE(level == ml_cpu_compute::SimdLevel::kNone ||
                level == ml_cpu_compute::SimdLevel::kAVX2 ||
                level == ml_cpu_compute::SimdLevel::kAVX512);
    EXPECT_STREQ(ml_cpu_compute::BinaryCompareInt64Implementation(count),
                 level == ml_cpu_compute::SimdLevel::kAVX512 ? "avx512"
                 : level == ml_cpu_compute::SimdLevel::kAVX2 ? "avx2"
                                                             : "scalar");
  }
}

#ifdef ML_CPU_COMPUTE_HAVE_AVX2
TEST(BinaryComparisonInt64, Avx2Signed) {
  if (ml_cpu_compute::DetectSimdLevel() < ml_cpu_compute::SimdLevel::kAVX2) {
    GTEST_SKIP() << "AVX2 unavailable";
  }
  Check<std::int64_t>(ml_cpu_compute::BinaryCompareInt64_AVX2);
}

TEST(BinaryComparisonInt64, Avx2Unsigned) {
  if (ml_cpu_compute::DetectSimdLevel() < ml_cpu_compute::SimdLevel::kAVX2) {
    GTEST_SKIP() << "AVX2 unavailable";
  }
  Check<std::uint64_t>(ml_cpu_compute::BinaryCompareUInt64_AVX2);
}
#endif

#ifdef ML_CPU_COMPUTE_HAVE_AVX512
TEST(BinaryComparisonInt64, Avx512Signed) {
  if (ml_cpu_compute::DetectSimdLevel() < ml_cpu_compute::SimdLevel::kAVX512) {
    GTEST_SKIP() << "AVX-512F unavailable";
  }
  Check<std::int64_t>(ml_cpu_compute::BinaryCompareInt64_AVX512);
}

TEST(BinaryComparisonInt64, Avx512Unsigned) {
  if (ml_cpu_compute::DetectSimdLevel() < ml_cpu_compute::SimdLevel::kAVX512) {
    GTEST_SKIP() << "AVX-512F unavailable";
  }
  Check<std::uint64_t>(ml_cpu_compute::BinaryCompareUInt64_AVX512);
}
#endif

} // namespace
