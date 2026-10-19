// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "ml_cpu_compute/checked_arithmetic.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
namespace {

TEST(CheckedArithmetic, RejectsNegativeDimensions) {
  EXPECT_THROW(ml_cpu_compute::CheckedDimension(-1, "Test", "dimension"), std::invalid_argument);
}

TEST(CheckedArithmetic, RejectsSizeByteIndexAndOffsetOverflow) {
  constexpr std::size_t size_max = std::numeric_limits<std::size_t>::max();
  constexpr std::int64_t index_max = std::numeric_limits<std::int64_t>::max();
  EXPECT_THROW(ml_cpu_compute::CheckedMultiply(size_max, 2, "Test", "shape"),
               std::invalid_argument);
  EXPECT_THROW(ml_cpu_compute::CheckedByteSize(size_max / 2 + 1, 2, "Test", "bytes"),
               std::invalid_argument);
  EXPECT_THROW(ml_cpu_compute::CheckedIndexMultiply(index_max / 2 + 1, 2, "Test", "stride"),
               std::invalid_argument);
  EXPECT_THROW(ml_cpu_compute::CheckedIndexAdd(index_max, 1, "Test", "offset"),
               std::invalid_argument);
  EXPECT_THROW(ml_cpu_compute::CheckedStride(
                   static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) + 1, "Test",
                   "stride"),
               std::invalid_argument);
}

TEST(CheckedArithmetic, ComputesBoundaryValues) {
  constexpr std::size_t size_max = std::numeric_limits<std::size_t>::max();
  constexpr std::int64_t index_max = std::numeric_limits<std::int64_t>::max();
  EXPECT_EQ(ml_cpu_compute::CheckedProduct({size_max / 2, 2}, "Test", "shape"),
            size_max - size_max % 2);
  EXPECT_EQ(ml_cpu_compute::CheckedIndexMultiply(index_max / 7, 7, "Test", "stride"),
            index_max - index_max % 7);
  EXPECT_EQ(ml_cpu_compute::CheckedIndexAdd(index_max - 1, 1, "Test", "offset"), index_max);
}

} // namespace
