// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/tensor/nonzero_kernel.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace {

void CheckIndices(const std::vector<uint8_t> &input, const std::vector<int64_t> &shape,
                  int64_t count, const std::vector<int64_t> &expected) {
  const auto total = static_cast<int64_t>(input.size());
  ASSERT_EQ(onnx_light_cpu::CountNonZeroBool(input.data(), total), count);
  const auto rank = static_cast<int64_t>(shape.size());
  ASSERT_EQ(expected.size(), static_cast<std::size_t>(rank * count));
  std::vector<int64_t> output(expected.size() + 2, -1);
  onnx_light_cpu::WriteNonZeroBoolIndices(input.data(), total, shape.data(), rank, count,
                                          output.data() + 1);
  EXPECT_EQ(output.front(), -1);
  EXPECT_EQ(output.back(), -1);
  EXPECT_EQ(std::vector<int64_t>(output.begin() + 1, output.end() - 1), expected);
}

TEST(NonZeroKernel, RowMajorCoordinates) {
  CheckIndices({0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1}, {2, 2, 3}, 4,
               {0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 2, 2});
  CheckIndices({0, 1, 0, 1, 1, 0}, {1, 2, 1, 3}, 3, {0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1});
  CheckIndices({1, 0, 2, 0, 255}, {5}, 3, {0, 2, 4});
}

TEST(NonZeroKernel, ChangingCountsAndDenseInput) {
  CheckIndices({0, 0, 0, 0, 0, 0}, {2, 3}, 0, {});
  CheckIndices({0, 0, 0, 0, 0, 1}, {2, 3}, 1, {1, 2});
  CheckIndices({1, 1, 1, 1, 1, 1}, {2, 3}, 6, {0, 0, 0, 1, 1, 1, 0, 1, 2, 0, 1, 2});
  CheckIndices({0, 0, 0, 0, 0, 0}, {2, 3}, 0, {});
}

TEST(NonZeroKernel, ScalarsAndEmptyDimensions) {
  CheckIndices({0}, {}, 0, {});
  CheckIndices({1}, {}, 1, {});
  CheckIndices({}, {0, 3}, 0, {});
  CheckIndices({}, {2, 0, 4}, 0, {});
  CheckIndices({}, {2, 3, 0}, 0, {});
  EXPECT_EQ(onnx_light_cpu::CountNonZeroBool(nullptr, 0), 0);
  onnx_light_cpu::WriteNonZeroBoolIndices(nullptr, 0, nullptr, 3, 0, nullptr);
  onnx_light_cpu::WriteNonZeroBoolIndices(nullptr, 1, nullptr, 0, 1, nullptr);
}

} // namespace
