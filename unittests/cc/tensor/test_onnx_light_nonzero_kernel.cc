// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/tensor/nonzero_kernel.h"

#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include <gtest/gtest.h>

#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
using rt_ns::DataType;
using rt_ns::Shape;
using rt_ns::Tensor;

TEST(OnnxLightNonZeroKernel, MatchesBuiltinAcrossShapesAndCounts) {
  const rt_ns::KernelContext ctx{rt_ns::DefaultOpset(13)};
  const onnx_light_cpu::NonZeroKernel kernel{ctx};
  const ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::NonZero reference{ctx};
  for (const Shape &shape : {Shape{}, Shape{7}, Shape{2, 5}, Shape{2, 3, 4}, Shape{1, 2, 1, 3},
                             Shape{0}, Shape{0, 3}, Shape{2, 0, 4}, Shape{2, 3, 0}}) {
    for (int pattern : {0, 1, 2, 0}) {
      std::vector<uint8_t> values(static_cast<std::size_t>(shape.product()));
      for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = pattern == 1 || (pattern == 2 && i % 3 == 1);
      }
      const Tensor input = Tensor::FromBool("X", shape, values);
      const Tensor expected = reference(input);
      const Tensor actual = kernel(input);
      ASSERT_EQ(actual.data_type, DataType::INT64);
      ASSERT_EQ(actual.shape, expected.shape);
      ASSERT_EQ(actual.size_bytes(), expected.size_bytes());
      if (actual.size_bytes() != 0) {
        EXPECT_EQ(std::memcmp(actual.bytes(), expected.bytes(), actual.size_bytes()), 0);
      }
    }
  }
}

TEST(OnnxLightNonZeroKernel, RowMajorCoordinatesAndExactAllocation) {
  const rt_ns::KernelContext ctx{rt_ns::DefaultOpset(13)};
  const onnx_light_cpu::NonZeroKernel kernel{ctx};
  const Tensor input = Tensor::FromBool("X", {2, 2, 3}, {0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1});
  const Tensor actual = kernel(input);
  ASSERT_EQ(actual.shape, (Shape{3, 4}));
  ASSERT_EQ(actual.size_bytes(), 12 * sizeof(int64_t));
  const std::vector<int64_t> expected = {0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 2, 2};
  EXPECT_EQ(std::memcmp(actual.bytes(), expected.data(), actual.size_bytes()), 0);
  for (uint8_t value : {0, 1}) {
    const Tensor scalar = kernel(Tensor::FromBool("X", {}, {value}));
    EXPECT_EQ(scalar.shape, (Shape{0, value}));
    EXPECT_EQ(scalar.size_bytes(), 0u);
  }
}

TEST(OnnxLightNonZeroKernel, RejectsMalformedInputsBeforeReading) {
  const rt_ns::KernelContext ctx{rt_ns::DefaultOpset(13)};
  const onnx_light_cpu::NonZeroKernel kernel{ctx};
  const Tensor input = Tensor::FromBool("X", {2}, {0, 1});
  Tensor bad = input;
  bad.data_type = DataType::UINT8;
  EXPECT_THROW(kernel(bad), std::invalid_argument);
  for (const Shape &shape : {Shape{3}, Shape{1}, Shape{-1}, Shape{0, -1},
                             Shape{std::numeric_limits<int64_t>::max(), 2}}) {
    bad = input;
    bad.shape = shape;
    EXPECT_THROW(kernel(bad), std::invalid_argument);
  }
}

} // namespace
