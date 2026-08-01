// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/logical/logical_kernels.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace {

TEST(NotBool, EmptyInput) {
  std::uint8_t dummy = 1;
  onnx_light_cpu::NotBool(&dummy, &dummy, 0);
  EXPECT_EQ(dummy, 1);
}

TEST(NotBool, SingleElement) {
  std::uint8_t in = 0;
  std::uint8_t out = 5;
  onnx_light_cpu::NotBool(&in, &out, 1);
  EXPECT_EQ(out, 1);
  in = 1;
  onnx_light_cpu::NotBool(&in, &out, 1);
  EXPECT_EQ(out, 0);
}

TEST(NotBool, Basic) {
  std::vector<std::uint8_t> in = {0, 1, 0, 1, 1, 0};
  std::vector<std::uint8_t> out(in.size(), 42);
  onnx_light_cpu::NotBool(in.data(), out.data(), in.size());
  const std::vector<std::uint8_t> expected = {1, 0, 1, 0, 0, 1};
  EXPECT_EQ(out, expected);
}

TEST(NotBool, NonZeroBytesTreatedAsTrue) {
  // Any non-zero byte is logically true, so its negation is 0 (matching
  // numpy.logical_not for non-canonical bool storage).
  std::vector<std::uint8_t> in = {0, 2, 5, 255, 0, 128};
  std::vector<std::uint8_t> out(in.size(), 9);
  onnx_light_cpu::NotBool(in.data(), out.data(), in.size());
  const std::vector<std::uint8_t> expected = {1, 0, 0, 0, 1, 0};
  EXPECT_EQ(out, expected);
}

TEST(NotBool, LargeArray) {
  // Exercise the SIMD loop and remainder handling for various sizes.
  for (std::size_t size : {7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 1000, 1023, 1024, 1025}) {
    std::vector<std::uint8_t> in(size);
    for (std::size_t i = 0; i < size; ++i) {
      in[i] = static_cast<std::uint8_t>(i % 2);
    }
    std::vector<std::uint8_t> out(size, 42);
    onnx_light_cpu::NotBool(in.data(), out.data(), size);
    for (std::size_t i = 0; i < size; ++i) {
      EXPECT_EQ(out[i], static_cast<std::uint8_t>(in[i] == 0 ? 1 : 0))
          << "at index " << i << " size=" << size;
    }
  }
}

} // namespace
