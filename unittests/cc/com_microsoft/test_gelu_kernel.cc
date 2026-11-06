// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "ml_cpu_compute/com_microsoft/gelu.h"
#include "ml_cpu_compute/math/half_conversion.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

double ReferenceGelu(double x) { return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0))); }

TEST(GeluKernel, PortableTypesAndInPlace) {
  const std::vector<float> input = {-20, -3, -1, -0.0f, 0, 0.5f, 1, 3, 20};
  auto floats = input;
  ml_cpu_compute::GeluFloat32(floats.data(), floats.data(), floats.size());
  std::vector<double> doubles(input.begin(), input.end());
  ml_cpu_compute::GeluFloat64(doubles.data(), doubles.data(), doubles.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    EXPECT_NEAR(floats[i], ReferenceGelu(input[i]), 1e-6);
    EXPECT_NEAR(doubles[i], ReferenceGelu(input[i]), 1e-12);
  }
  EXPECT_TRUE(std::signbit(floats[3]));
  for (bool bfloat16 : {false, true}) {
    const auto encode = bfloat16 ? ml_cpu_compute::detail::FloatToBFloat16Bits
                                 : ml_cpu_compute::detail::FloatToFloat16Bits;
    const auto decode = bfloat16 ? ml_cpu_compute::detail::Bfloat16BitsToFloat
                                 : ml_cpu_compute::detail::Float16BitsToFloat;
    std::vector<std::uint16_t> half;
    for (float x : input) {
      half.push_back(encode(x));
    }
    if (bfloat16) {
      ml_cpu_compute::GeluBFloat16(half.data(), half.data(), half.size());
    } else {
      ml_cpu_compute::GeluFloat16(half.data(), half.data(), half.size());
    }
    for (std::size_t i = 0; i < input.size(); ++i) {
      EXPECT_NEAR(decode(half[i]), ReferenceGelu(input[i]), bfloat16 ? 0.01 : 0.001);
    }
  }
}

TEST(GeluKernel, EmptyAndOverflow) {
  ml_cpu_compute::GeluFloat32(nullptr, nullptr, 0);
  ml_cpu_compute::GeluFloat64(nullptr, nullptr, 0);
  ml_cpu_compute::GeluFloat16(nullptr, nullptr, 0);
  ml_cpu_compute::GeluBFloat16(nullptr, nullptr, 0);
  const auto count = std::numeric_limits<std::size_t>::max();
  EXPECT_THROW(ml_cpu_compute::GeluFloat32(nullptr, nullptr, count), std::invalid_argument);
  EXPECT_THROW(ml_cpu_compute::GeluFloat64(nullptr, nullptr, count), std::invalid_argument);
  EXPECT_THROW(ml_cpu_compute::GeluFloat16(nullptr, nullptr, count), std::invalid_argument);
  EXPECT_THROW(ml_cpu_compute::GeluBFloat16(nullptr, nullptr, count), std::invalid_argument);
}

} // namespace
