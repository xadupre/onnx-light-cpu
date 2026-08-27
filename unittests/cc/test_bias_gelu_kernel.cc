// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/com_microsoft/bias_gelu.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

using onnx_light_cpu::BiasGeluBFloat16;
using onnx_light_cpu::BiasGeluExecutionTuning;
using onnx_light_cpu::BiasGeluFloat16;
using onnx_light_cpu::BiasGeluFloat32;
using onnx_light_cpu::BiasGeluFloat32WithTuning;
using onnx_light_cpu::BiasGeluFloat64;

// Exact erf-based reference GELU, independent of the kernel's own
// ``GeluExact``.
double ReferenceGelu(double z) { return 0.5 * z * (1.0 + std::erf(z / std::sqrt(2.0))); }

TEST(BiasGelu, Float32MatchesExactErfReference) {
  constexpr std::size_t outer = 2;
  constexpr std::size_t inner = 4;
  const std::vector<float> a = {-3.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f, -0.5f, 4.0f};
  const std::vector<float> bias = {0.1f, -0.2f, 0.3f, -0.4f};
  std::vector<float> output(outer * inner);
  BiasGeluFloat32(a.data(), bias.data(), output.data(), outer, inner);
  for (std::size_t row = 0; row < outer; ++row) {
    for (std::size_t col = 0; col < inner; ++col) {
      const double z = static_cast<double>(a[row * inner + col]) + bias[col];
      EXPECT_NEAR(output[row * inner + col], ReferenceGelu(z), 1e-5) << row << "," << col;
    }
  }
}

TEST(BiasGelu, Float64MatchesExactErfReference) {
  constexpr std::size_t outer = 3;
  constexpr std::size_t inner = 3;
  std::vector<double> a(outer * inner);
  std::vector<double> bias = {0.5, -0.5, 0.25};
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<double>(static_cast<int>(i % 11) - 5) * 0.3;
  }
  std::vector<double> output(outer * inner);
  BiasGeluFloat64(a.data(), bias.data(), output.data(), outer, inner);
  for (std::size_t row = 0; row < outer; ++row) {
    for (std::size_t col = 0; col < inner; ++col) {
      const double z = a[row * inner + col] + bias[col];
      EXPECT_NEAR(output[row * inner + col], ReferenceGelu(z), 1e-9) << row << "," << col;
    }
  }
}

TEST(BiasGelu, Float16MatchesConvertedReferenceWithinHalfTolerance) {
  constexpr std::size_t outer = 2;
  constexpr std::size_t inner = 5;
  std::vector<float> a_values(outer * inner);
  std::vector<float> bias_values(inner);
  for (std::size_t i = 0; i < a_values.size(); ++i) {
    a_values[i] = static_cast<float>(static_cast<int>(i % 9) - 4) * 0.5f;
  }
  for (std::size_t i = 0; i < bias_values.size(); ++i) {
    bias_values[i] = static_cast<float>(static_cast<int>(i % 5) - 2) * 0.25f;
  }
  std::vector<std::uint16_t> a_bits(a_values.size());
  std::vector<std::uint16_t> bias_bits(bias_values.size());
  onnx_light_cpu::detail::ConvertFloat32ToFloat16(a_values.data(), a_bits.data(), a_values.size());
  onnx_light_cpu::detail::ConvertFloat32ToFloat16(bias_values.data(), bias_bits.data(),
                                                  bias_values.size());
  std::vector<std::uint16_t> output(outer * inner);
  BiasGeluFloat16(a_bits.data(), bias_bits.data(), output.data(), outer, inner);
  for (std::size_t row = 0; row < outer; ++row) {
    for (std::size_t col = 0; col < inner; ++col) {
      const std::size_t index = row * inner + col;
      const float rounded_a = onnx_light_cpu::detail::Float16BitsToFloat(a_bits[index]);
      const float rounded_bias = onnx_light_cpu::detail::Float16BitsToFloat(bias_bits[col]);
      const float actual = onnx_light_cpu::detail::Float16BitsToFloat(output[index]);
      EXPECT_NEAR(actual, ReferenceGelu(rounded_a + rounded_bias), 5e-3) << row << "," << col;
    }
  }
}

TEST(BiasGelu, BFloat16MatchesConvertedReferenceWithinBFloat16Tolerance) {
  constexpr std::size_t outer = 2;
  constexpr std::size_t inner = 5;
  std::vector<float> a_values(outer * inner);
  std::vector<float> bias_values(inner);
  for (std::size_t i = 0; i < a_values.size(); ++i) {
    a_values[i] = static_cast<float>(static_cast<int>(i % 9) - 4) * 0.5f;
  }
  for (std::size_t i = 0; i < bias_values.size(); ++i) {
    bias_values[i] = static_cast<float>(static_cast<int>(i % 5) - 2) * 0.25f;
  }
  std::vector<std::uint16_t> a_bits(a_values.size());
  std::vector<std::uint16_t> bias_bits(bias_values.size());
  onnx_light_cpu::detail::ConvertFloat32ToBFloat16(a_values.data(), a_bits.data(), a_values.size());
  onnx_light_cpu::detail::ConvertFloat32ToBFloat16(bias_values.data(), bias_bits.data(),
                                                   bias_values.size());
  std::vector<std::uint16_t> output(outer * inner);
  BiasGeluBFloat16(a_bits.data(), bias_bits.data(), output.data(), outer, inner);
  for (std::size_t row = 0; row < outer; ++row) {
    for (std::size_t col = 0; col < inner; ++col) {
      const std::size_t index = row * inner + col;
      const float rounded_a = onnx_light_cpu::detail::Bfloat16BitsToFloat(a_bits[index]);
      const float rounded_bias = onnx_light_cpu::detail::Bfloat16BitsToFloat(bias_bits[col]);
      const float actual = onnx_light_cpu::detail::Bfloat16BitsToFloat(output[index]);
      EXPECT_NEAR(actual, ReferenceGelu(rounded_a + rounded_bias), 5e-2) << row << "," << col;
    }
  }
}

TEST(BiasGelu, ZeroOuterOrInnerProducesNoWrites) {
  std::vector<float> sentinel(1, -42.0f);
  BiasGeluFloat32(nullptr, nullptr, sentinel.data(), 0, 4);
  EXPECT_EQ(sentinel[0], -42.0f);
  BiasGeluFloat32(nullptr, nullptr, sentinel.data(), 4, 0);
  EXPECT_EQ(sentinel[0], -42.0f);
}

TEST(BiasGelu, TunedDispatchWithParallelThresholdMatchesUntuned) {
  constexpr std::size_t outer = 64;
  constexpr std::size_t inner = 16;
  std::vector<float> a(outer * inner);
  std::vector<float> bias(inner);
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<float>(static_cast<int>(i % 23) - 11) * 0.125f;
  }
  for (std::size_t i = 0; i < bias.size(); ++i) {
    bias[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.25f;
  }
  std::vector<float> serial(outer * inner);
  std::vector<float> parallel(outer * inner);
  BiasGeluFloat32(a.data(), bias.data(), serial.data(), outer, inner);

  BiasGeluExecutionTuning tuning;
  tuning.parallel_threshold_bytes = 1;
  tuning.target_block_bytes = 1;
  tuning.use_cost_model = false;
  BiasGeluFloat32WithTuning(a.data(), bias.data(), parallel.data(), outer, inner, tuning);
  for (std::size_t i = 0; i < serial.size(); ++i) {
    EXPECT_NEAR(serial[i], parallel[i], 1e-5f) << i;
  }
}

} // namespace
