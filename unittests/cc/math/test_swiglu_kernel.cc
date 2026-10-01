// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/impl/math/math_kernels.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

float Reference(float gate, float value, float alpha) {
  return gate * value / (1.0f + std::exp(-alpha * gate));
}

TEST(SwiGLU, Float32UsesAlphaAndExpPrimitiveSemantics) {
  const std::array<float, 7> gate = {-3.0f, -1.0f, -0.0f, 0.5f, 1.0f, 2.0f, 4.0f};
  const std::array<float, 7> value = {0.5f, -2.0f, 3.0f, 1.0f, -1.0f, 2.0f, 0.25f};
  std::array<float, 7> output{};
  onnx_light_cpu::SwiGLUFloat32(gate.data(), value.data(), output.data(), output.size(), 0.5f);
  for (std::size_t i = 0; i < output.size(); ++i) {
    EXPECT_NEAR(output[i], Reference(gate[i], value[i], 0.5f), 2e-6f) << i;
  }
}

TEST(SwiGLU, Float16AndBFloat16MatchConvertedReferenceAcrossVectorTail) {
  constexpr std::size_t count = 263;
  std::vector<float> gate(count);
  std::vector<float> value(count);
  for (std::size_t i = 0; i < count; ++i) {
    gate[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 4.0f;
    value[i] = static_cast<float>(static_cast<int>(i % 11) - 5) / 3.0f;
  }

  for (bool bfloat16 : {false, true}) {
    std::vector<std::uint16_t> gate_bits(count);
    std::vector<std::uint16_t> value_bits(count);
    std::vector<std::uint16_t> output(count);
    if (bfloat16) {
      onnx_light_cpu::detail::ConvertFloat32ToBFloat16(gate.data(), gate_bits.data(), count);
      onnx_light_cpu::detail::ConvertFloat32ToBFloat16(value.data(), value_bits.data(), count);
      onnx_light_cpu::SwiGLUBFloat16(gate_bits.data(), value_bits.data(), output.data(), count,
                                     1.0f);
    } else {
      onnx_light_cpu::detail::ConvertFloat32ToFloat16(gate.data(), gate_bits.data(), count);
      onnx_light_cpu::detail::ConvertFloat32ToFloat16(value.data(), value_bits.data(), count);
      onnx_light_cpu::SwiGLUFloat16(gate_bits.data(), value_bits.data(), output.data(), count,
                                    1.0f);
    }
    for (std::size_t i = 0; i < count; ++i) {
      const float actual = bfloat16 ? onnx_light_cpu::detail::Bfloat16BitsToFloat(output[i])
                                    : onnx_light_cpu::detail::Float16BitsToFloat(output[i]);
      const float rounded_gate = bfloat16
                                     ? onnx_light_cpu::detail::Bfloat16BitsToFloat(gate_bits[i])
                                     : onnx_light_cpu::detail::Float16BitsToFloat(gate_bits[i]);
      const float rounded_value = bfloat16
                                      ? onnx_light_cpu::detail::Bfloat16BitsToFloat(value_bits[i])
                                      : onnx_light_cpu::detail::Float16BitsToFloat(value_bits[i]);
      const float tolerance = bfloat16 ? 2e-2f : 3e-3f;
      EXPECT_NEAR(actual, Reference(rounded_gate, rounded_value, 1.0f), tolerance) << i;
    }
  }
}

TEST(SwiGLU, BFloat16TailsAliasingAndSpecialValues) {
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const std::array<float, 12> gates = {-inf, -8.0f, -1.0f, -0.0f, 0.0f,      0.25f,
                                       1.0f, 8.0f,  inf,   nan,   0x1p-126f, -0x1p-126f};
  const std::array<float, 9> values = {-inf, -2.0f, -0.0f, 0.0f, 0.5f, 2.0f, inf, nan, 0x1p-126f};
  const auto encode = [](float value) {
    return onnx_light_cpu::detail::FloatToBFloat16Bits(value);
  };
  const auto decode = [](std::uint16_t value) {
    return onnx_light_cpu::detail::Bfloat16BitsToFloat(value);
  };
  const auto check = [&](std::uint16_t actual_bits, std::uint16_t gate_bits,
                         std::uint16_t value_bits, float alpha) {
    const float expected = Reference(decode(gate_bits), decode(value_bits), alpha);
    const float actual = decode(actual_bits);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(actual));
    } else if (std::isinf(expected)) {
      EXPECT_EQ(actual, expected);
    } else {
      EXPECT_NEAR(
          actual, expected,
          std::max(std::abs(expected) * 2e-2f, onnx_light_cpu::detail::Bfloat16BitsToFloat(1)));
    }
  };

  for (const std::size_t count : {0u, 1u, 15u, 16u, 17u, 31u, 32u, 33u, 255u, 256u, 257u}) {
    for (const float alpha : {0.5f, 1.0f}) {
      SCOPED_TRACE(::testing::Message() << "count=" << count << " alpha=" << alpha);
      std::vector<std::uint16_t> gate(count + 2, 42);
      std::vector<std::uint16_t> value(count + 2, 42);
      std::vector<std::uint16_t> output(count + 2, 42);
      for (std::size_t index = 0; index < count; ++index) {
        gate[index + 1] = encode(gates[index % gates.size()]);
        value[index + 1] = encode(values[index % values.size()]);
      }
      onnx_light_cpu::SwiGLUBFloat16(gate.data() + 1, value.data() + 1, output.data() + 1, count,
                                     alpha);
      for (std::size_t index = 0; index < count; ++index) {
        check(output[index + 1], gate[index + 1], value[index + 1], alpha);
      }
      auto aliased_gate = gate;
      onnx_light_cpu::SwiGLUBFloat16(aliased_gate.data() + 1, value.data() + 1,
                                     aliased_gate.data() + 1, count, alpha);
      auto aliased_value = value;
      onnx_light_cpu::SwiGLUBFloat16(gate.data() + 1, aliased_value.data() + 1,
                                     aliased_value.data() + 1, count, alpha);
      for (std::size_t index = 0; index < count; ++index) {
        check(aliased_gate[index + 1], gate[index + 1], value[index + 1], alpha);
        check(aliased_value[index + 1], gate[index + 1], value[index + 1], alpha);
      }
      for (const auto *buffer : {&gate, &value, &output, &aliased_gate, &aliased_value}) {
        EXPECT_EQ(buffer->front(), 42);
        EXPECT_EQ(buffer->back(), 42);
      }
    }
  }
}

} // namespace
