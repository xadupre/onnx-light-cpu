// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstring>

namespace onnx_light_cpu::detail {

inline float Float16BitsToFloat(std::uint16_t value) {
  const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000u) << 16;
  std::uint32_t exponent = (value >> 10) & 0x1fu;
  std::uint32_t mantissa = value & 0x3ffu;
  std::uint32_t bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      exponent = 127u - 15u + 1u;
      while ((mantissa & 0x400u) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x3ffu;
      bits = sign | (exponent << 23) | (mantissa << 13);
    }
  } else if (exponent == 0x1f) {
    bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    bits = sign | ((exponent - 15u + 127u) << 23) | (mantissa << 13);
  }
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

inline std::uint16_t FloatToFloat16Bits(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint16_t sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000u);
  const std::uint32_t biased = (bits >> 23) & 0xffu;
  const std::uint32_t mantissa = bits & 0x7fffffu;
  if (biased == 0xff) {
    return static_cast<std::uint16_t>(sign | (mantissa != 0 ? 0x7e00u : 0x7c00u));
  }

  const std::int32_t exponent = static_cast<std::int32_t>(biased) - 127 + 15;
  if (exponent >= 0x1f) {
    return static_cast<std::uint16_t>(sign | 0x7c00u);
  }
  if (exponent <= 0) {
    if (exponent < -10) {
      return sign;
    }
    const std::uint32_t normalized = mantissa | 0x800000u;
    const int shift = 14 - exponent;
    std::uint32_t half_mantissa = normalized >> shift;
    const std::uint32_t remainder = normalized & ((1u << shift) - 1u);
    const std::uint32_t halfway = 1u << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (half_mantissa & 1u) != 0)) {
      ++half_mantissa;
    }
    return static_cast<std::uint16_t>(sign | half_mantissa);
  }

  const auto half_mantissa = static_cast<std::uint16_t>(mantissa >> 13);
  const std::uint32_t remainder = mantissa & 0x1fffu;
  std::uint16_t half = static_cast<std::uint16_t>(
      sign | (static_cast<std::uint32_t>(exponent) << 10) | half_mantissa);
  if (remainder > 0x1000u || (remainder == 0x1000u && (half_mantissa & 1u) != 0)) {
    ++half;
  }
  return half;
}

inline std::uint16_t FloatToBFloat16Bits(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7fffffffu) > 0x7f800000u) {
    return static_cast<std::uint16_t>((bits >> 16) | 0x0040u);
  }
  const std::uint32_t rounding_bias = 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<std::uint16_t>((bits + rounding_bias) >> 16);
}

} // namespace onnx_light_cpu::detail
