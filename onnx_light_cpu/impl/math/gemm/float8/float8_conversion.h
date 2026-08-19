// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Exact scalar decoders for the four ONNX Float8 formats (Roadmap PR09.5). Each
// format is treated as a separate packing format: the GEMM engine decodes the
// one-byte pattern to ``float`` while packing, rather than branching inside the
// FP32 inner loop. The decoders below reproduce the reference ``ml_dtypes`` /
// ONNX conversion bit for bit for all 256 byte patterns of every format
// (including the format-specific NaN encodings), so a decode lookup table built
// from them is exact.

#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>

namespace onnx_light_cpu::detail {

/// The four Float8 storage formats defined by ONNX. ``FN`` variants are finite
/// (no infinities); ``FNUZ`` variants additionally have a single (unsigned)
/// zero and encode NaN only as the ``0x80`` sign-bit-only pattern.
enum class Float8Format {
  kE4M3FN,   ///< 1-4-3, bias 7, finite, NaN at 0x7f / 0xff.
  kE4M3FNUZ, ///< 1-4-3, bias 8, finite, single zero, NaN at 0x80.
  kE5M2,     ///< 1-5-2, bias 15, IEEE-like with infinities and NaNs.
  kE5M2FNUZ, ///< 1-5-2, bias 16, finite, single zero, NaN at 0x80.
};

namespace float8_internal {

inline float BitsToFloat(std::uint32_t bits) { return std::bit_cast<float>(bits); }

// Shared decode for the two ``FN``/``FNUZ`` non-IEEE formats parameterized by
// the exponent width. ``exponent_bits`` is 4 (E4M3) or 5 (E5M2); the mantissa
// occupies the remaining ``7 - exponent_bits`` bits. ``bias`` is the exponent
// bias and ``nan_is_sign_only`` selects the FNUZ NaN encoding (only ``0x80``)
// versus the E4M3FN encoding (exponent and mantissa all ones).
inline float DecodeFinite(std::uint8_t value, unsigned exponent_bits, unsigned bias,
                          bool nan_is_sign_only) {
  const unsigned mantissa_bits = 7u - exponent_bits;
  const std::uint32_t sign = static_cast<std::uint32_t>(value >> 7) & 0x1u;
  const std::uint32_t exponent =
      (static_cast<std::uint32_t>(value) >> mantissa_bits) & ((1u << exponent_bits) - 1u);
  const std::uint32_t mantissa = static_cast<std::uint32_t>(value) & ((1u << mantissa_bits) - 1u);
  const std::uint32_t float_sign = sign << 31;

  if (nan_is_sign_only) {
    // FNUZ: 0x80 (sign only) is the sole NaN; every other pattern is finite.
    if (exponent == 0u && mantissa == 0u) {
      return sign ? BitsToFloat(0xffc00000u) : 0.0f;
    }
  } else {
    // E4M3FN: exponent and mantissa all ones encodes NaN (0x7f / 0xff).
    if (exponent == ((1u << exponent_bits) - 1u) && mantissa == ((1u << mantissa_bits) - 1u)) {
      return BitsToFloat(float_sign | 0x7fc00000u);
    }
  }

  if (exponent == 0u) {
    if (mantissa == 0u) {
      return BitsToFloat(float_sign);
    }
    // Subnormal: value = (-1)^s * 2^(1 - bias) * (mantissa / 2^mantissa_bits).
    std::int32_t exp = 1 - static_cast<std::int32_t>(bias);
    std::uint32_t m = mantissa;
    while ((m & (1u << mantissa_bits)) == 0u) {
      m <<= 1;
      --exp;
    }
    m &= (1u << mantissa_bits) - 1u;
    const std::uint32_t float_exp = static_cast<std::uint32_t>(exp + 127);
    return BitsToFloat(float_sign | (float_exp << 23) | (m << (23u - mantissa_bits)));
  }

  const std::uint32_t float_exp = static_cast<std::uint32_t>(static_cast<std::int32_t>(exponent) -
                                                             static_cast<std::int32_t>(bias) + 127);
  return BitsToFloat(float_sign | (float_exp << 23) | (mantissa << (23u - mantissa_bits)));
}

} // namespace float8_internal

inline float Float8E4M3FNBitsToFloat(std::uint8_t value) {
  return float8_internal::DecodeFinite(value, 4u, 7u, /*nan_is_sign_only=*/false);
}

inline float Float8E4M3FNUZBitsToFloat(std::uint8_t value) {
  return float8_internal::DecodeFinite(value, 4u, 8u, /*nan_is_sign_only=*/true);
}

inline float Float8E5M2BitsToFloat(std::uint8_t value) {
  // 1-5-2 IEEE-like format: exponent all ones encodes infinity (mantissa 0) or
  // NaN, matching the standard half-precision layout scaled to two mantissa
  // bits.
  const std::uint32_t sign = static_cast<std::uint32_t>(value >> 7) & 0x1u;
  const std::uint32_t exponent = (static_cast<std::uint32_t>(value) >> 2) & 0x1fu;
  const std::uint32_t mantissa = static_cast<std::uint32_t>(value) & 0x3u;
  const std::uint32_t float_sign = sign << 31;
  if (exponent == 0x1fu) {
    return mantissa == 0u ? float8_internal::BitsToFloat(float_sign | 0x7f800000u)
                          : float8_internal::BitsToFloat(float_sign | 0x7fc00000u);
  }
  if (exponent == 0u) {
    if (mantissa == 0u) {
      return float8_internal::BitsToFloat(float_sign);
    }
    std::int32_t exp = 1 - 15;
    std::uint32_t m = mantissa;
    while ((m & 0x4u) == 0u) {
      m <<= 1;
      --exp;
    }
    m &= 0x3u;
    const std::uint32_t float_exp = static_cast<std::uint32_t>(exp + 127);
    return float8_internal::BitsToFloat(float_sign | (float_exp << 23) | (m << 21));
  }
  const std::uint32_t float_exp = exponent - 15u + 127u;
  return float8_internal::BitsToFloat(float_sign | (float_exp << 23) | (mantissa << 21));
}

inline float Float8E5M2FNUZBitsToFloat(std::uint8_t value) {
  return float8_internal::DecodeFinite(value, 5u, 16u, /*nan_is_sign_only=*/true);
}

/// Decodes one Float8 byte of ``format`` to ``float``.
inline float Float8BitsToFloat(Float8Format format, std::uint8_t value) {
  switch (format) {
  case Float8Format::kE4M3FN:
    return Float8E4M3FNBitsToFloat(value);
  case Float8Format::kE4M3FNUZ:
    return Float8E4M3FNUZBitsToFloat(value);
  case Float8Format::kE5M2:
    return Float8E5M2BitsToFloat(value);
  case Float8Format::kE5M2FNUZ:
    return Float8E5M2FNUZBitsToFloat(value);
  }
  return 0.0f;
}

/// Builds the exact 256-entry decode lookup table for ``format``. The GEMM
/// packing path gathers from this table (see ``PackConvertContiguous``), which
/// keeps the per-format decode out of the FP32 inner loop.
inline std::array<float, 256> BuildFloat8DecodeTable(Float8Format format) {
  std::array<float, 256> table{};
  for (int i = 0; i < 256; ++i) {
    table[static_cast<std::size_t>(i)] = Float8BitsToFloat(format, static_cast<std::uint8_t>(i));
  }
  return table;
}

} // namespace onnx_light_cpu::detail
