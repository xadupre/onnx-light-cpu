// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/math_kernels.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

// float16 <-> float32 helpers mirroring the reference implementation, used so
// the tests can express expectations in float and compare the rounded result.
float HalfToFloat(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
  std::uint32_t exp = (h >> 10) & 0x1fu;
  std::uint32_t mant = h & 0x3ffu;
  std::uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign;
    } else {
      exp = 127u - 15u + 1u;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x3ffu;
      f = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 0x1f) {
    f = sign | 0x7f800000u | (mant << 13);
  } else {
    f = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}

std::uint16_t FloatToHalf(float value) {
  std::uint32_t x;
  std::memcpy(&x, &value, sizeof(x));
  const std::uint16_t sign = static_cast<std::uint16_t>((x >> 16) & 0x8000u);
  const std::uint32_t biased = (x >> 23) & 0xffu;
  const std::uint32_t mant = x & 0x7fffffu;
  if (biased == 0xff) {
    return static_cast<std::uint16_t>(sign | (mant ? 0x7e00u : 0x7c00u));
  }
  const std::int32_t exp = static_cast<std::int32_t>(biased) - 127 + 15;
  if (exp >= 0x1f) {
    return static_cast<std::uint16_t>(sign | 0x7c00u);
  }
  if (exp <= 0) {
    if (exp < -10) {
      return sign;
    }
    std::uint32_t m = mant | 0x800000u;
    const int shift = 14 - exp;
    std::uint32_t half_mant = m >> shift;
    const std::uint32_t rem = m & ((1u << shift) - 1u);
    const std::uint32_t halfway = 1u << (shift - 1);
    if (rem > halfway || (rem == halfway && (half_mant & 1u))) {
      ++half_mant;
    }
    return static_cast<std::uint16_t>(sign | half_mant);
  }
  std::uint16_t half_mant = static_cast<std::uint16_t>(mant >> 13);
  const std::uint32_t rem = mant & 0x1fffu;
  std::uint16_t h =
      static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | half_mant);
  if (rem > 0x1000u || (rem == 0x1000u && (half_mant & 1u))) {
    ++h;
  }
  return h;
}

// ---------------------------------------------------------------------------
// ExpFloat32
// ---------------------------------------------------------------------------

TEST(ExpFloat32, EmptyInput) {
  float dummy = 1.0f;
  onnx_light_cpu::ExpFloat32(&dummy, &dummy, 0);
}

TEST(ExpFloat32, MatchesStdOverRange) {
  for (std::size_t size : {1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1000, 1023, 1024, 1025}) {
    std::vector<float> in(size);
    std::vector<float> out(size, -1.0f);
    for (std::size_t i = 0; i < size; ++i) {
      in[i] = -20.0f + 40.0f * (static_cast<float>(i) / static_cast<float>(size));
    }
    onnx_light_cpu::ExpFloat32(in.data(), out.data(), size);
    for (std::size_t i = 0; i < size; ++i) {
      const float ref = std::exp(in[i]);
      EXPECT_NEAR(out[i], ref, std::fabs(ref) * 1e-5f + 1e-7f)
          << "at index " << i << " size=" << size << " x=" << in[i];
    }
  }
}

TEST(ExpFloat32, SpecialValues) {
  const float inf = std::numeric_limits<float>::infinity();
  std::vector<float> in = {0.0f, -0.0f, 1.0f, -1.0f, inf, -inf, std::nanf(""), 200.0f, -200.0f};
  std::vector<float> out(in.size(), -1.0f);
  onnx_light_cpu::ExpFloat32(in.data(), out.data(), in.size());
  EXPECT_FLOAT_EQ(out[0], 1.0f);
  EXPECT_FLOAT_EQ(out[1], 1.0f);
  EXPECT_NEAR(out[2], std::exp(1.0f), 1e-4f);
  EXPECT_NEAR(out[3], std::exp(-1.0f), 1e-6f);
  EXPECT_EQ(out[4], inf);
  EXPECT_FLOAT_EQ(out[5], 0.0f);
  EXPECT_TRUE(std::isnan(out[6]));
  EXPECT_EQ(out[7], inf);
  EXPECT_FLOAT_EQ(out[8], 0.0f);
}

// ---------------------------------------------------------------------------
// LogFloat32
// ---------------------------------------------------------------------------

TEST(LogFloat32, MatchesStdOverRange) {
  for (std::size_t size : {1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1000, 1023, 1024, 1025}) {
    std::vector<float> in(size);
    std::vector<float> out(size, -1.0f);
    for (std::size_t i = 0; i < size; ++i) {
      in[i] = 1e-3f + 1000.0f * (static_cast<float>(i + 1) / static_cast<float>(size));
    }
    onnx_light_cpu::LogFloat32(in.data(), out.data(), size);
    for (std::size_t i = 0; i < size; ++i) {
      const float ref = std::log(in[i]);
      EXPECT_NEAR(out[i], ref, std::fabs(ref) * 1e-5f + 1e-6f)
          << "at index " << i << " size=" << size << " x=" << in[i];
    }
  }
}

TEST(LogFloat32, SpecialValues) {
  const float inf = std::numeric_limits<float>::infinity();
  std::vector<float> in = {1.0f, 0.0f, -0.0f, -1.0f, inf, std::nanf("")};
  std::vector<float> out(in.size(), -1.0f);
  onnx_light_cpu::LogFloat32(in.data(), out.data(), in.size());
  EXPECT_FLOAT_EQ(out[0], 0.0f);
  EXPECT_EQ(out[1], -inf);
  EXPECT_EQ(out[2], -inf);
  EXPECT_TRUE(std::isnan(out[3]));
  EXPECT_EQ(out[4], inf);
  EXPECT_TRUE(std::isnan(out[5]));
}

// ---------------------------------------------------------------------------
// ExpFloat64 / LogFloat64
// ---------------------------------------------------------------------------

TEST(ExpFloat64, MatchesStdOverRange) {
  for (std::size_t size : {1, 2, 3, 4, 5, 7, 8, 9, 100, 255, 256, 257}) {
    std::vector<double> in(size);
    std::vector<double> out(size, -1.0);
    for (std::size_t i = 0; i < size; ++i) {
      in[i] = -30.0 + 60.0 * (static_cast<double>(i) / static_cast<double>(size));
    }
    onnx_light_cpu::ExpFloat64(in.data(), out.data(), size);
    for (std::size_t i = 0; i < size; ++i) {
      const double ref = std::exp(in[i]);
      EXPECT_NEAR(out[i], ref, std::fabs(ref) * 1e-12 + 1e-300)
          << "at index " << i << " size=" << size << " x=" << in[i];
    }
  }
}

TEST(ExpFloat64, SpecialValues) {
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<double> in = {0.0, -0.0, 1.0, -1.0, inf, -inf, std::nan(""), 1000.0, -1000.0};
  std::vector<double> out(in.size(), -1.0);
  onnx_light_cpu::ExpFloat64(in.data(), out.data(), in.size());
  EXPECT_DOUBLE_EQ(out[0], 1.0);
  EXPECT_DOUBLE_EQ(out[1], 1.0);
  EXPECT_NEAR(out[2], std::exp(1.0), 1e-12);
  EXPECT_NEAR(out[3], std::exp(-1.0), 1e-15);
  EXPECT_EQ(out[4], inf);
  EXPECT_DOUBLE_EQ(out[5], 0.0);
  EXPECT_TRUE(std::isnan(out[6]));
  EXPECT_EQ(out[7], inf);
  EXPECT_DOUBLE_EQ(out[8], 0.0);
}

TEST(LogFloat64, MatchesStdOverRange) {
  for (std::size_t size : {1, 2, 3, 4, 5, 7, 8, 9, 100, 255, 256, 257}) {
    std::vector<double> in(size);
    std::vector<double> out(size, -1.0);
    for (std::size_t i = 0; i < size; ++i) {
      in[i] = 1e-6 + 1e6 * (static_cast<double>(i + 1) / static_cast<double>(size));
    }
    onnx_light_cpu::LogFloat64(in.data(), out.data(), size);
    for (std::size_t i = 0; i < size; ++i) {
      const double ref = std::log(in[i]);
      EXPECT_NEAR(out[i], ref, std::fabs(ref) * 1e-12 + 1e-12)
          << "at index " << i << " size=" << size << " x=" << in[i];
    }
  }
}

TEST(LogFloat64, SpecialValues) {
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<double> in = {1.0, 0.0, -0.0, -1.0, inf, std::nan("")};
  std::vector<double> out(in.size(), -1.0);
  onnx_light_cpu::LogFloat64(in.data(), out.data(), in.size());
  EXPECT_DOUBLE_EQ(out[0], 0.0);
  EXPECT_EQ(out[1], -inf);
  EXPECT_EQ(out[2], -inf);
  EXPECT_TRUE(std::isnan(out[3]));
  EXPECT_EQ(out[4], inf);
  EXPECT_TRUE(std::isnan(out[5]));
}

// ---------------------------------------------------------------------------
// ExpFloat16 / LogFloat16
// ---------------------------------------------------------------------------

TEST(ExpFloat16, MatchesReference) {
  std::vector<float> values = {-4.0f, -1.5f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f};
  std::vector<std::uint16_t> in(values.size());
  std::vector<std::uint16_t> out(values.size(), 0);
  for (std::size_t i = 0; i < values.size(); ++i) {
    in[i] = FloatToHalf(values[i]);
  }
  onnx_light_cpu::ExpFloat16(in.data(), out.data(), in.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    const std::uint16_t ref = FloatToHalf(std::exp(HalfToFloat(in[i])));
    EXPECT_EQ(out[i], ref) << "at index " << i;
  }
}

TEST(LogFloat16, MatchesReference) {
  std::vector<float> values = {0.25f, 0.5f, 1.0f, 2.0f, 3.0f, 5.0f, 10.0f, 100.0f, 1000.0f};
  std::vector<std::uint16_t> in(values.size());
  std::vector<std::uint16_t> out(values.size(), 0);
  for (std::size_t i = 0; i < values.size(); ++i) {
    in[i] = FloatToHalf(values[i]);
  }
  onnx_light_cpu::LogFloat16(in.data(), out.data(), in.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    const std::uint16_t ref = FloatToHalf(std::log(HalfToFloat(in[i])));
    EXPECT_EQ(out[i], ref) << "at index " << i;
  }
}

TEST(ExpLogParallel, LargeArrayMatchesStd) {
  // Large enough to cross the ParallelFor cost threshold for the compute-bound
  // Exp/Log kernels and split across blocks; results must match std within the
  // same tolerance used by the single-block tests.
  const std::size_t size = 200000;
  std::vector<float> in(size);
  std::vector<float> out(size, -1.0f);
  for (std::size_t i = 0; i < size; ++i) {
    in[i] = -8.0f + 16.0f * (static_cast<float>(i) / static_cast<float>(size));
  }

  onnx_light_cpu::ExpFloat32(in.data(), out.data(), size);
  for (std::size_t i = 0; i < size; ++i) {
    const float ref = std::exp(in[i]);
    EXPECT_NEAR(out[i], ref, std::fabs(ref) * 1e-5f + 1e-7f) << "exp at index " << i;
  }

  std::vector<float> pos(size);
  for (std::size_t i = 0; i < size; ++i) {
    pos[i] = 1e-3f + 10.0f * (static_cast<float>(i) / static_cast<float>(size));
  }
  onnx_light_cpu::LogFloat32(pos.data(), out.data(), size);
  for (std::size_t i = 0; i < size; ++i) {
    const float ref = std::log(pos[i]);
    EXPECT_NEAR(out[i], ref, std::fabs(ref) * 1e-5f + 1e-6f) << "log at index " << i;
  }
}

} // namespace
