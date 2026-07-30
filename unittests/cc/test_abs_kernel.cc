// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/cpu_kernels.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// DetectSimdLevel
// ---------------------------------------------------------------------------

TEST(SimdDetection, DetectsAtLeastScalar) {
  auto level = onnx_light_cpu::DetectSimdLevel();
  EXPECT_GE(static_cast<int>(level), 0);
}

// ---------------------------------------------------------------------------
// AbsFloat32
// ---------------------------------------------------------------------------

TEST(AbsFloat32, EmptyInput) {
  float dummy = 1.0f;
  onnx_light_cpu::AbsFloat32(&dummy, &dummy, 0);
}

TEST(AbsFloat32, SingleElement) {
  float in = -3.14f;
  float out = 0.0f;
  onnx_light_cpu::AbsFloat32(&in, &out, 1);
  EXPECT_FLOAT_EQ(out, 3.14f);
}

TEST(AbsFloat32, PositiveValues) {
  std::vector<float> in = {1.0f, 2.5f, 100.0f};
  std::vector<float> out(3, 0.0f);
  onnx_light_cpu::AbsFloat32(in.data(), out.data(), in.size());
  for (std::size_t i = 0; i < in.size(); ++i) {
    EXPECT_FLOAT_EQ(out[i], in[i]);
  }
}

TEST(AbsFloat32, NegativeValues) {
  std::vector<float> in = {-1.0f, -2.5f, -100.0f};
  std::vector<float> out(3, 0.0f);
  onnx_light_cpu::AbsFloat32(in.data(), out.data(), in.size());
  EXPECT_FLOAT_EQ(out[0], 1.0f);
  EXPECT_FLOAT_EQ(out[1], 2.5f);
  EXPECT_FLOAT_EQ(out[2], 100.0f);
}

TEST(AbsFloat32, MixedValues) {
  std::vector<float> in = {-1.0f, 0.0f, 3.0f, -7.5f, 0.001f};
  std::vector<float> out(in.size(), 0.0f);
  onnx_light_cpu::AbsFloat32(in.data(), out.data(), in.size());
  for (std::size_t i = 0; i < in.size(); ++i) {
    EXPECT_FLOAT_EQ(out[i], std::fabs(in[i]));
  }
}

TEST(AbsFloat32, LargeArray) {
  // Test with sizes that exercise SIMD loop and remainder.
  for (std::size_t size : {7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 1000, 1023, 1024, 1025}) {
    std::vector<float> in(size);
    for (std::size_t i = 0; i < size; ++i) {
      in[i] = static_cast<float>(i) - static_cast<float>(size / 2);
    }
    std::vector<float> out(size, -1.0f);
    onnx_light_cpu::AbsFloat32(in.data(), out.data(), size);
    for (std::size_t i = 0; i < size; ++i) {
      EXPECT_FLOAT_EQ(out[i], std::fabs(in[i])) << "at index " << i << " size=" << size;
    }
  }
}

TEST(AbsFloat32, NegativeZero) {
  float in = -0.0f;
  float out = -1.0f;
  onnx_light_cpu::AbsFloat32(&in, &out, 1);
  EXPECT_FLOAT_EQ(out, 0.0f);
  // Ensure the sign bit is cleared.
  EXPECT_FALSE(std::signbit(out));
}

TEST(AbsFloat32, InPlace) {
  std::vector<float> data = {-1.0f, -2.0f, 3.0f, -4.0f};
  onnx_light_cpu::AbsFloat32(data.data(), data.data(), data.size());
  EXPECT_FLOAT_EQ(data[0], 1.0f);
  EXPECT_FLOAT_EQ(data[1], 2.0f);
  EXPECT_FLOAT_EQ(data[2], 3.0f);
  EXPECT_FLOAT_EQ(data[3], 4.0f);
}

// ---------------------------------------------------------------------------
// AbsFloat64
// ---------------------------------------------------------------------------

TEST(AbsFloat64, MixedValues) {
  std::vector<double> in = {-1.0, 0.0, 3.0, -7.5, 0.001};
  std::vector<double> out(in.size(), 0.0);
  onnx_light_cpu::AbsFloat64(in.data(), out.data(), in.size());
  for (std::size_t i = 0; i < in.size(); ++i) {
    EXPECT_DOUBLE_EQ(out[i], std::fabs(in[i]));
  }
}

TEST(AbsFloat64, LargeArray) {
  for (std::size_t size : {3, 4, 5, 7, 8, 9, 100, 255, 256, 257}) {
    std::vector<double> in(size);
    for (std::size_t i = 0; i < size; ++i) {
      in[i] = static_cast<double>(i) - static_cast<double>(size / 2);
    }
    std::vector<double> out(size, -1.0);
    onnx_light_cpu::AbsFloat64(in.data(), out.data(), size);
    for (std::size_t i = 0; i < size; ++i) {
      EXPECT_DOUBLE_EQ(out[i], std::fabs(in[i])) << "at index " << i << " size=" << size;
    }
  }
}

// ---------------------------------------------------------------------------
// AbsInt32
// ---------------------------------------------------------------------------

TEST(AbsInt32, MixedValues) {
  std::vector<int32_t> in = {-1, 0, 3, -7, 100, -2147483647};
  std::vector<int32_t> out(in.size(), 0);
  onnx_light_cpu::AbsInt32(in.data(), out.data(), in.size());
  EXPECT_EQ(out[0], 1);
  EXPECT_EQ(out[1], 0);
  EXPECT_EQ(out[2], 3);
  EXPECT_EQ(out[3], 7);
  EXPECT_EQ(out[4], 100);
  EXPECT_EQ(out[5], 2147483647);
}

TEST(AbsInt32, LargeArray) {
  for (std::size_t size : {7, 8, 9, 15, 16, 17, 100, 255, 256, 257}) {
    std::vector<int32_t> in(size);
    for (std::size_t i = 0; i < size; ++i) {
      in[i] = static_cast<int32_t>(i) - static_cast<int32_t>(size / 2);
    }
    std::vector<int32_t> out(size, -1);
    onnx_light_cpu::AbsInt32(in.data(), out.data(), size);
    for (std::size_t i = 0; i < size; ++i) {
      EXPECT_EQ(out[i], std::abs(in[i])) << "at index " << i << " size=" << size;
    }
  }
}

// ---------------------------------------------------------------------------
// AbsInt64
// ---------------------------------------------------------------------------

TEST(AbsInt64, MixedValues) {
  std::vector<int64_t> in = {-1, 0, 3, -7, 100, -9223372036854775807LL};
  std::vector<int64_t> out(in.size(), 0);
  onnx_light_cpu::AbsInt64(in.data(), out.data(), in.size());
  EXPECT_EQ(out[0], 1);
  EXPECT_EQ(out[1], 0);
  EXPECT_EQ(out[2], 3);
  EXPECT_EQ(out[3], 7);
  EXPECT_EQ(out[4], 100);
  EXPECT_EQ(out[5], 9223372036854775807LL);
}

TEST(AbsInt64, LargeArray) {
  for (std::size_t size : {1, 2, 3, 4, 5, 7, 8, 9, 100, 255, 256, 257}) {
    std::vector<int64_t> in(size);
    for (std::size_t i = 0; i < size; ++i) {
      in[i] = static_cast<int64_t>(i) - static_cast<int64_t>(size / 2);
    }
    std::vector<int64_t> out(size, -1);
    onnx_light_cpu::AbsInt64(in.data(), out.data(), size);
    for (std::size_t i = 0; i < size; ++i) {
      EXPECT_EQ(out[i], std::abs(in[i])) << "at index " << i << " size=" << size;
    }
  }
}

} // namespace
