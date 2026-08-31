// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/binary/binary_arithmetic_kernel.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace {

// Binary PR02 -- exercises every FP32/FP64 Add/Sub/Mul/Div SIMD dispatch
// entry point (contiguous and both scalar-broadcast directions) against a
// portable scalar reference computed independently in the test itself.
// ``DispatchFn`` covers every declared shape via lambdas so a single
// parameterized body can drive all three loop families.

template <typename T> struct BinaryArithSpec {
  std::string name;
  std::function<T(T, T)> reference;
  std::function<void(const T *, const T *, T *, std::size_t)> contiguous;
  std::function<void(T, const T *, T *, std::size_t)> left_scalar;
  std::function<void(const T *, T, T *, std::size_t)> right_scalar;
};

template <typename T> std::vector<BinaryArithSpec<T>> AllOps();

// Explicit per-type instantiation avoids depending on template argument
// deduction picking the float overloads above for double.
template <> std::vector<BinaryArithSpec<float>> AllOps<float>() {
  using onnx_light_cpu::BinaryAddFloat32Contiguous;
  using onnx_light_cpu::BinaryAddFloat32LeftScalar;
  using onnx_light_cpu::BinaryAddFloat32RightScalar;
  using onnx_light_cpu::BinaryDivFloat32Contiguous;
  using onnx_light_cpu::BinaryDivFloat32LeftScalar;
  using onnx_light_cpu::BinaryDivFloat32RightScalar;
  using onnx_light_cpu::BinaryMulFloat32Contiguous;
  using onnx_light_cpu::BinaryMulFloat32LeftScalar;
  using onnx_light_cpu::BinaryMulFloat32RightScalar;
  using onnx_light_cpu::BinarySubFloat32Contiguous;
  using onnx_light_cpu::BinarySubFloat32LeftScalar;
  using onnx_light_cpu::BinarySubFloat32RightScalar;
  return {
      {"Add", [](float a, float b) { return a + b; }, BinaryAddFloat32Contiguous,
       BinaryAddFloat32LeftScalar, BinaryAddFloat32RightScalar},
      {"Sub", [](float a, float b) { return a - b; }, BinarySubFloat32Contiguous,
       BinarySubFloat32LeftScalar, BinarySubFloat32RightScalar},
      {"Mul", [](float a, float b) { return a * b; }, BinaryMulFloat32Contiguous,
       BinaryMulFloat32LeftScalar, BinaryMulFloat32RightScalar},
      {"Div", [](float a, float b) { return a / b; }, BinaryDivFloat32Contiguous,
       BinaryDivFloat32LeftScalar, BinaryDivFloat32RightScalar},
  };
}

template <> std::vector<BinaryArithSpec<double>> AllOps<double>() {
  using onnx_light_cpu::BinaryAddFloat64Contiguous;
  using onnx_light_cpu::BinaryAddFloat64LeftScalar;
  using onnx_light_cpu::BinaryAddFloat64RightScalar;
  using onnx_light_cpu::BinaryDivFloat64Contiguous;
  using onnx_light_cpu::BinaryDivFloat64LeftScalar;
  using onnx_light_cpu::BinaryDivFloat64RightScalar;
  using onnx_light_cpu::BinaryMulFloat64Contiguous;
  using onnx_light_cpu::BinaryMulFloat64LeftScalar;
  using onnx_light_cpu::BinaryMulFloat64RightScalar;
  using onnx_light_cpu::BinarySubFloat64Contiguous;
  using onnx_light_cpu::BinarySubFloat64LeftScalar;
  using onnx_light_cpu::BinarySubFloat64RightScalar;
  return {
      {"Add", [](double a, double b) { return a + b; }, BinaryAddFloat64Contiguous,
       BinaryAddFloat64LeftScalar, BinaryAddFloat64RightScalar},
      {"Sub", [](double a, double b) { return a - b; }, BinarySubFloat64Contiguous,
       BinarySubFloat64LeftScalar, BinarySubFloat64RightScalar},
      {"Mul", [](double a, double b) { return a * b; }, BinaryMulFloat64Contiguous,
       BinaryMulFloat64LeftScalar, BinaryMulFloat64RightScalar},
      {"Div", [](double a, double b) { return a / b; }, BinaryDivFloat64Contiguous,
       BinaryDivFloat64LeftScalar, BinaryDivFloat64RightScalar},
  };
}

template <typename T> std::vector<T> SpecialValues() {
  return {
      static_cast<T>(0.0),
      static_cast<T>(-0.0),
      static_cast<T>(1.0),
      static_cast<T>(-1.0),
      static_cast<T>(3.5),
      static_cast<T>(-2.25),
      std::numeric_limits<T>::infinity(),
      -std::numeric_limits<T>::infinity(),
      std::numeric_limits<T>::quiet_NaN(),
      std::numeric_limits<T>::min(),
      std::numeric_limits<T>::max(),
  };
}

// Bit-exact comparison: IEEE-754 add/sub/mul/div are correctly rounded, so a
// SIMD path and the scalar reference must agree bit-for-bit (NaN payload bits
// aside, where only "is a NaN" is required to match).
template <typename T> void ExpectBitwiseOrNaNEqual(T actual, T expected, std::size_t index) {
  if (std::isnan(expected)) {
    EXPECT_TRUE(std::isnan(actual)) << "index=" << index;
    return;
  }
  using Bits = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;
  EXPECT_EQ(std::bit_cast<Bits>(actual), std::bit_cast<Bits>(expected)) << "index=" << index;
}

template <typename T> void RunContiguous(const BinaryArithSpec<T> &op, std::size_t count) {
  std::vector<T> left(count);
  std::vector<T> right(count);
  const std::vector<T> special = SpecialValues<T>();
  for (std::size_t i = 0; i < count; ++i) {
    left[i] = special[i % special.size()];
    right[i] = special[(i + 3) % special.size()];
  }
  std::vector<T> actual(count, static_cast<T>(-123.0));
  op.contiguous(left.data(), right.data(), actual.data(), count);
  for (std::size_t i = 0; i < count; ++i) {
    ExpectBitwiseOrNaNEqual<T>(actual[i], op.reference(left[i], right[i]), i);
  }
}

template <typename T> void RunLeftScalar(const BinaryArithSpec<T> &op, std::size_t count) {
  const std::vector<T> special = SpecialValues<T>();
  for (std::size_t s = 0; s < special.size(); ++s) {
    const T left = special[s];
    std::vector<T> right(count);
    for (std::size_t i = 0; i < count; ++i) {
      right[i] = special[(i + s) % special.size()];
    }
    std::vector<T> actual(count, static_cast<T>(-123.0));
    op.left_scalar(left, right.data(), actual.data(), count);
    for (std::size_t i = 0; i < count; ++i) {
      ExpectBitwiseOrNaNEqual<T>(actual[i], op.reference(left, right[i]), i);
    }
  }
}

template <typename T> void RunRightScalar(const BinaryArithSpec<T> &op, std::size_t count) {
  const std::vector<T> special = SpecialValues<T>();
  for (std::size_t s = 0; s < special.size(); ++s) {
    const T right = special[s];
    std::vector<T> left(count);
    for (std::size_t i = 0; i < count; ++i) {
      left[i] = special[(i + s) % special.size()];
    }
    std::vector<T> actual(count, static_cast<T>(-123.0));
    op.right_scalar(left.data(), right, actual.data(), count);
    for (std::size_t i = 0; i < count; ++i) {
      ExpectBitwiseOrNaNEqual<T>(actual[i], op.reference(left[i], right), i);
    }
  }
}

// SIMD tail sizes: 0 (no-op) and sizes spanning every vector width boundary
// used by SSE2 (4x float/2x double), AVX2 (8x/4x), AVX-512 (16x/8x), NEON
// (4x/2x), and SVE (variable width, generally <= 16x float/8x double on
// current hardware).
constexpr std::array<std::size_t, 18> kTailSizes = {
    0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
};

template <typename T> void RunAllShapesAndSizes() {
  for (const BinaryArithSpec<T> &op : AllOps<T>()) {
    for (std::size_t count : kTailSizes) {
      RunContiguous<T>(op, count);
      RunLeftScalar<T>(op, count);
      RunRightScalar<T>(op, count);
    }
  }
}

TEST(BinaryArithmeticKernel, Float32MatchesScalarReferenceAcrossTailSizes) {
  RunAllShapesAndSizes<float>();
}

TEST(BinaryArithmeticKernel, Float32PReluMatchesScalarReferenceAcrossTailSizes) {
  const auto reference = [](float x, float slope) { return x < 0.0f ? x * slope : x; };
  const std::vector<float> special = SpecialValues<float>();
  for (const std::size_t count : kTailSizes) {
    std::vector<float> left(count);
    std::vector<float> right(count);
    std::vector<float> actual(count);
    for (std::size_t i = 0; i < count; ++i) {
      left[i] = special[i % special.size()];
      right[i] = special[(i + 3) % special.size()];
    }

    onnx_light_cpu::BinaryPReluFloat32Contiguous(left.data(), right.data(), actual.data(), count);
    for (std::size_t i = 0; i < count; ++i) {
      ExpectBitwiseOrNaNEqual(actual[i], reference(left[i], right[i]), i);
    }

    for (const float left_scalar : special) {
      onnx_light_cpu::BinaryPReluFloat32LeftScalar(left_scalar, right.data(), actual.data(), count);
      for (std::size_t i = 0; i < count; ++i) {
        ExpectBitwiseOrNaNEqual(actual[i], reference(left_scalar, right[i]), i);
      }
    }

    const float right_scalar = 3.5f;
    onnx_light_cpu::BinaryPReluFloat32RightScalar(left.data(), right_scalar, actual.data(), count);
    for (std::size_t i = 0; i < count; ++i) {
      ExpectBitwiseOrNaNEqual(actual[i], reference(left[i], right_scalar), i);
    }
  }
}

TEST(BinaryArithmeticKernel, Float64MatchesScalarReferenceAcrossTailSizes) {
  RunAllShapesAndSizes<double>();
}

TEST(BinaryArithmeticKernel, PreservesOperandOrderForNonCommutativeOps) {
  // Sub/Div must not silently swap operands when vectorized.
  std::vector<float> left = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
  std::vector<float> right = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> sub_out(left.size());
  std::vector<float> div_out(left.size());
  onnx_light_cpu::BinarySubFloat32Contiguous(left.data(), right.data(), sub_out.data(),
                                             left.size());
  onnx_light_cpu::BinaryDivFloat32Contiguous(left.data(), right.data(), div_out.data(),
                                             left.size());
  for (std::size_t i = 0; i < left.size(); ++i) {
    EXPECT_FLOAT_EQ(sub_out[i], left[i] - right[i]) << "index=" << i;
    EXPECT_FLOAT_EQ(div_out[i], left[i] / right[i]) << "index=" << i;
  }

  std::vector<float> left_scalar_out(right.size());
  std::vector<float> right_scalar_out(left.size());
  onnx_light_cpu::BinarySubFloat32LeftScalar(100.0f, right.data(), left_scalar_out.data(),
                                             right.size());
  onnx_light_cpu::BinarySubFloat32RightScalar(left.data(), 100.0f, right_scalar_out.data(),
                                              left.size());
  for (std::size_t i = 0; i < right.size(); ++i) {
    EXPECT_FLOAT_EQ(left_scalar_out[i], 100.0f - right[i]) << "index=" << i;
  }
  for (std::size_t i = 0; i < left.size(); ++i) {
    EXPECT_FLOAT_EQ(right_scalar_out[i], left[i] - 100.0f) << "index=" << i;
  }
}

} // namespace
