// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/rms_normalization.h"

#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

using onnx_light_cpu::RmsNormalizationBFloat16;
using onnx_light_cpu::RmsNormalizationFloat16;
using onnx_light_cpu::RmsNormalizationFloat32;
using onnx_light_cpu::detail::Bfloat16BitsToFloat;
using onnx_light_cpu::detail::Float16BitsToFloat;
using onnx_light_cpu::detail::FloatToBFloat16Bits;
using onnx_light_cpu::detail::FloatToFloat16Bits;

float ReferenceRmsNormalize(const std::vector<float> &row, const std::vector<float> &scale,
                            std::size_t column, float epsilon) {
  double sum_squares = 0.0;
  for (float value : row) {
    sum_squares += static_cast<double>(value) * value;
  }
  const double inverse_rms =
      1.0 / std::sqrt(sum_squares / static_cast<double>(row.size()) + epsilon);
  return static_cast<float>(row[column] * inverse_rms * scale[column]);
}

TEST(RmsNormalizationKernel, Float32MatchesReferenceAcrossWidths) {
  // Widths crossing the AVX2 8/32-wide unroll boundaries and a scalar tail.
  for (std::size_t width : {std::size_t{1}, std::size_t{7}, std::size_t{8}, std::size_t{31},
                            std::size_t{32}, std::size_t{33}, std::size_t{64}, std::size_t{896}}) {
    std::vector<float> input(width);
    std::vector<float> scale(width);
    for (std::size_t i = 0; i < width; ++i) {
      input[i] = std::sin(static_cast<float>(i) * 0.37f) * 3.0f;
      scale[i] = 1.0f + 0.01f * static_cast<float>(i % 5);
    }
    std::vector<float> output(width);
    RmsNormalizationFloat32(input.data(), scale.data(), output.data(), 1, width, 1e-6f);
    for (std::size_t column = 0; column < width; ++column) {
      EXPECT_NEAR(output[column], ReferenceRmsNormalize(input, scale, column, 1e-6f), 1e-4f)
          << "width=" << width << " column=" << column;
    }
  }
}

TEST(RmsNormalizationKernel, Float16MatchesReferenceAcrossWidths) {
  for (std::size_t width : {std::size_t{1}, std::size_t{7}, std::size_t{8}, std::size_t{31},
                            std::size_t{32}, std::size_t{33}, std::size_t{64}, std::size_t{896}}) {
    std::vector<std::uint16_t> input(width);
    std::vector<std::uint16_t> scale(width);
    std::vector<float> reference_input(width);
    std::vector<float> reference_scale(width);
    for (std::size_t i = 0; i < width; ++i) {
      const float value = std::sin(static_cast<float>(i) * 0.37f) * 3.0f;
      const float weight = 1.0f + 0.01f * static_cast<float>(i % 5);
      input[i] = FloatToFloat16Bits(value);
      scale[i] = FloatToFloat16Bits(weight);
      reference_input[i] = Float16BitsToFloat(input[i]);
      reference_scale[i] = Float16BitsToFloat(scale[i]);
    }

    std::vector<std::uint16_t> output(width);
    RmsNormalizationFloat16(input.data(), scale.data(), output.data(), 1, width, 1e-6f);
    for (std::size_t column = 0; column < width; ++column) {
      const float actual = Float16BitsToFloat(output[column]);
      const float expected = ReferenceRmsNormalize(reference_input, reference_scale, column, 1e-6f);
      EXPECT_NEAR(actual, expected, 5e-2f) << "width=" << width << " column=" << column;
    }
  }
}

TEST(RmsNormalizationKernel, OptionalInverseRmsPreservesOutput) {
  for (std::size_t width : {1U, 7U, 8U, 9U, 31U, 32U, 33U, 127U, 128U, 129U}) {
    constexpr std::size_t rows = 3;
    std::vector<float> input(rows * width);
    std::vector<float> scale(width, 1.5F);
    for (std::size_t row = 0; row < rows; ++row) {
      std::fill_n(input.data() + row * width, width, static_cast<float>(row + 1));
    }
    std::vector<float> expected(input.size()), actual(input.size()), without_inverse(input.size());
    std::vector<float> inverse(rows + 2, -1.0F);
    RmsNormalizationFloat32(input.data(), scale.data(), expected.data(), rows, width, 0.0F);
    RmsNormalizationFloat32(input.data(), scale.data(), actual.data(), rows, width, 0.0F,
                            inverse.data() + 1);
    RmsNormalizationFloat32(input.data(), scale.data(), without_inverse.data(), rows, width, 0.0F,
                            nullptr);
    EXPECT_EQ(actual, expected);
    EXPECT_EQ(without_inverse, expected);
    EXPECT_EQ(inverse.front(), -1.0F);
    EXPECT_EQ(inverse.back(), -1.0F);
    for (std::size_t row = 0; row < rows; ++row) {
      EXPECT_FLOAT_EQ(inverse[row + 1], 1.0F / static_cast<float>(row + 1));
    }
  }
  float inverse = -1.0F;
  RmsNormalizationFloat32(nullptr, nullptr, nullptr, 0, 8, 1.0e-6F, &inverse);
  EXPECT_EQ(inverse, -1.0F);
}

TEST(RmsNormalizationKernel, BFloat16MatchesReferenceAcrossWidths) {
  for (std::size_t width : {std::size_t{1}, std::size_t{7}, std::size_t{8}, std::size_t{31},
                            std::size_t{32}, std::size_t{33}, std::size_t{64}, std::size_t{896}}) {
    std::vector<std::uint16_t> input(width);
    std::vector<std::uint16_t> scale(width);
    std::vector<float> reference_input(width);
    std::vector<float> reference_scale(width);
    for (std::size_t i = 0; i < width; ++i) {
      const float value = std::sin(static_cast<float>(i) * 0.37f) * 3.0f;
      const float weight = 1.0f + 0.01f * static_cast<float>(i % 5);
      input[i] = FloatToBFloat16Bits(value);
      scale[i] = FloatToBFloat16Bits(weight);
      reference_input[i] = Bfloat16BitsToFloat(input[i]);
      reference_scale[i] = Bfloat16BitsToFloat(scale[i]);
    }
    std::vector<std::uint16_t> output(width);
    RmsNormalizationBFloat16(input.data(), scale.data(), output.data(), 1, width, 1e-6f);
    for (std::size_t column = 0; column < width; ++column) {
      const float actual = Bfloat16BitsToFloat(output[column]);
      const float expected = ReferenceRmsNormalize(reference_input, reference_scale, column, 1e-6f);
      EXPECT_NEAR(actual, expected, 2e-1f) << "width=" << width << " column=" << column;
    }
  }
}

TEST(RmsNormalizationKernel, MultipleRowsAreIndependent) {
  constexpr std::size_t rows = 3;
  constexpr std::size_t width = 40;
  std::vector<float> input(rows * width);
  std::vector<float> scale(width, 1.0f);
  for (std::size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>(i % 7) - 3.0f;
  }
  std::vector<float> output(rows * width);
  RmsNormalizationFloat32(input.data(), scale.data(), output.data(), rows, width, 1e-6f);
  for (std::size_t row = 0; row < rows; ++row) {
    const std::vector<float> row_input(input.begin() + static_cast<std::ptrdiff_t>(row * width),
                                       input.begin() +
                                           static_cast<std::ptrdiff_t>((row + 1) * width));
    for (std::size_t column = 0; column < width; ++column) {
      EXPECT_NEAR(output[row * width + column],
                  ReferenceRmsNormalize(row_input, scale, column, 1e-6f), 1e-4f)
          << "row=" << row << " column=" << column;
    }
  }
}

#if defined(ONNX_LIGHT_CPU_HAVE_RMS_F16C)
using onnx_light_cpu::RmsNormalizationFloat16_F16C;

TEST(RmsNormalizationKernel, Float16F16CDirectPathMatchesReferenceAcrossWidths) {
  // Widths crossing the 4-way-unrolled reduction's 8/32-wide boundaries.
  for (std::size_t width : {std::size_t{1}, std::size_t{7}, std::size_t{8}, std::size_t{31},
                            std::size_t{32}, std::size_t{33}, std::size_t{64}, std::size_t{896}}) {
    std::vector<std::uint16_t> input(width);
    std::vector<std::uint16_t> scale(width);
    std::vector<float> reference_input(width);
    std::vector<float> reference_scale(width);
    for (std::size_t i = 0; i < width; ++i) {
      const float value = std::sin(static_cast<float>(i) * 0.37f) * 3.0f;
      const float weight = 1.0f + 0.01f * static_cast<float>(i % 5);
      input[i] = FloatToFloat16Bits(value);
      scale[i] = FloatToFloat16Bits(weight);
      reference_input[i] = Float16BitsToFloat(input[i]);
      reference_scale[i] = Float16BitsToFloat(scale[i]);
    }
    std::vector<std::uint16_t> output(width);
    RmsNormalizationFloat16_F16C(input.data(), scale.data(), output.data(), 0, 1, width, 1e-6f);
    for (std::size_t column = 0; column < width; ++column) {
      const float actual = Float16BitsToFloat(output[column]);
      const float expected = ReferenceRmsNormalize(reference_input, reference_scale, column, 1e-6f);
      EXPECT_NEAR(actual, expected, 5e-2f) << "width=" << width << " column=" << column;
    }
  }
}

TEST(RmsNormalizationKernel, Float16F16CPreservesSpecialValues) {
  constexpr std::size_t width = 16;
  std::vector<std::uint16_t> input(width);
  std::vector<std::uint16_t> scale(width);
  for (std::size_t i = 0; i < width; ++i) {
    input[i] = FloatToFloat16Bits(1.0f);
    scale[i] = FloatToFloat16Bits(1.0f);
  }
  input[3] = FloatToFloat16Bits(std::numeric_limits<float>::infinity());
  std::vector<std::uint16_t> output(width);
  RmsNormalizationFloat16_F16C(input.data(), scale.data(), output.data(), 0, 1, width, 1e-6f);
  // An infinite value drives the mean square (and thus the inverse RMS) to
  // zero; only the infinite lane itself produces 0 * inf = NaN, every other
  // lane normalizes to zero.
  for (std::size_t column = 0; column < width; ++column) {
    const float actual = Float16BitsToFloat(output[column]);
    if (column == 3) {
      EXPECT_TRUE(std::isnan(actual)) << "column=" << column;
    } else {
      EXPECT_EQ(actual, 0.0f) << "column=" << column;
    }
  }
}
#endif

} // namespace
