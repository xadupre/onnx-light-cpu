// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/impl/math/normalization_kernel.h"
#include "onnx_light_cpu/impl/simd_level.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#endif

namespace {

using namespace onnx_light_cpu;
using detail::Float16BitsToFloat;
using detail::FloatToFloat16Bits;

TEST(NormalizationPrimitives, DispatchLabelsMatchRequiredCpuFeatures) {
  const char *half_path = NormalizationFloat16Path();
  if (std::strcmp(half_path, "f16c") == 0) {
    EXPECT_TRUE(CpuSupportsF16C());
    EXPECT_GE(DetectSimdLevel(), SimdLevel::kAVX);
  } else {
    EXPECT_STREQ(half_path, "scalar");
  }
  const char *double_path = NormalizationFloat64Path();
  if (std::strcmp(double_path, "avx") == 0) {
    EXPECT_GE(DetectSimdLevel(), SimdLevel::kAVX);
  } else {
    EXPECT_STREQ(double_path, "scalar");
  }
}

TEST(NormalizationPrimitives, EmptyReductionsRejectAndEmptyAffineDoesNotDereference) {
  EXPECT_THROW(ComputeNormalizationMeanSquareFloat16(nullptr, 0), std::invalid_argument);
  EXPECT_THROW(ComputeNormalizationMeanSquareFloat64(nullptr, 0), std::invalid_argument);
  EXPECT_THROW(ComputeNormalizationMeanSquareFloat64StashFloat32(nullptr, 0),
               std::invalid_argument);
  ApplyRmsNormalizationFloat32(nullptr, nullptr, nullptr, 0, 1.0F);
  ApplyNormalizationAffineFloat16(nullptr, nullptr, nullptr, 0, 1.0F);
  ApplyNormalizationAffineFloat64(nullptr, nullptr, nullptr, 0, 1.0);
  ApplyNormalizationAffineFloat64StashFloat32(nullptr, nullptr, nullptr, 0, 1.0F);
}

TEST(NormalizationPrimitives, Float32RmsAffineHandlesEveryTailAndUnalignedInput) {
  constexpr float multiplier = 0.731234F;
  for (std::size_t count = 1; count <= 33; ++count) {
    SCOPED_TRACE(count);
    std::vector<float> input(count + 1), scale(count + 1);
    std::vector<float> output(count + 2, -999.0F);
    for (std::size_t i = 0; i < count; ++i) {
      input[i + 1] = (static_cast<float>(i % 11) - 5.0F) * 0.25F;
      scale[i + 1] = 0.5F + static_cast<float>(i % 5) * 0.25F;
    }
    ApplyRmsNormalizationFloat32(input.data() + 1, scale.data() + 1, output.data() + 1, count,
                                 multiplier);
    for (std::size_t i = 0; i < count; ++i) {
      EXPECT_FLOAT_EQ(output[i + 1], input[i + 1] * multiplier * scale[i + 1]);
    }
    EXPECT_EQ(output.front(), -999.0F);
    EXPECT_EQ(output.back(), -999.0F);
    ApplyRmsNormalizationFloat32(input.data() + 1, scale.data() + 1, input.data() + 1, count,
                                 multiplier);
    for (std::size_t i = 0; i < count; ++i) {
      EXPECT_EQ(input[i + 1], output[i + 1]);
    }
  }
}

TEST(NormalizationPrimitives, HalfEveryTailAndUnalignedInPlaceAffine) {
  for (std::size_t count = 1; count <= 33; ++count) {
    SCOPED_TRACE(count);
    std::vector<std::uint16_t> input(count + 1), scale(count + 1);
    std::vector<std::uint16_t> output(count + 2, 0x1234);
    float squares = 0.0F;
    for (std::size_t i = 0; i < count; ++i) {
      const float value = (static_cast<float>(i % 11) - 5.0F) * 0.25F;
      input[i + 1] = FloatToFloat16Bits(value);
      scale[i + 1] = FloatToFloat16Bits(0.5F + static_cast<float>(i % 5) * 0.25F);
      squares += value * value;
    }
    EXPECT_FLOAT_EQ(ComputeNormalizationMeanSquareFloat16(input.data() + 1, count),
                    squares / static_cast<float>(count));
    ApplyNormalizationAffineFloat16(input.data() + 1, scale.data() + 1, output.data() + 1, count,
                                    0.731234F);
    for (std::size_t i = 0; i < count; ++i) {
      const float value = Float16BitsToFloat(input[i + 1]) * 0.731234F;
      EXPECT_EQ(output[i + 1], FloatToFloat16Bits(value * Float16BitsToFloat(scale[i + 1])));
    }
    EXPECT_EQ(output.front(), 0x1234);
    EXPECT_EQ(output.back(), 0x1234);
    ApplyNormalizationAffineFloat16(input.data() + 1, scale.data() + 1, input.data() + 1, count,
                                    0.731234F);
    for (std::size_t i = 0; i < count; ++i) {
      EXPECT_EQ(input[i + 1], output[i + 1]);
    }
  }
}

TEST(NormalizationPrimitives, HalfRoundsOnlyAfterScale) {
  for (std::size_t count = 1; count <= 33; ++count) {
    const auto value = FloatToFloat16Bits(0.3F), weight = FloatToFloat16Bits(0.7F);
    std::vector<std::uint16_t> input(count, value), scale(count, weight), output(count);
    const float normalized = Float16BitsToFloat(value) * 0.731234F;
    const auto expected = FloatToFloat16Bits(normalized * Float16BitsToFloat(weight));
    const auto intermediate_rounded = FloatToFloat16Bits(
        Float16BitsToFloat(FloatToFloat16Bits(normalized)) * Float16BitsToFloat(weight));
    ASSERT_NE(expected, intermediate_rounded);
    ApplyNormalizationAffineFloat16(input.data(), scale.data(), output.data(), count, 0.731234F);
    for (const auto actual : output) {
      EXPECT_EQ(actual, expected) << count;
    }
  }
}

TEST(NormalizationPrimitives, HalfMomentsAndNormalizationFastPathsHandleTails) {
  for (std::size_t count : {31, 32, 33, 63, 64, 65}) {
    SCOPED_TRACE(count);
    std::vector<std::uint16_t> input(count), scale(count), bias(count), output(count);
    float sums[4] = {};
    float square_sums[4] = {};
    for (std::size_t i = 0; i < count; ++i) {
      const float value = (static_cast<float>(i % 17) - 8.0F) * 0.125F;
      input[i] = FloatToFloat16Bits(value);
      scale[i] = FloatToFloat16Bits(0.5F + static_cast<float>(i % 5) * 0.125F);
      bias[i] = FloatToFloat16Bits((static_cast<float>(i % 3) - 1.0F) * 0.25F);
      sums[i & 3] += value;
      square_sums[i & 3] += value * value;
    }
    const float mean = (sums[0] + sums[1] + sums[2] + sums[3]) / static_cast<float>(count);
    const float second_moment =
        (square_sums[0] + square_sums[1] + square_sums[2] + square_sums[3]) /
        static_cast<float>(count);
    const auto moments = ComputeNormalizationMomentsFloat16(input.data(), count);
    EXPECT_NEAR(moments.mean, mean, 1.0e-6F);
    EXPECT_NEAR(moments.variance, second_moment - mean * mean, 1.0e-6F);

    ApplyNormalizationScaleBiasFloat16(input.data(), output.data(), count, 0.75F, -0.125F);
    for (std::size_t i = 0; i < count; ++i) {
      const float value = Float16BitsToFloat(input[i]) * 0.75F - 0.125F;
      EXPECT_EQ(output[i], FloatToFloat16Bits(value));
    }

    ApplyLayerNormalizationFloat16(input.data(), scale.data(), bias.data(), output.data(), count,
                                   mean, 0.75F);
    for (std::size_t i = 0; i < count; ++i) {
      const float normalized =
          Float16BitsToFloat(FloatToFloat16Bits((Float16BitsToFloat(input[i]) - mean) * 0.75F));
      const float scaled =
          Float16BitsToFloat(FloatToFloat16Bits(normalized * Float16BitsToFloat(scale[i])));
      EXPECT_EQ(output[i], FloatToFloat16Bits(scaled + Float16BitsToFloat(bias[i])));
    }

    ApplyGroupNormalizationFloat16(input.data(), output.data(), count, mean, 0.75F, 0.625F, -0.25F);
    for (std::size_t i = 0; i < count; ++i) {
      const float normalized =
          Float16BitsToFloat(FloatToFloat16Bits((Float16BitsToFloat(input[i]) - mean) * 0.75F));
      const float scaled = Float16BitsToFloat(FloatToFloat16Bits(normalized * 0.625F));
      EXPECT_EQ(output[i], FloatToFloat16Bits(scaled - 0.25F));
    }

    std::vector<std::uint16_t> lp_input(2 * count), lp_output(2 * count);
    for (std::size_t i = 0; i < lp_input.size(); ++i) {
      lp_input[i] = input[i % count];
    }
    LpNormalizationFloat16(lp_input.data(), lp_output.data(), 2, count);
    double square_sum = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
      const double value = Float16BitsToFloat(input[i]);
      square_sum += value * value;
    }
    const float inverse = static_cast<float>(1.0 / std::sqrt(square_sum));
    for (std::size_t i = 0; i < lp_output.size(); ++i) {
      EXPECT_EQ(lp_output[i], FloatToFloat16Bits(Float16BitsToFloat(lp_input[i]) * inverse));
    }
  }
}

TEST(NormalizationPrimitives, HalfBatchedNormalizationMatchesPerSlicePrimitives) {
  constexpr std::size_t kChannels = 3;
  constexpr std::size_t kSlices = 5;
  constexpr std::size_t kWidth = 33;
  std::vector<std::uint16_t> input(kSlices * kWidth), scale(kWidth), bias(kWidth);
  for (std::size_t i = 0; i < input.size(); ++i) {
    input[i] = FloatToFloat16Bits((static_cast<float>(i % 29) - 14.0F) * 0.0625F);
  }
  for (std::size_t i = 0; i < kWidth; ++i) {
    scale[i] = FloatToFloat16Bits(0.5F + static_cast<float>(i % 7) * 0.125F);
    bias[i] = FloatToFloat16Bits((static_cast<float>(i % 5) - 2.0F) * 0.0625F);
  }

  std::vector<std::uint16_t> expected(kSlices * kWidth, 0x1234);
  std::vector<std::uint16_t> actual = expected;
  for (std::size_t slice = 1; slice < 4; ++slice) {
    const auto moments = ComputeNormalizationMomentsFloat16(input.data() + slice * kWidth, kWidth);
    const std::size_t channel = slice % kChannels;
    const float multiplier =
        Float16BitsToFloat(scale[channel]) / std::sqrt(moments.variance + 1.0e-5F);
    const float offset = Float16BitsToFloat(bias[channel]) - moments.mean * multiplier;
    ApplyNormalizationScaleBiasFloat16(input.data() + slice * kWidth,
                                       expected.data() + slice * kWidth, kWidth, multiplier,
                                       offset);
  }
  InstanceNormalizationFloat16(input.data(), scale.data(), bias.data(), actual.data(), 1, 4,
                               kChannels, kWidth, 1.0e-5F);
  EXPECT_EQ(actual, expected);

  std::fill(expected.begin(), expected.end(), 0x1234);
  actual = expected;
  std::vector<float> expected_mean(kSlices, -999.0F), expected_inv(kSlices, -999.0F);
  std::vector<float> actual_mean = expected_mean, actual_inv = expected_inv;
  for (std::size_t row = 1; row < 4; ++row) {
    const auto moments = ComputeNormalizationMomentsFloat16(input.data() + row * kWidth, kWidth);
    const float inverse = 1.0F / std::sqrt(moments.variance + 1.0e-5F);
    expected_mean[row] = moments.mean;
    expected_inv[row] = inverse;
    ApplyLayerNormalizationFloat16(input.data() + row * kWidth, scale.data(), bias.data(),
                                   expected.data() + row * kWidth, kWidth, moments.mean, inverse);
  }
  LayerNormalizationFloat16Rows(input.data(), scale.data(), bias.data(), actual.data(),
                                actual_mean.data(), actual_inv.data(), 1, 4, kWidth, 1.0e-5F);
  EXPECT_EQ(actual, expected);
  EXPECT_EQ(actual_mean, expected_mean);
  EXPECT_EQ(actual_inv, expected_inv);

  std::fill(expected.begin(), expected.end(), 0x1234);
  actual = expected;
  for (std::size_t channel = 0; channel < kChannels; ++channel) {
    ApplyGroupNormalizationFloat16(
        input.data() + channel * kWidth, expected.data() + channel * kWidth, kWidth, 0.125F, 0.75F,
        Float16BitsToFloat(scale[channel]), Float16BitsToFloat(bias[channel]));
  }
  ApplyGroupNormalizationFloat16Channels(input.data(), scale.data(), bias.data(), actual.data(),
                                         kChannels, kWidth, 0.125F, 0.75F);
  EXPECT_EQ(actual, expected);
}

TEST(NormalizationPrimitives, HalfSubnormalsAndNanCanonicalization) {
  for (std::uint16_t bits : {0x0001, 0x03ff, 0x8001, 0x83ff, 0x7e55, 0xfe55}) {
    for (std::size_t count = 1; count <= 33; ++count) {
      std::vector<std::uint16_t> input(count, bits), scale(count, 0x3c00), output(count);
      ApplyNormalizationAffineFloat16(input.data(), scale.data(), output.data(), count, 1.0F);
      const auto expected = FloatToFloat16Bits(Float16BitsToFloat(bits));
      for (const auto actual : output) {
        EXPECT_EQ(actual, expected) << "bits=" << bits << " count=" << count;
      }
    }
  }
}

#if defined(__x86_64__) || defined(_M_X64)
TEST(NormalizationPrimitives, HalfConversionDoesNotRaiseUnmaskedExceptions) {
  for (std::size_t count = 1; count <= 33; ++count) {
    std::vector<std::uint16_t> input(count, 0x7bff), scale(count, 0x4000), output(count);
    const unsigned int original = _mm_getcsr();
    const unsigned int unmasked = original & ~(_MM_MASK_MASK | _MM_EXCEPT_MASK);
    _mm_setcsr(unmasked);
    ApplyNormalizationAffineFloat16(input.data(), scale.data(), output.data(), count, 1.0F);
    const unsigned int after = _mm_getcsr();
    _mm_setcsr(original);
    EXPECT_EQ(after, unmasked);
    for (const auto actual : output) {
      EXPECT_EQ(actual, 0x7c00);
    }
  }
}
#endif

TEST(NormalizationPrimitives, DoubleEveryTailAndUnalignedInPlaceAffine) {
  for (std::size_t count = 1; count <= 33; ++count) {
    SCOPED_TRACE(count);
    std::vector<double> input(count + 1), scale(count + 1), output(count + 2, -999.0);
    double squares = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
      input[i + 1] = (static_cast<double>(i % 11) - 5.0) * 0.25;
      scale[i + 1] = 0.5 + static_cast<double>(i % 5) * 0.25;
      squares += input[i + 1] * input[i + 1];
    }
    EXPECT_DOUBLE_EQ(ComputeNormalizationMeanSquareFloat64(input.data() + 1, count),
                     squares / static_cast<double>(count));
    EXPECT_FLOAT_EQ(ComputeNormalizationMeanSquareFloat64StashFloat32(input.data() + 1, count),
                    static_cast<float>(squares) / static_cast<float>(count));
    ApplyNormalizationAffineFloat64(input.data() + 1, scale.data() + 1, output.data() + 1, count,
                                    0.731234567890123);
    for (std::size_t i = 0; i < count; ++i) {
      EXPECT_DOUBLE_EQ(output[i + 1], input[i + 1] * 0.731234567890123 * scale[i + 1]);
    }
    EXPECT_EQ(output.front(), -999.0);
    EXPECT_EQ(output.back(), -999.0);
    ApplyNormalizationAffineFloat64(input.data() + 1, scale.data() + 1, input.data() + 1, count,
                                    0.731234567890123);
    for (std::size_t i = 0; i < count; ++i) {
      EXPECT_EQ(input[i + 1], output[i + 1]);
    }
    ApplyNormalizationAffineFloat64StashFloat32(input.data() + 1, scale.data() + 1,
                                                output.data() + 1, count, 0.731234F);
    for (std::size_t i = 0; i < count; ++i) {
      const float normalized = static_cast<float>(input[i + 1]) * 0.731234F;
      EXPECT_EQ(output[i + 1], static_cast<double>(normalized * static_cast<float>(scale[i + 1])));
    }
    ApplyNormalizationAffineFloat64StashFloat32(input.data() + 1, scale.data() + 1,
                                                input.data() + 1, count, 0.731234F);
    for (std::size_t i = 0; i < count; ++i) {
      EXPECT_EQ(input[i + 1], output[i + 1]);
    }
  }
}

TEST(NormalizationPrimitives, FloatStashCastsInputsBeforeSquareAndScaleBeforeMultiply) {
  for (std::size_t count = 1; count <= 33; ++count) {
    SCOPED_TRACE(count);
    std::vector<double> input(count, 1.0e30), scale(count, 1.0), output(count);
    EXPECT_TRUE(std::isfinite(ComputeNormalizationMeanSquareFloat64(input.data(), count)));
    EXPECT_TRUE(std::isinf(ComputeNormalizationMeanSquareFloat64StashFloat32(input.data(), count)));
    std::fill(input.begin(), input.end(), 1.0e40);
    ApplyNormalizationAffineFloat64StashFloat32(input.data(), scale.data(), output.data(), count,
                                                1.0e-20F);
    for (const auto actual : output) {
      EXPECT_TRUE(std::isinf(actual));
    }
    std::fill(input.begin(), input.end(), 1.0);
    std::fill(scale.begin(), scale.end(), 1.0e40);
    ApplyNormalizationAffineFloat64StashFloat32(input.data(), scale.data(), output.data(), count,
                                                1.0e-20F);
    for (const auto actual : output) {
      EXPECT_TRUE(std::isinf(actual));
    }
    std::fill(input.begin(), input.end(), 1.0e30);
    std::fill(scale.begin(), scale.end(), 1.0e-30);
    ApplyNormalizationAffineFloat64StashFloat32(input.data(), scale.data(), output.data(), count,
                                                1.0e10F);
    for (const auto actual : output) {
      EXPECT_TRUE(std::isinf(actual));
    }
    ApplyNormalizationAffineFloat64(input.data(), scale.data(), output.data(), count, 1.0e10);
    for (const auto actual : output) {
      EXPECT_NEAR(actual, 1.0e10, 1.0e-5);
    }
    std::fill(input.begin(), input.end(), 1.0 + 0x1p-30);
    EXPECT_EQ(ComputeNormalizationMeanSquareFloat64StashFloat32(input.data(), count), 1.0F);
    EXPECT_GT(ComputeNormalizationMeanSquareFloat64(input.data(), count), 1.0);
  }
}

TEST(NormalizationPrimitives, IeeeValuesAcrossVectorAndScalarTails) {
  for (std::size_t count = 1; count <= 33; ++count) {
    SCOPED_TRACE(count);
    for (double special :
         {std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
          std::numeric_limits<double>::quiet_NaN()}) {
      std::vector<double> input(count, 1.0), scale(count, 1.0), output(count);
      input.back() = special;
      const double squares = ComputeNormalizationMeanSquareFloat64(input.data(), count);
      const float float_squares =
          ComputeNormalizationMeanSquareFloat64StashFloat32(input.data(), count);
      EXPECT_EQ(std::isnan(squares), std::isnan(special));
      EXPECT_EQ(std::isnan(float_squares), std::isnan(special));
      EXPECT_EQ(std::isinf(squares), std::isinf(special));
      EXPECT_EQ(std::isinf(float_squares), std::isinf(special));
      ApplyNormalizationAffineFloat64(input.data(), scale.data(), output.data(), count, 0.0);
      EXPECT_TRUE(std::isnan(output.back()));
      ApplyNormalizationAffineFloat64StashFloat32(input.data(), scale.data(), output.data(), count,
                                                  0.0F);
      EXPECT_TRUE(std::isnan(output.back()));
      std::vector<std::uint16_t> half_input(count, FloatToFloat16Bits(1.0F));
      std::vector<std::uint16_t> half_scale(count, FloatToFloat16Bits(1.0F)), half_output(count);
      half_input.back() = FloatToFloat16Bits(static_cast<float>(special));
      const float half_squares = ComputeNormalizationMeanSquareFloat16(half_input.data(), count);
      EXPECT_EQ(std::isnan(half_squares), std::isnan(special));
      EXPECT_EQ(std::isinf(half_squares), std::isinf(special));
      ApplyNormalizationAffineFloat16(half_input.data(), half_scale.data(), half_output.data(),
                                      count, 0.0F);
      EXPECT_TRUE(std::isnan(Float16BitsToFloat(half_output.back())));
    }
    std::vector<double> zero(count, -0.0), scale(count, 1.0), output(count);
    ApplyNormalizationAffineFloat64(zero.data(), scale.data(), output.data(), count, 1.0);
    for (const auto actual : output) {
      EXPECT_TRUE(std::signbit(actual));
    }
    ApplyNormalizationAffineFloat64StashFloat32(zero.data(), scale.data(), output.data(), count,
                                                1.0F);
    for (const auto actual : output) {
      EXPECT_TRUE(std::signbit(actual));
    }
    std::vector<std::uint16_t> half_zero(count, 0x8000), half_scale(count, 0x3c00),
        half_output(count);
    ApplyNormalizationAffineFloat16(half_zero.data(), half_scale.data(), half_output.data(), count,
                                    1.0F);
    EXPECT_EQ(half_output, half_zero);
  }
}

} // namespace
