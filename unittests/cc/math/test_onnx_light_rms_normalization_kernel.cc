// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/math/rms_normalization_kernel.h"

#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/impl/math/rms_normalization.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

rt_ns::KernelContext MakeContext() {
  return rt_ns::KernelContext(rt_ns::OpsetId(std::string(), 23));
}

float RoundFloat(rt_ns::DataType type, float value) {
  if (type == rt_ns::DataType::FLOAT16) {
    return onnx_light_cpu::detail::Float16BitsToFloat(
        onnx_light_cpu::detail::FloatToFloat16Bits(value));
  }
  if (type == rt_ns::DataType::BFLOAT16) {
    return onnx_light_cpu::detail::Bfloat16BitsToFloat(
        onnx_light_cpu::detail::FloatToBFloat16Bits(value));
  }
  return value;
}

TEST(OnnxLightRmsNormalizationKernel, NormalizesLastAxisAndAppliesScale) {
  const onnx_light_cpu::RmsNormalizationKernel kernel(MakeContext());
  const rt_ns::Tensor x = rt_ns::Tensor::FromFloat("x", {2, 2}, {3.0f, 4.0f, 0.0f, 5.0f});
  const rt_ns::Tensor scale = rt_ns::Tensor::FromFloat("scale", {2}, {2.0f, 0.5f});
  const rt_ns::Tensor y = kernel(x, scale, -1, 0.0f, 1);

  ASSERT_EQ(y.shape, x.shape);
  const float *values = y.AsFloat();
  const float first_inverse_rms = 1.0f / std::sqrt(12.5f);
  const float second_inverse_rms = 1.0f / std::sqrt(12.5f);
  EXPECT_NEAR(values[0], 3.0f * first_inverse_rms * 2.0f, 1.0e-6f);
  EXPECT_NEAR(values[1], 4.0f * first_inverse_rms * 0.5f, 1.0e-6f);
  EXPECT_NEAR(values[2], 0.0f, 1.0e-6f);
  EXPECT_NEAR(values[3], 5.0f * second_inverse_rms * 0.5f, 1.0e-6f);
}

TEST(OnnxLightRmsNormalizationKernel, UsesFloatStashForDoubleInput) {
  const onnx_light_cpu::RmsNormalizationKernel kernel(MakeContext());
  const rt_ns::Tensor x = rt_ns::Tensor::FromDouble("x", {1, 2}, {100000001.0, 100000002.0});
  const rt_ns::Tensor scale = rt_ns::Tensor::FromDouble("scale", {2}, {1.0, 1.0});
  const rt_ns::Tensor y = kernel(x, scale, -1, 0.0F, 1);

  const auto *values = reinterpret_cast<const double *>(y.bytes());
  const float collapsed = static_cast<float>(100000001.0);
  const float expected = collapsed / std::sqrt(collapsed * collapsed);
  EXPECT_DOUBLE_EQ(values[0], static_cast<double>(expected));
  EXPECT_DOUBLE_EQ(values[1], static_cast<double>(expected));
}

TEST(OnnxLightRmsNormalizationKernel, SupportsIndependentInputAndScaleTypes) {
  const onnx_light_cpu::RmsNormalizationKernel kernel(MakeContext());
  const float inverse_rms = 1.0F / std::sqrt(12.5F);

  const rt_ns::Tensor x_float16 = rt_ns::MakeFloat16Tensor("x", {1, 2}, {3.0F, 4.0F});
  const rt_ns::Tensor scale_float32 = rt_ns::Tensor::FromFloat("scale", {2}, {2.0F, 0.5F});
  const rt_ns::Tensor y_float32 = kernel(x_float16, scale_float32, -1, 0.0F, 1);
  ASSERT_EQ(y_float32.data_type, static_cast<std::int32_t>(rt_ns::DataType::FLOAT));
  EXPECT_FLOAT_EQ(y_float32.AsFloat()[0],
                  RoundFloat(rt_ns::DataType::FLOAT16, 3.0F * inverse_rms) * 2.0F);
  EXPECT_FLOAT_EQ(y_float32.AsFloat()[1],
                  RoundFloat(rt_ns::DataType::FLOAT16, 4.0F * inverse_rms) * 0.5F);

  const rt_ns::Tensor x_float32 = rt_ns::Tensor::FromFloat("x", {1, 2}, {3.0F, 4.0F});
  const rt_ns::Tensor scale_float16 = rt_ns::MakeFloat16Tensor("scale", {2}, {2.0F, 0.5F});
  const rt_ns::Tensor y_float16 = kernel(x_float32, scale_float16, -1, 0.0F, 1);
  ASSERT_EQ(y_float16.data_type, static_cast<std::int32_t>(rt_ns::DataType::FLOAT16));
  const auto *values = reinterpret_cast<const std::uint16_t *>(y_float16.bytes());
  const float normalized0 = RoundFloat(rt_ns::DataType::FLOAT16, 3.0F * inverse_rms);
  const float normalized1 = RoundFloat(rt_ns::DataType::FLOAT16, 4.0F * inverse_rms);
  EXPECT_EQ(values[0], onnx_light_cpu::detail::FloatToFloat16Bits(normalized0 * 2.0F));
  EXPECT_EQ(values[1], onnx_light_cpu::detail::FloatToFloat16Bits(normalized1 * 0.5F));
}

TEST(OnnxLightRmsNormalizationKernel, RoundsNormalizedValueBeforeScaling) {
  for (rt_ns::DataType type : {rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16}) {
    const onnx_light_cpu::RmsNormalizationKernel kernel(MakeContext());
    const std::vector<float> inputs = {-3.0F, -2.0F, -1.0F, -0.5F};
    const rt_ns::Tensor x = type == rt_ns::DataType::FLOAT16
                                ? rt_ns::MakeFloat16Tensor("x", {1, 4}, inputs)
                                : rt_ns::MakeBfloat16Tensor("x", {1, 4}, inputs);
    const rt_ns::Tensor scale =
        type == rt_ns::DataType::FLOAT16
            ? rt_ns::MakeFloat16Tensor("scale", {4}, {0.3F, 0.3F, 0.3F, 0.3F})
            : rt_ns::MakeBfloat16Tensor("scale", {4}, {0.3F, 0.3F, 0.3F, 0.3F});
    const rt_ns::Tensor y = kernel(x, scale);

    const float inverse_rms = 1.0F / std::sqrt(3.5625F + 1.0e-5F);
    const float stored_scale = RoundFloat(type, 0.3F);
    const auto *values = reinterpret_cast<const std::uint16_t *>(y.bytes());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      const float normalized = inputs[i] * inverse_rms;
      const float expected = RoundFloat(type, RoundFloat(type, normalized) * stored_scale);
      const float actual = type == rt_ns::DataType::FLOAT16
                               ? onnx_light_cpu::detail::Float16BitsToFloat(values[i])
                               : onnx_light_cpu::detail::Bfloat16BitsToFloat(values[i]);
      EXPECT_FLOAT_EQ(actual, expected);
    }
    const float unrounded = RoundFloat(type, inputs[0] * inverse_rms * stored_scale);
    const float actual = type == rt_ns::DataType::FLOAT16
                             ? onnx_light_cpu::detail::Float16BitsToFloat(values[0])
                             : onnx_light_cpu::detail::Bfloat16BitsToFloat(values[0]);
    EXPECT_NE(actual, unrounded);
  }
}

TEST(OnnxLightRmsNormalizationKernel, RejectsUnsupportedShapeAndStashType) {
  const onnx_light_cpu::RmsNormalizationKernel kernel(MakeContext());
  const rt_ns::Tensor x = rt_ns::Tensor::FromFloat("x", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  const rt_ns::Tensor wrong_scale = rt_ns::Tensor::FromFloat("scale", {3}, {1.0f, 1.0f, 1.0f});
  EXPECT_THROW(kernel(x, wrong_scale), std::invalid_argument);

  const rt_ns::Tensor scale = rt_ns::Tensor::FromFloat("scale", {2}, {1.0f, 1.0f});
  EXPECT_THROW(kernel(x, scale, -1, 1.0e-5f, 0), std::invalid_argument);
}

TEST(OnnxLightRmsNormalizationKernel, Float16FusedPathHandlesVectorTails) {
  constexpr std::size_t kRows = 3;
  constexpr std::size_t kWidth = 13;
  constexpr float kEpsilon = 1.0e-5F;
  std::vector<std::uint16_t> input(kRows * kWidth);
  std::vector<std::uint16_t> scale(kWidth);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] =
        onnx_light_cpu::detail::FloatToFloat16Bits(static_cast<float>(index % 17) * 0.125F - 1.0F);
  }
  for (std::size_t column = 0; column < kWidth; ++column) {
    scale[column] =
        onnx_light_cpu::detail::FloatToFloat16Bits(0.5F + static_cast<float>(column) * 0.03125F);
  }

  std::vector<std::uint16_t> output(input.size());
  onnx_light_cpu::RmsNormalizationFloat16(input.data(), scale.data(), output.data(), kRows, kWidth,
                                          kEpsilon);
  for (std::size_t row = 0; row < kRows; ++row) {
    float sum_squares = 0.0F;
    for (std::size_t column = 0; column < kWidth; ++column) {
      const float value = onnx_light_cpu::detail::Float16BitsToFloat(input[row * kWidth + column]);
      sum_squares += value * value;
    }
    const float inverse_rms = 1.0F / std::sqrt(sum_squares / static_cast<float>(kWidth) + kEpsilon);
    for (std::size_t column = 0; column < kWidth; ++column) {
      const float value = onnx_light_cpu::detail::Float16BitsToFloat(input[row * kWidth + column]);
      const float weight = onnx_light_cpu::detail::Float16BitsToFloat(scale[column]);
      const float normalized = onnx_light_cpu::detail::Float16BitsToFloat(
          onnx_light_cpu::detail::FloatToFloat16Bits(value * inverse_rms));
      const std::uint16_t expected =
          onnx_light_cpu::detail::FloatToFloat16Bits(normalized * weight);
      EXPECT_EQ(output[row * kWidth + column], expected) << "row=" << row << ", column=" << column;
    }
  }
}

TEST(OnnxLightRmsNormalizationKernel, BFloat16BulkPathMatchesFloatReference) {
  const onnx_light_cpu::RmsNormalizationKernel kernel(MakeContext());
  const rt_ns::Tensor x = rt_ns::MakeBfloat16Tensor("x", {1, 1, 3}, {1.0F, -2.0F, 3.0F});
  const rt_ns::Tensor scale = rt_ns::MakeBfloat16Tensor("scale", {3}, {0.5F, 1.0F, 1.5F});
  const rt_ns::Tensor y = kernel(x, scale, -1, 1.0e-5F, 1);

  const float inverse_rms = 1.0F / std::sqrt(14.0F / 3.0F + 1.0e-5F);
  const auto *values = reinterpret_cast<const std::uint16_t *>(y.bytes());
  for (std::size_t index = 0; index < 3; ++index) {
    const float input = onnx_light_cpu::detail::Bfloat16BitsToFloat(
        reinterpret_cast<const std::uint16_t *>(x.bytes())[index]);
    const float weight = onnx_light_cpu::detail::Bfloat16BitsToFloat(
        reinterpret_cast<const std::uint16_t *>(scale.bytes())[index]);
    const float normalized = onnx_light_cpu::detail::Bfloat16BitsToFloat(
        onnx_light_cpu::detail::FloatToBFloat16Bits(input * inverse_rms));
    EXPECT_EQ(values[index], onnx_light_cpu::detail::FloatToBFloat16Bits(normalized * weight));
  }
}

TEST(OnnxLightRmsNormalizationKernel, BroadcastsScaleAcrossNormalizedDimensions) {
  const onnx_light_cpu::RmsNormalizationKernel kernel(MakeContext());
  const rt_ns::Tensor x = rt_ns::Tensor::FromFloat("x", {2, 2}, {3.0F, 4.0F, 0.0F, 5.0F});
  const rt_ns::Tensor scale = rt_ns::Tensor::FromFloat("scale", {1}, {2.0F});
  const rt_ns::Tensor y = kernel(x, scale, -1, 0.0F, 1);

  EXPECT_EQ(y.data_type, static_cast<std::int32_t>(rt_ns::DataType::FLOAT));
  EXPECT_NEAR(y.AsFloat()[0], 3.0 / std::sqrt(12.5) * 2.0, 1.0e-6);
  EXPECT_NEAR(y.AsFloat()[1], 4.0 / std::sqrt(12.5) * 2.0, 1.0e-6);
  EXPECT_NEAR(y.AsFloat()[2], 0.0, 1.0e-6);
  EXPECT_NEAR(y.AsFloat()[3], 5.0 / std::sqrt(12.5) * 2.0, 1.0e-6);
}

TEST(OnnxLightRmsNormalizationKernel, BroadcastsRowDependentScaleAgainstFullInput) {
  const onnx_light_cpu::RmsNormalizationKernel kernel(MakeContext());
  const rt_ns::Tensor x =
      rt_ns::Tensor::FromFloat("x", {2, 3}, {3.0F, 4.0F, 0.0F, 0.0F, 0.0F, 5.0F});
  const rt_ns::Tensor scale = rt_ns::Tensor::FromFloat("scale", {2, 1}, {2.0F, 0.5F});
  const rt_ns::Tensor y = kernel(x, scale, -1, 0.0F, 1);

  const float first_inverse_rms = 1.0F / std::sqrt(25.0F / 3.0F);
  const float second_inverse_rms = 1.0F / std::sqrt(25.0F / 3.0F);
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 3.0F * first_inverse_rms * 2.0F);
  EXPECT_FLOAT_EQ(y.AsFloat()[1], 4.0F * first_inverse_rms * 2.0F);
  EXPECT_FLOAT_EQ(y.AsFloat()[2], 0.0F);
  EXPECT_FLOAT_EQ(y.AsFloat()[3], 0.0F);
  EXPECT_FLOAT_EQ(y.AsFloat()[4], 0.0F);
  EXPECT_FLOAT_EQ(y.AsFloat()[5], 5.0F * second_inverse_rms * 0.5F);

  const rt_ns::Tensor full_scale =
      rt_ns::Tensor::FromFloat("scale", {2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const rt_ns::Tensor full_y = kernel(x, full_scale, -1, 0.0F, 1);
  EXPECT_FLOAT_EQ(full_y.AsFloat()[0], 3.0F * first_inverse_rms);
  EXPECT_FLOAT_EQ(full_y.AsFloat()[1], 4.0F * first_inverse_rms * 2.0F);
  EXPECT_FLOAT_EQ(full_y.AsFloat()[5], 5.0F * second_inverse_rms * 6.0F);
}

} // namespace
