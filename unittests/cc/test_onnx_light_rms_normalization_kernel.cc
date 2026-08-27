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

TEST(OnnxLightRmsNormalizationKernel, RejectsUnsupportedShapeAndStashType) {
  const onnx_light_cpu::RmsNormalizationKernel kernel(MakeContext());
  const rt_ns::Tensor x = rt_ns::Tensor::FromFloat("x", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  const rt_ns::Tensor wrong_scale = rt_ns::Tensor::FromFloat("scale", {1}, {1.0f});
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
      const float expected = value * inverse_rms * weight;
      const float actual =
          onnx_light_cpu::detail::Float16BitsToFloat(output[row * kWidth + column]);
      EXPECT_NEAR(actual, expected, 1.0e-3F) << "row=" << row << ", column=" << column;
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
  EXPECT_NEAR(onnx_light_cpu::detail::Bfloat16BitsToFloat(values[0]), 0.5F * inverse_rms, 1.0e-2F);
  EXPECT_NEAR(onnx_light_cpu::detail::Bfloat16BitsToFloat(values[1]), -2.0F * inverse_rms, 1.0e-2F);
  EXPECT_NEAR(onnx_light_cpu::detail::Bfloat16BitsToFloat(values[2]), 4.5F * inverse_rms, 1.0e-2F);
}

} // namespace
