// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/math/rms_normalization_kernel.h"

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>

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

} // namespace
