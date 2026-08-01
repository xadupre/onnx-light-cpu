// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/onnx_light/exp_log_kernel.h"

#include "onnx_core/runtime/kernel_context.h"
#include "onnx_core/runtime/simple_tensor.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

rt_ns::KernelContext MakeCtx() { return rt_ns::KernelContext(rt_ns::OpsetId(std::string(), 18)); }

TEST(OnnxLightExpKernel, Float32) {
  onnx_light_cpu::ExpKernel kernel(MakeCtx());
  const std::vector<float> values = {-1.0f, 0.0f, 1.0f, 2.0f};
  const rt_ns::Tensor x = rt_ns::Tensor::FromFloat("x", {4}, values);
  const rt_ns::Tensor y = kernel(x);
  ASSERT_EQ(y.element_count(), 4);
  const float *py = y.AsFloat();
  for (std::size_t i = 0; i < values.size(); ++i) {
    EXPECT_NEAR(py[i], std::exp(values[i]), 1e-4f * std::abs(std::exp(values[i])) + 1e-5f);
  }
}

TEST(OnnxLightExpKernel, Double) {
  onnx_light_cpu::ExpKernel kernel(MakeCtx());
  const std::vector<double> values = {-2.0, 0.0, 0.5, 3.0};
  const rt_ns::Tensor x = rt_ns::Tensor::FromDouble("x", {4}, values);
  const rt_ns::Tensor y = kernel(x);
  ASSERT_EQ(y.element_count(), 4);
  const double *py = y.AsDouble();
  for (std::size_t i = 0; i < values.size(); ++i) {
    EXPECT_NEAR(py[i], std::exp(values[i]), 1e-9 * std::abs(std::exp(values[i])) + 1e-12);
  }
}

TEST(OnnxLightLogKernel, Float32) {
  onnx_light_cpu::LogKernel kernel(MakeCtx());
  const std::vector<float> values = {0.5f, 1.0f, 2.0f, 10.0f};
  const rt_ns::Tensor x = rt_ns::Tensor::FromFloat("x", {4}, values);
  const rt_ns::Tensor y = kernel(x);
  ASSERT_EQ(y.element_count(), 4);
  const float *py = y.AsFloat();
  for (std::size_t i = 0; i < values.size(); ++i) {
    EXPECT_NEAR(py[i], std::log(values[i]), 1e-4f * std::abs(std::log(values[i])) + 1e-5f);
  }
}

TEST(OnnxLightLogKernel, Double) {
  onnx_light_cpu::LogKernel kernel(MakeCtx());
  const std::vector<double> values = {0.25, 1.0, 2.0, 100.0};
  const rt_ns::Tensor x = rt_ns::Tensor::FromDouble("x", {4}, values);
  const rt_ns::Tensor y = kernel(x);
  ASSERT_EQ(y.element_count(), 4);
  const double *py = y.AsDouble();
  for (std::size_t i = 0; i < values.size(); ++i) {
    EXPECT_NEAR(py[i], std::log(values[i]), 1e-9 * std::abs(std::log(values[i])) + 1e-12);
  }
}

TEST(OnnxLightExpLogKernel, Roundtrip) {
  onnx_light_cpu::ExpKernel exp_kernel(MakeCtx());
  onnx_light_cpu::LogKernel log_kernel(MakeCtx());
  const std::vector<float> values = {0.0f, 1.0f, 2.0f, 3.0f};
  const rt_ns::Tensor x = rt_ns::Tensor::FromFloat("x", {4}, values);
  const rt_ns::Tensor y = log_kernel(exp_kernel(x));
  const float *py = y.AsFloat();
  for (std::size_t i = 0; i < values.size(); ++i) {
    EXPECT_NEAR(py[i], values[i], 1e-3f);
  }
}

} // namespace
