// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/logical/not_kernel.h"

#include "onnx_core/runtime/kernel_context.h"
#include "onnx_core/runtime/simple_tensor.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

rt_ns::KernelContext MakeCtx() { return rt_ns::KernelContext(rt_ns::OpsetId(std::string(), 18)); }

TEST(OnnxLightNotKernel, Basic) {
  onnx_light_cpu::NotKernel kernel(MakeCtx());
  const std::vector<std::uint8_t> values = {0, 1, 0, 1, 1, 0};
  const rt_ns::Tensor x = rt_ns::Tensor::FromBool("x", {6}, values);
  const rt_ns::Tensor y = kernel(x);
  ASSERT_EQ(y.element_count(), 6);
  const std::uint8_t *py = y.AsBool();
  const std::vector<std::uint8_t> expected = {1, 0, 1, 0, 0, 1};
  for (std::size_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(py[i], expected[i]) << "at index " << i;
  }
}

TEST(OnnxLightNotKernel, Multidimensional) {
  onnx_light_cpu::NotKernel kernel(MakeCtx());
  const std::vector<std::uint8_t> values = {1, 0, 0, 1};
  const rt_ns::Tensor x = rt_ns::Tensor::FromBool("x", {2, 2}, values);
  const rt_ns::Tensor y = kernel(x);
  ASSERT_EQ(y.element_count(), 4);
  const std::uint8_t *py = y.AsBool();
  const std::vector<std::uint8_t> expected = {0, 1, 1, 0};
  for (std::size_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(py[i], expected[i]) << "at index " << i;
  }
}

} // namespace
