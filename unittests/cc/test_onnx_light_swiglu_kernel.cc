// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/math/swiglu_kernel.h"

#include "onnx_core/runtime/memory/simple_tensor.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

TEST(OnnxLightSwiGLUKernel, RejectsBroadcastableUnequalShapes) {
  const rt_ns::OpsetId opset(std::string(), 28);
  const onnx_light_cpu::SwiGLUKernel kernel(rt_ns::KernelContext{opset});
  const rt_ns::Tensor gate = rt_ns::Tensor::FromFloat("", {2, 3}, std::vector<float>(6, 1.0f));
  const rt_ns::Tensor value = rt_ns::Tensor::FromFloat("", {3}, std::vector<float>(3, 1.0f));
  EXPECT_THROW((void)kernel(gate, value), std::invalid_argument);
}

} // namespace
