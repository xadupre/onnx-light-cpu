#include "onnx_light_cpu/kernels/math/matmul_kernel.h"

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"

#include <gtest/gtest.h>

#include <string>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

rt_ns::KernelContext MakeCtx() { return rt_ns::KernelContext(rt_ns::OpsetId(std::string(), 18)); }

TEST(MatMulKernel, BroadcastsBatches) {
  onnx_light_cpu::MatMulKernel kernel(MakeCtx());
  const auto a = rt_ns::Tensor::From<float>("a", {2, 1, 2}, {1, 2, 3, 4});
  const auto b = rt_ns::Tensor::From<float>("b", {1, 2, 2}, {5, 6, 7, 8});

  const auto y = kernel(a, b);

  EXPECT_EQ(y.shape, (rt_ns::Shape{2, 1, 2}));
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 19.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[1], 22.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[2], 43.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[3], 50.0f);
}

TEST(MatMulKernel, SupportsVectorDot) {
  onnx_light_cpu::MatMulKernel kernel(MakeCtx());
  const auto vector_a = rt_ns::Tensor::From<float>("va", {2}, {2, 3});
  const auto vector_b = rt_ns::Tensor::From<float>("vb", {2}, {4, 5});
  const auto dot = kernel(vector_a, vector_b);
  EXPECT_TRUE(dot.shape.empty());
  EXPECT_FLOAT_EQ(dot.AsFloat()[0], 23.0f);
}

} // namespace
