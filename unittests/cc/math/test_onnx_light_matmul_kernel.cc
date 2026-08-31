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

TEST(MatMulKernel, RuntimeCacheReusesShapesAndInvalidatesOnShapeChange) {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("MatMul");
  node.add_input("A");
  node.add_input("B");
  node.add_output("Y");
  rt_ns::RuntimeContext runtime(MakeCtx());
  runtime.Set("A", rt_ns::Tensor::From<float>("A", {2, 2}, {1, 2, 3, 4}));
  runtime.Set("B", rt_ns::Tensor::From<float>("B", {2, 2}, {1, 0, 0, 1}));

  onnx_light_cpu::MatMulKernel kernel(MakeCtx());
  kernel.set_node(node);
  kernel.Run(runtime);
  EXPECT_FLOAT_EQ(runtime.Get("Y").AsFloat()[3], 4.0f);

  rt_ns::RuntimeContext same_shape_runtime(MakeCtx());
  same_shape_runtime.Set("A", rt_ns::Tensor::From<float>("A", {2, 2}, {5, 6, 7, 8}));
  same_shape_runtime.Set("B", rt_ns::Tensor::From<float>("B", {2, 2}, {1, 0, 0, 1}));
  kernel.Run(same_shape_runtime);
  EXPECT_FLOAT_EQ(same_shape_runtime.Get("Y").AsFloat()[0], 5.0f);
  EXPECT_FLOAT_EQ(same_shape_runtime.Get("Y").AsFloat()[3], 8.0f);

  rt_ns::RuntimeContext changed_shape_runtime(MakeCtx());
  changed_shape_runtime.Set("A", rt_ns::Tensor::From<float>("A", {1, 2}, {2, 3}));
  changed_shape_runtime.Set("B", rt_ns::Tensor::From<float>("B", {2, 1}, {4, 5}));
  kernel.Run(changed_shape_runtime);
  EXPECT_EQ(changed_shape_runtime.Get("Y").shape, (rt_ns::Shape{1, 1}));
  EXPECT_FLOAT_EQ(changed_shape_runtime.Get("Y").AsFloat()[0], 23.0f);
}

} // namespace
