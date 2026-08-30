// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/register_kernels.h"

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_light_cpu/kernels/com_microsoft/naive_bias_gelu_kernel.h"

#include <gtest/gtest.h>

#include <vector>

namespace {

// ``RegisterAllKernels`` installs every onnx-light-cpu kernel class into
// onnx-light's shared ``KernelDispatchTable``. The call must succeed and be
// safe to invoke more than once (re-registration overrides the existing
// entries with the same factories).
TEST(OnnxLightRegisterKernels, RegisterAllKernels) {
  EXPECT_NO_THROW(onnx_light_cpu::RegisterAllKernels());
  EXPECT_NO_THROW(onnx_light_cpu::RegisterAllKernels());
}

TEST(OnnxLightRegisterKernels, NaivePolicyInstallsNaiveMicrosoftFactories) {
  namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
  onnx_light_cpu::RegisterAllKernels(onnx_light_cpu::MicrosoftKernelImplementation::NAIVE);
  const auto &table = rt_ns::KernelDispatchTable();
  const auto factory = table.find("com.microsoft:BiasGelu");
  ASSERT_NE(factory, table.end());

  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_domain("com.microsoft");
  node.set_op_type("BiasGelu");
  node.add_input("a");
  node.add_input("b");
  node.add_output("c");
  rt_ns::RuntimeContext runtime(rt_ns::KernelContext(rt_ns::OpsetId("com.microsoft", 1)));
  runtime.Set("a", rt_ns::Tensor::FromFloat("a", {1, 2}, {-1.0f, 2.0f}));
  runtime.Set("b", rt_ns::Tensor::FromFloat("b", {2}, {0.25f, -0.5f}));

  std::unique_ptr<rt_ns::KernelBase> kernel = factory->second(node, runtime);
  EXPECT_NE(dynamic_cast<onnx_light_cpu::NaiveBiasGeluKernel *>(kernel.get()), nullptr);

  onnx_light_cpu::RegisterAllKernels();
}

TEST(OnnxLightRegisterKernels, RegisteredFactoriesConstructWithoutSessionExecutor) {
  namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
  onnx_light_cpu::RegisterAllKernels();
  const auto &table = rt_ns::KernelDispatchTable();
  const auto factory = table.find("ai.onnx:Abs");
  ASSERT_NE(factory, table.end());
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("Abs");
  node.add_input("x");
  node.add_output("y");
  rt_ns::RuntimeContext runtime(rt_ns::KernelContext(rt_ns::DefaultOpset(18)));
  runtime.Set("x", rt_ns::Tensor::FromFloat("x", {4}, {-1.0f, 2.0f, -3.5f, 4.0f}));

  std::unique_ptr<rt_ns::KernelBase> kernel;
  ASSERT_NO_THROW(kernel = factory->second(node, runtime));
  ASSERT_NE(kernel, nullptr);
  EXPECT_NO_THROW(kernel->Run(runtime));
  const float *y = runtime.Get("y").AsFloat();
  EXPECT_FLOAT_EQ(y[0], 1.0f);
  EXPECT_FLOAT_EQ(y[1], 2.0f);
  EXPECT_FLOAT_EQ(y[2], 3.5f);
  EXPECT_FLOAT_EQ(y[3], 4.0f);
}

TEST(OnnxLightRegisterKernels, VariadicFactoryUsesOneCommonBroadcastPlan) {
  namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
  onnx_light_cpu::RegisterAllKernels();
  const auto factory = rt_ns::KernelDispatchTable().find("ai.onnx:Sum");
  ASSERT_NE(factory, rt_ns::KernelDispatchTable().end());
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("Sum");
  node.add_input("a");
  node.add_input("b");
  node.add_input("c");
  node.add_output("y");
  rt_ns::RuntimeContext runtime(rt_ns::KernelContext(rt_ns::DefaultOpset(18)));
  runtime.Set("a", rt_ns::Tensor::FromFloat("a", {2, 1}, {1.0f, 2.0f}));
  runtime.Set("b", rt_ns::Tensor::FromFloat("b", {1, 3}, {10.0f, 20.0f, 30.0f}));
  runtime.Set("c", rt_ns::Tensor::FromFloat("c", {}, {100.0f}));

  std::unique_ptr<rt_ns::KernelBase> kernel = factory->second(node, runtime);
  ASSERT_NE(kernel, nullptr);
  ASSERT_NO_THROW(kernel->Run(runtime));
  const rt_ns::Tensor &output = runtime.Get("y");
  EXPECT_EQ(output.shape, (rt_ns::Shape{2, 3}));
  const float *values = output.AsFloat();
  EXPECT_EQ(std::vector<float>(values, values + 6),
            (std::vector<float>{111.0f, 121.0f, 131.0f, 112.0f, 122.0f, 132.0f}));
}

} // namespace
