// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/register_kernels.h"

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

#include <gtest/gtest.h>

namespace {

// ``RegisterAllKernels`` installs every onnx-light-cpu kernel class into
// onnx-light's shared ``KernelDispatchTable``. The call must succeed and be
// safe to invoke more than once (re-registration overrides the existing
// entries with the same factories).
TEST(OnnxLightRegisterKernels, RegisterAllKernels) {
  EXPECT_NO_THROW(onnx_light_cpu::RegisterAllKernels());
  EXPECT_NO_THROW(onnx_light_cpu::RegisterAllKernels());
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

} // namespace
