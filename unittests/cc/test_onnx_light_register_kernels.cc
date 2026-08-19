// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/register_kernels.h"

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
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

TEST(OnnxLightRegisterKernels, RegisteredFactoriesRequireSessionExecutor) {
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

  EXPECT_THROW(factory->second(node, runtime), std::invalid_argument);
}

} // namespace
