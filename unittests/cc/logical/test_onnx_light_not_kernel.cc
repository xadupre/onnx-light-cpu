// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/logical/not_kernel.h"

#include "onnx_light_cpu/impl/execution.h"

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/tuning/kernel_tuning.h"
#include "onnx_core/symbolic/sym_tensor.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

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

struct InlineExecutor {
  std::int64_t dispatches = 0;
  std::int64_t blocks = 0;

  static void Run(void *context, std::int64_t num_blocks, void *task_context,
                  onnx_light_cpu::ExecutionBlockFn task) {
    auto &self = *static_cast<InlineExecutor *>(context);
    ++self.dispatches;
    self.blocks = num_blocks;
    for (std::int64_t block = 0; block < num_blocks; ++block) {
      task(task_context, block);
    }
  }
};

TEST(OnnxLightNotKernel, RegistersAndAppliesValidatedTuning) {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("Not");
  node.add_input("x");
  node.add_output("y");
  onnx_light_cpu::NotKernel kernel(node, MakeCtx());
  onnx_light_cpu::NotKernel::RegisterTuningSchemas();

  const auto key = kernel.TuningKey(static_cast<int32_t>(rt_ns::DataType::BOOL));
  EXPECT_EQ(key.library, "onnx_light_cpu");
  EXPECT_EQ(key.kernel, "Not");
  EXPECT_EQ(key.implementation, "simd_dispatch");
  EXPECT_EQ(key.tuning_abi, onnx_light_cpu::NotKernel::kTuningAbi);
  const auto schema = rt_ns::GetKernelTuningRegistry().FindSchema(key);
  ASSERT_NE(schema, nullptr);
  auto parameters = schema->portable_defaults();
  EXPECT_EQ(parameters.Get<int64_t>("parallel.threshold_bytes"), 2 * 1024 * 1024);
  EXPECT_EQ(parameters.Get<int64_t>("parallel.target_block_bytes"), 256 * 1024);
  EXPECT_EQ(parameters.Get<int64_t>("parallel.max_participants"), 32);
  EXPECT_EQ(parameters.Get<int64_t>("parallel.preferred_participants"), 0);
  EXPECT_EQ(parameters.Get<int64_t>("parallel.cost_model"), 1);

  constexpr std::size_t count = 256;
  const rt_ns::Tensor x =
      rt_ns::Tensor::FromBool("x", {count}, std::vector<std::uint8_t>(count, 0));
  InlineExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 8, &InlineExecutor::Run};
  onnx_light_cpu::ExecutionExecutorScope scope(&view);

  const rt_ns::Tensor serial = kernel(x);
  EXPECT_EQ(executor.dispatches, 0);
  for (std::uint8_t value : std::span(serial.AsBool(), count)) {
    EXPECT_EQ(value, 1);
  }

  parameters.values["parallel.threshold_bytes"] = int64_t{1};
  parameters.values["parallel.target_block_bytes"] = int64_t{16};
  parameters.values["parallel.max_participants"] = int64_t{4};
  parameters.values["parallel.preferred_participants"] = int64_t{0};
  parameters.values["parallel.cost_model"] = int64_t{0};
  EXPECT_NO_THROW(schema->Validate(parameters));
  EXPECT_NO_THROW(kernel.Configure(parameters));
  const rt_ns::Tensor parallel = kernel(x);
  EXPECT_EQ(executor.dispatches, 1);
  EXPECT_EQ(executor.blocks, 4);
  for (std::uint8_t value : std::span(parallel.AsBool(), count)) {
    EXPECT_EQ(value, 1);
  }

  parameters.values["parallel.target_block_bytes"] = int64_t{0};
  EXPECT_THROW(schema->Validate(parameters), std::invalid_argument);
  EXPECT_THROW(kernel.Configure(parameters), std::invalid_argument);
  parameters = schema->portable_defaults();
  parameters.values["parallel.max_participants"] = int64_t{-1};
  EXPECT_THROW(schema->Validate(parameters), std::invalid_argument);
  parameters = schema->portable_defaults();
  parameters.values["parallel.preferred_participants"] = int64_t{-1};
  EXPECT_THROW(schema->Validate(parameters), std::invalid_argument);
  parameters = schema->portable_defaults();
  parameters.values["parallel.cost_model"] = int64_t{2};
  EXPECT_THROW(schema->Validate(parameters), std::invalid_argument);
  parameters = schema->portable_defaults();
  parameters.values["parallel.threshold_bytes"] = int64_t{-1};
  EXPECT_THROW(schema->Validate(parameters), std::invalid_argument);
  EXPECT_THROW(kernel.Configure(parameters), std::invalid_argument);
  EXPECT_EQ(kernel.TuningKey(static_cast<int32_t>(rt_ns::DataType::FLOAT)).device,
            sym_ns::Device::kUndefined);
}

} // namespace
