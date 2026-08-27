// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/math/abs_kernel.h"
#include "onnx_light_cpu/kernels/register_kernels.h"

#include "onnx_core/compute/raw_buffer_allocator.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/tuning/cpu_executor.h"
#include "onnx_core/runtime/tuning/kernel_tuning.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_light_cpu/impl/execution.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

rt_ns::KernelContext MakeCtx() { return rt_ns::KernelContext(rt_ns::OpsetId(std::string(), 18)); }

TEST(OnnxLightAbsKernel, Float32) {
  onnx_light_cpu::AbsKernel kernel(MakeCtx());
  const std::vector<float> values = {-1.0f, 0.0f, 3.0f, -7.5f};
  const rt_ns::Tensor x = rt_ns::Tensor::FromFloat("x", {4}, values);
  const rt_ns::Tensor y = kernel(x);
  ASSERT_EQ(y.element_count(), 4);
  const float *py = y.AsFloat();
  for (std::size_t i = 0; i < values.size(); ++i) {
    EXPECT_FLOAT_EQ(py[i], std::fabs(values[i]));
  }
}

TEST(OnnxLightAbsKernel, Float64) {
  onnx_light_cpu::AbsKernel kernel(MakeCtx());
  const std::vector<double> values = {-1.0, 0.0, 3.0, -7.5, 0.001};
  const rt_ns::Tensor x = rt_ns::Tensor::FromDouble("x", {5}, values);
  const rt_ns::Tensor y = kernel(x);
  ASSERT_EQ(y.element_count(), 5);
  const double *py = y.AsDouble();
  for (std::size_t i = 0; i < values.size(); ++i) {
    EXPECT_DOUBLE_EQ(py[i], std::fabs(values[i]));
  }
}

TEST(OnnxLightAbsKernel, Int64) {
  onnx_light_cpu::AbsKernel kernel(MakeCtx());
  const std::vector<int64_t> values = {-1, 0, 3, -7, 100};
  const rt_ns::Tensor x = rt_ns::Tensor::FromInt64("x", {5}, values);
  const rt_ns::Tensor y = kernel(x);
  ASSERT_EQ(y.element_count(), 5);
  const int64_t *py = y.AsInt64();
  for (std::size_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(py[i], values[i] < 0 ? -values[i] : values[i]);
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

TEST(OnnxLightAbsKernel, RegistersAndAppliesValidatedTuning) {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("Abs");
  node.add_input("x");
  node.add_output("y");
  onnx_light_cpu::AbsKernel kernel(node, MakeCtx());
  onnx_light_cpu::AbsKernel::RegisterTuningSchemas();

  for (rt_ns::DataType type :
       {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE, rt_ns::DataType::FLOAT16,
        rt_ns::DataType::BFLOAT16, rt_ns::DataType::INT8, rt_ns::DataType::INT16,
        rt_ns::DataType::INT32, rt_ns::DataType::INT64}) {
    const auto key = kernel.TuningKey(static_cast<int32_t>(type));
    EXPECT_EQ(key.library, "onnx_light_cpu");
    EXPECT_EQ(key.kernel, "Abs");
    EXPECT_EQ(key.implementation, "simd_dispatch");
    EXPECT_EQ(key.tuning_abi, onnx_light_cpu::AbsKernel::kTuningAbi);
    EXPECT_NE(rt_ns::GetKernelTuningRegistry().FindSchema(key), nullptr);
  }

  const auto key = kernel.TuningKey(static_cast<int32_t>(rt_ns::DataType::FLOAT));
  const auto schema = rt_ns::GetKernelTuningRegistry().FindSchema(key);
  ASSERT_NE(schema, nullptr);
  auto parameters = schema->portable_defaults();
  EXPECT_EQ(parameters.Get<int64_t>("parallel.threshold_bytes"), 2 * 1024 * 1024);
  EXPECT_EQ(parameters.Get<int64_t>("parallel.target_block_bytes"), 256 * 1024);
  EXPECT_EQ(parameters.Get<int64_t>("parallel.max_participants"), 32);
  EXPECT_EQ(parameters.Get<int64_t>("parallel.preferred_participants"), 0);
  EXPECT_EQ(parameters.Get<int64_t>("memory.streaming_store_threshold_bytes"), 0);
  EXPECT_EQ(parameters.Get<int64_t>("parallel.cost_model"), 1);

  parameters.values["parallel.threshold_bytes"] = int64_t{1};
  parameters.values["parallel.target_block_bytes"] = int64_t{64};
  parameters.values["parallel.max_participants"] = int64_t{2};
  parameters.values["parallel.preferred_participants"] = int64_t{2};
  parameters.values["memory.streaming_store_threshold_bytes"] = int64_t{0};
  parameters.values["parallel.cost_model"] = int64_t{0};
  EXPECT_NO_THROW(kernel.Configure(parameters));

  constexpr std::size_t count = 256;
  const rt_ns::Tensor x = rt_ns::Tensor::FromFloat("x", {count}, std::vector<float>(count, -3.0f));
  InlineExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 8, &InlineExecutor::Run};
  onnx_light_cpu::ExecutionExecutorScope scope(&view);
  const rt_ns::Tensor y = kernel(x);
  EXPECT_EQ(executor.dispatches, 1);
  EXPECT_EQ(executor.blocks, 2);
  for (float value : std::span(y.AsFloat(), count)) {
    EXPECT_EQ(value, 3.0f);
  }

  parameters.values["parallel.max_participants"] = int64_t{-1};
  EXPECT_THROW(schema->Validate(parameters), std::invalid_argument);
  EXPECT_THROW(kernel.Configure(parameters), std::invalid_argument);
  parameters = schema->portable_defaults();
  parameters.values["parallel.preferred_participants"] = int64_t{-1};
  EXPECT_THROW(schema->Validate(parameters), std::invalid_argument);
  parameters = schema->portable_defaults();
  parameters.values["memory.streaming_store_threshold_bytes"] = int64_t{-1};
  EXPECT_THROW(schema->Validate(parameters), std::invalid_argument);
  parameters = schema->portable_defaults();
  parameters.values["parallel.cost_model"] = int64_t{2};
  EXPECT_THROW(schema->Validate(parameters), std::invalid_argument);
  parameters = schema->portable_defaults();
  parameters.values["parallel.target_block_bytes"] = int64_t{0};
  EXPECT_THROW(schema->Validate(parameters), std::invalid_argument);
  EXPECT_EQ(kernel.TuningKey(static_cast<int32_t>(rt_ns::DataType::STRING)).device,
            sym_ns::Device::kUndefined);
}

TEST(OnnxLightAbsKernel, OutputUsesSlotAllocator) {
  onnx_light_cpu::AbsKernel kernel(MakeCtx());
  const rt_ns::Tensor x = rt_ns::Tensor::FromFloat("x", {2}, {-1.0f, 3.0f});
  rt_ns::ExecutionArena execution_arena(1);
  auto io_arena = rt_ns::IOArena::Create(1);
  rt_ns::RuntimeContextOptions options;
  options.allocator = &execution_arena;
  options.io_allocator = io_arena.get();
  rt_ns::RuntimeContext rt(options);
  rt.set_output_slot_io_roles({true});

  {
    const rt_ns::Tensor y = kernel(x, &rt);
    EXPECT_EQ(y.allocation_owner(), io_arena.get());
    EXPECT_EQ(execution_arena.allocated_count(), 0);
    EXPECT_EQ(io_arena->allocated_count(), 1);
  }

  rt.set_output_slot_io_roles({false});
  {
    const rt_ns::Tensor y = kernel(x, &rt);
    EXPECT_EQ(y.allocation_owner(), &execution_arena);
    EXPECT_EQ(execution_arena.allocated_count(), 1);
    EXPECT_EQ(io_arena->allocated_count(), 0);
  }
}

// Exercises a large direct kernel call, which remains serial unless a
// registered session installs its executor.
TEST(OnnxLightAbsKernel, Float32LargeParallel) {
  onnx_light_cpu::AbsKernel kernel(MakeCtx());
  const int64_t n = 100000;
  std::vector<float> values(static_cast<std::size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    values[static_cast<std::size_t>(i)] = static_cast<float>((i % 2 == 0 ? -1 : 1) * (i % 97));
  }
  const rt_ns::Tensor x = rt_ns::Tensor::FromFloat("x", {n}, values);
  const rt_ns::Tensor y = kernel(x);
  ASSERT_EQ(y.element_count(), n);
  const float *py = y.AsFloat();
  for (std::size_t i = 0; i < values.size(); ++i) {
    EXPECT_FLOAT_EQ(py[i], std::fabs(values[i]));
  }
}

TEST(OnnxLightAbsKernel, RegisteredKernelUsesSessionExecutorWithoutPrivatePool) {
  onnx_light_cpu::RegisterAllKernels();
  const auto &table = rt_ns::KernelDispatchTable();
  const auto factory = table.find("ai.onnx:Abs");
  ASSERT_NE(factory, table.end());

  if (rt_ns::ProcessVisibleLogicalProcessors().size() < 2) {
    GTEST_SKIP() << "Session executor dispatch test requires two visible processors.";
  }
  const uint32_t participants = 2;
  rt_ns::CpuExecutionPolicy policy;
  policy.num_threads = static_cast<int32_t>(participants);
  policy.affinity_policy = rt_ns::CpuAffinityPolicy::kNone;
  rt_ns::CpuExecutorRegistry registry(1);
  std::shared_ptr<rt_ns::CpuExecutor> executor = registry.Acquire(policy);
  executor->EnableCounters();

  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("Abs");
  node.add_input("x");
  node.add_output("y");
  rt_ns::RuntimeContext runtime(rt_ns::KernelContext(rt_ns::DefaultOpset(18)));
  runtime.set_cpu_executor(executor.get());
  const int64_t n = 2097152;
  std::vector<float> values(static_cast<std::size_t>(n), -3.0f);
  runtime.Set("x", rt_ns::Tensor::FromFloat("x", {n}, values));
  std::unique_ptr<rt_ns::KernelBase> kernel = factory->second(node, runtime);
  {
    rt_ns::CpuExecutorScope scope(executor.get());
    kernel->Run(runtime);
  }

  EXPECT_EQ(runtime.Get("y").element_count(), n);
  EXPECT_GE(executor->counters().dispatches, 1u);
}

} // namespace
