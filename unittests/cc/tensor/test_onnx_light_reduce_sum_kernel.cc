// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/register_kernels.h"

#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/kernels/math/reduce_sum_kernel.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
using ONNX_LIGHT_NAMESPACE::NodeProto;

struct InlineExecutor {
  int64_t calls = 0;
  int64_t blocks = 0;

  static void Run(void *context, int64_t block_count, void *task_context,
                  onnx_light_cpu::ExecutionBlockFn task) {
    auto &executor = *static_cast<InlineExecutor *>(context);
    ++executor.calls;
    executor.blocks += block_count;
    for (int64_t block = 0; block < block_count; ++block) {
      task(task_context, block);
    }
  }
};

NodeProto MakeNode(const std::vector<std::string> &inputs, bool keepdims = true,
                   bool noop_with_empty_axes = false) {
  NodeProto node;
  node.set_op_type("ReduceSum");
  for (const auto &input : inputs) {
    node.add_input(input);
  }
  node.add_output("y");
  auto *keepdims_attribute = node.add_attribute();
  keepdims_attribute->set_name("keepdims");
  keepdims_attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::INT);
  keepdims_attribute->set_i(keepdims ? 1 : 0);
  if (noop_with_empty_axes) {
    auto *noop_attribute = node.add_attribute();
    noop_attribute->set_name("noop_with_empty_axes");
    noop_attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::INT);
    noop_attribute->set_i(1);
  }
  return node;
}

void RunKernel(NodeProto &node, rt_ns::RuntimeContext &runtime) {
  ASSERT_TRUE(onnx_light_cpu::RegisterKernelForSession(runtime, "", "ReduceSum"));
  runtime.set_kernel_usage_enabled(true);
  rt_ns::RunNode(node, runtime);
}

TEST(OnnxLightReduceSumKernel, ReducesInt64AxesAndRecordsCpuDispatch) {
  rt_ns::RuntimeContext runtime(rt_ns::KernelContext(rt_ns::DefaultOpset(13)));
  runtime.Set("data", rt_ns::Tensor::FromInt64("data", {2, 3}, {1, 0, 1, 0, 1, 1}));
  runtime.Set("axes", rt_ns::Tensor::FromInt64("axes", {1}, {1}));
  NodeProto node = MakeNode({"data", "axes"});

  RunKernel(node, runtime);

  const auto &output = runtime.Get("y");
  EXPECT_EQ(output.data_type, rt_ns::DataType::INT64);
  EXPECT_EQ(output.shape, (rt_ns::Shape{2, 1}));
  EXPECT_EQ(std::vector<int64_t>(output.AsInt64(), output.AsInt64() + 2),
            (std::vector<int64_t>{2, 2}));
  EXPECT_EQ(runtime.GetKernelUsage(),
            (std::vector<std::string>{onnx_light_cpu::ReduceSumKernel::kName}));
}

TEST(OnnxLightReduceSumKernel, SupportsLegacyAxesAttribute) {
  rt_ns::RuntimeContext runtime(rt_ns::KernelContext(rt_ns::DefaultOpset(11)));
  runtime.Set("data", rt_ns::Tensor::FromInt64("data", {2, 3}, {1, 2, 3, 4, 5, 6}));
  NodeProto node = MakeNode({"data"}, false);
  auto *axes_attribute = node.add_attribute();
  axes_attribute->set_name("axes");
  axes_attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::INTS);
  axes_attribute->add_ints(1);

  RunKernel(node, runtime);

  const auto &output = runtime.Get("y");
  EXPECT_EQ(output.shape, (rt_ns::Shape{2}));
  EXPECT_EQ(std::vector<int64_t>(output.AsInt64(), output.AsInt64() + 2),
            (std::vector<int64_t>{6, 15}));
  EXPECT_EQ(runtime.GetKernelUsage(),
            (std::vector<std::string>{onnx_light_cpu::ReduceSumKernel::kName}));
}

TEST(OnnxLightReduceSumKernel, PreservesLargeIntegerPrecisionAndWrapsOverflow) {
  constexpr int64_t beyond_double_precision = int64_t{1} << 53;
  {
    rt_ns::RuntimeContext runtime(rt_ns::KernelContext(rt_ns::DefaultOpset(13)));
    runtime.Set("data",
                rt_ns::Tensor::FromInt64("data", {2}, {beyond_double_precision, int64_t{1}}));
    runtime.Set("axes", rt_ns::Tensor::FromInt64("axes", {1}, {0}));
    NodeProto node = MakeNode({"data", "axes"}, false);

    RunKernel(node, runtime);

    EXPECT_EQ(runtime.Get("y").AsInt64()[0], beyond_double_precision + 1);
  }
  {
    rt_ns::RuntimeContext runtime(rt_ns::KernelContext(rt_ns::DefaultOpset(13)));
    runtime.Set("data", rt_ns::Tensor::FromInt64(
                            "data", {2}, {std::numeric_limits<int64_t>::max(), int64_t{1}}));
    runtime.Set("axes", rt_ns::Tensor::FromInt64("axes", {1}, {0}));
    NodeProto node = MakeNode({"data", "axes"}, false);

    RunKernel(node, runtime);

    EXPECT_EQ(runtime.Get("y").AsInt64()[0], std::numeric_limits<int64_t>::min());
  }
}

TEST(OnnxLightReduceSumKernel, SupportsNegativeAxesKeepdimsFalseAndEmptyAxes) {
  {
    rt_ns::RuntimeContext runtime(rt_ns::KernelContext(rt_ns::DefaultOpset(13)));
    runtime.Set("data", rt_ns::Tensor::FromInt64("data", {2, 3}, {1, 2, 3, 4, 5, 6}));
    runtime.Set("axes", rt_ns::Tensor::FromInt64("axes", {1}, {-1}));
    NodeProto node = MakeNode({"data", "axes"}, false);

    RunKernel(node, runtime);

    const auto &output = runtime.Get("y");
    EXPECT_EQ(output.shape, (rt_ns::Shape{2}));
    EXPECT_EQ(std::vector<int64_t>(output.AsInt64(), output.AsInt64() + 2),
              (std::vector<int64_t>{6, 15}));
  }
  {
    rt_ns::RuntimeContext runtime(rt_ns::KernelContext(rt_ns::DefaultOpset(13)));
    runtime.Set("data", rt_ns::Tensor::FromInt64("data", {2, 2}, {1, 2, 3, 4}));
    runtime.Set("axes", rt_ns::Tensor::FromInt64("axes", {0}, {}));
    NodeProto node = MakeNode({"data", "axes"});

    RunKernel(node, runtime);

    const auto &output = runtime.Get("y");
    EXPECT_EQ(output.shape, (rt_ns::Shape{1, 1}));
    EXPECT_EQ(output.AsInt64()[0], 10);
  }
}

TEST(OnnxLightReduceSumKernel, EmptyAxisNoopAndZeroSizedReduction) {
  {
    rt_ns::RuntimeContext runtime(rt_ns::KernelContext(rt_ns::DefaultOpset(13)));
    runtime.Set("data", rt_ns::Tensor::FromInt64("data", {2, 2}, {1, 2, 3, 4}));
    runtime.Set("axes", rt_ns::Tensor::FromInt64("axes", {0}, {}));
    NodeProto node = MakeNode({"data", "axes"}, true, true);

    RunKernel(node, runtime);

    const auto &output = runtime.Get("y");
    EXPECT_EQ(output.shape, (rt_ns::Shape{2, 2}));
    EXPECT_EQ(std::vector<int64_t>(output.AsInt64(), output.AsInt64() + 4),
              (std::vector<int64_t>{1, 2, 3, 4}));
  }
  {
    rt_ns::RuntimeContext runtime(rt_ns::KernelContext(rt_ns::DefaultOpset(13)));
    runtime.Set("data", rt_ns::Tensor::FromInt64("data", {2, 0, 3}, {}));
    runtime.Set("axes", rt_ns::Tensor::FromInt64("axes", {1}, {1}));
    NodeProto node = MakeNode({"data", "axes"});

    RunKernel(node, runtime);

    const auto &output = runtime.Get("y");
    EXPECT_EQ(output.shape, (rt_ns::Shape{2, 1, 3}));
    EXPECT_EQ(std::vector<int64_t>(output.AsInt64(), output.AsInt64() + 6),
              (std::vector<int64_t>{0, 0, 0, 0, 0, 0}));
  }
}

TEST(OnnxLightReduceSumKernel, DelegatesOtherTypesAndRejectsInvalidAxes) {
  {
    rt_ns::RuntimeContext runtime(rt_ns::KernelContext(rt_ns::DefaultOpset(13)));
    runtime.Set("data", rt_ns::Tensor::FromFloat("data", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}));
    runtime.Set("axes", rt_ns::Tensor::FromInt64("axes", {1}, {1}));
    NodeProto node = MakeNode({"data", "axes"});

    RunKernel(node, runtime);

    const auto &output = runtime.Get("y");
    EXPECT_EQ(output.shape, (rt_ns::Shape{2, 1}));
    EXPECT_EQ(std::vector<float>(output.AsFloat(), output.AsFloat() + 2),
              (std::vector<float>{3.0f, 7.0f}));
    EXPECT_TRUE(runtime.GetKernelUsage().empty());
  }
  {
    rt_ns::RuntimeContext runtime(rt_ns::KernelContext(rt_ns::DefaultOpset(13)));
    runtime.Set("data", rt_ns::Tensor::FromInt64("data", {2, 2}, {1, 2, 3, 4}));
    runtime.Set("axes", rt_ns::Tensor::FromInt64("axes", {1}, {2}));
    NodeProto node = MakeNode({"data", "axes"});
    ASSERT_TRUE(onnx_light_cpu::RegisterKernelForSession(runtime, "", "ReduceSum"));
    EXPECT_THROW(rt_ns::RunNode(node, runtime), std::invalid_argument);
  }
  {
    rt_ns::RuntimeContext runtime(rt_ns::KernelContext(rt_ns::DefaultOpset(13)));
    runtime.Set("data", rt_ns::Tensor::FromInt64("data", {2, 2}, {1, 2, 3, 4}));
    runtime.Set("axes", rt_ns::Tensor::FromInt64("axes", {1, 1}, {1}));
    NodeProto node = MakeNode({"data", "axes"});
    ASSERT_TRUE(onnx_light_cpu::RegisterKernelForSession(runtime, "", "ReduceSum"));
    EXPECT_THROW(rt_ns::RunNode(node, runtime), std::invalid_argument);
  }
}

TEST(OnnxLightReduceSumKernel, UsesRuntimeExecutorOnlyForSufficientOutputWork) {
  InlineExecutor executor;
  const onnx_light_cpu::ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run};
  const onnx_light_cpu::ExecutionExecutorScope scope(&view);
  {
    rt_ns::RuntimeContext runtime(rt_ns::KernelContext(rt_ns::DefaultOpset(13)));
    runtime.Set("data", rt_ns::Tensor::FromInt64("data", {2, 2}, {1, 2, 3, 4}));
    runtime.Set("axes", rt_ns::Tensor::FromInt64("axes", {1}, {1}));
    NodeProto node = MakeNode({"data", "axes"});

    RunKernel(node, runtime);
    EXPECT_EQ(executor.calls, 0);
  }
  {
    constexpr int64_t rows = 35000;
    rt_ns::RuntimeContext runtime(rt_ns::KernelContext(rt_ns::DefaultOpset(13)));
    runtime.Set("data",
                rt_ns::Tensor::FromInt64("data", {rows, 2}, std::vector<int64_t>(rows * 2, 1)));
    runtime.Set("axes", rt_ns::Tensor::FromInt64("axes", {1}, {1}));
    NodeProto node = MakeNode({"data", "axes"});

    RunKernel(node, runtime);
    EXPECT_EQ(executor.calls, 1);
    EXPECT_EQ(executor.blocks, 2);
    const auto &output = runtime.Get("y");
    ASSERT_EQ(output.shape, (rt_ns::Shape{rows, 1}));
    EXPECT_EQ(output.AsInt64()[rows - 1], 2);
  }
}

} // namespace
