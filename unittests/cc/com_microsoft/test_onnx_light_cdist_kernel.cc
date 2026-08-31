// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/com_microsoft/cdist_kernel.h"
#include "onnx_light_cpu/kernels/com_microsoft/naive_cdist_kernel.h"
#include "onnx_light_cpu/kernels/register_kernels.h"

#include "onnx_core/compute/raw_buffer_allocator.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/tuning/kernel_tuning.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_light_cpu/impl/execution.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

rt_ns::KernelContext MakeCtx() { return rt_ns::KernelContext(rt_ns::OpsetId("com.microsoft", 1)); }

TEST(OnnxLightCDistKernel, Float32DefaultMetricIsSqEuclidean) {
  onnx_light_cpu::CDistKernel kernel(MakeCtx());
  const rt_ns::Tensor a =
      rt_ns::Tensor::FromFloat("A", {3, 2}, {0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 2.0f});
  const rt_ns::Tensor b = rt_ns::Tensor::FromFloat("B", {2, 2}, {0.0f, 0.0f, 1.0f, 0.0f});
  const rt_ns::Tensor c = kernel(a, b);
  const rt_ns::Tensor reference = onnx_light_cpu::NaiveCDistKernel(MakeCtx())(a, b);
  ASSERT_EQ(c.shape.size(), 2u);
  EXPECT_EQ(c.shape[0], 3);
  EXPECT_EQ(c.shape[1], 2);
  const float *pc = c.AsFloat();
  const std::vector<float> expected = {0.0f, 1.0f, 2.0f, 1.0f, 8.0f, 5.0f};
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(pc[i], expected[i], 1e-5f) << i;
    EXPECT_NEAR(pc[i], reference.AsFloat()[i], 1e-5f) << i;
  }
}

TEST(OnnxLightCDistKernel, Float32EuclideanTakesSquareRoot) {
  onnx_light_cpu::CDistKernel kernel(MakeCtx());
  const rt_ns::Tensor a =
      rt_ns::Tensor::FromFloat("A", {3, 2}, {0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 2.0f});
  const rt_ns::Tensor b = rt_ns::Tensor::FromFloat("B", {2, 2}, {0.0f, 0.0f, 1.0f, 0.0f});
  const rt_ns::Tensor c = kernel(a, b, "euclidean");
  const rt_ns::Tensor reference = onnx_light_cpu::NaiveCDistKernel(MakeCtx())(a, b, "euclidean");
  const float *pc = c.AsFloat();
  const std::vector<float> expected = {0.0f,           1.0f, std::sqrt(2.0f), 1.0f, std::sqrt(8.0f),
                                       std::sqrt(5.0f)};
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(pc[i], expected[i], 1e-5f) << i;
    EXPECT_NEAR(pc[i], reference.AsFloat()[i], 1e-5f) << i;
  }
}

TEST(OnnxLightCDistKernel, Float64Works) {
  onnx_light_cpu::CDistKernel kernel(MakeCtx());
  const rt_ns::Tensor a = rt_ns::Tensor::FromDouble("A", {2, 3}, {1.0, 2.0, 3.0, -1.0, 0.0, 1.0});
  const rt_ns::Tensor b = rt_ns::Tensor::FromDouble("B", {1, 3}, {0.0, 0.0, 0.0});
  const rt_ns::Tensor c = kernel(a, b, "sqeuclidean");
  const rt_ns::Tensor reference = onnx_light_cpu::NaiveCDistKernel(MakeCtx())(a, b, "sqeuclidean");
  ASSERT_EQ(c.element_count(), 2);
  const double *pc = c.AsDouble();
  EXPECT_NEAR(pc[0], 14.0, 1e-9);
  EXPECT_NEAR(pc[1], 2.0, 1e-9);
  EXPECT_NEAR(pc[0], reference.AsDouble()[0], 1e-9);
  EXPECT_NEAR(pc[1], reference.AsDouble()[1], 1e-9);
}

TEST(OnnxLightCDistKernel, RejectsInvalidMetric) {
  onnx_light_cpu::CDistKernel kernel(MakeCtx());
  const rt_ns::Tensor a = rt_ns::Tensor::FromFloat("A", {2, 2}, {0.0f, 0.0f, 1.0f, 1.0f});
  const rt_ns::Tensor b = rt_ns::Tensor::FromFloat("B", {2, 2}, {0.0f, 0.0f, 1.0f, 1.0f});
  EXPECT_THROW((void)kernel(a, b, "cityblock"), std::invalid_argument);
}

TEST(OnnxLightCDistKernel, RejectsRankMismatch) {
  onnx_light_cpu::CDistKernel kernel(MakeCtx());
  const rt_ns::Tensor a = rt_ns::Tensor::FromFloat("A", {2, 2, 2}, std::vector<float>(8, 1.0f));
  const rt_ns::Tensor b = rt_ns::Tensor::FromFloat("B", {2, 2}, {0.0f, 0.0f, 1.0f, 1.0f});
  EXPECT_THROW((void)kernel(a, b), std::invalid_argument);
}

TEST(OnnxLightCDistKernel, RejectsMismatchedFeatureDimension) {
  onnx_light_cpu::CDistKernel kernel(MakeCtx());
  const rt_ns::Tensor a = rt_ns::Tensor::FromFloat("A", {2, 3}, std::vector<float>(6, 1.0f));
  const rt_ns::Tensor b = rt_ns::Tensor::FromFloat("B", {2, 2}, std::vector<float>(4, 1.0f));
  EXPECT_THROW((void)kernel(a, b), std::invalid_argument);
}

TEST(OnnxLightCDistKernel, RejectsZeroFeatureDimensionLikeOnnxRuntime) {
  onnx_light_cpu::CDistKernel kernel(MakeCtx());
  const rt_ns::Tensor a = rt_ns::Tensor::FromFloat("A", {2, 0}, std::vector<float>{});
  const rt_ns::Tensor b = rt_ns::Tensor::FromFloat("B", {3, 0}, std::vector<float>{});
  EXPECT_THROW((void)kernel(a, b), std::invalid_argument);
}

TEST(OnnxLightCDistKernel, RejectsMismatchedDataTypes) {
  onnx_light_cpu::CDistKernel kernel(MakeCtx());
  const rt_ns::Tensor a = rt_ns::Tensor::FromFloat("A", {2, 2}, std::vector<float>(4, 1.0f));
  const rt_ns::Tensor b = rt_ns::Tensor::FromDouble("B", {2, 2}, std::vector<double>(4, 1.0));
  EXPECT_THROW((void)kernel(a, b), std::invalid_argument);
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

TEST(OnnxLightCDistKernel, RegistersAndAppliesValidatedTuning) {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("CDist");
  node.set_domain("com.microsoft");
  node.add_input("A");
  node.add_input("B");
  node.add_output("C");
  onnx_light_cpu::CDistKernel kernel(node, MakeCtx());
  onnx_light_cpu::CDistKernel::RegisterTuningSchemas();

  for (rt_ns::DataType type : {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE}) {
    const auto key = kernel.TuningKey(static_cast<int32_t>(type));
    EXPECT_EQ(key.library, "onnx_light_cpu");
    EXPECT_EQ(key.kernel, "CDist");
    EXPECT_EQ(key.implementation, "row_dispatch");
    EXPECT_EQ(key.tuning_abi, onnx_light_cpu::CDistKernel::kTuningAbi);
    const auto schema = rt_ns::GetKernelTuningRegistry().FindSchema(key);
    ASSERT_NE(schema, nullptr);
  }

  const auto key = kernel.TuningKey(static_cast<int32_t>(rt_ns::DataType::FLOAT));
  const auto schema = rt_ns::GetKernelTuningRegistry().FindSchema(key);
  ASSERT_NE(schema, nullptr);
  auto parameters = schema->portable_defaults();
  EXPECT_EQ(parameters.Get<int64_t>("parallel.threshold_bytes"), 128 * 1024);
  EXPECT_EQ(parameters.Get<int64_t>("parallel.target_block_bytes"), 16 * 1024);
  EXPECT_EQ(parameters.Get<int64_t>("parallel.max_participants"), 16);
  EXPECT_EQ(parameters.Get<int64_t>("parallel.preferred_participants"), 0);
  EXPECT_EQ(parameters.Get<int64_t>("parallel.cost_model"), 1);

  parameters.values["parallel.threshold_bytes"] = int64_t{1};
  parameters.values["parallel.target_block_bytes"] = int64_t{1};
  parameters.values["parallel.max_participants"] = int64_t{2};
  parameters.values["parallel.preferred_participants"] = int64_t{2};
  parameters.values["parallel.cost_model"] = int64_t{0};
  EXPECT_NO_THROW(kernel.Configure(parameters));

  constexpr std::size_t m = 33;
  constexpr std::size_t k = 7;
  constexpr std::size_t n = 19;
  std::vector<float> a_values(m * n);
  std::vector<float> b_values(k * n);
  for (std::size_t i = 0; i < a_values.size(); ++i) {
    a_values[i] = static_cast<float>(static_cast<int>(i % 29) - 14) / 9.0f;
  }
  for (std::size_t i = 0; i < b_values.size(); ++i) {
    b_values[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 11.0f;
  }
  const rt_ns::Tensor a =
      rt_ns::Tensor::FromFloat("A", {static_cast<int64_t>(m), static_cast<int64_t>(n)}, a_values);
  const rt_ns::Tensor b =
      rt_ns::Tensor::FromFloat("B", {static_cast<int64_t>(k), static_cast<int64_t>(n)}, b_values);
  InlineExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 8, &InlineExecutor::Run};
  onnx_light_cpu::ExecutionExecutorScope scope(&view);
  const rt_ns::Tensor c = kernel(a, b);
  const rt_ns::Tensor reference = onnx_light_cpu::NaiveCDistKernel(MakeCtx())(a, b);
  EXPECT_EQ(executor.dispatches, 1);
  EXPECT_GT(executor.blocks, 0);
  for (std::size_t i = 0; i < m * k; ++i) {
    EXPECT_NEAR(c.AsFloat()[i], reference.AsFloat()[i], 2e-5f) << i;
  }

  parameters.values["parallel.max_participants"] = int64_t{-1};
  EXPECT_THROW(schema->Validate(parameters), std::invalid_argument);
  EXPECT_THROW(kernel.Configure(parameters), std::invalid_argument);
  parameters = schema->portable_defaults();
  parameters.values["parallel.cost_model"] = int64_t{2};
  EXPECT_THROW(schema->Validate(parameters), std::invalid_argument);
  parameters = schema->portable_defaults();
  parameters.values["parallel.target_block_bytes"] = int64_t{0};
  EXPECT_THROW(schema->Validate(parameters), std::invalid_argument);
  EXPECT_EQ(kernel.TuningKey(static_cast<int32_t>(rt_ns::DataType::STRING)).device,
            sym_ns::Device::kUndefined);
}

TEST(OnnxLightCDistKernel, RunReadsMetricAttributeAndWritesOutput) {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("CDist");
  node.set_domain("com.microsoft");
  node.add_input("A");
  node.add_input("B");
  node.add_output("C");
  auto *metric = node.add_attribute();
  metric->set_name("metric");
  metric->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::STRING);
  metric->set_s("euclidean");

  rt_ns::RuntimeContext rt(rt_ns::KernelContext(rt_ns::OpsetId("com.microsoft", 1)));
  rt.Set("A", rt_ns::Tensor::FromFloat("A", {2, 2}, {0.0f, 0.0f, 3.0f, 4.0f}));
  rt.Set("B", rt_ns::Tensor::FromFloat("B", {1, 2}, {0.0f, 0.0f}));

  onnx_light_cpu::CDistKernel kernel(node, MakeCtx());
  kernel.Run(rt);
  const rt_ns::Tensor &c = rt.Get("C");
  ASSERT_EQ(c.element_count(), 2);
  EXPECT_NEAR(c.AsFloat()[0], 0.0f, 1e-5f);
  EXPECT_NEAR(c.AsFloat()[1], 5.0f, 1e-5f);
}

TEST(OnnxLightCDistKernel, RegisteredUnderMicrosoftDomain) {
  onnx_light_cpu::RegisterAllKernels();
  const auto &table = rt_ns::KernelDispatchTable();
  EXPECT_NE(table.find("com.microsoft:CDist"), table.end());
}

} // namespace
