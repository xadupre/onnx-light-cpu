// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/com_microsoft/bias_gelu_kernel.h"
#include "onnx_light_cpu/kernels/register_kernels.h"

#include "onnx_core/compute/raw_buffer_allocator.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/tuning/kernel_tuning.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

rt_ns::KernelContext MakeCtx() { return rt_ns::KernelContext(rt_ns::OpsetId("com.microsoft", 1)); }

double ReferenceGelu(double z) { return 0.5 * z * (1.0 + std::erf(z / std::sqrt(2.0))); }

TEST(OnnxLightBiasGeluKernel, Float32BroadcastsBiasOverLastDimension) {
  onnx_light_cpu::BiasGeluKernel kernel(MakeCtx());
  const rt_ns::Tensor a =
      rt_ns::Tensor::FromFloat("A", {2, 3}, {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 3.0f});
  const rt_ns::Tensor b = rt_ns::Tensor::FromFloat("B", {3}, {0.1f, -0.2f, 0.3f});
  const rt_ns::Tensor c = kernel(a, b);
  ASSERT_EQ(c.shape.size(), 2u);
  EXPECT_EQ(c.shape[0], 2);
  EXPECT_EQ(c.shape[1], 3);
  const float *values_a = a.AsFloat();
  const float *values_b = b.AsFloat();
  const float *pc = c.AsFloat();
  for (std::size_t row = 0; row < 2; ++row) {
    for (std::size_t col = 0; col < 3; ++col) {
      const double z = static_cast<double>(values_a[row * 3 + col]) + values_b[col];
      EXPECT_NEAR(pc[row * 3 + col], ReferenceGelu(z), 1e-5) << row << "," << col;
    }
  }
}

TEST(OnnxLightBiasGeluKernel, Float64Works) {
  onnx_light_cpu::BiasGeluKernel kernel(MakeCtx());
  const rt_ns::Tensor a = rt_ns::Tensor::FromDouble("A", {1, 4}, {-3.0, -1.0, 1.0, 4.0});
  const rt_ns::Tensor b = rt_ns::Tensor::FromDouble("B", {4}, {0.5, -0.5, 0.25, -0.25});
  const rt_ns::Tensor c = kernel(a, b);
  ASSERT_EQ(c.element_count(), 4);
  const double *pc = c.AsDouble();
  const double *values_a = a.AsDouble();
  const double *values_b = b.AsDouble();
  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT_NEAR(pc[i], ReferenceGelu(values_a[i] + values_b[i]), 1e-9) << i;
  }
}

TEST(OnnxLightBiasGeluKernel, Float16Works) {
  onnx_light_cpu::BiasGeluKernel kernel(MakeCtx());
  const rt_ns::Tensor a = rt_ns::MakeFloat16Tensor("A", {1, 3}, {-2.0f, 0.5f, 3.0f});
  const rt_ns::Tensor b = rt_ns::MakeFloat16Tensor("B", {3}, {0.1f, -0.2f, 0.3f});
  const rt_ns::Tensor c = kernel(a, b);
  ASSERT_EQ(c.element_count(), 3);
  const auto *pa = reinterpret_cast<const std::uint16_t *>(a.bytes());
  const auto *pb = reinterpret_cast<const std::uint16_t *>(b.bytes());
  const auto *pc = reinterpret_cast<const std::uint16_t *>(c.bytes());
  for (std::size_t i = 0; i < 3; ++i) {
    const float rounded_a = onnx_light_cpu::detail::Float16BitsToFloat(pa[i]);
    const float rounded_b = onnx_light_cpu::detail::Float16BitsToFloat(pb[i]);
    const float actual = onnx_light_cpu::detail::Float16BitsToFloat(pc[i]);
    EXPECT_NEAR(actual, ReferenceGelu(rounded_a + rounded_b), 5e-3) << i;
  }
}

TEST(OnnxLightBiasGeluKernel, BFloat16Works) {
  onnx_light_cpu::BiasGeluKernel kernel(MakeCtx());
  const rt_ns::Tensor a = rt_ns::MakeBfloat16Tensor("A", {1, 3}, {-2.0f, 0.5f, 3.0f});
  const rt_ns::Tensor b = rt_ns::MakeBfloat16Tensor("B", {3}, {0.1f, -0.2f, 0.3f});
  const rt_ns::Tensor c = kernel(a, b);
  ASSERT_EQ(c.element_count(), 3);
  const auto *pa = reinterpret_cast<const std::uint16_t *>(a.bytes());
  const auto *pb = reinterpret_cast<const std::uint16_t *>(b.bytes());
  const auto *pc = reinterpret_cast<const std::uint16_t *>(c.bytes());
  for (std::size_t i = 0; i < 3; ++i) {
    const float rounded_a = onnx_light_cpu::detail::Bfloat16BitsToFloat(pa[i]);
    const float rounded_b = onnx_light_cpu::detail::Bfloat16BitsToFloat(pb[i]);
    const float actual = onnx_light_cpu::detail::Bfloat16BitsToFloat(pc[i]);
    EXPECT_NEAR(actual, ReferenceGelu(rounded_a + rounded_b), 5e-2) << i;
  }
}

TEST(OnnxLightBiasGeluKernel, RejectsRankZeroA) {
  onnx_light_cpu::BiasGeluKernel kernel(MakeCtx());
  const rt_ns::Tensor a = rt_ns::Tensor::FromFloat("A", {}, {1.0f});
  const rt_ns::Tensor b = rt_ns::Tensor::FromFloat("B", {1}, {0.0f});
  EXPECT_THROW((void)kernel(a, b), std::invalid_argument);
}

TEST(OnnxLightBiasGeluKernel, RejectsNonRank1Bias) {
  onnx_light_cpu::BiasGeluKernel kernel(MakeCtx());
  const rt_ns::Tensor a = rt_ns::Tensor::FromFloat("A", {2, 3}, std::vector<float>(6, 1.0f));
  const rt_ns::Tensor b = rt_ns::Tensor::FromFloat("B", {1, 3}, std::vector<float>(3, 1.0f));
  EXPECT_THROW((void)kernel(a, b), std::invalid_argument);
}

TEST(OnnxLightBiasGeluKernel, RejectsBiasLengthMismatch) {
  onnx_light_cpu::BiasGeluKernel kernel(MakeCtx());
  const rt_ns::Tensor a = rt_ns::Tensor::FromFloat("A", {2, 3}, std::vector<float>(6, 1.0f));
  const rt_ns::Tensor b = rt_ns::Tensor::FromFloat("B", {4}, std::vector<float>(4, 1.0f));
  EXPECT_THROW((void)kernel(a, b), std::invalid_argument);
}

TEST(OnnxLightBiasGeluKernel, RejectsMismatchedDataTypes) {
  onnx_light_cpu::BiasGeluKernel kernel(MakeCtx());
  const rt_ns::Tensor a = rt_ns::Tensor::FromFloat("A", {1, 3}, std::vector<float>(3, 1.0f));
  const rt_ns::Tensor b = rt_ns::Tensor::FromDouble("B", {3}, std::vector<double>(3, 1.0));
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

TEST(OnnxLightBiasGeluKernel, RegistersAndAppliesValidatedTuning) {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("BiasGelu");
  node.set_domain("com.microsoft");
  node.add_input("A");
  node.add_input("B");
  node.add_output("C");
  onnx_light_cpu::BiasGeluKernel kernel(node, MakeCtx());
  onnx_light_cpu::BiasGeluKernel::RegisterTuningSchemas();

  for (rt_ns::DataType type : {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE,
                               rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16}) {
    const auto key = kernel.TuningKey(static_cast<int32_t>(type));
    EXPECT_EQ(key.library, "onnx_light_cpu");
    EXPECT_EQ(key.kernel, "BiasGelu");
    EXPECT_EQ(key.implementation, "scalar_row_dispatch");
    EXPECT_EQ(key.tuning_abi, onnx_light_cpu::BiasGeluKernel::kTuningAbi);
    const auto schema = rt_ns::GetKernelTuningRegistry().FindSchema(key);
    ASSERT_NE(schema, nullptr);
  }

  const auto key = kernel.TuningKey(static_cast<int32_t>(rt_ns::DataType::FLOAT));
  const auto schema = rt_ns::GetKernelTuningRegistry().FindSchema(key);
  ASSERT_NE(schema, nullptr);
  auto parameters = schema->portable_defaults();
  EXPECT_EQ(parameters.Get<int64_t>("parallel.threshold_bytes"), 256 * 1024);
  EXPECT_EQ(parameters.Get<int64_t>("parallel.target_block_bytes"), 64 * 1024);
  EXPECT_EQ(parameters.Get<int64_t>("parallel.max_participants"), 0);
  EXPECT_EQ(parameters.Get<int64_t>("parallel.preferred_participants"), 0);
  EXPECT_EQ(parameters.Get<int64_t>("parallel.cost_model"), 1);

  parameters.values["parallel.threshold_bytes"] = int64_t{1};
  parameters.values["parallel.target_block_bytes"] = int64_t{1};
  parameters.values["parallel.max_participants"] = int64_t{2};
  parameters.values["parallel.preferred_participants"] = int64_t{2};
  parameters.values["parallel.cost_model"] = int64_t{0};
  EXPECT_NO_THROW(kernel.Configure(parameters));

  constexpr std::size_t outer = 32;
  constexpr std::size_t inner = 4;
  const rt_ns::Tensor a =
      rt_ns::Tensor::FromFloat("A", {static_cast<int64_t>(outer), static_cast<int64_t>(inner)},
                               std::vector<float>(outer * inner, 0.0f));
  const rt_ns::Tensor b =
      rt_ns::Tensor::FromFloat("B", {static_cast<int64_t>(inner)}, std::vector<float>(inner, 0.0f));
  InlineExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 8, &InlineExecutor::Run};
  onnx_light_cpu::ExecutionExecutorScope scope(&view);
  const rt_ns::Tensor c = kernel(a, b);
  EXPECT_EQ(executor.dispatches, 1);
  EXPECT_GT(executor.blocks, 0);
  for (float value : std::span(c.AsFloat(), outer * inner)) {
    EXPECT_NEAR(value, 0.0f, 1e-6f);
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

TEST(OnnxLightBiasGeluKernel, RunWritesOutputFromNodeInputs) {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("BiasGelu");
  node.set_domain("com.microsoft");
  node.add_input("A");
  node.add_input("B");
  node.add_output("C");

  rt_ns::RuntimeContext rt(rt_ns::KernelContext(rt_ns::OpsetId("com.microsoft", 1)));
  rt.Set("A", rt_ns::Tensor::FromFloat("A", {1, 2}, {1.0f, -1.0f}));
  rt.Set("B", rt_ns::Tensor::FromFloat("B", {2}, {0.0f, 0.0f}));

  onnx_light_cpu::BiasGeluKernel kernel(node, MakeCtx());
  kernel.Run(rt);
  const rt_ns::Tensor &c = rt.Get("C");
  ASSERT_EQ(c.element_count(), 2);
  EXPECT_NEAR(c.AsFloat()[0], ReferenceGelu(1.0), 1e-5f);
  EXPECT_NEAR(c.AsFloat()[1], ReferenceGelu(-1.0), 1e-5f);
}

TEST(OnnxLightBiasGeluKernel, RegisteredUnderMicrosoftDomain) {
  onnx_light_cpu::RegisterAllKernels();
  const auto &table = rt_ns::KernelDispatchTable();
  EXPECT_NE(table.find("com.microsoft:BiasGelu"), table.end());
}

} // namespace
