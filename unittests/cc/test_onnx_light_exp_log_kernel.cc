// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/math/exp_log_kernel.h"

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/tuning/kernel_tuning.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_light_cpu/impl/execution.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

rt_ns::KernelContext MakeCtx() { return rt_ns::KernelContext(rt_ns::OpsetId(std::string(), 18)); }

TEST(OnnxLightExpKernel, Float32) {
  onnx_light_cpu::ExpKernel kernel(MakeCtx());
  const std::vector<float> values = {-1.0f, 0.0f, 1.0f, 2.0f};
  const rt_ns::Tensor x = rt_ns::Tensor::FromFloat("x", {4}, values);
  const rt_ns::Tensor y = kernel(x);
  ASSERT_EQ(y.element_count(), 4);
  const float *py = y.AsFloat();
  for (std::size_t i = 0; i < values.size(); ++i) {
    EXPECT_NEAR(py[i], std::exp(values[i]), 1e-4f * std::abs(std::exp(values[i])) + 1e-5f);
  }
}

TEST(OnnxLightExpKernel, Double) {
  onnx_light_cpu::ExpKernel kernel(MakeCtx());
  const std::vector<double> values = {-2.0, 0.0, 0.5, 3.0};
  const rt_ns::Tensor x = rt_ns::Tensor::FromDouble("x", {4}, values);
  const rt_ns::Tensor y = kernel(x);
  ASSERT_EQ(y.element_count(), 4);
  const double *py = y.AsDouble();
  for (std::size_t i = 0; i < values.size(); ++i) {
    EXPECT_NEAR(py[i], std::exp(values[i]), 1e-9 * std::abs(std::exp(values[i])) + 1e-12);
  }
}

TEST(OnnxLightLogKernel, Float32) {
  onnx_light_cpu::LogKernel kernel(MakeCtx());
  const std::vector<float> values = {0.5f, 1.0f, 2.0f, 10.0f};
  const rt_ns::Tensor x = rt_ns::Tensor::FromFloat("x", {4}, values);
  const rt_ns::Tensor y = kernel(x);
  ASSERT_EQ(y.element_count(), 4);
  const float *py = y.AsFloat();
  for (std::size_t i = 0; i < values.size(); ++i) {
    EXPECT_NEAR(py[i], std::log(values[i]), 1e-4f * std::abs(std::log(values[i])) + 1e-5f);
  }
}

TEST(OnnxLightLogKernel, Double) {
  onnx_light_cpu::LogKernel kernel(MakeCtx());
  const std::vector<double> values = {0.25, 1.0, 2.0, 100.0};
  const rt_ns::Tensor x = rt_ns::Tensor::FromDouble("x", {4}, values);
  const rt_ns::Tensor y = kernel(x);
  ASSERT_EQ(y.element_count(), 4);
  const double *py = y.AsDouble();
  for (std::size_t i = 0; i < values.size(); ++i) {
    EXPECT_NEAR(py[i], std::log(values[i]), 1e-9 * std::abs(std::log(values[i])) + 1e-12);
  }
}

TEST(OnnxLightExpLogKernel, Roundtrip) {
  onnx_light_cpu::ExpKernel exp_kernel(MakeCtx());
  onnx_light_cpu::LogKernel log_kernel(MakeCtx());
  const std::vector<float> values = {0.0f, 1.0f, 2.0f, 3.0f};
  const rt_ns::Tensor x = rt_ns::Tensor::FromFloat("x", {4}, values);
  const rt_ns::Tensor y = log_kernel(exp_kernel(x));
  const float *py = y.AsFloat();
  for (std::size_t i = 0; i < values.size(); ++i) {
    EXPECT_NEAR(py[i], values[i], 1e-3f);
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

template <typename Kernel> void CheckTuningSchema(const char *op_type) {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type(op_type);
  node.add_input("x");
  node.add_output("y");
  Kernel kernel(node, MakeCtx());
  Kernel::RegisterTuningSchemas();

  for (rt_ns::DataType type : {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE,
                               rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16}) {
    const auto key = kernel.TuningKey(static_cast<int32_t>(type));
    EXPECT_EQ(key.library, "onnx_light_cpu");
    EXPECT_EQ(key.kernel, op_type);
    EXPECT_EQ(key.implementation, "simd_dispatch");
    EXPECT_EQ(key.tuning_abi, Kernel::kTuningAbi);
    const auto schema = rt_ns::GetKernelTuningRegistry().FindSchema(key);
    ASSERT_NE(schema, nullptr);
    const auto defaults = schema->portable_defaults();
    const bool half = type == rt_ns::DataType::FLOAT16 || type == rt_ns::DataType::BFLOAT16;
    const bool log_bfloat16 =
        std::string_view(op_type) == "Log" && type == rt_ns::DataType::BFLOAT16;
    EXPECT_EQ(defaults.template Get<int64_t>("parallel.threshold_bytes"), log_bfloat16 ? 512 * 1024
                                                                          : half       ? 1024 * 1024
                                                                                 : 2 * 1024 * 1024);
    EXPECT_EQ(defaults.template Get<int64_t>("parallel.target_block_bytes"), log_bfloat16
                                                                                 ? 64 * 1024
                                                                             : half ? 128 * 1024
                                                                                    : 256 * 1024);
    EXPECT_EQ(defaults.template Get<int64_t>("parallel.max_participants"), 32);
    EXPECT_EQ(defaults.template Get<int64_t>("parallel.cost_model"), 1);
  }
  EXPECT_EQ(kernel.TuningKey(static_cast<int32_t>(rt_ns::DataType::INT32)).device,
            sym_ns::Device::kUndefined);

  const auto key = kernel.TuningKey(static_cast<int32_t>(rt_ns::DataType::FLOAT));
  const auto schema = rt_ns::GetKernelTuningRegistry().FindSchema(key);
  auto parameters = schema->portable_defaults();
  parameters.values["parallel.threshold_bytes"] = int64_t{1};
  parameters.values["parallel.target_block_bytes"] = int64_t{64};
  parameters.values["parallel.max_participants"] = int64_t{3};
  parameters.values["parallel.cost_model"] = int64_t{0};
  EXPECT_NO_THROW(kernel.Configure(parameters));

  constexpr std::size_t count = 256;
  const rt_ns::Tensor x = rt_ns::Tensor::FromFloat("x", {count}, std::vector<float>(count, 1.0f));
  InlineExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 8, &InlineExecutor::Run};
  onnx_light_cpu::ExecutionExecutorScope scope(&view);
  EXPECT_EQ(kernel(x).element_count(), count);
  EXPECT_EQ(executor.dispatches, 1);
  EXPECT_EQ(executor.blocks, 3);

  parameters.values["parallel.max_participants"] = int64_t{-1};
  EXPECT_THROW(schema->Validate(parameters), std::invalid_argument);
  EXPECT_THROW(kernel.Configure(parameters), std::invalid_argument);
  parameters = schema->portable_defaults();
  parameters.values["parallel.cost_model"] = int64_t{2};
  EXPECT_THROW(schema->Validate(parameters), std::invalid_argument);
  parameters = schema->portable_defaults();
  parameters.values["parallel.target_block_bytes"] = int64_t{0};
  EXPECT_THROW(schema->Validate(parameters), std::invalid_argument);
}

TEST(OnnxLightExpLogKernel, RegistersAndAppliesValidatedTuning) {
  CheckTuningSchema<onnx_light_cpu::ExpKernel>("Exp");
  CheckTuningSchema<onnx_light_cpu::LogKernel>("Log");
}

} // namespace
