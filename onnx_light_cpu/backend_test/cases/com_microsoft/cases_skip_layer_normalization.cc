// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/com_microsoft/include_com_microsoft_cases.h"
#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"
#include "onnx_light_cpu/kernels/com_microsoft/skip_layer_normalization_kernel.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/kernel_context.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace onnx_light_cpu::backend_test {
namespace {
namespace bt = ONNX_LIGHT_NAMESPACE::core::backend_test;
namespace rt = ONNX_LIGHT_NAMESPACE::core::runtime;

void RegisterCase(std::vector<TestCase> &registry, const std::string &tag, const rt::Shape &shape,
                  const rt::Shape &skip_shape, rt::DataType type, bool beta, bool bias,
                  bool residual, bool statistics) {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("SkipLayerNormalization");
  node.set_domain(kMicrosoftDomain);
  auto *epsilon = node.add_attribute();
  epsilon->set_name("epsilon");
  epsilon->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::FLOAT);
  epsilon->set_f(1.0e-5F);
  node.add_input("input");
  node.add_input("skip");
  node.add_input("gamma");
  node.add_input(beta ? "beta" : "");
  node.add_input(bias ? "bias" : "");
  node.add_output("output");
  if (statistics || residual) {
    node.add_output(statistics ? "mean" : "");
    node.add_output(statistics ? "inv_std_var" : "");
    node.add_output(residual ? "input_skip_bias_sum" : "");
  }

  std::int64_t count = 1;
  std::int64_t skip_count = 1;
  for (const auto dimension : shape) {
    count *= dimension;
  }
  for (const auto dimension : skip_shape) {
    skip_count *= dimension;
  }
  const auto width = shape.back();
  std::vector<std::int64_t> input_counts{count, skip_count, width};
  if (beta) {
    input_counts.push_back(width);
  }
  if (bias) {
    input_counts.push_back(width);
  }
  std::vector<std::int64_t> output_counts{count};
  const std::vector<std::int64_t> output_shape(shape.begin(), shape.end());
  std::vector<bt::TypeSpec> output_types{
      bt::TensorTypeSpec(static_cast<std::int32_t>(type), output_shape)};
  if (statistics) {
    auto statistics_shape = output_shape;
    statistics_shape.back() = 1;
    for (int slot = 0; slot < 2; ++slot) {
      output_counts.push_back(count / width);
      output_types.push_back(
          bt::TensorTypeSpec(static_cast<std::int32_t>(rt::DataType::FLOAT), statistics_shape));
    }
  }
  if (residual) {
    output_counts.push_back(count);
    output_types.push_back(bt::TensorTypeSpec(static_cast<std::int32_t>(type), output_shape));
  }

  const rt::OpsetId opset(kMicrosoftDomain, 1);
  const std::string name = "test_cpu_skip_layer_normalization_" + tag + "_" + DataTypeSuffix(type);
  bt::Expect(
      registry, node, name, {rt::DefaultOpset(23), opset}, input_counts, output_counts,
      [=](bool) -> bt::IoData {
        std::vector<rt::Tensor> inputs;
        inputs.push_back(MakeBenchmarkTensor(type, shape, 987654321ULL));
        inputs.push_back(MakeBenchmarkTensor(type, skip_shape, 135792468ULL));
        inputs.push_back(MakeBenchmarkTensor(type, {width}, 246813579ULL));
        if (beta) {
          inputs.push_back(MakeBenchmarkTensor(type, {width}, 112358132ULL));
        }
        if (bias) {
          inputs.push_back(MakeBenchmarkTensor(type, {width}, 314159265ULL));
        }
        std::size_t index = 3;
        const rt::Tensor *beta_tensor = beta ? &inputs[index++] : nullptr;
        const rt::Tensor *bias_tensor = bias ? &inputs[index] : nullptr;
        auto result = SkipLayerNormalizationKernel{rt::KernelContext{opset}}(
            inputs[0], inputs[1], inputs[2], beta_tensor, bias_tensor, 1.0e-5F, residual,
            statistics, statistics);
        std::vector<rt::Tensor> outputs;
        outputs.push_back(std::move(result.output));
        if (statistics) {
          outputs.push_back(std::move(result.mean));
          outputs.push_back(std::move(result.inv_std_var));
        }
        if (residual) {
          outputs.push_back(std::move(result.input_skip_bias_sum));
        }
        return bt::IoData{std::move(inputs), std::move(outputs)};
      },
      "backend-test", bt::TestCaseTag::AI_RT, output_types);
}
} // namespace

void RegisterCpuSkipLayerNormalizationCases(std::vector<TestCase> &registry, TestMode mode) {
  if (mode == TestMode::BENCHMARK) {
    for (const auto type : {rt::DataType::FLOAT, rt::DataType::FLOAT16}) {
      RegisterCase(registry, "decoder_1x128x4096", {1, 128, 4096}, {1, 128, 4096}, type, true, true,
                   true, false);
    }
    return;
  }
  for (const auto type : {rt::DataType::FLOAT, rt::DataType::FLOAT16, rt::DataType::BFLOAT16}) {
    RegisterCase(registry, "rank2", {3, 5}, {3, 5}, type, false, false, false, false);
    RegisterCase(registry, "rank3_beta_bias_outputs", {2, 3, 17}, {2, 3, 17}, type, true, true,
                 true, true);
    RegisterCase(registry, "broadcast_rank2_skip", {2, 3, 7}, {3, 7}, type, true, false, true,
                 false);
    RegisterCase(registry, "broadcast_rank3_skip", {2, 3, 7}, {1, 3, 7}, type, false, true, true,
                 false);
    RegisterCase(registry, "empty_rows", {0, 3, 5}, {0, 3, 5}, type, false, false, true, false);
  }
}

} // namespace onnx_light_cpu::backend_test
