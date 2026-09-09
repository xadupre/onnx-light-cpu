// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/com_microsoft/include_com_microsoft_cases.h"
#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"
#include "onnx_light_cpu/kernels/com_microsoft/naive_skip_simplified_layer_normalization_kernel.h"
#include "onnx_light_cpu/kernels/com_microsoft/skip_simplified_layer_normalization_kernel.h"
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
                  const rt::Shape &skip_shape, rt::DataType type, bool bias, bool residual,
                  bool benchmark, bool statistics = false) {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("SkipSimplifiedLayerNormalization");
  node.set_domain(kMicrosoftDomain);
  auto *epsilon = node.add_attribute();
  epsilon->set_name("epsilon");
  epsilon->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::FLOAT);
  epsilon->set_f(1.0e-5F);
  node.add_input("input");
  node.add_input("skip");
  node.add_input("gamma");
  node.add_input(bias ? "bias" : "");
  node.add_output("output");
  if (residual) {
    node.add_output(statistics ? "mean" : "");
    node.add_output(statistics ? "inv_std_var" : "");
    node.add_output("input_skip_bias_sum");
  }
  std::int64_t count = 1;
  std::int64_t skip_count = 1;
  for (auto dimension : shape) {
    count *= dimension;
  }
  for (auto dimension : skip_shape) {
    skip_count *= dimension;
  }
  const auto width = shape.back();
  std::vector<std::int64_t> input_counts{count, skip_count, width};
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
    for (int slot = 1; slot < 3; ++slot) {
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
  const std::string name = "test_cpu_skip_simplified_layer_normalization_" + tag + "_" +
                           DataTypeSuffix(type) + (benchmark ? "_benchmark" : "");
  bt::Expect(
      registry, node, name, {rt::DefaultOpset(23), opset}, input_counts, output_counts,
      [=](bool generate_expected_outputs) -> bt::IoData {
        std::vector<rt::Tensor> inputs;
        inputs.push_back(MakeBenchmarkTensor(type, shape, 987654321ULL));
        inputs.push_back(MakeBenchmarkTensor(type, skip_shape, 135792468ULL));
        inputs.push_back(MakeBenchmarkTensor(type, {width}, 246813579ULL));
        if (bias) {
          inputs.push_back(MakeBenchmarkTensor(type, {width}, 314159265ULL));
        }
        if (!generate_expected_outputs) {
          return bt::IoData{std::move(inputs), {}, {}, false};
        }
        const auto *bias_tensor = bias ? &inputs[3] : nullptr;
        SkipSimplifiedLayerNormalizationResult result;
        if (benchmark) {
          result = SkipSimplifiedLayerNormalizationKernel{rt::KernelContext{opset}}(
              inputs[0], inputs[1], inputs[2], bias_tensor, 1.0e-5F, residual, statistics,
              statistics);
        } else {
          result = NaiveSkipSimplifiedLayerNormalizationKernel{rt::KernelContext{opset}}(
              inputs[0], inputs[1], inputs[2], bias_tensor, 1.0e-5F, residual, statistics,
              statistics);
        }
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

void RegisterCpuSkipSimplifiedLayerNormalizationCases(std::vector<TestCase> &registry,
                                                      TestMode mode) {
  for (auto type : {rt::DataType::FLOAT, rt::DataType::FLOAT16, rt::DataType::BFLOAT16}) {
    if (mode == TestMode::BENCHMARK) {
      RegisterCase(registry, "decode_1x1x4096", {1, 1, 4096}, {1, 1, 4096}, type, false, true,
                   true);
      RegisterCase(registry, "prefill_1x128x4096", {1, 128, 4096}, {1, 128, 4096}, type, true, true,
                   true);
    } else {
      RegisterCase(registry, "rank2", {3, 5}, {3, 5}, type, false, false, false);
      RegisterCase(registry, "rank3_bias_residual_statistics", {2, 3, 17}, {2, 3, 17}, type, true,
                   true, false, true);
      RegisterCase(registry, "broadcast_rank2_skip", {2, 3, 7}, {3, 7}, type, false, true, false);
      RegisterCase(registry, "broadcast_rank3_skip", {2, 3, 7}, {1, 3, 7}, type, true, true, false);
      RegisterCase(registry, "empty_rows", {0, 3, 5}, {0, 3, 5}, type, false, true, false);
    }
  }
}
} // namespace onnx_light_cpu::backend_test
