// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/math/include_math_cases.h"

#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"
#include "onnx_light_cpu/kernels/math/rms_normalization_kernel.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/kernel_context.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace onnx_light_cpu::backend_test {
namespace {

namespace bt_ns = ONNX_LIGHT_NAMESPACE::core::backend_test;
namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

using bt_ns::Expect;
using bt_ns::IoData;
using bt_ns::TestCase;
using bt_ns::TestMode;
using ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::DataType;
using rt_ns::DefaultOpset;
using rt_ns::KernelContext;
using rt_ns::OpsetId;
using rt_ns::Tensor;

struct RmsNormalizationShape {
  const char *name;
  rt_ns::Shape input_shape;
  rt_ns::Shape scale_shape;
  std::int64_t axis;
  std::uint64_t seed;
};

std::int64_t ElementCount(const rt_ns::Shape &shape) {
  std::int64_t count = 1;
  for (std::int64_t dimension : shape) {
    count *= dimension;
  }
  return count;
}

NodeProto MakeRmsNormalizationNode(std::int64_t normalized_axis) {
  NodeProto node;
  node.set_op_type("RMSNormalization");
  node.add_input("X");
  node.add_input("scale");
  node.add_output("Y");
  auto *axis = node.add_attribute();
  axis->set_name("axis");
  axis->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::AttributeType::INT);
  axis->set_i(normalized_axis);
  auto *epsilon = node.add_attribute();
  epsilon->set_name("epsilon");
  epsilon->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::AttributeType::FLOAT);
  epsilon->set_f(1.0e-6f);
  auto *stash_type = node.add_attribute();
  stash_type->set_name("stash_type");
  stash_type->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::AttributeType::INT);
  stash_type->set_i(1);
  return node;
}

void RegisterRmsNormalizationCase(std::vector<TestCase> &registry, const OpsetId &opset,
                                  const RmsNormalizationShape &shape, DataType data_type,
                                  bool benchmark) {
  const std::string name = "test_cpu_rms_normalization_" + std::string(shape.name) + "_" +
                           DataTypeSuffix(data_type) + (benchmark ? "_benchmark" : "");
  const std::int64_t x_count = ElementCount(shape.input_shape);
  const std::int64_t scale_count = ElementCount(shape.scale_shape);
  const NodeProto node = MakeRmsNormalizationNode(shape.axis);
  if (benchmark) {
    Expect(registry, node, name, {opset}, {x_count, scale_count}, {x_count},
           [shape, data_type](bool generate_expected_outputs) -> IoData {
             Tensor x = MakeBenchmarkTensor(data_type, shape.input_shape, shape.seed);
             Tensor scale = MakeBenchmarkTensor(data_type, shape.scale_shape, shape.seed + 1);
             if (!generate_expected_outputs) {
               return IoData{{std::move(x), std::move(scale)}, {}, {}, false};
             }
             const RmsNormalizationKernel kernel{KernelContext{DefaultOpset(23)}};
             Tensor y = kernel(x, scale, shape.axis, 1.0e-6f, 1);
             return IoData{{std::move(x), std::move(scale)}, {std::move(y)}};
           },
           "backend-test", bt_ns::TestCaseTag::NONE,
           {bt_ns::TensorTypeSpec(static_cast<std::int32_t>(data_type), shape.input_shape)});
  } else {
    Expect(registry, node, name, {opset}, {x_count, scale_count}, {x_count},
           [shape, data_type]() -> IoData {
             const RmsNormalizationKernel kernel{KernelContext{DefaultOpset(23)}};
             Tensor x = MakeBenchmarkTensor(data_type, shape.input_shape, shape.seed);
             Tensor scale = MakeBenchmarkTensor(data_type, shape.scale_shape, shape.seed + 1);
             Tensor y = kernel(x, scale, shape.axis, 1.0e-6f, 1);
             return IoData{{std::move(x), std::move(scale)}, {std::move(y)}};
           });
  }
  if (data_type == DataType::FLOAT16 || data_type == DataType::BFLOAT16) {
    registry.back().rtol = 2.0e-2;
    registry.back().atol = 2.0e-2;
  }
}

} // namespace

void RegisterCpuRmsNormalizationCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(23);
  if (mode == TestMode::BENCHMARK) {
    for (const RmsNormalizationShape &shape :
         {RmsNormalizationShape{"small_r8_w32_axis1", {8, 32}, {32}, -1, 611},
          RmsNormalizationShape{"r64_w512", {1, 64, 512}, {512}, -1, 616},
          RmsNormalizationShape{"wide_r1_w4096_axis1", {1, 4096}, {4096}, -1, 621},
          RmsNormalizationShape{"llm_r128_w4096_axis1", {128, 4096}, {4096}, -1, 631},
          RmsNormalizationShape{"tall_r4096_w128_axis1", {4096, 128}, {128}, -1, 641},
          RmsNormalizationShape{"rank3_n4_r16_w512_axis2", {4, 16, 512}, {512}, 2, 651},
          RmsNormalizationShape{"rank3_n2_h8_w16_axis1", {2, 8, 16}, {8, 16}, 1, 661},
          RmsNormalizationShape{"rank4_n2_c4_h8_w16_axis2", {2, 4, 8, 16}, {8, 16}, 2, 671}}) {
      RegisterRmsNormalizationCase(registry, opset, shape, DataType::FLOAT, true);
    }
    for (const RmsNormalizationShape &shape :
         {RmsNormalizationShape{"llm_qwen3_8b_hidden4096_s1", {1, 1, 4096}, {4096}, -1, 681},
          RmsNormalizationShape{"llm_qwen3_8b_hidden4096_s128", {1, 128, 4096}, {4096}, -1, 691},
          RmsNormalizationShape{"llm_qwen3_8b_q_norm_hd128_s1_qh32", {1, 32, 128}, {128}, -1, 701},
          RmsNormalizationShape{
              "llm_qwen3_8b_q_norm_hd128_s128_qh32", {1, 4096, 128}, {128}, -1, 711},
          RmsNormalizationShape{"llm_qwen3_8b_k_norm_hd128_s1_kvh8", {1, 8, 128}, {128}, -1, 721},
          RmsNormalizationShape{
              "llm_qwen3_8b_k_norm_hd128_s128_kvh8", {1, 1024, 128}, {128}, -1, 731}}) {
      RegisterRmsNormalizationCase(registry, opset, shape, DataType::FLOAT16, true);
    }
    RegisterRmsNormalizationCase(registry, opset, {"r64_w512", {1, 64, 512}, {512}, -1, 741},
                                 DataType::BFLOAT16, true);
    for (const RmsNormalizationShape &shape :
         {RmsNormalizationShape{"llm_qwen3_6_27b_hidden5120_s1", {1, 1, 5120}, {5120}, -1, 751},
          RmsNormalizationShape{"llm_qwen3_6_27b_hidden5120_s128", {1, 128, 5120}, {5120}, -1, 761},
          RmsNormalizationShape{"llm_qwen3_6_27b_hidden5120_s512", {1, 512, 5120}, {5120}, -1, 771},
          RmsNormalizationShape{
              "llm_qwen3_6_27b_q_norm_hd256_s1_qh24", {1, 24, 256}, {256}, -1, 781},
          RmsNormalizationShape{
              "llm_qwen3_6_27b_q_norm_hd256_s128_qh24", {1, 3072, 256}, {256}, -1, 791},
          RmsNormalizationShape{
              "llm_qwen3_6_27b_k_norm_hd256_s1_kvh4", {1, 4, 256}, {256}, -1, 801},
          RmsNormalizationShape{
              "llm_qwen3_6_27b_k_norm_hd256_s128_kvh4", {1, 512, 256}, {256}, -1, 811},
          RmsNormalizationShape{"llm_qwen3_6_35b_a3b_hidden2048_s1", {1, 1, 2048}, {2048}, -1, 821},
          RmsNormalizationShape{
              "llm_qwen3_6_35b_a3b_hidden2048_s128", {1, 128, 2048}, {2048}, -1, 831},
          RmsNormalizationShape{
              "llm_qwen3_6_35b_a3b_hidden2048_s512", {1, 512, 2048}, {2048}, -1, 841},
          RmsNormalizationShape{
              "llm_qwen3_6_35b_a3b_q_norm_hd256_s1_qh16", {1, 16, 256}, {256}, -1, 851},
          RmsNormalizationShape{
              "llm_qwen3_6_35b_a3b_q_norm_hd256_s128_qh16", {1, 2048, 256}, {256}, -1, 861},
          RmsNormalizationShape{
              "llm_qwen3_6_35b_a3b_k_norm_hd256_s1_kvh2", {1, 2, 256}, {256}, -1, 871},
          RmsNormalizationShape{
              "llm_qwen3_6_35b_a3b_k_norm_hd256_s128_kvh2", {1, 256, 256}, {256}, -1, 881}}) {
      RegisterRmsNormalizationCase(registry, opset, shape, DataType::BFLOAT16, true);
    }
    return;
  }
  for (DataType data_type : {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16}) {
    RegisterRmsNormalizationCase(registry, opset, {"small", {1, 2, 4}, {4}, -1, 611}, data_type,
                                 false);
  }

  for (const std::pair<DataType, DataType> &types :
       {std::pair{DataType::FLOAT16, DataType::FLOAT},
        std::pair{DataType::FLOAT, DataType::FLOAT16}}) {
    NodeProto node = MakeRmsNormalizationNode(-1);
    const std::string name = "test_cpu_rms_normalization_mixed_x_" +
                             std::string(DataTypeSuffix(types.first)) + "_scale_" +
                             DataTypeSuffix(types.second);
    Expect(registry, std::move(node), name, {opset}, [types]() -> IoData {
      const RmsNormalizationKernel kernel{KernelContext{DefaultOpset(23)}};
      Tensor x = MakeBenchmarkTensor(types.first, {1, 2, 4}, 751);
      Tensor scale = MakeBenchmarkTensor(types.second, {4}, 752);
      Tensor y = kernel(x, scale, -1, 1.0e-6F, 1);
      return IoData{{std::move(x), std::move(scale)}, {std::move(y)}};
    });
    registry.back().rtol = 2.0e-2;
    registry.back().atol = 2.0e-2;
  }
}

} // namespace onnx_light_cpu::backend_test
