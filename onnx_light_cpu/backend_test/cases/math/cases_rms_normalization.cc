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
  std::int64_t rows;
  std::int64_t width;
};

NodeProto MakeRmsNormalizationNode() {
  NodeProto node;
  node.set_op_type("RMSNormalization");
  node.add_input("X");
  node.add_input("scale");
  node.add_output("Y");
  auto *axis = node.add_attribute();
  axis->set_name("axis");
  axis->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::AttributeType::INT);
  axis->set_i(-1);
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
  const std::string name = "test_cpu_rmsnormalization_" + std::string(shape.name) + "_" +
                           DataTypeSuffix(data_type) + (benchmark ? "_benchmark" : "");
  const std::int64_t x_count = shape.rows * shape.width;
  const NodeProto node = MakeRmsNormalizationNode();
  Expect(registry, node, name, {opset}, {x_count, shape.width}, {x_count},
         [shape, data_type]() -> IoData {
           const RmsNormalizationKernel kernel{KernelContext{DefaultOpset(23)}};
           Tensor x = MakeBenchmarkTensor(data_type, {1, shape.rows, shape.width}, 611);
           Tensor scale = MakeBenchmarkTensor(data_type, {shape.width}, 612);
           Tensor y = kernel(x, scale, -1, 1.0e-6f, 1);
           return IoData{{std::move(x), std::move(scale)}, {std::move(y)}};
         });
}

} // namespace

void RegisterCpuRmsNormalizationCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(23);
  if (mode == TestMode::BENCHMARK) {
    for (const RmsNormalizationShape &shape :
         {RmsNormalizationShape{"llm_qwen3_8b_hidden4096_s1", 1, 4096},
          RmsNormalizationShape{"llm_qwen3_8b_hidden4096_s128", 128, 4096},
          RmsNormalizationShape{"llm_qwen3_8b_q_norm_hd128_s1_qh32", 32, 128},
          RmsNormalizationShape{"llm_qwen3_8b_q_norm_hd128_s128_qh32", 4096, 128},
          RmsNormalizationShape{"llm_qwen3_8b_k_norm_hd128_s1_kvh8", 8, 128},
          RmsNormalizationShape{"llm_qwen3_8b_k_norm_hd128_s128_kvh8", 1024, 128}}) {
      RegisterRmsNormalizationCase(registry, opset, shape, DataType::FLOAT16, true);
    }
    return;
  }
  RegisterRmsNormalizationCase(registry, opset, {"small", 2, 4}, DataType::FLOAT, false);
}

} // namespace onnx_light_cpu::backend_test
