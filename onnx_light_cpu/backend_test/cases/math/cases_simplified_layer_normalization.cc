// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/math/include_math_cases.h"

#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"
#include "onnx_light_cpu/backend_test/cases/math/simplified_layer_normalization_reference.h"

#include "onnx_proto/onnx_helper.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace onnx_light_cpu::backend_test {
namespace {

namespace bt = ONNX_LIGHT_NAMESPACE::core::backend_test;
using rt_ns::DataType;
using rt_ns::Shape;

struct CaseShape {
  const char *name;
  Shape x;
  Shape scale;
  std::int64_t axis;
};

void RegisterCase(std::vector<TestCase> &registry, const CaseShape &shape, DataType x_type,
                  DataType scale_type, std::int64_t stash_type, bool inverse, bool benchmark) {
  auto node = MakeNode("SimplifiedLayerNormalization", {"X", "scale"}, {"Y"});
  if (inverse) {
    node.add_output("inv_std_var");
  }
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "axis", shape.axis);
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "epsilon", 1.0e-6F);
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "stash_type", stash_type);
  const auto axis =
      shape.axis < 0 ? shape.axis + static_cast<std::int64_t>(shape.x.size()) : shape.axis;
  Shape stats_shape = shape.x;
  std::fill(stats_shape.begin() + axis, stats_shape.end(), 1);
  const std::string name = "test_cpu_simplified_layer_normalization_" + std::string(shape.name) +
                           "_x_" + DataTypeSuffix(x_type) + "_stash" + std::to_string(stash_type) +
                           (inverse ? "_with_inv_" : "_y_") + DataTypeSuffix(scale_type) +
                           (benchmark ? "_benchmark" : "");
  std::vector<std::int64_t> output_sizes{shape.x.product()};
  std::vector<bt::TypeSpec> output_types{
      bt::TensorTypeSpec(static_cast<std::int32_t>(scale_type), shape.x)};
  if (inverse) {
    output_sizes.push_back(stats_shape.product());
    output_types.push_back(bt::TensorTypeSpec(static_cast<std::int32_t>(stash_type), stats_shape));
  }
  auto build = [shape, x_type, scale_type, stash_type, inverse](bool expected) -> bt::IoData {
    auto x = MakeBenchmarkTensor(x_type, shape.x, 941);
    auto scale = MakeBenchmarkTensor(scale_type, shape.scale, 942, true);
    if (!expected) {
      return bt::IoData{{std::move(x), std::move(scale)}, {}, {}, false};
    }
    const rt_ns::KernelContext ctx{rt_ns::DefaultOpset(23)};
    auto outputs = ReferenceSimplifiedLayerNormalization(x, scale, shape.axis, 1.0e-6F, stash_type,
                                                         inverse, ctx);
    return bt::IoData{{std::move(x), std::move(scale)}, std::move(outputs)};
  };
  bt::Expect(registry, std::move(node), name, {rt_ns::DefaultOpset(23)},
             {shape.x.product(), shape.scale.product()}, output_sizes, std::move(build),
             "backend-test", bt::TestCaseTag::NONE, output_types);
  if (!benchmark) {
    registry.back().set_expected_outputs_generated(true);
  }
  registry.back().rtol = scale_type == DataType::BFLOAT16  ? 1.0e-2
                         : scale_type == DataType::FLOAT16 ? 2.0e-3
                                                           : 1.0e-5;
  registry.back().atol = scale_type == DataType::BFLOAT16  ? 1.0e-3
                         : scale_type == DataType::FLOAT16 ? 1.0e-4
                                                           : 1.0e-6;
}

} // namespace

void RegisterCpuSimplifiedLayerNormalizationCases(std::vector<TestCase> &registry, TestMode mode) {
  const std::vector<DataType> types{DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16,
                                    DataType::BFLOAT16};
  if (mode == TestMode::BENCHMARK) {
    for (const auto &shape : {
             CaseShape{"decode_w128", {1, 128}, {128}, -1},
             CaseShape{"small_r8_w32", {8, 32}, {32}, -1},
             CaseShape{"medium_r64_w512", {64, 512}, {512}, -1},
             CaseShape{"qwen_hidden1024_decode", {1, 1, 1024}, {1024}, -1},
             CaseShape{"qwen_hidden2048_prefill", {1, 128, 2048}, {2048}, -1},
             CaseShape{"qwen_hidden4096_decode", {1, 1, 4096}, {4096}, -1},
             CaseShape{"qwen_hidden4096_prefill", {1, 128, 4096}, {4096}, -1},
             CaseShape{"qwen_hidden5120_prefill", {1, 128, 5120}, {5120}, -1},
             CaseShape{"qwen_q_norm", {1, 4096, 128}, {128}, -1},
             CaseShape{"tail_r257_w65", {257, 65}, {65}, -1},
             CaseShape{"suffix_axis1", {2, 8, 16}, {8, 16}, 1},
             CaseShape{"outer_broadcast", {4, 64, 128}, {4, 1, 128}, -1},
         }) {
      for (const auto type : types) {
        for (bool inverse : {false, true}) {
          RegisterCase(registry, shape, type, type, 1, inverse, true);
        }
      }
    }
    return;
  }
  for (const auto &shape : {
           CaseShape{"width1", {2, 1}, {1}, -1},
           CaseShape{"tail7", {3, 7}, {7}, -1},
           CaseShape{"width8", {3, 8}, {8}, -1},
           CaseShape{"tail9", {3, 9}, {9}, -1},
           CaseShape{"tail31", {2, 31}, {31}, 1},
           CaseShape{"width32", {2, 32}, {32}, 1},
           CaseShape{"tail33", {2, 33}, {33}, -1},
           CaseShape{"axis0", {2, 3, 4}, {2, 3, 4}, 0},
           CaseShape{"suffix_axis1", {2, 3, 4}, {3, 4}, 1},
           CaseShape{"inner_broadcast", {2, 3, 4}, {4}, -2},
           CaseShape{"outer_broadcast", {2, 3, 4}, {2, 1, 4}, 1},
           CaseShape{"scalar_scale", {2, 3, 4}, {}, -1},
           CaseShape{"empty_outer", {0, 4}, {4}, -1},
       }) {
    for (const auto x_type : types) {
      for (const auto scale_type : types) {
        for (std::int64_t stash : {1, 11}) {
          for (bool inverse : {false, true}) {
            RegisterCase(registry, shape, x_type, scale_type, stash, inverse, false);
          }
        }
      }
    }
  }
}

} // namespace onnx_light_cpu::backend_test
