// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/tensor/include_tensor_cases.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/kernels/kernels/logical/include_logical_kernels.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace onnx_light_cpu::backend_test {
namespace {

namespace bt_ns = ONNX_LIGHT_NAMESPACE::core::backend_test;
namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace reference_ns = ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel;
using rt_ns::DataType;
using rt_ns::Shape;
using rt_ns::Tensor;

void RegisterNonZeroCase(std::vector<TestCase> &registry, const std::string &layout,
                         const Shape &shape, int pattern, bool benchmark) {
  const auto opset = rt_ns::DefaultOpset(13);
  const int64_t total = shape.product();
  const int64_t count = pattern == 0 ? 0 : (pattern == 1 ? total : (total + 2) / 3);
  const Shape output_shape{static_cast<int64_t>(shape.size()), count};
  const std::string name = "test_cpu_nonzero_" + layout + "_" +
                           (pattern == 0 ? "zero" : (pattern == 1 ? "all" : "mixed")) + "_bool" +
                           (benchmark ? "_benchmark" : "");
  const auto build = [shape, total, pattern, opset](bool expected) -> bt_ns::IoData {
    std::vector<uint8_t> values(static_cast<std::size_t>(total));
    for (int64_t i = 0; i < total; ++i) {
      values[i] = pattern == 1 || (pattern == 2 && i % 3 == 0);
    }
    Tensor input = Tensor::FromBool("X", shape, values);
    if (!expected) {
      return bt_ns::IoData{{std::move(input)}, {}, {}, false};
    }
    const reference_ns::NonZero reference{rt_ns::KernelContext{opset}};
    Tensor output = reference(input);
    return bt_ns::IoData{{std::move(input)}, {std::move(output)}};
  };
  bt_ns::Expect(registry, ONNX_LIGHT_NAMESPACE::MakeNode("NonZero", {"X"}, {"Y"}), name, {opset},
                {total}, {output_shape.product()}, build, "backend-test", bt_ns::TestCaseTag::NONE,
                {bt_ns::TensorTypeSpec(DataType::INT64, output_shape)});
}

void RegisterTokenPositionsCase(std::vector<TestCase> &registry) {
  const std::string name = "test_cpu_nonzero_multimodal_token_positions_bool";
  TestCase test_case(name, name, bt_ns::TestCaseKind::MODEL);
  test_case.declared_input_element_counts = {16, 1};
  test_case.declared_output_element_counts = {32};
  auto build = [name](bool expected) {
    const auto opset = rt_ns::DefaultOpset(13);
    bt_ns::BuiltCase built;
    bt_ns::InitModel(built.model, 10, {opset});
    auto *graph = built.model.add_graph();
    graph->set_name(name);
    ONNX_LIGHT_NAMESPACE::AddNode(*graph, "Equal", {"input_ids", "image_token_id"}, {"mask"});
    ONNX_LIGHT_NAMESPACE::AddNode(*graph, "NonZero", {"mask"}, {"indices"});
    ONNX_LIGHT_NAMESPACE::AddNode(*graph, "Transpose", {"indices"}, {"positions"});
    bt_ns::AppendValueInfo(*graph->add_input(), "input_ids", DataType::INT64,
                           {bt_ns::DimSpec("batch"), bt_ns::DimSpec("sequence")});
    bt_ns::AppendValueInfo(*graph->add_input(), "image_token_id", DataType::INT64, {});
    bt_ns::AppendValueInfo(*graph->add_output(), "positions", DataType::INT64,
                           {bt_ns::DimSpec("count"), bt_ns::DimSpec(int64_t{2})});
    // Six and ten placeholders cover rectangular and multiple-image counts.
    for (int64_t count : {0, 1, 6, 10, 16, 0}) {
      std::vector<int64_t> ids(16, 7);
      for (int64_t i = 0; i < count; ++i) {
        ids[(i * 7 + 3) % 16] = 99;
      }
      Tensor input = Tensor::FromInt64("input_ids", {2, 8}, ids);
      Tensor token = Tensor::FromInt64("image_token_id", {}, {99});
      rt_ns::Tensors outputs;
      if (expected) {
        const rt_ns::KernelContext ctx{opset};
        const reference_ns::Equal equal{ctx};
        const reference_ns::NonZero nonzero{ctx};
        const reference_ns::Transpose transpose{ctx};
        outputs.push_back(transpose(nonzero(equal(input, token)), {1, 0}));
        outputs.back().name = "positions";
      }
      built.data_sets.push_back(
          bt_ns::DataSet{{std::move(input), std::move(token)}, std::move(outputs), expected});
    }
    return built;
  };
  test_case.build = std::move(build);
  registry.push_back(std::move(test_case));
}

} // namespace

void RegisterCpuNonZeroCases(std::vector<TestCase> &registry, TestMode mode) {
  if (mode == TestMode::BENCHMARK) {
    for (int pattern : {0, 1, 2}) {
      RegisterNonZeroCase(registry, "tokens", {4, 4096}, pattern, true);
    }
    return;
  }
  const std::vector<std::pair<std::string, Shape>> shapes = {
      {"scalar", {}},         {"vector", {7}},         {"matrix", {2, 5}},
      {"rank3", {2, 3, 4}},   {"empty_first", {0, 3}}, {"empty_middle", {2, 0, 4}},
      {"empty_last", {2, 0}},
  };
  for (const auto &[name, shape] : shapes) {
    for (int pattern : {0, 1, 2}) {
      RegisterNonZeroCase(registry, name, shape, pattern, false);
    }
  }
  RegisterTokenPositionsCase(registry);
}

} // namespace onnx_light_cpu::backend_test
