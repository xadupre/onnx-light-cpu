// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/tensor/include_tensor_cases.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/kernels/kernels/logical/include_logical_kernels.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace onnx_light_cpu::backend_test {
namespace {

namespace bt_ns = ONNX_LIGHT_NAMESPACE::core::backend_test;
namespace reference_ns = ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel;
using rt_ns::DataType;
using rt_ns::Shape;
using rt_ns::Tensor;

struct ScatterShape {
  const char *name;
  Shape data;
  Shape indices;
  std::vector<int64_t> values;
};

void RegisterScatterCase(std::vector<TestCase> &registry, const ScatterShape &shape,
                         DataType data_type, DataType index_type, bool benchmark) {
  const auto opset = rt_ns::DefaultOpset(13);
  Shape updates_shape;
  updates_shape.insert(updates_shape.end(), shape.indices.begin(), shape.indices.end() - 1);
  updates_shape.insert(updates_shape.end(), shape.data.begin() + shape.indices.back(),
                       shape.data.end());
  const std::string name = "test_cpu_scatternd_" + std::string(shape.name) +
                           (index_type == DataType::INT32 ? "_indices32_" : "_indices64_") +
                           DataTypeSuffix(data_type) + (benchmark ? "_benchmark" : "");
  const auto build = [shape, updates_shape, data_type, index_type,
                      opset](bool expected) -> bt_ns::IoData {
    Tensor data = MakeBenchmarkTensor(data_type, shape.data, 719);
    Tensor updates = MakeBenchmarkTensor(data_type, updates_shape, 911);
    Tensor indices =
        index_type == DataType::INT32
            ? Tensor::FromInt32("indices", shape.indices,
                                std::vector<int32_t>(shape.values.begin(), shape.values.end()))
            : Tensor::FromInt64("indices", shape.indices, shape.values);
    if (!expected) {
      return bt_ns::IoData{
          {std::move(data), std::move(indices), std::move(updates)}, {}, {}, false};
    }
    // The built-in oracle accepts only INT64 indices, unlike our byte-copy kernel.
    Tensor oracle_indices = Tensor::FromInt64("indices", shape.indices, shape.values);
    const rt_ns::KernelContext ctx{opset};
    const reference_ns::ScatterND scatter{ctx};
    Tensor output = scatter(data, oracle_indices, updates, {});
    return bt_ns::IoData{{std::move(data), std::move(indices), std::move(updates)},
                         {std::move(output)}};
  };
  bt_ns::Expect(registry, MakeNode("ScatterND", {"data", "indices", "updates"}, {"output"}), name,
                {opset}, {shape.data.product(), shape.indices.product(), updates_shape.product()},
                {shape.data.product()}, build, "backend-test", bt_ns::TestCaseTag::NONE,
                {bt_ns::TensorTypeSpec(data_type, shape.data)});
}

void RegisterEmbeddingCase(std::vector<TestCase> &registry, DataType data_type) {
  const std::string name =
      "test_cpu_scatternd_multimodal_embeddings_" + std::string(DataTypeSuffix(data_type));
  TestCase test_case(name, name, bt_ns::TestCaseKind::MODEL);
  test_case.declared_input_element_counts = {100 * 6656, 16, 1, 6 * 6656};
  test_case.declared_output_element_counts = {16 * 6656};
  const auto build = [name, data_type](bool expected) {
    const auto opset = rt_ns::DefaultOpset(13);
    bt_ns::BuiltCase built;
    bt_ns::InitModel(built.model, 10, {opset});
    auto *graph = built.model.add_graph();
    graph->set_name(name);
    ONNX_LIGHT_NAMESPACE::AddNode(*graph, "Gather", {"embedding_table", "input_ids"},
                                  {"embeddings"});
    ONNX_LIGHT_NAMESPACE::AddNode(*graph, "Equal", {"input_ids", "image_token_id"}, {"mask"});
    ONNX_LIGHT_NAMESPACE::AddNode(*graph, "NonZero", {"mask"}, {"indices"});
    ONNX_LIGHT_NAMESPACE::AddNode(*graph, "Transpose", {"indices"}, {"positions"});
    ONNX_LIGHT_NAMESPACE::AddNode(*graph, "ScatterND", {"embeddings", "positions", "visual"},
                                  {"output"});
    bt_ns::AppendValueInfo(*graph->add_input(), "embedding_table", data_type,
                           {bt_ns::DimSpec(int64_t{100}), bt_ns::DimSpec(int64_t{6656})});
    bt_ns::AppendValueInfo(*graph->add_input(), "input_ids", DataType::INT64,
                           {bt_ns::DimSpec("tokens")});
    bt_ns::AppendValueInfo(*graph->add_input(), "image_token_id", DataType::INT64, {});
    bt_ns::AppendValueInfo(*graph->add_input(), "visual", data_type,
                           {bt_ns::DimSpec("visual_count"), bt_ns::DimSpec(int64_t{6656})});
    bt_ns::AppendValueInfo(*graph->add_output(), "output", data_type,
                           {bt_ns::DimSpec("tokens"), bt_ns::DimSpec(int64_t{6656})});
    // Delimiters retain their text embeddings; only the three patch tokens per
    // image are replaced by visual embeddings.
    for (const int64_t images : {0, 1, 2, 0}) {
      const int64_t count = images * 3;
      Tensor table = MakeBenchmarkTensor(data_type, {100, 6656}, 731);
      table.name = "embedding_table";
      std::vector<int64_t> ids(16, 7);
      for (int64_t image = 0; image < images; ++image) {
        const int64_t start = 1 + image * 7;
        ids[start] = 97;
        for (int64_t patch = 1; patch <= 3; ++patch) {
          ids[start + patch] = 99;
        }
        ids[start + 4] = 98;
      }
      Tensor input = Tensor::FromInt64("input_ids", {16}, ids);
      Tensor token = Tensor::FromInt64("image_token_id", {}, {99});
      Tensor visual = MakeBenchmarkTensor(data_type, {count, 6656}, 947);
      visual.name = "visual";
      rt_ns::Tensors outputs;
      if (expected) {
        const rt_ns::KernelContext ctx{opset};
        const reference_ns::Gather gather{ctx};
        const reference_ns::Equal equal{ctx};
        const reference_ns::NonZero nonzero{ctx};
        const reference_ns::Transpose transpose{ctx};
        const reference_ns::ScatterND scatter{ctx};
        Tensor embeddings = gather(table, input, 0);
        Tensor positions = transpose(nonzero(equal(input, token)), {1, 0});
        outputs.push_back(scatter(embeddings, positions, visual, {}));
        outputs.back().name = "output";
      }
      built.data_sets.push_back(
          bt_ns::DataSet{{std::move(table), std::move(input), std::move(token), std::move(visual)},
                         std::move(outputs),
                         expected});
    }
    return built;
  };
  test_case.build = build;
  registry.push_back(std::move(test_case));
}

} // namespace

void RegisterCpuScatterNDCases(std::vector<TestCase> &registry, TestMode mode) {
  const bool benchmark = mode == TestMode::BENCHMARK;
  const std::vector<ScatterShape> shapes =
      benchmark ? std::vector<ScatterShape>{
                      {"small_rows", {32, 64}, {4, 1}, {0, 7, 15, 31}},
                      {"hidden6656_one_row", {128, 6656}, {1, 1}, {63}},
                      {"hidden6656_rows", {128, 6656}, {8, 1}, {0, 3, 17, 31, 48, 63, 99, 127}},
                  }
                : std::vector<ScatterShape>{
                      {"scalars", {2, 3}, {3, 2}, {0, 0, 1, 2, 0, 1}},
                      {"scalar_update", {2, 3}, {2}, {1, 2}},
                      {"slices", {3, 2, 4}, {2, 1}, {0, 2}},
                      {"negative", {2, 3}, {2, 2}, {-2, -1, -1, -3}},
                      {"tuple_grid", {3, 4, 2}, {2, 2, 2}, {0, 0, 0, 3, 2, 1, 1, 2}},
                      {"empty_updates", {3, 4}, {0, 1}, {}},
                      {"empty_data", {0, 4}, {0, 1}, {}},
                      {"empty_slices", {3, 0}, {2, 1}, {0, 2}},
                      {"hidden6656", {2, 8, 6656}, {3, 2}, {0, 1, 1, 0, 1, 7}},
                  };
  for (const DataType data_type : {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16}) {
    for (const DataType index_type : {DataType::INT32, DataType::INT64}) {
      for (const auto &shape : shapes) {
        RegisterScatterCase(registry, shape, data_type, index_type, benchmark);
      }
    }
    if (!benchmark) {
      RegisterEmbeddingCase(registry, data_type);
    }
  }
}

} // namespace onnx_light_cpu::backend_test
