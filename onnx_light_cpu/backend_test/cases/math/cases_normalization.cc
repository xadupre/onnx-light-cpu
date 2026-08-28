// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/math/include_math_cases.h"

#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"
#include "onnx_light_cpu/kernels/math/normalization_kernel.h"

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
using rt_ns::Shape;
using rt_ns::Tensor;

void AddIntAttribute(NodeProto &node, const char *name, std::int64_t value) {
  auto *attribute = node.add_attribute();
  attribute->set_name(name);
  attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::INT);
  attribute->set_i(value);
}

void AddFloatAttribute(NodeProto &node, const char *name, float value) {
  auto *attribute = node.add_attribute();
  attribute->set_name(name);
  attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::FLOAT);
  attribute->set_f(value);
}

void AddIntsAttribute(NodeProto &node, const char *name, const std::vector<std::int64_t> &values) {
  auto *attribute = node.add_attribute();
  attribute->set_name(name);
  attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::INTS);
  for (std::int64_t value : values) {
    attribute->add_ints(value);
  }
}

NodeProto MakeNode(const char *op_type, std::initializer_list<const char *> inputs,
                   std::initializer_list<const char *> outputs) {
  NodeProto node;
  node.set_op_type(op_type);
  for (const char *input : inputs) {
    node.add_input(input);
  }
  for (const char *output : outputs) {
    node.add_output(output);
  }
  return node;
}

std::int64_t ElementCount(const Shape &shape) {
  std::int64_t count = 1;
  for (std::int64_t dimension : shape) {
    count *= dimension;
  }
  return count;
}

Tensor MakeTypedTensor(DataType data_type, const Shape &shape, const std::vector<float> &values) {
  switch (data_type) {
  case DataType::FLOAT:
    return Tensor::FromFloat("", shape, values);
  case DataType::DOUBLE:
    return Tensor::FromDouble("", shape, std::vector<double>(values.begin(), values.end()));
  case DataType::FLOAT16:
    return rt_ns::MakeFloat16Tensor("", shape, values);
  case DataType::BFLOAT16:
    return rt_ns::MakeBfloat16Tensor("", shape, values);
  default:
    throw std::invalid_argument("normalization case requires a floating-point data type");
  }
}

Tensor MakeConstantTensor(DataType data_type, const Shape &shape, float value) {
  return MakeTypedTensor(data_type, shape,
                         std::vector<float>(static_cast<std::size_t>(ElementCount(shape)), value));
}

void SetNormalizationTolerance(std::vector<TestCase> &registry, DataType data_type) {
  if (data_type == DataType::FLOAT16 || data_type == DataType::BFLOAT16) {
    registry.back().rtol = 2.0e-2;
    registry.back().atol = 2.0e-2;
  }
}

struct BatchBenchmark {
  const char *name;
  Shape shape;
  bool training;
  DataType data_type;
  std::uint64_t seed;
};

void RegisterBatchBenchmark(std::vector<TestCase> &registry, const OpsetId &opset,
                            const BatchBenchmark &benchmark) {
  const std::int64_t count = ElementCount(benchmark.shape);
  const std::int64_t channels = benchmark.shape[1];
  NodeProto node = MakeNode(
      "BatchNormalization", {"X", "scale", "B", "mean", "variance"},
      benchmark.training ? std::initializer_list<const char *>{"Y", "running_mean", "running_var"}
                         : std::initializer_list<const char *>{"Y"});
  if (benchmark.training) {
    AddIntAttribute(node, "training_mode", 1);
  }
  const std::string name = "test_cpu_batchnormalization_" + std::string(benchmark.name) + "_" +
                           DataTypeSuffix(benchmark.data_type) + "_benchmark";
  const std::vector<std::int64_t> output_counts =
      benchmark.training ? std::vector<std::int64_t>{count, channels, channels}
                         : std::vector<std::int64_t>{count};
  Expect(registry, std::move(node), name, {opset}, {count, channels, channels, channels, channels},
         output_counts, [benchmark, channels]() -> IoData {
           const BatchNormalizationKernel kernel{KernelContext{DefaultOpset(15)}};
           Tensor x = MakeBenchmarkTensor(benchmark.data_type, benchmark.shape, benchmark.seed);
           Tensor scale = MakeBenchmarkTensor(benchmark.data_type, {channels}, benchmark.seed + 1);
           Tensor bias = MakeBenchmarkTensor(benchmark.data_type, {channels}, benchmark.seed + 2);
           Tensor mean = MakeBenchmarkTensor(benchmark.data_type, {channels}, benchmark.seed + 3);
           Tensor variance = MakeConstantTensor(benchmark.data_type, {channels}, 1.0F);
           if (benchmark.training) {
             BatchNormalizationResult result = kernel.Compute(x, scale, bias, mean, variance, true);
             return IoData{{std::move(x), std::move(scale), std::move(bias), std::move(mean),
                            std::move(variance)},
                           {std::move(result.y), std::move(*result.running_mean),
                            std::move(*result.running_variance)}};
           }
           Tensor y = kernel(x, scale, bias, mean, variance);
           return IoData{{std::move(x), std::move(scale), std::move(bias), std::move(mean),
                          std::move(variance)},
                         {std::move(y)}};
         });
  SetNormalizationTolerance(registry, benchmark.data_type);
}

struct GroupBenchmark {
  const char *name;
  Shape shape;
  std::int64_t groups;
  DataType data_type;
  std::uint64_t seed;
};

void RegisterGroupBenchmark(std::vector<TestCase> &registry, const OpsetId &opset,
                            const GroupBenchmark &benchmark) {
  const std::int64_t count = ElementCount(benchmark.shape);
  const std::int64_t channels = benchmark.shape[1];
  NodeProto node = MakeNode("GroupNormalization", {"X", "scale", "bias"}, {"Y"});
  AddIntAttribute(node, "num_groups", benchmark.groups);
  const std::string name = "test_cpu_groupnormalization_" + std::string(benchmark.name) + "_" +
                           DataTypeSuffix(benchmark.data_type) + "_benchmark";
  Expect(registry, std::move(node), name, {opset}, {count, channels, channels}, {count},
         [benchmark, channels]() -> IoData {
           const GroupNormalizationKernel kernel{KernelContext{DefaultOpset(21)}};
           Tensor x = MakeBenchmarkTensor(benchmark.data_type, benchmark.shape, benchmark.seed);
           Tensor scale = MakeBenchmarkTensor(benchmark.data_type, {channels}, benchmark.seed + 1);
           Tensor bias = MakeBenchmarkTensor(benchmark.data_type, {channels}, benchmark.seed + 2);
           Tensor y = kernel(x, scale, bias, benchmark.groups);
           return IoData{{std::move(x), std::move(scale), std::move(bias)}, {std::move(y)}};
         });
  SetNormalizationTolerance(registry, benchmark.data_type);
}

struct InstanceBenchmark {
  const char *name;
  Shape shape;
  DataType data_type;
  std::uint64_t seed;
};

void RegisterInstanceBenchmark(std::vector<TestCase> &registry, const OpsetId &opset,
                               const InstanceBenchmark &benchmark) {
  const std::int64_t count = ElementCount(benchmark.shape);
  const std::int64_t channels = benchmark.shape[1];
  NodeProto node = MakeNode("InstanceNormalization", {"X", "scale", "B"}, {"Y"});
  const std::string name = "test_cpu_instancenormalization_" + std::string(benchmark.name) + "_" +
                           DataTypeSuffix(benchmark.data_type) + "_benchmark";
  Expect(registry, std::move(node), name, {opset}, {count, channels, channels}, {count},
         [benchmark, channels]() -> IoData {
           const InstanceNormalizationKernel kernel{KernelContext{DefaultOpset(22)}};
           Tensor x = MakeBenchmarkTensor(benchmark.data_type, benchmark.shape, benchmark.seed);
           Tensor scale = MakeBenchmarkTensor(benchmark.data_type, {channels}, benchmark.seed + 1);
           Tensor bias = MakeBenchmarkTensor(benchmark.data_type, {channels}, benchmark.seed + 2);
           Tensor y = kernel(x, scale, bias);
           return IoData{{std::move(x), std::move(scale), std::move(bias)}, {std::move(y)}};
         });
  SetNormalizationTolerance(registry, benchmark.data_type);
}

struct LayerBenchmark {
  const char *name;
  Shape shape;
  Shape affine_shape;
  std::int64_t axis;
  bool bias;
  DataType data_type;
  std::uint64_t seed;
};

void RegisterLayerBenchmark(std::vector<TestCase> &registry, const OpsetId &opset,
                            const LayerBenchmark &benchmark) {
  const std::int64_t count = ElementCount(benchmark.shape);
  const std::int64_t affine_count = ElementCount(benchmark.affine_shape);
  NodeProto node = benchmark.bias ? MakeNode("LayerNormalization", {"X", "Scale", "B"}, {"Y"})
                                  : MakeNode("LayerNormalization", {"X", "Scale"}, {"Y"});
  AddIntAttribute(node, "axis", benchmark.axis);
  const std::string name = "test_cpu_layernormalization_" + std::string(benchmark.name) + "_" +
                           DataTypeSuffix(benchmark.data_type) + "_benchmark";
  const std::vector<std::int64_t> input_counts =
      benchmark.bias ? std::vector<std::int64_t>{count, affine_count, affine_count}
                     : std::vector<std::int64_t>{count, affine_count};
  Expect(registry, std::move(node), name, {opset}, input_counts, {count}, [benchmark]() -> IoData {
    const LayerNormalizationKernel kernel{KernelContext{DefaultOpset(17)}};
    Tensor x = MakeBenchmarkTensor(benchmark.data_type, benchmark.shape, benchmark.seed);
    Tensor scale =
        MakeBenchmarkTensor(benchmark.data_type, benchmark.affine_shape, benchmark.seed + 1);
    if (benchmark.bias) {
      Tensor bias =
          MakeBenchmarkTensor(benchmark.data_type, benchmark.affine_shape, benchmark.seed + 2);
      LayerNormalizationResult result = kernel(x, scale, &bias, benchmark.axis);
      return IoData{{std::move(x), std::move(scale), std::move(bias)}, {std::move(result.y)}};
    }
    LayerNormalizationResult result = kernel(x, scale, nullptr, benchmark.axis);
    return IoData{{std::move(x), std::move(scale)}, {std::move(result.y)}};
  });
  SetNormalizationTolerance(registry, benchmark.data_type);
}

struct LpBenchmark {
  const char *name;
  Shape shape;
  std::int64_t axis;
  std::int64_t p;
  DataType data_type;
  std::uint64_t seed;
};

void RegisterLpBenchmark(std::vector<TestCase> &registry, const OpsetId &opset,
                         const LpBenchmark &benchmark) {
  const std::int64_t count = ElementCount(benchmark.shape);
  NodeProto node = MakeNode("LpNormalization", {"X"}, {"Y"});
  AddIntAttribute(node, "axis", benchmark.axis);
  AddIntAttribute(node, "p", benchmark.p);
  const std::string name = "test_cpu_lpnormalization_" + std::string(benchmark.name) + "_" +
                           DataTypeSuffix(benchmark.data_type) + "_benchmark";
  Expect(registry, std::move(node), name, {opset}, {count}, {count}, [benchmark]() -> IoData {
    const LpNormalizationKernel kernel{KernelContext{DefaultOpset(22)}};
    Tensor x = MakeBenchmarkTensor(benchmark.data_type, benchmark.shape, benchmark.seed);
    Tensor y = kernel(x, benchmark.axis, benchmark.p);
    return IoData{{std::move(x)}, {std::move(y)}};
  });
  SetNormalizationTolerance(registry, benchmark.data_type);
}

struct MvnBenchmark {
  const char *name;
  Shape shape;
  std::vector<std::int64_t> axes;
  bool default_axes;
  DataType data_type;
  std::uint64_t seed;
};

void RegisterMvnBenchmark(std::vector<TestCase> &registry, const OpsetId &opset,
                          const MvnBenchmark &benchmark) {
  const std::int64_t count = ElementCount(benchmark.shape);
  NodeProto node = MakeNode("MeanVarianceNormalization", {"X"}, {"Y"});
  if (!benchmark.default_axes) {
    AddIntsAttribute(node, "axes", benchmark.axes);
  }
  const std::string name = "test_cpu_meanvariancenormalization_" + std::string(benchmark.name) +
                           "_" + DataTypeSuffix(benchmark.data_type) + "_benchmark";
  Expect(registry, std::move(node), name, {opset}, {count}, {count}, [benchmark]() -> IoData {
    const MeanVarianceNormalizationKernel kernel{KernelContext{DefaultOpset(13)}};
    Tensor x = MakeBenchmarkTensor(benchmark.data_type, benchmark.shape, benchmark.seed);
    Tensor y = kernel(x, benchmark.axes);
    return IoData{{std::move(x)}, {std::move(y)}};
  });
  SetNormalizationTolerance(registry, benchmark.data_type);
}

} // namespace

void RegisterCpuBatchNormalizationCases(std::vector<TestCase> &registry, TestMode mode) {
  const auto opset = DefaultOpset(15);
  const BatchNormalizationKernel kernel{KernelContext{opset}};
  NodeProto node = MakeNode("BatchNormalization", {"X", "scale", "B", "mean", "variance"}, {"Y"});
  if (mode == TestMode::BENCHMARK) {
    for (const BatchBenchmark &benchmark :
         {BatchBenchmark{"n32_c128_rank2", {32, 128}, false, DataType::FLOAT, 701},
          BatchBenchmark{"n8_c16_l64_rank3", {8, 16, 64}, false, DataType::FLOAT, 711},
          BatchBenchmark{"n4_c32_h16_w16_rank4", {4, 32, 16, 16}, false, DataType::FLOAT, 721},
          BatchBenchmark{"n8_c64_h8_w8_rank4", {8, 64, 8, 8}, false, DataType::FLOAT, 731},
          BatchBenchmark{"n2_c16_d4_h8_w8_rank5", {2, 16, 4, 8, 8}, false, DataType::FLOAT, 741},
          BatchBenchmark{"training_n4_c32_h8_w8_rank4", {4, 32, 8, 8}, true, DataType::FLOAT, 751},
          BatchBenchmark{"n4_c16_h8_w8_rank4", {4, 16, 8, 8}, false, DataType::FLOAT16, 761},
          BatchBenchmark{"n4_c16_h8_w8_rank4", {4, 16, 8, 8}, false, DataType::BFLOAT16, 771}}) {
      RegisterBatchBenchmark(registry, opset, benchmark);
    }
    return;
  }
  for (DataType data_type : {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16}) {
    NodeProto inference_node =
        MakeNode("BatchNormalization", {"X", "scale", "B", "mean", "variance"}, {"Y"});
    const std::string name =
        "test_cpu_batchnormalization_small_" + std::string(DataTypeSuffix(data_type));
    Expect(registry, std::move(inference_node), name, {opset}, [kernel, data_type]() -> IoData {
      Tensor x = MakeTypedTensor(data_type, {1, 2, 1, 3}, {-1.0F, 0.0F, 1.0F, 2.0F, 3.0F, 4.0F});
      Tensor scale = MakeTypedTensor(data_type, {2}, {1.0F, 1.5F});
      Tensor bias = MakeTypedTensor(data_type, {2}, {0.0F, 1.0F});
      Tensor mean = MakeTypedTensor(data_type, {2}, {0.0F, 3.0F});
      Tensor variance = MakeTypedTensor(data_type, {2}, {1.0F, 1.5F});
      Tensor y = kernel(x, scale, bias, mean, variance);
      return IoData{
          {std::move(x), std::move(scale), std::move(bias), std::move(mean), std::move(variance)},
          {std::move(y)}};
    });
    SetNormalizationTolerance(registry, data_type);
  }

  NodeProto training_node = MakeNode("BatchNormalization", {"X", "scale", "B", "mean", "variance"},
                                     {"Y", "running_mean", "running_var"});
  AddIntAttribute(training_node, "training_mode", 1);
  AddFloatAttribute(training_node, "epsilon", 0.0F);
  AddFloatAttribute(training_node, "momentum", 0.75F);
  Expect(registry, std::move(training_node), "test_cpu_batchnormalization_training_float32",
         {opset}, [kernel]() -> IoData {
           Tensor x =
               Tensor::FromFloat("", {2, 2, 2}, {1.0F, 3.0F, 2.0F, 6.0F, 5.0F, 7.0F, 4.0F, 8.0F});
           Tensor scale = Tensor::FromFloat("", {2}, {2.0F, 0.5F});
           Tensor bias = Tensor::FromFloat("", {2}, {1.0F, -1.0F});
           Tensor mean = Tensor::FromFloat("", {2}, {10.0F, 20.0F});
           Tensor variance = Tensor::FromFloat("", {2}, {9.0F, 13.0F});
           BatchNormalizationResult result =
               kernel.Compute(x, scale, bias, mean, variance, true, 0.0F, 0.75F);
           return IoData{{std::move(x), std::move(scale), std::move(bias), std::move(mean),
                          std::move(variance)},
                         {std::move(result.y), std::move(*result.running_mean),
                          std::move(*result.running_variance)}};
         });
}

void RegisterCpuGroupNormalizationCases(std::vector<TestCase> &registry, TestMode mode) {
  const auto opset = DefaultOpset(21);
  const GroupNormalizationKernel kernel{KernelContext{opset}};
  NodeProto node = MakeNode("GroupNormalization", {"X", "scale", "bias"}, {"Y"});
  AddIntAttribute(node, "num_groups", mode == TestMode::BENCHMARK ? 8 : 2);
  if (mode == TestMode::BENCHMARK) {
    for (const GroupBenchmark &benchmark :
         {GroupBenchmark{"n2_c16_g4_l64_rank3", {2, 16, 64}, 4, DataType::FLOAT, 801},
          GroupBenchmark{"n4_c32_g8_h16_w16_rank4", {4, 32, 16, 16}, 8, DataType::FLOAT, 811},
          GroupBenchmark{"n1_c64_g32_h8_w8_rank4", {1, 64, 8, 8}, 32, DataType::FLOAT, 821},
          GroupBenchmark{"n2_c24_g6_d4_h8_w8_rank5", {2, 24, 4, 8, 8}, 6, DataType::FLOAT, 831},
          GroupBenchmark{"n2_c16_g4_h8_w8_rank4", {2, 16, 8, 8}, 4, DataType::FLOAT16, 841},
          GroupBenchmark{"n2_c16_g4_h8_w8_rank4", {2, 16, 8, 8}, 4, DataType::BFLOAT16, 851}}) {
      RegisterGroupBenchmark(registry, opset, benchmark);
    }
    return;
  }
  for (DataType data_type : {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16}) {
    NodeProto test_node = MakeNode("GroupNormalization", {"X", "scale", "bias"}, {"Y"});
    AddIntAttribute(test_node, "num_groups", 2);
    const std::string name =
        "test_cpu_groupnormalization_small_" + std::string(DataTypeSuffix(data_type));
    Expect(registry, std::move(test_node), name, {opset}, [kernel, data_type]() -> IoData {
      Tensor x =
          MakeTypedTensor(data_type, {1, 4, 2}, {-2.0F, -1.0F, 0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F});
      Tensor scale = MakeTypedTensor(data_type, {4}, {0.5F, 1.0F, 1.5F, 2.0F});
      Tensor bias = MakeTypedTensor(data_type, {4}, {-0.25F, 0.0F, 0.25F, 0.5F});
      Tensor y = kernel(x, scale, bias, 2);
      return IoData{{std::move(x), std::move(scale), std::move(bias)}, {std::move(y)}};
    });
    SetNormalizationTolerance(registry, data_type);
  }

  const OpsetId opset18 = DefaultOpset(18);
  NodeProto opset18_node = MakeNode("GroupNormalization", {"X", "scale", "bias"}, {"Y"});
  AddIntAttribute(opset18_node, "num_groups", 2);
  Expect(registry, std::move(opset18_node),
         "test_cpu_groupnormalization_opset18_group_affine_float32", {opset18},
         [opset18]() -> IoData {
           const GroupNormalizationKernel opset18_kernel{KernelContext{opset18}};
           Tensor x = Tensor::FromFloat("", {1, 4, 2},
                                        {-3.0F, -1.0F, 1.0F, 3.0F, 10.0F, 12.0F, 14.0F, 16.0F});
           Tensor scale = Tensor::FromFloat("", {2}, {2.0F, 0.5F});
           Tensor bias = Tensor::FromFloat("", {2}, {1.0F, -2.0F});
           Tensor y = opset18_kernel(x, scale, bias, 2, 0.0F);
           return IoData{{std::move(x), std::move(scale), std::move(bias)}, {std::move(y)}};
         });
}

void RegisterCpuInstanceNormalizationCases(std::vector<TestCase> &registry, TestMode mode) {
  const auto opset = DefaultOpset(22);
  const InstanceNormalizationKernel kernel{KernelContext{opset}};
  NodeProto node = MakeNode("InstanceNormalization", {"X", "scale", "B"}, {"Y"});
  if (mode == TestMode::BENCHMARK) {
    for (const InstanceBenchmark &benchmark :
         {InstanceBenchmark{"n8_c16_l128_rank3", {8, 16, 128}, DataType::FLOAT, 901},
          InstanceBenchmark{"n4_c32_h16_w16_rank4", {4, 32, 16, 16}, DataType::FLOAT, 911},
          InstanceBenchmark{"n2_c8_h64_w8_rank4", {2, 8, 64, 8}, DataType::FLOAT, 921},
          InstanceBenchmark{"n2_c16_d4_h8_w8_rank5", {2, 16, 4, 8, 8}, DataType::FLOAT, 931},
          InstanceBenchmark{"n2_c16_h8_w8_rank4", {2, 16, 8, 8}, DataType::FLOAT16, 941},
          InstanceBenchmark{"n2_c16_h8_w8_rank4", {2, 16, 8, 8}, DataType::BFLOAT16, 951}}) {
      RegisterInstanceBenchmark(registry, opset, benchmark);
    }
    return;
  }
  for (DataType data_type : {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16}) {
    NodeProto test_node = MakeNode("InstanceNormalization", {"X", "scale", "B"}, {"Y"});
    const std::string name =
        "test_cpu_instancenormalization_small_" + std::string(DataTypeSuffix(data_type));
    Expect(registry, std::move(test_node), name, {opset}, [kernel, data_type]() -> IoData {
      Tensor x = MakeTypedTensor(data_type, {1, 2, 3}, {-1.0F, 0.0F, 1.0F, 2.0F, 3.0F, 4.0F});
      Tensor scale = MakeTypedTensor(data_type, {2}, {1.0F, 1.5F});
      Tensor bias = MakeTypedTensor(data_type, {2}, {0.0F, 1.0F});
      Tensor y = kernel(x, scale, bias);
      return IoData{{std::move(x), std::move(scale), std::move(bias)}, {std::move(y)}};
    });
    SetNormalizationTolerance(registry, data_type);
  }
}

void RegisterCpuLayerNormalizationCases(std::vector<TestCase> &registry, TestMode mode) {
  const auto opset = DefaultOpset(17);
  const LayerNormalizationKernel kernel{KernelContext{opset}};
  NodeProto node = MakeNode("LayerNormalization", {"X", "Scale"}, {"Y"});
  AddIntAttribute(node, "axis", -1);
  if (mode == TestMode::BENCHMARK) {
    for (const LayerBenchmark &benchmark :
         {LayerBenchmark{
              "small_r256_w128_axis1", {256, 128}, {128}, -1, false, DataType::FLOAT, 1001},
          LayerBenchmark{"wide_r1_w4096_axis1", {1, 4096}, {4096}, -1, true, DataType::FLOAT, 1011},
          LayerBenchmark{
              "llm_r128_w4096_axis1", {128, 4096}, {4096}, -1, false, DataType::FLOAT, 1021},
          LayerBenchmark{
              "tall_r4096_w128_axis1", {4096, 128}, {128}, -1, false, DataType::FLOAT, 1031},
          LayerBenchmark{
              "rank3_n8_s16_w512_axis2", {8, 16, 512}, {512}, 2, true, DataType::FLOAT, 1041},
          LayerBenchmark{
              "rank3_n4_d8_w64_axis1", {4, 8, 64}, {8, 64}, 1, false, DataType::FLOAT, 1051},
          LayerBenchmark{"r64_w512_axis1", {64, 512}, {512}, -1, false, DataType::FLOAT16, 1061},
          LayerBenchmark{
              "r64_w512_axis1", {64, 512}, {512}, -1, false, DataType::BFLOAT16, 1071}}) {
      RegisterLayerBenchmark(registry, opset, benchmark);
    }
    return;
  }
  for (DataType data_type : {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16}) {
    NodeProto test_node =
        MakeNode("LayerNormalization", {"X", "Scale", "B"}, {"Y", "Mean", "InvStdDev"});
    AddIntAttribute(test_node, "axis", -1);
    const std::string name =
        "test_cpu_layernormalization_axis1_" + std::string(DataTypeSuffix(data_type));
    Expect(registry, std::move(test_node), name, {opset}, [kernel, data_type]() -> IoData {
      Tensor x = MakeTypedTensor(data_type, {2, 3}, {-1.0F, 0.0F, 1.0F, 2.0F, 4.0F, 8.0F});
      Tensor scale = MakeTypedTensor(data_type, {3}, {0.5F, 1.0F, 1.5F});
      Tensor bias = MakeTypedTensor(data_type, {3}, {-0.25F, 0.0F, 0.25F});
      LayerNormalizationResult result = kernel(x, scale, &bias, -1, 1.0e-5F, 1, true, true);
      return IoData{{std::move(x), std::move(scale), std::move(bias)},
                    {std::move(result.y), std::move(*result.mean), std::move(*result.inv_std_dev)}};
    });
    SetNormalizationTolerance(registry, data_type);
  }
}

void RegisterCpuLpNormalizationCases(std::vector<TestCase> &registry, TestMode mode) {
  const auto opset = DefaultOpset(22);
  const LpNormalizationKernel kernel{KernelContext{opset}};
  NodeProto node = MakeNode("LpNormalization", {"X"}, {"Y"});
  if (mode == TestMode::BENCHMARK) {
    for (const LpBenchmark &benchmark :
         {LpBenchmark{"r256_w128_axis1_p2", {256, 128}, -1, 2, DataType::FLOAT, 1101},
          LpBenchmark{"b16_r32_w128_axis2_p2", {16, 32, 128}, -1, 2, DataType::FLOAT, 1111},
          LpBenchmark{"b8_r64_w32_axis1_p2", {8, 64, 32}, 1, 2, DataType::FLOAT, 1121},
          LpBenchmark{"n4_c8_h16_w16_axis1_p1", {4, 8, 16, 16}, 1, 1, DataType::FLOAT, 1131},
          LpBenchmark{"wide_r16_w4096_axis1_p1", {16, 4096}, -1, 1, DataType::FLOAT, 1141},
          LpBenchmark{"b8_r16_w128_axis2_p2", {8, 16, 128}, -1, 2, DataType::FLOAT16, 1151},
          LpBenchmark{"b8_r16_w128_axis2_p2", {8, 16, 128}, -1, 2, DataType::BFLOAT16, 1161}}) {
      RegisterLpBenchmark(registry, opset, benchmark);
    }
    return;
  }
  for (DataType data_type : {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16}) {
    NodeProto test_node = MakeNode("LpNormalization", {"X"}, {"Y"});
    AddIntAttribute(test_node, "axis", 0);
    AddIntAttribute(test_node, "p", 1);
    const std::string name =
        "test_cpu_lpnormalization_axis0_p1_" + std::string(DataTypeSuffix(data_type));
    Expect(registry, std::move(test_node), name, {opset}, [kernel, data_type]() -> IoData {
      Tensor x = MakeTypedTensor(data_type, {2, 3}, {1.0F, -2.0F, 3.0F, 4.0F, 5.0F, -6.0F});
      Tensor y = kernel(x, 0, 1);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
    SetNormalizationTolerance(registry, data_type);
  }
}

void RegisterCpuMeanVarianceNormalizationCases(std::vector<TestCase> &registry, TestMode mode) {
  const auto opset = DefaultOpset(13);
  const MeanVarianceNormalizationKernel kernel{KernelContext{opset}};
  NodeProto node = MakeNode("MeanVarianceNormalization", {"X"}, {"Y"});
  if (mode == TestMode::BENCHMARK) {
    for (const MvnBenchmark &benchmark :
         {MvnBenchmark{"r256_w128_axes1_rank2", {256, 128}, {1}, false, DataType::FLOAT, 1201},
          MvnBenchmark{"r64_w32_axes0_1_rank2", {64, 32}, {0, 1}, false, DataType::FLOAT, 1211},
          MvnBenchmark{
              "n8_c32_l128_axes0_2_rank3", {8, 32, 128}, {0, 2}, false, DataType::FLOAT, 1221},
          MvnBenchmark{"n8_c32_h16_w16_default_rank4",
                       {8, 32, 16, 16},
                       {0, 2, 3},
                       true,
                       DataType::FLOAT,
                       1231},
          MvnBenchmark{
              "n4_c16_h32_w8_axes2_3_rank4", {4, 16, 32, 8}, {2, 3}, false, DataType::FLOAT, 1241},
          MvnBenchmark{"n2_c8_d4_h8_w8_axes0_2_3_4_rank5",
                       {2, 8, 4, 8, 8},
                       {0, 2, 3, 4},
                       false,
                       DataType::FLOAT,
                       1251},
          MvnBenchmark{"n4_c16_h8_w8_default_rank4",
                       {4, 16, 8, 8},
                       {0, 2, 3},
                       true,
                       DataType::FLOAT16,
                       1261},
          MvnBenchmark{"n4_c16_h8_w8_default_rank4",
                       {4, 16, 8, 8},
                       {0, 2, 3},
                       true,
                       DataType::BFLOAT16,
                       1271}}) {
      RegisterMvnBenchmark(registry, opset, benchmark);
    }
    return;
  }
  for (DataType data_type : {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16}) {
    NodeProto test_node = MakeNode("MeanVarianceNormalization", {"X"}, {"Y"});
    AddIntsAttribute(test_node, "axes", {1});
    const std::string name =
        "test_cpu_meanvariancenormalization_axis1_" + std::string(DataTypeSuffix(data_type));
    Expect(registry, std::move(test_node), name, {opset}, [kernel, data_type]() -> IoData {
      Tensor x = MakeTypedTensor(data_type, {2, 3}, {1.0F, 3.0F, 5.0F, 2.0F, 4.0F, 8.0F});
      Tensor y = kernel(x, {1});
      return IoData{{std::move(x)}, {std::move(y)}};
    });
    SetNormalizationTolerance(registry, data_type);
  }
}

} // namespace onnx_light_cpu::backend_test
