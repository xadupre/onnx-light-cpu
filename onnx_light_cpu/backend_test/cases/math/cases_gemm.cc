// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/math/include_math_cases.h"

#include "onnx_light_cpu/kernels/math/gemm_kernel.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/kernels/random.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace onnx_light_cpu::backend_test {

namespace {

namespace bt_ns = ONNX_LIGHT_NAMESPACE::core::backend_test;
namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

using bt_ns::AppendValueInfo;
using bt_ns::BuiltCase;
using bt_ns::DataSet;
using bt_ns::DimSpec;
using bt_ns::Expect;
using bt_ns::InitModel;
using bt_ns::IoData;
using bt_ns::TestCase;
using bt_ns::TestMode;
using ONNX_LIGHT_NAMESPACE::AddNode;
using ONNX_LIGHT_NAMESPACE::GraphProto;
using ONNX_LIGHT_NAMESPACE::ModelProto;
using ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::DataType;
using rt_ns::DefaultOpset;
using rt_ns::KernelContext;
using rt_ns::OpsetId;
using rt_ns::Randn;
using rt_ns::Tensor;

constexpr int64_t kDefaultIrVersion = 10;

enum class BiasShape {
  kNone,
  kScalar,
  kRow,
  kColumn,
  kMatrix,
};

// Builds a plain (no-attribute) two-input ``Gemm`` NodeProto: Y = A @ B.
NodeProto MakeGemmNode() {
  NodeProto node;
  node.set_op_type("Gemm");
  node.add_input("A");
  node.add_input("B");
  node.add_output("Y");
  return node;
}

void AddIntAttribute(NodeProto &node, const char *name, int64_t value) {
  auto *attribute = node.add_attribute();
  attribute->set_name(name);
  attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::AttributeType::INT);
  attribute->set_i(value);
}

// Encodes ``values`` as a BFLOAT16 ``Tensor`` (raw 16-bit bit patterns),
// mirroring how onnx-light's own backend test cases build bfloat16 inputs.
Tensor MakeBfloat16Tensor(const std::vector<int64_t> &shape, const std::vector<float> &values) {
  std::vector<std::uint8_t> raw(values.size() * sizeof(std::uint16_t));
  auto *dst = reinterpret_cast<std::uint16_t *>(raw.data());
  for (std::size_t i = 0; i < values.size(); ++i) {
    dst[i] = rt_ns::FloatToBfloat16Bits(values[i]);
  }
  return Tensor("", static_cast<std::int32_t>(DataType::BFLOAT16), shape, std::move(raw));
}

// Builds a ``Tensor`` of the requested element type from ``float`` values so
// each benchmark shape can be exercised for float32, float16 and bfloat16.
Tensor MakeGemmTensor(DataType dtype, const std::vector<int64_t> &shape,
                      const std::vector<float> &values) {
  switch (dtype) {
  case DataType::FLOAT:
    return Tensor::FromFloat("", shape, values);
  case DataType::DOUBLE: {
    std::vector<double> doubles(values.begin(), values.end());
    return Tensor::FromDouble("", shape, doubles);
  }
  case DataType::FLOAT16:
    return rt_ns::MakeFloat16Tensor("", shape, values);
  case DataType::BFLOAT16:
    return MakeBfloat16Tensor(shape, values);
  default:
    return Tensor::FromFloat("", shape, values);
  }
}

// Short element-type tag inserted into benchmark case names.
const char *GemmDtypeSuffix(DataType dtype) {
  switch (dtype) {
  case DataType::FLOAT:
    return "float32";
  case DataType::DOUBLE:
    return "float64";
  case DataType::FLOAT16:
    return "float16";
  case DataType::BFLOAT16:
    return "bfloat16";
  default:
    return "float32";
  }
}

const char *GemmBiasSuffix(BiasShape bias_shape) {
  switch (bias_shape) {
  case BiasShape::kNone:
    return "none";
  case BiasShape::kScalar:
    return "scalar";
  case BiasShape::kRow:
    return "row";
  case BiasShape::kColumn:
    return "column";
  case BiasShape::kMatrix:
    return "matrix";
  }
  return "none";
}

void RegisterGemmBenchmark(std::vector<TestCase> &registry,
                           const std::shared_ptr<onnx_light_cpu::GemmKernel> &kernel,
                           const OpsetId &opset, const std::string &base_name, DataType dtype,
                           int64_t m, int64_t n, int64_t k, bool trans_a = false,
                           bool trans_b = false, BiasShape bias_shape = BiasShape::kNone) {
  const std::string name = base_name + "_" + GemmDtypeSuffix(dtype) + "_transA_" +
                           (trans_a ? "1" : "0") + "_transB_" + (trans_b ? "1" : "0") + "_bias_" +
                           GemmBiasSuffix(bias_shape) + "_benchmark";
  NodeProto node = MakeGemmNode();
  if (trans_a) {
    AddIntAttribute(node, "transA", 1);
  }
  if (trans_b) {
    AddIntAttribute(node, "transB", 1);
  }
  const std::vector<int64_t> a_shape =
      trans_a ? std::vector<int64_t>{k, m} : std::vector<int64_t>{m, k};
  const std::vector<int64_t> b_shape =
      trans_b ? std::vector<int64_t>{n, k} : std::vector<int64_t>{k, n};
  const int64_t a_count = m * k;
  const int64_t b_count = k * n;
  const int64_t y_count = m * n;

  if (bias_shape != BiasShape::kNone) {
    node.add_input("C");
    std::vector<int64_t> c_shape;
    switch (bias_shape) {
    case BiasShape::kScalar:
      break;
    case BiasShape::kRow:
      c_shape = {1, n};
      break;
    case BiasShape::kColumn:
      c_shape = {m, 1};
      break;
    case BiasShape::kMatrix:
      c_shape = {m, n};
      break;
    case BiasShape::kNone:
      break;
    }
    const int64_t c_count =
        bias_shape == BiasShape::kMatrix
            ? m * n
            : (bias_shape == BiasShape::kRow ? n : (bias_shape == BiasShape::kColumn ? m : 1));
    Expect(registry, std::move(node), name, {opset}, {a_count, b_count, c_count}, {y_count},
           [kernel, dtype, a_shape, b_shape, c_shape, trans_a, trans_b, a_count, b_count,
            c_count]() -> IoData {
             Tensor a = MakeGemmTensor(dtype, a_shape, Randn<float>(a_shape, 433 + a_count));
             Tensor b = MakeGemmTensor(dtype, b_shape, Randn<float>(b_shape, 434 + b_count));
             Tensor c = MakeGemmTensor(dtype, c_shape, Randn<float>(c_shape, 435 + c_count));
             Tensor y = (*kernel)(a, b, c, 1.0f, 1.0f, trans_a, trans_b);
             return IoData{{std::move(a), std::move(b), std::move(c)}, {std::move(y)}};
           });
    return;
  }

  Expect(registry, std::move(node), name, {opset}, {a_count, b_count}, {y_count},
         [kernel, dtype, a_shape, b_shape, trans_a, trans_b, a_count, b_count]() -> IoData {
           Tensor a = MakeGemmTensor(dtype, a_shape, Randn<float>(a_shape, 433 + a_count));
           Tensor b = MakeGemmTensor(dtype, b_shape, Randn<float>(b_shape, 434 + b_count));
           Tensor y = (*kernel)(a, b, 1.0f, trans_a, trans_b);
           return IoData{{std::move(a), std::move(b)}, {std::move(y)}};
         });
}

void RegisterChainedGemmCase(std::vector<TestCase> &registry, const OpsetId &opset,
                             const std::string &name, int64_t m, int64_t k, int64_t n1, int64_t n2,
                             int64_t n3) {
  TestCase test_case(name, name, "model", "gemm_chain");
  test_case.declared_input_element_counts = {m * k, k * n1, n1 * n2, n2 * n3};
  test_case.declared_output_element_counts = {m * n3};
  test_case.build = [opset, name, m, k, n1, n2, n3]() {
    BuiltCase built;
    InitModel(built.model, kDefaultIrVersion, {opset});
    GraphProto *graph = built.model.add_graph();
    graph->set_name(name);
    AddNode(*graph, "Gemm", {"A", "B1"}, {"Y1"});
    AddNode(*graph, "Gemm", {"Y1", "B2"}, {"Y2"});
    AddNode(*graph, "Gemm", {"Y2", "B3"}, {"Y"});

    AppendValueInfo(*graph->add_input(), "A", DataType::FLOAT, {DimSpec(m), DimSpec(k)});
    AppendValueInfo(*graph->add_input(), "B1", DataType::FLOAT, {DimSpec(k), DimSpec(n1)});
    AppendValueInfo(*graph->add_input(), "B2", DataType::FLOAT, {DimSpec(n1), DimSpec(n2)});
    AppendValueInfo(*graph->add_input(), "B3", DataType::FLOAT, {DimSpec(n2), DimSpec(n3)});
    AppendValueInfo(*graph->add_value_info(), "Y1", DataType::FLOAT, {DimSpec(m), DimSpec(n1)});
    AppendValueInfo(*graph->add_value_info(), "Y2", DataType::FLOAT, {DimSpec(m), DimSpec(n2)});
    AppendValueInfo(*graph->add_output(), "Y", DataType::FLOAT, {DimSpec(m), DimSpec(n3)});

    Tensor a = Tensor::FromFloat("A", {m, k}, Randn<float>({m, k}, 701));
    Tensor b1 = Tensor::FromFloat("B1", {k, n1}, Randn<float>({k, n1}, 702));
    Tensor b2 = Tensor::FromFloat("B2", {n1, n2}, Randn<float>({n1, n2}, 703));
    Tensor b3 = Tensor::FromFloat("B3", {n2, n3}, Randn<float>({n2, n3}, 704));
    GemmKernel first(KernelContext{opset});
    GemmKernel second(KernelContext{opset});
    GemmKernel third(KernelContext{opset});
    Tensor y1 = first(a, b1, 1.0f, false, false);
    Tensor y2 = second(y1, b2, 1.0f, false, false);
    Tensor y = third(y2, b3, 1.0f, false, false);
    y.name = "Y";
    built.data_sets.push_back(
        DataSet{{std::move(a), std::move(b1), std::move(b2), std::move(b3)}, {std::move(y)}});
    return built;
  };
  registry.emplace_back(std::move(test_case));
}

} // namespace

// Gemm — Y = A @ B (default attributes), covering float32, float64, float16,
// bfloat16 (every element type ``GemmKernel`` implements).
void RegisterCpuGemmCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);
  const auto gemm_kernel = std::make_shared<onnx_light_cpu::GemmKernel>(KernelContext{opset});

  if (mode == TestMode::BENCHMARK) {
    RegisterChainedGemmCase(registry, opset, "test_cpu_gemm_chain_square_projection_benchmark", 512,
                            512, 1024, 256, 512);
    RegisterChainedGemmCase(registry, opset, "test_cpu_gemm_chain_transformer_benchmark", 128, 768,
                            3072, 768, 3072);
    RegisterChainedGemmCase(registry, opset, "test_cpu_gemm_chain_alternating_benchmark", 256, 1024,
                            256, 2048, 128);
    // Every prepared code path is exercised for each element type the
    // ``GemmKernel`` implements (float32, float64, float16, bfloat16), except
    // for the largest square float16 cases, which are prohibitively slow in
    // the onnxruntime comparison.
    for (const DataType dtype :
         {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16}) {
      RegisterGemmBenchmark(registry, gemm_kernel, opset, "test_cpu_gemm_direct", dtype, 32, 128,
                            16);
      RegisterGemmBenchmark(registry, gemm_kernel, opset, "test_cpu_gemm_direct_k32", dtype, 64,
                            256, 32);
      RegisterGemmBenchmark(registry, gemm_kernel, opset, "test_cpu_gemm_tiny_dynamic", dtype, 1,
                            64, 64);
      RegisterGemmBenchmark(registry, gemm_kernel, opset, "test_cpu_gemm_square_128", dtype, 128,
                            128, 128);
      RegisterGemmBenchmark(registry, gemm_kernel, opset, "test_cpu_gemm_square_256", dtype, 256,
                            256, 256);
      RegisterGemmBenchmark(registry, gemm_kernel, opset, "test_cpu_gemm_square_512", dtype, 512,
                            512, 512);
      RegisterGemmBenchmark(registry, gemm_kernel, opset, "test_cpu_gemm_square_1024", dtype, 1024,
                            1024, 1024);
      if (dtype != DataType::FLOAT16) {
        RegisterGemmBenchmark(registry, gemm_kernel, opset, "test_cpu_gemm_square_2048", dtype,
                              2048, 2048, 2048);
        RegisterGemmBenchmark(registry, gemm_kernel, opset, "test_cpu_gemm_square_4096", dtype,
                              4096, 4096, 4096);
      }
      RegisterGemmBenchmark(registry, gemm_kernel, opset, "test_cpu_gemm_square_256", dtype, 256,
                            256, 256, false, false, BiasShape::kScalar);
      RegisterGemmBenchmark(registry, gemm_kernel, opset, "test_cpu_gemm_square_256", dtype, 256,
                            256, 256, false, false, BiasShape::kRow);
      RegisterGemmBenchmark(registry, gemm_kernel, opset, "test_cpu_gemm_square_256", dtype, 256,
                            256, 256, false, false, BiasShape::kColumn);
      RegisterGemmBenchmark(registry, gemm_kernel, opset, "test_cpu_gemm_square_256", dtype, 256,
                            256, 256, false, false, BiasShape::kMatrix);
      RegisterGemmBenchmark(registry, gemm_kernel, opset, "test_cpu_gemm_skinny_m", dtype, 1, 1024,
                            1024);
      RegisterGemmBenchmark(registry, gemm_kernel, opset, "test_cpu_gemm_skinny_m_small", dtype, 1,
                            256, 256);
      RegisterGemmBenchmark(registry, gemm_kernel, opset, "test_cpu_gemm_skinny_n", dtype, 1024, 1,
                            1024);
      RegisterGemmBenchmark(registry, gemm_kernel, opset, "test_cpu_gemm_skinny_n_small", dtype,
                            256, 1, 256);
      RegisterGemmBenchmark(registry, gemm_kernel, opset, "test_cpu_gemm_large_k", dtype, 32, 32,
                            4096);
      RegisterGemmBenchmark(registry, gemm_kernel, opset, "test_cpu_gemm_large_k_1024", dtype, 32,
                            32, 1024);
      RegisterGemmBenchmark(registry, gemm_kernel, opset, "test_cpu_gemm_large_k_16384", dtype, 32,
                            32, 16384);
      RegisterGemmBenchmark(registry, gemm_kernel, opset, "test_cpu_gemm_split_k", dtype, 2, 2,
                            4096);
      RegisterGemmBenchmark(registry, gemm_kernel, opset, "test_cpu_gemm_split_k_16384", dtype, 2,
                            2, 16384);
      RegisterGemmBenchmark(registry, gemm_kernel, opset, "test_cpu_gemm_big_m8192_n128_k128",
                            dtype, 8192, 128, 128);
      RegisterGemmBenchmark(registry, gemm_kernel, opset, "test_cpu_gemm_big_m128_n8192_k128",
                            dtype, 128, 8192, 128);
      RegisterGemmBenchmark(registry, gemm_kernel, opset, "test_cpu_gemm_big_m128_n128_k8192",
                            dtype, 128, 128, 8192);
      RegisterGemmBenchmark(registry, gemm_kernel, opset, "test_cpu_gemm_big_m16384_n128_k128",
                            dtype, 16384, 128, 128);
      RegisterGemmBenchmark(registry, gemm_kernel, opset, "test_cpu_gemm_big_m128_n16384_k128",
                            dtype, 128, 16384, 128);
      RegisterGemmBenchmark(registry, gemm_kernel, opset, "test_cpu_gemm_big_m128_n128_k16384",
                            dtype, 128, 128, 16384);
      RegisterGemmBenchmark(registry, gemm_kernel, opset, "test_cpu_gemm_square_128", dtype, 128,
                            128, 128, true);
      RegisterGemmBenchmark(registry, gemm_kernel, opset, "test_cpu_gemm_square_128", dtype, 128,
                            128, 128, false, true);
      RegisterGemmBenchmark(registry, gemm_kernel, opset, "test_cpu_gemm_transformer_projection",
                            dtype, 128, 3072, 768);
      RegisterGemmBenchmark(registry, gemm_kernel, opset,
                            "test_cpu_gemm_transformer_projection_decode", dtype, 1, 3072, 768);
    }
    return;
  }

  RegisterChainedGemmCase(registry, opset, "test_cpu_gemm_chain_rectangular", 8, 16, 12, 20, 6);
  RegisterChainedGemmCase(registry, opset, "test_cpu_gemm_chain_widen_narrow", 16, 8, 32, 4, 16);
  RegisterChainedGemmCase(registry, opset, "test_cpu_gemm_chain_alternating", 4, 24, 8, 32, 12);

  const std::vector<int64_t> a_shape = {2, 3};
  const std::vector<int64_t> b_shape = {3, 4};
  const std::vector<float> a = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  const std::vector<float> b = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f,  5.0f,
                                6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};

  Expect(registry, MakeGemmNode(), "test_cpu_gemm_float32", {opset}, [=]() -> IoData {
    Tensor ta = Tensor::FromFloat("", a_shape, a);
    Tensor tb = Tensor::FromFloat("", b_shape, b);
    return IoData{{ta, tb}, {(*gemm_kernel)(ta, tb, 1.0f, false, false)}};
  });
  Expect(registry, MakeGemmNode(), "test_cpu_gemm_float64", {opset}, [=]() -> IoData {
    Tensor ta = Tensor::FromDouble("", a_shape, {0.0, 1.0, 2.0, 3.0, 4.0, 5.0});
    Tensor tb = Tensor::FromDouble("", b_shape,
                                   {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0});
    return IoData{{ta, tb}, {(*gemm_kernel)(ta, tb, 1.0f, false, false)}};
  });
  Expect(registry, MakeGemmNode(), "test_cpu_gemm_float16", {opset}, [=]() -> IoData {
    Tensor ta = rt_ns::MakeFloat16Tensor("", a_shape, a);
    Tensor tb = rt_ns::MakeFloat16Tensor("", b_shape, b);
    return IoData{{ta, tb}, {(*gemm_kernel)(ta, tb, 1.0f, false, false)}};
  });
  Expect(registry, MakeGemmNode(), "test_cpu_gemm_bfloat16", {opset}, [=]() -> IoData {
    Tensor ta = MakeBfloat16Tensor(a_shape, a);
    Tensor tb = MakeBfloat16Tensor(b_shape, b);
    return IoData{{ta, tb}, {(*gemm_kernel)(ta, tb, 1.0f, false, false)}};
  });
}

} // namespace onnx_light_cpu::backend_test
