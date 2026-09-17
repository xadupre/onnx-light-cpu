// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/com_microsoft/include_com_microsoft_cases.h"

#include "onnx_light_cpu/kernels/com_microsoft/matmul_nbits_kernel.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_proto/onnx_helper.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace onnx_light_cpu::backend_test {
namespace {

namespace bt_ns = ONNX_LIGHT_NAMESPACE::core::backend_test;
namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

using bt_ns::Expect;
using bt_ns::IoData;
using rt_ns::DefaultOpset;
using rt_ns::KernelContext;
using rt_ns::OpsetId;
using rt_ns::Tensor;

ONNX_LIGHT_NAMESPACE::NodeProto MakeMatMulNBitsNode(std::int64_t k, std::int64_t n,
                                                    bool with_bias) {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("MatMulNBits");
  node.set_domain(kMicrosoftDomain);
  node.add_input("A");
  node.add_input("B");
  node.add_input("scales");
  if (with_bias) {
    node.add_input("");
    node.add_input("");
    node.add_input("bias");
  }
  node.add_output("Y");
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "K", k);
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "N", n);
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "bits", std::int64_t{4});
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "block_size", std::int64_t{32});
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "accuracy_level", std::int64_t{4});
  return node;
}

std::vector<std::uint8_t> MakePackedWeights(std::int64_t k, std::int64_t n) {
  const std::int64_t blocks = (k + 31) / 32;
  std::vector<std::uint8_t> packed(static_cast<std::size_t>(n * blocks * 16));
  for (std::int64_t column = 0; column < n; ++column) {
    for (std::int64_t block = 0; block < blocks; ++block) {
      for (std::int64_t byte = 0; byte < 16; ++byte) {
        const std::uint8_t low =
            static_cast<std::uint8_t>((column * 3 + block + byte * 2 + 5) & 15);
        const std::uint8_t high = static_cast<std::uint8_t>(
            (column * 3 + block + byte * 2 + 6) & 15);
        packed[static_cast<std::size_t>((column * blocks + block) * 16 + byte)] =
            static_cast<std::uint8_t>(low | (high << 4));
      }
    }
  }
  return packed;
}

IoData MakeBenchmarkData(std::int64_t m, std::int64_t k, std::int64_t n,
                         bool generate_expected_outputs) {
  const std::int64_t blocks = (k + 31) / 32;
  std::vector<float> a_values(static_cast<std::size_t>(m * k));
  for (std::size_t i = 0; i < a_values.size(); ++i) {
    a_values[i] = static_cast<float>(static_cast<int>(i % 31) - 15) / 16.0f;
  }
  std::vector<float> scale_values(static_cast<std::size_t>(n * blocks));
  for (std::size_t i = 0; i < scale_values.size(); ++i) {
    scale_values[i] = 0.0025f * static_cast<float>(1 + i % 7);
  }
  Tensor a = Tensor::FromFloat("", {m, k}, a_values);
  Tensor b = Tensor::FromUint8("", {n, blocks, 16}, MakePackedWeights(k, n));
  Tensor scales = Tensor::FromFloat("", {n, blocks}, scale_values);
  if (!generate_expected_outputs) {
    return IoData{{std::move(a), std::move(b), std::move(scales)}, {}, {}, false};
  }
  const MatMulNBitsKernel kernel(
      MakeMatMulNBitsNode(k, n, false), KernelContext{OpsetId(kMicrosoftDomain, 1)});
  Tensor y = kernel(a, b, scales);
  return IoData{{std::move(a), std::move(b), std::move(scales)}, {std::move(y)}};
}

} // namespace

void RegisterCpuMatMulNBitsCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId microsoft_opset(kMicrosoftDomain, 1);
  if (mode == TestMode::BENCHMARK) {
    struct Shape {
      const char *name;
      std::int64_t m;
      std::int64_t k;
      std::int64_t n;
    };
    const Shape shapes[] = {
        {"qwen2_qkv_decode", 1, 4096, 6144},
        {"qwen2_gate_up_decode", 1, 4096, 11008},
        {"qwen3_qkv_short_prefill", 8, 1024, 4096},
    };
    for (const Shape &shape : shapes) {
      const std::string name = "test_cpu_matmulnbits_" + std::string(shape.name) + "_m" +
                               std::to_string(shape.m) + "_k" + std::to_string(shape.k) + "_n" +
                               std::to_string(shape.n) +
                               "_bits4_block32_accuracy4_float32_benchmark";
      Expect(
          registry, MakeMatMulNBitsNode(shape.k, shape.n, false), name,
          {DefaultOpset(26), microsoft_opset}, {shape.m * shape.k, shape.n * (shape.k / 32) * 16,
                                                shape.n * (shape.k / 32)},
          {shape.m * shape.n},
          [=](bool generate_expected_outputs) {
            return MakeBenchmarkData(shape.m, shape.k, shape.n, generate_expected_outputs);
          },
          "backend-test", bt_ns::TestCaseTag::AI_RT,
          {bt_ns::TensorTypeSpec(static_cast<std::int32_t>(rt_ns::DataType::FLOAT),
                                 {shape.m, shape.n})});
    }
    return;
  }

  Expect(
      registry, MakeMatMulNBitsNode(32, 2, true), "test_cpu_matmulnbits_bits4_block32_bias_float32",
      {DefaultOpset(26), microsoft_opset},
      []() {
        std::vector<std::uint8_t> packed(32, 0x88);
        std::fill(packed.begin() + 16, packed.end(), 0x99);
        return IoData{{Tensor::FromFloat("", {1, 32}, std::vector<float>(32, 1.0f)),
                       Tensor::FromUint8("", {2, 1, 16}, packed),
                       Tensor::FromFloat("", {2, 1}, {1.0f, 0.5f}),
                       Tensor::FromFloat("", {2}, {1.0f, -1.0f})},
                      {Tensor::FromFloat("", {1, 2}, {1.0f, 15.0f})}};
      },
      "backend-test", bt_ns::TestCaseTag::AI_RT);

  Expect(
      registry, MakeMatMulNBitsNode(33, 1, false),
      "test_cpu_matmulnbits_partial_block_float32", {DefaultOpset(26), microsoft_opset},
      []() {
        std::vector<std::uint8_t> packed(32, 0x88);
        packed[16] = 0x8a;
        return IoData{{Tensor::FromFloat("", {1, 33}, std::vector<float>(33, 1.0f)),
                       Tensor::FromUint8("", {1, 2, 16}, packed),
                       Tensor::FromFloat("", {1, 2}, {1.0f, 0.25f})},
                      {Tensor::FromFloat("", {1, 1}, {0.5f})}};
      },
      "backend-test", bt_ns::TestCaseTag::AI_RT);
}

} // namespace onnx_light_cpu::backend_test
