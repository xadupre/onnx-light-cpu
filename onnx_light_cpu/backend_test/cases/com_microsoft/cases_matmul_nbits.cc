// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/com_microsoft/include_com_microsoft_cases.h"
#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"

#include "onnx_light_cpu/kernels/com_microsoft/matmul_nbits_kernel.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_proto/onnx_helper.h"

#include <cstddef>
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
using rt_ns::DataType;
using rt_ns::DefaultOpset;
using rt_ns::KernelContext;
using rt_ns::OpsetId;
using rt_ns::Tensor;

ONNX_LIGHT_NAMESPACE::NodeProto MakeMatMulNBitsNode(std::int64_t k, std::int64_t n,
                                                    std::int64_t bits, bool with_bias) {
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
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "bits", bits);
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "block_size", std::int64_t{32});
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "accuracy_level", std::int64_t{4});
  return node;
}

std::vector<std::uint8_t> MakePackedWeights(std::int64_t k, std::int64_t n, std::int64_t bits) {
  const std::int64_t blocks = (k + 31) / 32;
  const std::int64_t blob_size = 4 * bits;
  const std::int64_t values_per_byte = 8 / bits;
  const std::uint8_t mask = static_cast<std::uint8_t>((1U << bits) - 1U);
  std::vector<std::uint8_t> packed(static_cast<std::size_t>(n * blocks * blob_size));
  for (std::int64_t column = 0; column < n; ++column) {
    for (std::int64_t block = 0; block < blocks; ++block) {
      for (std::int64_t offset = 0; offset < 32; ++offset) {
        const std::uint8_t value =
            static_cast<std::uint8_t>((column * 3 + block + offset + 1) & mask);
        const std::size_t byte = static_cast<std::size_t>((column * blocks + block) * blob_size +
                                                          offset / values_per_byte);
        packed[byte] |= static_cast<std::uint8_t>(
            value << (static_cast<std::uint64_t>(offset % values_per_byte) * bits));
      }
    }
  }
  return packed;
}

IoData MakeCaseData(std::int64_t m, std::int64_t k, std::int64_t n, std::int64_t bits,
                    DataType data_type, bool with_bias, const KernelContext *kernel_context) {
  const std::int64_t blocks = (k + 31) / 32;
  Tensor a = MakeBenchmarkTensor(data_type, {m, k}, 7301 + bits);
  Tensor scales = MakeBenchmarkTensor(data_type, {n, blocks}, 7311 + bits);
  std::vector<std::uint8_t> packed_values = MakePackedWeights(k, n, bits);
  Tensor b = Tensor::FromUint8("", {n, blocks, 4 * bits}, packed_values);
  std::vector<Tensor> inputs;
  inputs.push_back(std::move(a));
  inputs.push_back(std::move(b));
  inputs.push_back(std::move(scales));
  if (with_bias) {
    inputs.push_back(MakeBenchmarkTensor(data_type, {n}, 7321 + bits));
  }
  if (kernel_context == nullptr) {
    return IoData{std::move(inputs), {}, {}, false};
  }
  const MatMulNBitsKernel kernel{MakeMatMulNBitsNode(k, n, bits, with_bias), *kernel_context};
  Tensor y = kernel(inputs[0], inputs[1], inputs[2], with_bias ? &inputs[3] : nullptr);
  return IoData{std::move(inputs), {std::move(y)}};
}

void SetTolerance(std::vector<TestCase> &registry, DataType data_type) {
  registry.back().rtol = data_type == DataType::BFLOAT16  ? 2.0e-2
                         : data_type == DataType::FLOAT16 ? 3.0e-3
                                                          : 1.0e-5;
  registry.back().atol = data_type == DataType::BFLOAT16  ? 2.0e-2
                         : data_type == DataType::FLOAT16 ? 3.0e-3
                                                          : 1.0e-6;
}

void RegisterBenchmark(std::vector<TestCase> &registry, const char *shape_name, std::int64_t m,
                       std::int64_t k, std::int64_t n, std::int64_t bits, DataType data_type) {
  const std::int64_t blocks = (k + 31) / 32;
  const std::string name = "test_cpu_matmulnbits_" + std::string(shape_name) + "_m" +
                           std::to_string(m) + "_k" + std::to_string(k) + "_n" + std::to_string(n) +
                           "_bits" + std::to_string(bits) + "_block32_accuracy4_" +
                           DataTypeSuffix(data_type) + "_benchmark";
  Expect(registry, MakeMatMulNBitsNode(k, n, bits, false), name,
         {DefaultOpset(26), OpsetId(kMicrosoftDomain, 1)},
         {m * k, n * blocks * 4 * bits, n * blocks}, {m * n},
         [=](bool generate_expected_outputs) {
           if (generate_expected_outputs) {
             const KernelContext kernel_context{OpsetId(kMicrosoftDomain, 1)};
             return MakeCaseData(m, k, n, bits, data_type, false, &kernel_context);
           }
           return MakeCaseData(m, k, n, bits, data_type, false, nullptr);
         },
         "backend-test", bt_ns::TestCaseTag::AI_RT,
         {bt_ns::TensorTypeSpec(static_cast<std::int32_t>(data_type), {m, n})});
  SetTolerance(registry, data_type);
}

} // namespace

void RegisterCpuMatMulNBitsCases(std::vector<TestCase> &registry, TestMode mode) {
  constexpr DataType data_types[] = {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16};
  constexpr std::int64_t bit_widths[] = {2, 4, 8};
  if (mode == TestMode::BENCHMARK) {
    for (DataType data_type : data_types) {
      for (std::int64_t bits : bit_widths) {
        RegisterBenchmark(registry, "type_coverage", 2, 256, 128, bits, data_type);
      }
    }
    RegisterBenchmark(registry, "qwen2_qkv_decode", 1, 4096, 6144, 4, DataType::FLOAT);
    RegisterBenchmark(registry, "qwen2_gate_up_decode", 1, 4096, 11008, 4, DataType::FLOAT);
    RegisterBenchmark(registry, "qwen3_qkv_short_prefill", 8, 1024, 4096, 4, DataType::FLOAT);
    struct Projection {
      const char *name;
      std::int64_t k;
      std::int64_t n;
    };
    // Keep the 202048-column vocabulary projection in the opt-in Python parity tool.
    constexpr Projection muse_projections[] = {{"muse_glimmer_attention_gate", 6656, 4096},
                                               {"muse_glimmer_q", 6656, 4096},
                                               {"muse_glimmer_k", 6656, 256},
                                               {"muse_glimmer_v", 6656, 256},
                                               {"muse_glimmer_attention_output", 4096, 6656},
                                               {"muse_glimmer_gate", 6656, 19968},
                                               {"muse_glimmer_up", 6656, 19968},
                                               {"muse_glimmer_down", 19968, 6656}};
    for (const Projection &projection : muse_projections) {
      for (std::int64_t m : {1, 8, 128}) {
        for (DataType data_type : data_types) {
          RegisterBenchmark(registry, projection.name, m, projection.k, projection.n, 4, data_type);
        }
      }
    }
    return;
  }

  for (DataType data_type : data_types) {
    for (std::int64_t bits : bit_widths) {
      const std::string name = "test_cpu_matmulnbits_partial_block_bias_bits" +
                               std::to_string(bits) + "_" + DataTypeSuffix(data_type);
      Expect(
          registry, MakeMatMulNBitsNode(33, 3, bits, true), name,
          {DefaultOpset(26), OpsetId(kMicrosoftDomain, 1)},
          [=]() {
            const KernelContext kernel_context{OpsetId(kMicrosoftDomain, 1)};
            return MakeCaseData(2, 33, 3, bits, data_type, true, &kernel_context);
          },
          "backend-test", bt_ns::TestCaseTag::AI_RT);
      SetTolerance(registry, data_type);
    }
  }
}

} // namespace onnx_light_cpu::backend_test
