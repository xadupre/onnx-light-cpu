// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "ml_cpu_compute/com_microsoft/matmul_nbits.h"
#include "ml_cpu_compute/execution.h"
#include "ml_cpu_compute/math/half_conversion.h"
#include "onnx_light_cpu/kernels/com_microsoft/matmul_nbits_int3.h"
#include "onnx_light_cpu/kernels/com_microsoft/matmul_nbits_kernel.h"
#include "onnx_light_cpu/kernels/register_kernels.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"
#include "onnx_light_cpu/shapes/com_microsoft/shape_inference.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/kernels/value_store.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_session.h"
#include "onnx_core/runtime/tuning/kernel_tuning.h"
#include "onnx_core/shapes/shape_inference.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

struct InlineExecutor {
  std::int64_t dispatches = 0;
  std::int64_t blocks = 0;

  static void Run(void *context, std::int64_t num_blocks, void *task_context,
                  ml_cpu_compute::ExecutionBlockFn task) {
    auto &self = *static_cast<InlineExecutor *>(context);
    ++self.dispatches;
    self.blocks = num_blocks;
    for (std::int64_t block = 0; block < num_blocks; ++block) {
      task(task_context, block);
    }
  }
};

ONNX_LIGHT_NAMESPACE::NodeProto MakeNode(std::int64_t k, std::int64_t n, std::int64_t bits = 4,
                                         std::int64_t block_size = 32) {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_domain("com.microsoft");
  node.set_op_type("MatMulNBits");
  node.add_input("A");
  node.add_input("B");
  node.add_input("scales");
  node.add_output("Y");
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "K", k);
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "N", n);
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "bits", bits);
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "block_size", block_size);
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "accuracy_level", std::int64_t{4});
  return node;
}

rt_ns::Tensor MakeTensor(rt_ns::DataType data_type, const rt_ns::Shape &shape,
                         const std::vector<float> &values) {
  if (data_type == rt_ns::DataType::FLOAT16) {
    return rt_ns::MakeFloat16Tensor("", shape, values);
  }
  if (data_type == rt_ns::DataType::BFLOAT16) {
    return rt_ns::MakeBfloat16Tensor("", shape, values);
  }
  return rt_ns::Tensor::FromFloat("", shape, values);
}

ONNX_LIGHT_NAMESPACE::EncodedValueProto
MakeEncodedWeights(rt_ns::DataType data_type, std::int64_t k, std::int64_t n, std::int64_t bits = 4,
                   std::int64_t block_size = 32, double zero_point = -1.0) {
  const auto format = bits == 2   ? rt_ns::QuantizationFormat::kOrtMatmulnbitsInt2
                      : bits == 4 ? rt_ns::QuantizationFormat::kOrtMatmulnbitsInt4
                                  : rt_ns::QuantizationFormat::kOrtMatmulnbitsInt8;
  rt_ns::QuantizationPlan plan = rt_ns::MakeMatMulNBitsPlan(format, k, n, block_size);
  if (zero_point >= 0.0) {
    for (auto &block : plan.runs[0].blocks) {
      block.zero_point = zero_point;
    }
  }
  return rt_ns::QuantizeTensor(
             MakeTensor(data_type, {k, n},
                        std::vector<float>(static_cast<std::size_t>(k * n), 1.0f)),
             plan)
      .Encoded();
}

float ReadValue(const rt_ns::Tensor &tensor, std::size_t index) {
  if (tensor.data_type == static_cast<std::int32_t>(rt_ns::DataType::FLOAT)) {
    return tensor.AsFloat()[index];
  }
  const auto value = reinterpret_cast<const std::uint16_t *>(tensor.bytes())[index];
  return tensor.data_type == static_cast<std::int32_t>(rt_ns::DataType::FLOAT16)
             ? ml_cpu_compute::detail::Float16BitsToFloat(value)
             : ml_cpu_compute::detail::Bfloat16BitsToFloat(value);
}

ONNX_LIGHT_NAMESPACE::EncodedValueProto
MakeInt3Weights(rt_ns::DataType data_type, std::int64_t k, std::int64_t n,
                const std::vector<std::uint8_t> &codes, const std::vector<float> &scales,
                const std::vector<std::uint8_t> &zero_points = {}) {
  const std::size_t blocks = static_cast<std::size_t>((k + 31) / 32);
  const std::size_t count = static_cast<std::size_t>(n) * blocks;
  std::string raw(count * 12, '\0');
  for (std::size_t column = 0; column < static_cast<std::size_t>(n); ++column) {
    for (std::size_t p = 0; p < static_cast<std::size_t>(k); ++p) {
      const auto code = codes[column * static_cast<std::size_t>(k) + p];
      const std::size_t bit = (column * blocks + p / 32) * 96 + (p % 32) * 3;
      raw[bit / 8] |= static_cast<char>(code << (bit % 8));
      if (bit % 8 > 5) {
        raw[bit / 8 + 1] |= static_cast<char>(code >> (8 - bit % 8));
      }
    }
  }
  const rt_ns::Tensor scale_tensor =
      MakeTensor(data_type, {n, static_cast<std::int64_t>(blocks)}, scales);
  raw.append(reinterpret_cast<const char *>(scale_tensor.bytes()), scale_tensor.size_bytes());
  if (!zero_points.empty()) {
    raw.append(reinterpret_cast<const char *>(zero_points.data()), zero_points.size());
  }
  ONNX_LIGHT_NAMESPACE::EncodedValueProto encoded;
  auto *logical = encoded.mutable_logical_type()->mutable_tensor_type();
  logical->set_elem_type(static_cast<std::int32_t>(data_type));
  logical->mutable_shape()->add_dim()->set_dim_value(k);
  logical->mutable_shape()->add_dim()->set_dim_value(n);
  auto *root = encoded.mutable_struct_type();
  root->set_name(std::string(onnx_light_cpu::kMatMulNBitsInt3Format));
  auto *parameters = root->mutable_structure()->add_field();
  parameters->set_name("parameters");
  auto *constant = parameters->mutable_constant();
  constant->set_data_type(ONNX_LIGHT_NAMESPACE::TensorProto::INT64);
  constant->add_dims(2);
  constant->add_int64_data(3);
  constant->add_int64_data(32);
  const auto add_field = [root, n, blocks](const char *name, int type, std::int64_t third = -1) {
    auto *field = root->mutable_structure()->add_field();
    field->set_name(name);
    auto *tensor = field->mutable_type()->mutable_tensor_type();
    tensor->set_elem_type(type);
    tensor->mutable_shape()->add_dim()->set_dim_value(n);
    tensor->mutable_shape()->add_dim()->set_dim_value(static_cast<std::int64_t>(blocks));
    if (third > 0) {
      tensor->mutable_shape()->add_dim()->set_dim_value(third);
    }
  };
  add_field("B", ONNX_LIGHT_NAMESPACE::TensorProto::UINT8, 12);
  add_field("scales", static_cast<std::int32_t>(data_type));
  if (!zero_points.empty()) {
    add_field("zero_points", ONNX_LIGHT_NAMESPACE::TensorProto::UINT8);
  }
  encoded.set_raw_data(raw);
  return encoded;
}

TEST(MatMulNBitsKernel, EncodedInt3MatchesDequantizedMatMulAcrossBlockTails) {
  constexpr std::int64_t k = 35;
  constexpr std::int64_t n = 2;
  constexpr std::size_t rows = 3;
  std::vector<std::uint8_t> codes(k * n);
  std::vector<float> activations(rows * k);
  for (std::size_t i = 0; i < codes.size(); ++i) {
    codes[i] = static_cast<std::uint8_t>((i * 5 + 3) % 8);
  }
  for (std::size_t i = 0; i < activations.size(); ++i) {
    activations[i] = static_cast<float>(static_cast<int>(i % 7) - 3) / 4;
  }
  const std::vector<float> scales{0.5f, 0.25f, 1.0f, 2.0f};
  for (auto dtype : {rt_ns::DataType::FLOAT, rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16}) {
    for (const std::vector<std::uint8_t> zeros :
         {std::vector<std::uint8_t>{}, std::vector<std::uint8_t>{1, 7, 0, 3}}) {
      const auto encoded = MakeInt3Weights(dtype, k, n, codes, scales, zeros);
      const auto a = MakeTensor(dtype, {rows, k}, activations);
      const onnx_light_cpu::MatMulNBitsKernel kernel{
          MakeNode(k, n, 3), rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};
      const auto y = kernel(a, encoded);
      ASSERT_EQ(y.shape, (rt_ns::Shape{rows, n}));
      const auto actual_scales = MakeTensor(dtype, {n, 2}, scales);
      for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < n; ++column) {
          float expected = 0;
          for (std::size_t p = 0; p < k; ++p) {
            const std::size_t block = column * 2 + p / 32;
            const int zero = zeros.empty() ? 4 : zeros[block];
            expected += ReadValue(a, row * k + p) *
                        (static_cast<int>(codes[column * k + p]) - zero) *
                        ReadValue(actual_scales, block);
          }
          EXPECT_NEAR(ReadValue(y, row * n + column), expected,
                      dtype == rt_ns::DataType::FLOAT ? 1e-4f : 0.15f);
        }
      }
    }
  }
}

TEST(MatMulNBitsKernel, EncodedInt3RejectsMalformedMetadataAndPadding) {
  const auto a = rt_ns::Tensor::FromFloat("", {1, 35}, std::vector<float>(35, 1));
  const onnx_light_cpu::MatMulNBitsKernel kernel{
      MakeNode(35, 1, 3), rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};
  const auto valid = MakeInt3Weights(rt_ns::DataType::FLOAT, 35, 1,
                                     std::vector<std::uint8_t>(35, 7), {1, 1}, {4, 0});
  EXPECT_NO_THROW(kernel(a, valid));
  auto bad = valid;
  std::string payload(reinterpret_cast<const char *>(valid.raw_data().data()),
                      valid.raw_data().size());
  payload[13] |= static_cast<char>(0x80);
  bad.set_raw_data(payload);
  EXPECT_THROW(kernel(a, bad), std::invalid_argument);
  bad = valid;
  bad.set_raw_data(reinterpret_cast<const char *>(valid.raw_data().data()),
                   valid.raw_data().size() - 1);
  EXPECT_THROW(kernel(a, bad), std::invalid_argument);
  bad = valid;
  payload.assign(reinterpret_cast<const char *>(valid.raw_data().data()), valid.raw_data().size());
  payload.back() = 8;
  bad.set_raw_data(payload);
  EXPECT_THROW(kernel(a, bad), std::invalid_argument);
  bad = valid;
  *bad.mutable_struct_type()
       ->mutable_structure()
       ->mutable_field(0)
       ->mutable_constant()
       ->mutable_int64_data(1) = 64;
  EXPECT_THROW(kernel(a, bad), std::invalid_argument);
  bad = valid;
  bad.mutable_struct_type()
      ->mutable_structure()
      ->mutable_field(2)
      ->mutable_type()
      ->mutable_tensor_type()
      ->set_elem_type(ONNX_LIGHT_NAMESPACE::TensorProto::DOUBLE);
  EXPECT_THROW(kernel(a, bad), std::invalid_argument);
  const onnx_light_cpu::MatMulNBitsKernel wrong_geometry{
      MakeNode(35, 2, 3), rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};
  EXPECT_THROW(wrong_geometry(a, valid), std::invalid_argument);
  const auto raw = rt_ns::Tensor::FromUint8("", {1, 2, 12}, std::vector<std::uint8_t>(24));
  EXPECT_THROW(kernel(a, raw, MakeTensor(rt_ns::DataType::FLOAT, {1, 2}, {1, 1})),
               std::invalid_argument);
}

TEST(MatMulNBitsKernel, InfersEncodedInt3ShapeAndRejectsAttributeMismatch) {
  namespace shapes_ns = ONNX_LIGHT_NAMESPACE::core::shapes;
  namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;
  shapes_ns::ShapesContext ctx;
  ctx.Set("A", sym_ns::SymTensor(nullptr, sym_ns::TensorType::kFloat,
                                 {sym_ns::SymDim(2), sym_ns::SymDim(35)}));
  const auto encoded = MakeInt3Weights(rt_ns::DataType::FLOAT, 35, 2,
                                       std::vector<std::uint8_t>(70, 4), {1, 1, 1, 1});
  ctx.SetEncodedValue("B", encoded);
  auto node = MakeNode(35, 2, 3);
  node.ref_input().resize(2);
  onnx_light_cpu::ComputeShapeMatMulNBits(ctx, node);
  EXPECT_EQ(ctx.Get("Y").Shape(), (sym_ns::SymShape{sym_ns::SymDim(2), sym_ns::SymDim(2)}));
  auto wrong = MakeNode(35, 3, 3);
  wrong.ref_input().resize(2);
  EXPECT_THROW(onnx_light_cpu::ComputeShapeMatMulNBits(ctx, wrong), std::invalid_argument);
  wrong = MakeNode(35, 2, 4);
  wrong.ref_input().resize(2);
  EXPECT_THROW(onnx_light_cpu::ComputeShapeMatMulNBits(ctx, wrong), std::invalid_argument);
}

TEST(MatMulNBitsKernel, RunsPreparedEncodedInt3WithoutTensorScales) {
  auto node = MakeNode(35, 1, 3);
  node.ref_input().resize(2);
  onnx_light_cpu::MatMulNBitsKernel kernel{
      node, rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};
  rt_ns::RuntimeContext rt{rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};
  rt.PutValue("B",
              rt_ns::RuntimeValue(MakeInt3Weights(rt_ns::DataType::FLOAT, 35, 1,
                                                  std::vector<std::uint8_t>(35, 5), {0.5f, 2.0f})));
  rt.Put("A", rt_ns::Tensor::FromFloat("A", {1, 35}, std::vector<float>(35, 1)));
  ASSERT_TRUE(kernel.HasPreparations({"B"}));
  kernel.Prepare(rt, {"B"});
  kernel.Run(rt);
  EXPECT_FLOAT_EQ(rt.Get("Y").AsFloat()[0], 22.0f);
}

TEST(MatMulNBitsKernel, DecodesInt4AndAppliesPerBlockScalesAndBias) {
  std::vector<std::uint8_t> packed(64, 0x88);
  std::fill(packed.begin() + 32, packed.end(), 0x99);
  packed[16] = 0x8a;
  packed[48] = 0x87;
  const std::vector<float> a_values(33, 1.0f);
  const rt_ns::Tensor a = rt_ns::Tensor::FromFloat("", {1, 33}, a_values);
  const rt_ns::Tensor b = rt_ns::Tensor::FromUint8("", {2, 2, 16}, packed);
  const rt_ns::Tensor scales = rt_ns::Tensor::FromFloat("", {2, 2}, {1.0f, 0.25f, 0.5f, 2.0f});
  const rt_ns::Tensor bias = rt_ns::Tensor::FromFloat("", {2}, {1.0f, -1.0f});
  const onnx_light_cpu::MatMulNBitsKernel kernel{
      MakeNode(33, 2), rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};
  const rt_ns::Tensor y = kernel(a, b, scales, &bias);
  ASSERT_EQ(y.shape, (rt_ns::Shape{1, 2}));
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 1.5f);
  EXPECT_FLOAT_EQ(y.AsFloat()[1], 13.0f);
}

TEST(MatMulNBitsKernel, PreservesLeadingDimensionsAndFlatScales) {
  const std::vector<std::uint8_t> packed(16, 0x99);
  const rt_ns::Tensor a = rt_ns::Tensor::FromFloat("", {2, 1, 32}, std::vector<float>(64, 1.0f));
  const rt_ns::Tensor b = rt_ns::Tensor::FromUint8("", {1, 1, 16}, packed);
  const rt_ns::Tensor scales = rt_ns::Tensor::FromFloat("", {1}, {0.5f});
  const onnx_light_cpu::MatMulNBitsKernel kernel{
      MakeNode(32, 1), rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};
  const rt_ns::Tensor y = kernel(a, b, scales);
  EXPECT_EQ(y.shape, (rt_ns::Shape{2, 1, 1}));
  for (std::size_t i = 0; i < y.element_count(); ++i) {
    EXPECT_FLOAT_EQ(y.AsFloat()[i], 16.0f);
  }
}

TEST(MatMulNBitsKernel, Accuracy4DynamicPlanMatchesBlockQuantizationAndRefreshesWeights) {
  if (!ml_cpu_compute::MatMulNBitsAccuracy4Float32Available()) {
    GTEST_SKIP() << "The optimized accuracy-level-4 kernel is unavailable.";
  }
  constexpr std::size_t rows = 9;
  constexpr std::size_t k = 64;
  constexpr std::size_t n = 16;
  constexpr std::size_t blocks = k / 32;
  std::vector<float> a(rows * k), scales(n * blocks), bias(n);
  std::vector<std::uint8_t> packed(n * blocks * 16);
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<float>(static_cast<int>(i % 29) - 14) / 17.0f;
  }
  for (std::size_t i = 0; i < scales.size(); ++i) {
    scales[i] = static_cast<float>(i % 7 + 1) / 128.0f;
  }
  for (std::size_t i = 0; i < bias.size(); ++i) {
    bias[i] = static_cast<float>(i) / 32.0f;
  }
  for (std::size_t i = 0; i < packed.size(); ++i) {
    packed[i] = static_cast<std::uint8_t>(i * 37 + 11);
  }

  const rt_ns::Tensor input = rt_ns::Tensor::FromFloat("", {rows, k}, a);
  const rt_ns::Tensor scale_tensor = rt_ns::Tensor::FromFloat("", {n, blocks}, scales);
  const rt_ns::Tensor bias_tensor = rt_ns::Tensor::FromFloat("", {n}, bias);
  const onnx_light_cpu::MatMulNBitsKernel kernel{
      MakeNode(k, n), rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};

  auto check = [&](const std::vector<std::uint8_t> &weights) {
    const rt_ns::Tensor weight_tensor = rt_ns::Tensor::FromUint8("", {n, blocks, 16}, weights);
    const rt_ns::Tensor output = kernel(input, weight_tensor, scale_tensor, &bias_tensor);
    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t column = 0; column < n; ++column) {
        float expected = bias[column];
        for (std::size_t block = 0; block < blocks; ++block) {
          const float *activation = a.data() + row * k + block * 32;
          float maximum = 0.0f;
          for (std::size_t offset = 0; offset < 32; ++offset) {
            maximum = std::max(maximum, std::abs(activation[offset]));
          }
          const float activation_scale = maximum / 127.0f;
          std::int32_t dot = 0;
          for (std::size_t offset = 0; offset < 32; ++offset) {
            const auto quantized_activation = static_cast<std::int32_t>(
                std::nearbyint(activation[offset] * (maximum == 0.0f ? 0.0f : 127.0f / maximum)));
            const std::uint8_t byte = weights[(column * blocks + block) * 16 + offset / 2];
            const std::int32_t quantized_weight =
                static_cast<std::int32_t>((byte >> ((offset % 2) * 4)) & 15) - 8;
            dot += quantized_activation * quantized_weight;
          }
          expected += static_cast<float>(dot) * activation_scale * scales[column * blocks + block];
        }
        EXPECT_NEAR(output.AsFloat()[row * n + column], expected, 2e-6f);
      }
    }
  };

  check(packed);
  packed[0] ^= 0x0f;
  check(packed);
}

TEST(MatMulNBitsKernel, PreparesOnlyImmutableWeightsAndScales) {
  constexpr std::int64_t k = 64;
  constexpr std::int64_t n = 16;
  const ONNX_LIGHT_NAMESPACE::NodeProto node = MakeNode(k, n);
  onnx_light_cpu::MatMulNBitsKernel kernel{
      node, rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};

  EXPECT_EQ(kernel.HasPreparations({"B", "scales"}),
            ml_cpu_compute::MatMulNBitsAccuracy4Float32Available());
  EXPECT_FALSE(kernel.HasPreparations({"B"}));
  EXPECT_FALSE(kernel.HasPreparations({"scales"}));

  rt_ns::RuntimeContext rt{rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};
  rt.Put("B", rt_ns::Tensor::FromUint8(
                  "B", {n, k / 32, 16},
                  std::vector<std::uint8_t>(static_cast<std::size_t>(n * k / 2), 0x99)));
  rt.Put("scales",
         rt_ns::Tensor::FromFloat("scales", {n, k / 32},
                                  std::vector<float>(static_cast<std::size_t>(n * k / 32), 0.25f)));
  kernel.Prepare(rt, {"B", "scales"});
  EXPECT_EQ(rt.value_store().prepared_bytes(),
            ml_cpu_compute::MatMulNBitsAccuracy4Float32Available() ? 768U : 0U);

  const rt_ns::Tensor a =
      rt_ns::Tensor::FromFloat("A", {1, k}, std::vector<float>(static_cast<std::size_t>(k), 0.5f));
  const rt_ns::Tensor output = kernel(a, rt.Get("B"), rt.Get("scales"));
  for (std::size_t index = 0; index < output.element_count(); ++index) {
    EXPECT_FLOAT_EQ(output.AsFloat()[index], 8.0f);
  }
}

TEST(MatMulNBitsKernel, PreparesInt8OnlyForImmutableInputsAndAccountsForResidency) {
  constexpr std::int64_t k = 64;
  constexpr std::int64_t n = 13;
  const ONNX_LIGHT_NAMESPACE::NodeProto node = MakeNode(k, n, 8);
  onnx_light_cpu::MatMulNBitsKernel kernel{
      node, rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};
  const bool available = ml_cpu_compute::MatMulNBitsInt8Float32Available();
  EXPECT_EQ(kernel.HasPreparations({"B", "scales"}), available);
  EXPECT_FALSE(kernel.HasPreparations({"B"}));
  EXPECT_FALSE(kernel.HasPreparations({"scales"}));
  rt_ns::RuntimeContext rt{rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};
  rt.Put("B",
         rt_ns::Tensor::FromUint8("B", {n, k / 32, 32},
                                  std::vector<std::uint8_t>(static_cast<std::size_t>(n * k), 129)));
  rt.Put("scales",
         rt_ns::Tensor::FromFloat("scales", {n, k / 32},
                                  std::vector<float>(static_cast<std::size_t>(n * k / 32), 0.25f)));
  kernel.Prepare(rt, {"B", "scales"});
  const std::size_t expected_bytes = 2 * 2 * (256 + 8 * 2 * sizeof(float));
  EXPECT_EQ(rt.value_store().prepared_bytes(), available ? expected_bytes : 0U);
  kernel.Prepare(rt, {"B", "scales"});
  EXPECT_EQ(rt.value_store().prepared_bytes(), available ? expected_bytes : 0U);

  const rt_ns::Tensor a = rt_ns::Tensor::FromFloat("A", {3, k}, std::vector<float>(3 * k, 0.5f));
  const rt_ns::Tensor output = kernel(a, rt.Get("B"), rt.Get("scales"));
  for (std::size_t index = 0; index < output.element_count(); ++index) {
    EXPECT_FLOAT_EQ(output.AsFloat()[index], 8.0f);
  }
}

TEST(MatMulNBitsKernel, Int8HalfScalesKeepGenericFallback) {
  const ONNX_LIGHT_NAMESPACE::NodeProto node = MakeNode(32, 8, 8);
  onnx_light_cpu::MatMulNBitsKernel kernel{
      node, rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};
  rt_ns::RuntimeContext rt{rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};
  rt.Put("B", rt_ns::Tensor::FromUint8("B", {8, 1, 32}, std::vector<std::uint8_t>(256, 129)));
  rt.Put("scales", rt_ns::MakeFloat16Tensor("scales", {8, 1}, std::vector<float>(8, 0.25f)));
  kernel.Prepare(rt, {"B", "scales"});
  EXPECT_EQ(rt.value_store().prepared_bytes(), 0U);
  const rt_ns::Tensor a = rt_ns::MakeFloat16Tensor("A", {1, 32}, std::vector<float>(32, 0.5f));
  const rt_ns::Tensor y = kernel(a, rt.Get("B"), rt.Get("scales"));
  for (std::size_t i = 0; i < y.element_count(); ++i) {
    EXPECT_FLOAT_EQ(ReadValue(y, i), 4.0f);
  }
}

TEST(MatMulNBitsKernel, PreparedInt8HonorsOutputTuning) {
  constexpr std::int64_t k = 32;
  constexpr std::int64_t n = 16;
  constexpr std::int64_t rows = 9;
  const auto node = MakeNode(k, n, 8);
  onnx_light_cpu::MatMulNBitsKernel kernel{
      node, rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};
  if (!ml_cpu_compute::MatMulNBitsInt8Float32Available()) {
    GTEST_SKIP() << "The prepared INT8 kernel is unavailable.";
  }
  onnx_light_cpu::MatMulNBitsKernel::RegisterTuningSchemas();
  const auto key = kernel.TuningKey(static_cast<std::int32_t>(rt_ns::DataType::FLOAT));
  auto parameters = rt_ns::GetKernelTuningRegistry().FindSchema(key)->portable_defaults();
  parameters.values["parallel.threshold_outputs"] = rows * n + 1;
  parameters.values["parallel.target_block_outputs"] = n;
  parameters.values["parallel.max_participants"] = std::int64_t{4};
  kernel.Configure(parameters);
  rt_ns::RuntimeContext rt{rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};
  rt.Put("B", rt_ns::Tensor::FromUint8("B", {n, 1, 32}, std::vector<std::uint8_t>(n * k, 129)));
  rt.Put("scales", rt_ns::Tensor::FromFloat("scales", {n, 1}, std::vector<float>(n, 0.25f)));
  kernel.Prepare(rt, {"B", "scales"});
  const auto a = rt_ns::Tensor::FromFloat("A", {rows, k}, std::vector<float>(rows * k, 0.5f));
  InlineExecutor executor;
  ml_cpu_compute::ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run};
  ml_cpu_compute::ExecutionExecutorScope scope(&view);
  const auto check = [&] {
    const auto y = kernel(a, rt.Get("B"), rt.Get("scales"));
    for (std::size_t i = 0; i < y.element_count(); ++i) {
      EXPECT_FLOAT_EQ(y.AsFloat()[i], 4.0f);
    }
  };
  check();
  EXPECT_EQ(executor.dispatches, 0);
  parameters.values["parallel.threshold_outputs"] = rows * n;
  parameters.values["parallel.target_block_outputs"] = 3 * n;
  kernel.Configure(parameters);
  check();
  EXPECT_EQ(executor.dispatches, 1);
  EXPECT_EQ(executor.blocks, 3);
  parameters.values["parallel.target_block_outputs"] = rows * n;
  kernel.Configure(parameters);
  check();
  EXPECT_EQ(executor.dispatches, 1);
}

TEST(MatMulNBitsKernel, Accuracy4DynamicPlanSupportsHalfPrecisionInputsAndOutputs) {
  if (!ml_cpu_compute::MatMulNBitsAccuracy4Float32Available()) {
    GTEST_SKIP() << "The optimized accuracy-level-4 kernel is unavailable.";
  }
  constexpr std::size_t rows = 9;
  constexpr std::size_t k = 64;
  constexpr std::size_t n = 16;
  constexpr std::size_t blocks = k / 32;
  for (const rt_ns::DataType data_type : {rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16}) {
    const rt_ns::Tensor input =
        MakeTensor(data_type, {rows, k}, std::vector<float>(rows * k, 0.5f));
    const rt_ns::Tensor scales =
        MakeTensor(data_type, {n, blocks}, std::vector<float>(n * blocks, 0.25f));
    const rt_ns::Tensor bias = MakeTensor(data_type, {n}, std::vector<float>(n, 0.5f));
    const onnx_light_cpu::MatMulNBitsKernel kernel{
        MakeNode(k, n), rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};

    const auto check = [&](std::uint8_t packed, float expected) {
      const rt_ns::Tensor weights = rt_ns::Tensor::FromUint8(
          "", {n, blocks, 16}, std::vector<std::uint8_t>(n * blocks * 16, packed));
      const rt_ns::Tensor output = kernel(input, weights, scales, &bias);
      ASSERT_EQ(output.data_type, static_cast<std::int32_t>(data_type));
      for (std::size_t index = 0; index < rows * n; ++index) {
        EXPECT_FLOAT_EQ(ReadValue(output, index), expected);
      }
    };
    check(0x99, 8.5f);
    check(0xaa, 16.5f);
  }
}

TEST(MatMulNBitsKernel, SupportsEveryNonDoubleFloatAndIntWidth) {
  for (const rt_ns::DataType data_type :
       {rt_ns::DataType::FLOAT, rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16}) {
    for (const std::int64_t bits : {2, 4, 8}) {
      const std::size_t blob_size = static_cast<std::size_t>(4 * bits);
      const std::uint8_t packed_value = bits == 2 ? 0xff : bits == 4 ? 0x99 : 0x81;
      const rt_ns::Tensor a = MakeTensor(data_type, {1, 33}, std::vector<float>(33, 1.0f));
      const rt_ns::Tensor b =
          rt_ns::Tensor::FromUint8("", {1, 2, static_cast<std::int64_t>(blob_size)},
                                   std::vector<std::uint8_t>(2 * blob_size, packed_value));
      const rt_ns::Tensor scales = MakeTensor(data_type, {1, 2}, {0.5f, 0.5f});
      const rt_ns::Tensor bias = MakeTensor(data_type, {1}, {0.25f});
      const onnx_light_cpu::MatMulNBitsKernel kernel{
          MakeNode(33, 1, bits), rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};
      const rt_ns::Tensor y = kernel(a, b, scales, &bias);
      EXPECT_EQ(y.data_type, static_cast<std::int32_t>(data_type));
      EXPECT_FLOAT_EQ(ReadValue(y, 0), 16.75f)
          << "type=" << static_cast<int>(data_type) << ", bits=" << bits;
    }
  }
}

TEST(MatMulNBitsKernel, BorrowsEncodedWeightsWithoutRepackingThem) {
  constexpr std::int64_t k = 64;
  constexpr std::int64_t n = 16;
  for (const rt_ns::DataType data_type :
       {rt_ns::DataType::FLOAT, rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16}) {
    for (const std::int64_t bits : {2, 4, 8}) {
      ONNX_LIGHT_NAMESPACE::NodeProto node = MakeNode(k, n, bits);
      node.ref_input().resize(2);
      onnx_light_cpu::MatMulNBitsKernel kernel{
          node, rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};
      rt_ns::RuntimeContext rt{rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};
      rt.Put("A", MakeTensor(data_type, {1, k}, std::vector<float>(k, 1.0f)));
      rt.PutValue("B", rt_ns::RuntimeValue(MakeEncodedWeights(data_type, k, n, bits)));

      EXPECT_TRUE(kernel.HasPreparations({"B"}));
      kernel.Prepare(rt, {"B"});
      EXPECT_EQ(rt.value_store().prepared_bytes(), 0U);
      kernel.Run(rt);

      const rt_ns::Tensor &output = rt.Get("Y");
      ASSERT_EQ(output.shape, (rt_ns::Shape{1, n}));
      for (std::size_t index = 0; index < output.element_count(); ++index) {
        EXPECT_FLOAT_EQ(ReadValue(output, index), 64.0f)
            << "type=" << static_cast<int>(data_type) << ", bits=" << bits;
      }
    }
  }
}

TEST(MatMulNBitsKernel, RunsUnpreparedEncodedWeights) {
  constexpr std::int64_t k = 64;
  constexpr std::int64_t n = 16;
  const rt_ns::Tensor input = rt_ns::Tensor::FromFloat("", {1, k}, std::vector<float>(k, 1.0f));
  for (const std::int64_t bits : {2, 4, 8}) {
    ONNX_LIGHT_NAMESPACE::NodeProto node = MakeNode(k, n, bits);
    node.ref_input().resize(2);
    const onnx_light_cpu::MatMulNBitsKernel kernel{
        node, rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};

    const rt_ns::Tensor output =
        kernel(input, MakeEncodedWeights(rt_ns::DataType::FLOAT, k, n, bits));

    ASSERT_EQ(output.shape, (rt_ns::Shape{1, n}));
    for (std::size_t index = 0; index < output.element_count(); ++index) {
      EXPECT_FLOAT_EQ(output.AsFloat()[index], 64.0f) << "bits=" << bits;
    }
  }
}

TEST(MatMulNBitsKernel, RuntimeSessionPreparesEncodedWeightsWithoutSeparateScales) {
  constexpr std::int64_t k = 64;
  constexpr std::int64_t n = 16;
  for (const std::int64_t bits : {2, 4, 8}) {
    ONNX_LIGHT_NAMESPACE::ModelProto model;
    model.add_opset_import()->set_version(23);
    auto *microsoft_opset = model.add_opset_import();
    microsoft_opset->set_domain("com.microsoft");
    microsoft_opset->set_version(1);
    auto *graph = model.mutable_graph();
    graph->add_input()->set_name("A");
    graph->add_output()->set_name("Y");
    auto encoded = MakeEncodedWeights(rt_ns::DataType::FLOAT, k, n, bits);
    encoded.set_name("B");
    *graph->add_encoded_initializer() = std::move(encoded);
    ONNX_LIGHT_NAMESPACE::NodeProto node = MakeNode(k, n, bits);
    node.ref_input().resize(2);
    *graph->add_node() = std::move(node);

    rt_ns::RuntimeContext rt{rt_ns::KernelContext{rt_ns::DefaultOpset(23)}};
    ASSERT_TRUE(onnx_light_cpu::RegisterKernelForSession(rt, "com.microsoft", "MatMulNBits"));
    rt_ns::RuntimeSession session(model);
    for (float value : {1.0f, 2.0f}) {
      rt.Put("A", rt_ns::Tensor::FromFloat("A", {1, k}, std::vector<float>(k, value)));
      session.Run(rt);
      const rt_ns::Tensor &output = rt.Get("Y");
      ASSERT_EQ(output.shape, (rt_ns::Shape{1, n}));
      for (std::size_t index = 0; index < output.element_count(); ++index) {
        EXPECT_FLOAT_EQ(output.AsFloat()[index], 64.0f * value) << "bits=" << bits;
      }
    }
    EXPECT_EQ(session.prepared_bytes(), 0U);
  }
}

TEST(MatMulNBitsKernel, RejectsUnsupportedEncodedWeightContracts) {
  const rt_ns::Tensor input = rt_ns::Tensor::FromFloat("", {1, 32}, std::vector<float>(32, 1.0f));
  ONNX_LIGHT_NAMESPACE::NodeProto node = MakeNode(32, 1);
  node.ref_input().resize(2);
  onnx_light_cpu::MatMulNBitsKernel kernel{
      node, rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};

  EXPECT_THROW(kernel(input, MakeEncodedWeights(rt_ns::DataType::FLOAT, 32, 1, 4, 64)),
               std::invalid_argument);
  for (const auto &encoded : {MakeEncodedWeights(rt_ns::DataType::FLOAT, 32, 1, 4, 64)}) {
    rt_ns::RuntimeContext rt{rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};
    rt.PutValue("B", rt_ns::RuntimeValue(encoded));
    EXPECT_THROW(kernel.Prepare(rt, {"B"}), std::invalid_argument);
    EXPECT_EQ(rt.value_store().prepared_bytes(), 0U);
  }
}

TEST(MatMulNBitsKernel, RegistersTypeAndBitSpecificTuningSchemas) {
  onnx_light_cpu::MatMulNBitsKernel::RegisterTuningSchemas();
  for (const rt_ns::DataType data_type :
       {rt_ns::DataType::FLOAT, rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16}) {
    for (const std::int64_t bits : {2, 4, 8}) {
      const onnx_light_cpu::MatMulNBitsKernel kernel{
          MakeNode(32, 1, bits), rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};
      const auto key = kernel.TuningKey(static_cast<std::int32_t>(data_type));
      EXPECT_EQ(key.library, "onnx_light_cpu");
      EXPECT_EQ(key.kernel, "MatMulNBits");
      EXPECT_EQ(key.implementation, "packed_int" + std::to_string(bits));
      EXPECT_EQ(key.element_type, static_cast<std::int32_t>(data_type));
      EXPECT_EQ(key.tuning_abi, onnx_light_cpu::MatMulNBitsKernel::kTuningAbi);
      EXPECT_NE(rt_ns::GetKernelTuningRegistry().FindSchema(key), nullptr);
    }
  }
  const onnx_light_cpu::MatMulNBitsKernel kernel{
      MakeNode(32, 1), rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};
  EXPECT_TRUE(kernel.TuningKey(static_cast<std::int32_t>(rt_ns::DataType::DOUBLE)).library.empty());
}

TEST(MatMulNBitsKernel, RejectsUnsupportedContractAndInvalidShapes) {
  EXPECT_THROW((onnx_light_cpu::MatMulNBitsKernel{
                   MakeNode(32, 1, 5), rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}}),
               std::invalid_argument);
  EXPECT_THROW(
      (onnx_light_cpu::MatMulNBitsKernel{MakeNode(32, 1, 4, 48),
                                         rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}}),
      std::invalid_argument);

  const onnx_light_cpu::MatMulNBitsKernel kernel{
      MakeNode(32, 1), rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};
  const rt_ns::Tensor b =
      rt_ns::Tensor::FromUint8("", {1, 1, 16}, std::vector<std::uint8_t>(16, 0x88));
  const rt_ns::Tensor scales = rt_ns::Tensor::FromFloat("", {1, 1}, {1.0f});
  EXPECT_THROW(kernel(rt_ns::Tensor::FromDouble("", {1, 32}, std::vector<double>(32, 1.0)), b,
                      rt_ns::Tensor::FromDouble("", {1, 1}, {1.0})),
               std::invalid_argument);
  EXPECT_THROW(kernel(rt_ns::Tensor::FromFloat("", {1, 31}, std::vector<float>(31)), b, scales),
               std::invalid_argument);
  EXPECT_THROW(kernel(rt_ns::Tensor::FromFloat("", {1, 32}, std::vector<float>(32)),
                      rt_ns::Tensor::FromUint8("", {1, 1, 15}, std::vector<std::uint8_t>(15, 0x88)),
                      scales),
               std::invalid_argument);
}

TEST(MatMulNBitsKernel, LowLevelRejectsUnsupportedBlockSize) {
  float value = 0.0f;
  const std::uint8_t packed = 0x88;
  EXPECT_THROW(
      ml_cpu_compute::MatMulNBitsFloat32(&value, &packed, &value, nullptr, &value, 1, 1, 1, 24),
      std::invalid_argument);
}

TEST(MatMulNBitsKernel, Non32EncodedGeometryPreparationAndDequantizationParity) {
  for (const std::int64_t block_size : {16, 64, 128, 256}) {
    const std::int64_t k = 2 * block_size + 3;
    constexpr std::int64_t n = 16;
    for (const std::int64_t bits : {2, 4, 8}) {
      for (const auto dtype :
           {rt_ns::DataType::FLOAT, rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16}) {
        for (double zero : {-1.0, 1.0, 0.5}) {
          SCOPED_TRACE(::testing::Message() << block_size << ',' << bits << ',' << zero);
          const auto encoded = MakeEncodedWeights(dtype, k, n, bits, block_size, zero);
          const auto weights = rt_ns::DequantizeTensor(encoded);
          auto node = MakeNode(k, n, bits, block_size);
          onnx_light_cpu::MatMulNBitsKernel raw_kernel{
              node, rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};
          EXPECT_FALSE(raw_kernel.HasPreparations({"B", "scales"}));
          node.ref_input().resize(2);
          onnx_light_cpu::MatMulNBitsKernel kernel{
              node, rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};
          const auto a = MakeTensor(dtype, {2, k}, std::vector<float>(2 * k, 0.5f));
          const auto bias = MakeTensor(dtype, {n}, std::vector<float>(n, 0.25f));
          const auto output = kernel(a, encoded, &bias);
          for (std::size_t column = 0; column < n; ++column) {
            float expected = 0.25f;
            for (std::size_t p = 0; p < static_cast<std::size_t>(k); ++p) {
              expected += 0.5f * ReadValue(weights, p * n + column);
            }
            const auto rounded = MakeTensor(dtype, {1}, {expected});
            EXPECT_FLOAT_EQ(ReadValue(output, column), ReadValue(rounded, 0));
          }
          rt_ns::RuntimeContext rt{rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};
          rt.Put("A", a);
          rt.PutValue("B", rt_ns::RuntimeValue(encoded));
          EXPECT_TRUE(kernel.HasPreparations({"B"}));
          kernel.Prepare(rt, {"B"});
          kernel.Run(rt);
          const auto unprepared = kernel(a, encoded);
          for (std::size_t i = 0; i < unprepared.element_count(); ++i) {
            EXPECT_FLOAT_EQ(ReadValue(rt.Get("Y"), i), ReadValue(unprepared, i));
          }
          namespace shapes_ns = ONNX_LIGHT_NAMESPACE::core::shapes;
          namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;
          shapes_ns::ShapesContext ctx;
          const auto type = dtype == rt_ns::DataType::FLOAT     ? sym_ns::TensorType::kFloat
                            : dtype == rt_ns::DataType::FLOAT16 ? sym_ns::TensorType::kFloat16
                                                                : sym_ns::TensorType::kBfloat16;
          ctx.Set("A", sym_ns::SymTensor(nullptr, type, {sym_ns::SymDim(2), sym_ns::SymDim(k)}));
          ctx.SetEncodedValue("B", encoded);
          EXPECT_NO_THROW(onnx_light_cpu::ComputeShapeMatMulNBits(ctx, node));
          const onnx_light_cpu::MatMulNBitsKernel wrong_geometry{
              MakeNode(k, n, bits, block_size * 2),
              rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}};
          EXPECT_THROW(wrong_geometry(a, encoded), std::invalid_argument);
        }
      }
    }
  }
}

TEST(MatMulNBitsKernel, RejectsInvalidGroupsAndZeroPointShapes) {
  const rt_ns::KernelContext ctx{rt_ns::OpsetId("com.microsoft", 1)};
  for (const std::int64_t block :
       std::initializer_list<std::int64_t>{-32, 0, 8, 24, 48, INT64_C(4294967296)}) {
    EXPECT_THROW((onnx_light_cpu::MatMulNBitsKernel{MakeNode(128, 1, 2, block), ctx}),
                 std::invalid_argument);
  }
  EXPECT_THROW((onnx_light_cpu::MatMulNBitsKernel{MakeNode(128, 1, 3, 128), ctx}),
               std::invalid_argument);
  const onnx_light_cpu::MatMulNBitsKernel kernel{MakeNode(256, 16, 2, 128), ctx};
  const auto a = MakeTensor(rt_ns::DataType::FLOAT, {1, 256}, std::vector<float>(256, 1));
  const auto b = rt_ns::Tensor::FromUint8("", {16, 2, 32}, std::vector<std::uint8_t>(1024, 0x55));
  const auto scales = MakeTensor(rt_ns::DataType::FLOAT, {16, 2}, std::vector<float>(32, 1));
  const auto zeros = rt_ns::Tensor::FromUint8("", {16, 2}, std::vector<std::uint8_t>(32, 1));
  EXPECT_THROW(kernel(a, b, scales, nullptr, nullptr, &zeros), std::invalid_argument);
}

TEST(MatMulNBitsKernel, AlignedNon32TensorGroupsDoNotUseBlock32Preparation) {
  const rt_ns::KernelContext ctx{rt_ns::OpsetId("com.microsoft", 1)};
  for (const std::int64_t block : {16, 64, 128, 256}) {
    for (const std::int64_t bits : {2, 4, 8}) {
      const auto k = 2 * block;
      const std::uint8_t packed = bits == 2 ? 0xff : bits == 4 ? 0x99 : 0x81;
      const auto b =
          rt_ns::Tensor::FromUint8("", {16, 2, block * bits / 8},
                                   std::vector<std::uint8_t>(16 * 2 * block * bits / 8, packed));
      std::vector<float> values(32);
      for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = i % 2 == 0 ? 0.5f : 1.0f;
      }
      for (const auto dtype :
           {rt_ns::DataType::FLOAT, rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16}) {
        const auto node = MakeNode(k, 16, bits, block);
        onnx_light_cpu::MatMulNBitsKernel kernel{node, ctx};
        rt_ns::RuntimeContext rt{ctx};
        rt.Put("A", MakeTensor(dtype, {2, k}, std::vector<float>(2 * k, 1)));
        rt.Put("B", b);
        rt.Put("scales", MakeTensor(dtype, {16, 2}, values));
        EXPECT_FALSE(kernel.HasPreparations({"B", "scales"}));
        EXPECT_NO_THROW(kernel.Prepare(rt, {"B", "scales"}));
        kernel.Run(rt);
        EXPECT_EQ(rt.value_store().prepared_bytes(), 0U);
        for (std::size_t i = 0; i < 32; ++i) {
          EXPECT_FLOAT_EQ(ReadValue(rt.Get("Y"), i), 1.5f * block);
        }
      }
    }
  }
}

TEST(MatMulNBitsKernel, ExplicitZeroPointsDisableBlock32Preparation) {
  const rt_ns::KernelContext ctx{rt_ns::OpsetId("com.microsoft", 1)};
  for (const std::int64_t bits : {4, 8}) {
    auto node = MakeNode(64, 16, bits);
    node.add_input("zero_points");
    onnx_light_cpu::MatMulNBitsKernel kernel{node, ctx};
    const auto a = rt_ns::Tensor::FromFloat("", {1, 64}, std::vector<float>(64, 1));
    const auto b = rt_ns::Tensor::FromUint8(
        "", {16, 2, 4 * bits},
        std::vector<std::uint8_t>(16 * 2 * 4 * bits, bits == 4 ? 0x99 : 0x81));
    const auto scales = rt_ns::Tensor::FromFloat("", {16, 2}, std::vector<float>(32, 1));
    const auto stride = bits == 4 ? 1 : 2;
    const auto zeros = rt_ns::Tensor::FromUint8(
        "", {16, stride}, std::vector<std::uint8_t>(16 * stride, bits == 4 ? 0x77 : 0x7f));
    const auto direct = kernel(a, b, scales, nullptr, nullptr, &zeros);
    rt_ns::RuntimeContext rt{ctx};
    rt.Put("A", a);
    rt.Put("B", b);
    rt.Put("scales", scales);
    rt.Put("zero_points", zeros);
    EXPECT_FALSE(kernel.HasPreparations({"B", "scales", "zero_points"}));
    kernel.Prepare(rt, {"B", "scales", "zero_points"});
    kernel.Run(rt);
    EXPECT_EQ(rt.value_store().prepared_bytes(), 0U);
    for (std::size_t i = 0; i < 16; ++i) {
      EXPECT_FLOAT_EQ(ReadValue(direct, i), 128);
      EXPECT_FLOAT_EQ(ReadValue(rt.Get("Y"), i), 128);
    }
  }
}

TEST(MatMulNBitsKernel, SchemaAndShapeInferenceAcceptNon32ZeroPoints) {
  namespace shapes_ns = ONNX_LIGHT_NAMESPACE::core::shapes;
  namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;
  namespace schema_ns = ONNX_LIGHT_NAMESPACE::core::schema;
  const auto schemas = onnx_light_cpu::GetMicrosoftOpSchemasWithHistory("MatMulNBits", false);
  ASSERT_EQ(schemas.size(), 1U);
  for (const auto type :
       {sym_ns::TensorType::kFloat, sym_ns::TensorType::kFloat16, sym_ns::TensorType::kBfloat16}) {
    for (bool packed : {false, true}) {
      auto node = MakeNode(515, 3, 2, 128);
      node.add_input("zero_points");
      shapes_ns::ShapesContext ctx;
      ctx.Set("A", sym_ns::SymTensor(nullptr, type, {2, 515}));
      ctx.Set("B", sym_ns::SymTensor(nullptr, sym_ns::TensorType::kUint8, {3, 5, 32}));
      ctx.Set("scales", sym_ns::SymTensor(nullptr, type, {3, 5}));
      ctx.Set("zero_points", sym_ns::SymTensor(nullptr, packed ? sym_ns::TensorType::kUint8 : type,
                                               {3, packed ? 2 : 5}));
      const std::vector<std::optional<schema_ns::SchemaInputValue>> inputs = {
          schema_ns::SchemaInputValue(ctx.Get("A")), schema_ns::SchemaInputValue(ctx.Get("B")),
          schema_ns::SchemaInputValue(ctx.Get("scales")),
          schema_ns::SchemaInputValue(ctx.Get("zero_points"))};
      EXPECT_NO_THROW(schemas[0].Verify(node, &inputs));
      EXPECT_NO_THROW(onnx_light_cpu::ComputeShapeMatMulNBits(ctx, node));
      EXPECT_EQ(ctx.Get("Y").Shape(), sym_ns::SymShape({2, 3}));
      ctx.Set("zero_points", sym_ns::SymTensor(nullptr, type, {3, 5, 1}));
      EXPECT_THROW(onnx_light_cpu::ComputeShapeMatMulNBits(ctx, node), std::invalid_argument);
    }
  }
}

TEST(MatMulNBitsKernel, UsesRuntimeSchedulingAndSuppressesNestedParallelism) {
  const std::vector<float> a(32, 1.0f);
  const std::vector<std::uint8_t> packed(512 * 16, 0x88);
  const std::vector<float> scales(512, 1.0f);
  std::vector<float> output(512);
  InlineExecutor executor;
  ml_cpu_compute::ExecutionExecutorView view{&executor, 8, &InlineExecutor::Run};
  ml_cpu_compute::ExecutionExecutorScope scope(&view);

  ml_cpu_compute::MatMulNBitsFloat32(a.data(), packed.data(), scales.data(), nullptr, output.data(),
                                     1, 32, 2, 32);
  EXPECT_EQ(executor.dispatches, 0);

  ml_cpu_compute::MatMulNBitsFloat32(a.data(), packed.data(), scales.data(), nullptr, output.data(),
                                     1, 32, 512, 32);
  EXPECT_EQ(executor.dispatches, 1);
  EXPECT_GT(executor.blocks, 1);

  executor = {};
  {
    ml_cpu_compute::detail::ExecutionRegionScope region;
    ml_cpu_compute::MatMulNBitsFloat32(a.data(), packed.data(), scales.data(), nullptr,
                                       output.data(), 1, 32, 512, 32);
  }
  EXPECT_EQ(executor.dispatches, 0);
}

} // namespace
