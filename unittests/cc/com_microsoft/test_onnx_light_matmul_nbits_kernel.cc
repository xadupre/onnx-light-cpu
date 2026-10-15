// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/com_microsoft/matmul_nbits.h"
#include "onnx_light_cpu/impl/com_microsoft/matmul_nbits_int3.h"
#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/kernels/com_microsoft/matmul_nbits_kernel.h"
#include "onnx_light_cpu/kernels/register_kernels.h"
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
#include <string>
#include <unordered_set>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

struct InlineExecutor {
  std::int64_t dispatches = 0;
  std::int64_t blocks = 0;

  static void Run(void *context, std::int64_t num_blocks, void *task_context,
                  onnx_light_cpu::ExecutionBlockFn task) {
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
             ? onnx_light_cpu::detail::Float16BitsToFloat(value)
             : onnx_light_cpu::detail::Bfloat16BitsToFloat(value);
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
  std::string payload(valid.raw_data().data(), valid.raw_data().size());
  payload[13] |= static_cast<char>(0x80);
  bad.set_raw_data(payload);
  EXPECT_THROW(kernel(a, bad), std::invalid_argument);
  bad = valid;
  bad.set_raw_data(payload.data(), payload.size() - 1);
  EXPECT_THROW(kernel(a, bad), std::invalid_argument);
  bad = valid;
  payload.assign(valid.raw_data().data(), valid.raw_data().size());
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
  rt.PutValue("B", rt_ns::RuntimeValue(MakeInt3Weights(
                       rt_ns::DataType::FLOAT, 35, 1, std::vector<std::uint8_t>(35, 5),
                       {0.5f, 2.0f})));
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
  if (!onnx_light_cpu::MatMulNBitsAccuracy4Float32Available()) {
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
            onnx_light_cpu::MatMulNBitsAccuracy4Float32Available());
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
            onnx_light_cpu::MatMulNBitsAccuracy4Float32Available() ? 768U : 0U);

  const rt_ns::Tensor a =
      rt_ns::Tensor::FromFloat("A", {1, k}, std::vector<float>(static_cast<std::size_t>(k), 0.5f));
  const rt_ns::Tensor output = kernel(a, rt.Get("B"), rt.Get("scales"));
  for (std::size_t index = 0; index < output.element_count(); ++index) {
    EXPECT_FLOAT_EQ(output.AsFloat()[index], 8.0f);
  }
}

TEST(MatMulNBitsKernel, Accuracy4DynamicPlanSupportsHalfPrecisionInputsAndOutputs) {
  if (!onnx_light_cpu::MatMulNBitsAccuracy4Float32Available()) {
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
  EXPECT_THROW(kernel(input, MakeEncodedWeights(rt_ns::DataType::FLOAT, 32, 1, 4, 32, 7.0)),
               std::invalid_argument);

  for (const auto &encoded : {MakeEncodedWeights(rt_ns::DataType::FLOAT, 32, 1, 4, 64),
                              MakeEncodedWeights(rt_ns::DataType::FLOAT, 32, 1, 4, 32, 7.0)}) {
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
                   MakeNode(32, 1, 3), rt_ns::KernelContext{rt_ns::OpsetId("com.microsoft", 1)}}),
               std::invalid_argument);
  EXPECT_THROW(
      (onnx_light_cpu::MatMulNBitsKernel{MakeNode(32, 1, 4, 64),
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
      onnx_light_cpu::MatMulNBitsFloat32(&value, &packed, &value, nullptr, &value, 1, 1, 1, 16),
      std::invalid_argument);
}

TEST(MatMulNBitsKernel, UsesRuntimeSchedulingAndSuppressesNestedParallelism) {
  const std::vector<float> a(32, 1.0f);
  const std::vector<std::uint8_t> packed(512 * 16, 0x88);
  const std::vector<float> scales(512, 1.0f);
  std::vector<float> output(512);
  InlineExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 8, &InlineExecutor::Run};
  onnx_light_cpu::ExecutionExecutorScope scope(&view);

  onnx_light_cpu::MatMulNBitsFloat32(a.data(), packed.data(), scales.data(), nullptr, output.data(),
                                     1, 32, 2, 32);
  EXPECT_EQ(executor.dispatches, 0);

  onnx_light_cpu::MatMulNBitsFloat32(a.data(), packed.data(), scales.data(), nullptr, output.data(),
                                     1, 32, 512, 32);
  EXPECT_EQ(executor.dispatches, 1);
  EXPECT_GT(executor.blocks, 1);

  executor = {};
  {
    onnx_light_cpu::detail::ExecutionRegionScope region;
    onnx_light_cpu::MatMulNBitsFloat32(a.data(), packed.data(), scales.data(), nullptr,
                                       output.data(), 1, 32, 512, 32);
  }
  EXPECT_EQ(executor.dispatches, 0);
}

} // namespace
