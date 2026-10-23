// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/com_microsoft/gather_block_quantized_kernel.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"

#include "ml_cpu_compute/math/half_conversion.h"

#include <gtest/gtest.h>

#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

namespace runtime = ONNX_LIGHT_NAMESPACE::core::runtime;
using onnx_light_cpu::GatherBlockQuantizedAttributes;
using onnx_light_cpu::GatherBlockQuantizedKernel;
using runtime::DataType;
using runtime::Shape;
using runtime::Tensor;

runtime::KernelContext Context() {
  return runtime::KernelContext(runtime::OpsetId("com.microsoft", 1));
}

TEST(OnnxLightGatherBlockQuantizedKernel, ScalarAndMultidimensionalIndicesAllBitsAndTypes) {
  for (int64_t bits : {2, 4, 8}) {
    const int64_t components = 8 / bits;
    // The data value is always default_zp + 1.
    const unsigned value = (1u << (bits - 1)) + 1;
    uint8_t packed = 0;
    for (int64_t i = 0; i < components; ++i) {
      packed |= static_cast<uint8_t>(value << (bits * i));
    }
    const Tensor data =
        Tensor::From<uint8_t>("data", {2, 2, 1, 9}, std::vector<uint8_t>(36, packed));
    const int64_t blocks = (9 * components + 15) / 16;
    const Tensor scales =
        Tensor::FromFloat("scales", {2, 2, 1, blocks}, std::vector<float>(4 * blocks, 0.5f));
    GatherBlockQuantizedKernel kernel(Context(), {-4, -1, 16, bits});
    for (const Tensor &indices : {Tensor::From<int32_t>("indices", {}, {-1}),
                                  Tensor::From<int64_t>("indices", {1, 2, 2}, {1, -2, -3, 2})}) {
      const Tensor output = kernel(data, indices, scales);
      Shape expected = indices.shape;
      expected.push_back(2);
      expected.push_back(1);
      expected.push_back(9 * components);
      EXPECT_EQ(output.shape, expected);
      EXPECT_EQ(output.data_type, DataType::FLOAT);
      const int64_t valid_count =
          indices.shape.empty() ? output.element_count() : output.element_count() / 2;
      for (int64_t i = 0; i < output.element_count(); ++i) {
        EXPECT_EQ(output.AsFloat()[i], i < valid_count ? 0.5f : 0.0f);
      }
      Tensor half_scales = Tensor::From<uint16_t>(
          "scales", scales.shape,
          std::vector<uint16_t>(4 * blocks, ml_cpu_compute::detail::FloatToFloat16Bits(0.5f)));
      half_scales.data_type = DataType::FLOAT16;
      const Tensor half_output = kernel(data, indices, half_scales);
      EXPECT_EQ(half_output.shape, expected);
      EXPECT_EQ(half_output.data_type, DataType::FLOAT16);
      const auto *half_values = reinterpret_cast<const uint16_t *>(half_output.bytes());
      for (int64_t i = 0; i < half_output.element_count(); ++i) {
        EXPECT_EQ(ml_cpu_compute::detail::Float16BitsToFloat(half_values[i]), output.AsFloat()[i]);
      }
    }
  }
}

TEST(OnnxLightGatherBlockQuantizedKernel, ExplicitZeroPointsResetAtEveryLastDimensionRow) {
  const Tensor data = Tensor::From<uint8_t>("data", {2, 2, 17}, std::vector<uint8_t>(68, 0x99));
  const Tensor indices = Tensor::From<int64_t>("indices", {2}, {1, 0});
  const Tensor scales = Tensor::FromFloat("scales", {2, 2, 3}, std::vector<float>(12, 0.5f));
  const Tensor zp =
      Tensor::From<uint8_t>("zp", {2, 2, 2}, {0x00, 0xf0, 0x11, 0xf1, 0x22, 0xf2, 0x33, 0xf3});
  const Tensor output =
      GatherBlockQuantizedKernel(Context(), {0, -1, 16, 4})(data, indices, scales, &zp);
  ASSERT_EQ(output.shape, (Shape{2, 2, 34}));
  for (int64_t r = 0; r < 4; ++r) {
    const int64_t source = r < 2 ? r + 2 : r - 2;
    for (int64_t c = 0; c < 34; ++c) {
      EXPECT_EQ(output.AsFloat()[r * 34 + c], (9 - source) * 0.5f);
    }
  }
}

TEST(OnnxLightGatherBlockQuantizedKernel, EmptyShapesAndZeroSourceRows) {
  for (const Shape &shape : {Shape{0, 9}, Shape{2, 0}, Shape{2, 0, 9}}) {
    const Tensor data = Tensor::From<uint8_t>("data", shape, {});
    Shape scale_shape = shape;
    scale_shape.back() = shape.back() == 0 ? 0 : 1;
    const Tensor scales = Tensor::FromFloat("scales", scale_shape, {});
    GatherBlockQuantizedKernel kernel(Context(), {0, -1, 128, 4});
    const Tensor empty = kernel(data, Tensor::From<int32_t>("indices", {0, 2}, {}), scales);
    EXPECT_EQ(empty.element_count(), 0);
    const Tensor output = kernel(data, Tensor::From<int64_t>("indices", {}, {0}), scales);
    EXPECT_EQ(output.shape.size(), shape.size() - 1);
    for (int64_t i = 0; i < output.element_count(); ++i) {
      EXPECT_EQ(output.AsFloat()[i], 0);
    }
  }
}

TEST(OnnxLightGatherBlockQuantizedKernel, RejectsAttributesTypesShapesAndBuffers) {
  for (const auto &attributes :
       {GatherBlockQuantizedAttributes{0, 1, 0, 4}, GatherBlockQuantizedAttributes{0, 1, 24, 4},
        GatherBlockQuantizedAttributes{0, 1, 16, 3}}) {
    EXPECT_THROW(GatherBlockQuantizedKernel(Context(), attributes), std::invalid_argument);
  }
  const Tensor data = Tensor::From<uint8_t>("data", {2, 9}, std::vector<uint8_t>(18, 0x99));
  const Tensor indices = Tensor::From<int32_t>("indices", {1}, {0});
  const Tensor scales = Tensor::FromFloat("scales", {2, 1}, {1, 1});
  const GatherBlockQuantizedKernel kernel(Context());
  for (const auto &attributes :
       {GatherBlockQuantizedAttributes{1, 1, 128, 4}, GatherBlockQuantizedAttributes{0, 0, 128, 4},
        GatherBlockQuantizedAttributes{-3, 1, 128, 4},
        GatherBlockQuantizedAttributes{0, std::numeric_limits<int64_t>::min(), 128, 4}}) {
    EXPECT_THROW(GatherBlockQuantizedKernel(Context(), attributes)(data, indices, scales),
                 std::invalid_argument);
  }
  Tensor bad_data = data;
  bad_data.data_type = DataType::INT8;
  EXPECT_THROW(kernel(bad_data, indices, scales), std::invalid_argument);
  bad_data = data;
  bad_data.shape = {18};
  EXPECT_THROW(kernel(bad_data, indices, scales), std::invalid_argument);
  bad_data.shape = {2, -9};
  EXPECT_THROW(kernel(bad_data, indices, scales), std::invalid_argument);
  bad_data.shape = {2, 8};
  EXPECT_THROW(kernel(bad_data, indices, scales), std::invalid_argument);
  Tensor bad_scales = scales;
  bad_scales.data_type = DataType::BFLOAT16;
  EXPECT_THROW(kernel(data, indices, bad_scales), std::invalid_argument);
  bad_scales = Tensor::FromFloat("scales", {1, 2}, {1, 1});
  EXPECT_THROW(kernel(data, indices, bad_scales), std::invalid_argument);
  const Tensor bad_indices = Tensor::From<uint32_t>("indices", {1}, {0});
  EXPECT_THROW(kernel(data, bad_indices, scales), std::invalid_argument);
  const Tensor bad_zp = Tensor::From<uint8_t>("zp", {2, 2}, {0, 0, 0, 0});
  EXPECT_THROW(kernel(data, indices, scales, &bad_zp), std::invalid_argument);
}

TEST(OnnxLightGatherBlockQuantizedKernel, NodeDefaultsOptionalInputRunAndRegistration) {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("GatherBlockQuantized");
  node.set_domain("com.microsoft");
  node.add_input("data");
  node.add_input("indices");
  node.add_input("scales");
  node.add_input("");
  node.add_output("output");
  runtime::RuntimeContext rt(Context());
  rt.Set("data", Tensor::From<uint8_t>("data", {2, 2}, {0x99, 0xaa, 0xbb, 0xcc}));
  rt.Set("indices", Tensor::From<int32_t>("indices", {}, {-1}));
  rt.Set("scales", Tensor::FromFloat("scales", {2, 1}, {1, 2}));
  GatherBlockQuantizedKernel kernel(node, Context());
  kernel.Run(rt);
  const Tensor &output = rt.Get("output");
  ASSERT_EQ(output.shape, (Shape{4}));
  const std::vector<float> expected{6, 6, 8, 8};
  EXPECT_EQ(std::memcmp(output.bytes(), expected.data(), output.size_bytes()), 0);
  std::vector<onnx_light_cpu::KernelRegistration> registrations;
  {
    const onnx_light_cpu::KernelRegistrationScope scope(&registrations);
    onnx_light_cpu::RegisterGatherBlockQuantizedKernel();
  }
  ASSERT_EQ(registrations.size(), 1);
  EXPECT_EQ(registrations[0].domain, "com.microsoft");
  EXPECT_EQ(registrations[0].op_type, "GatherBlockQuantized");
  EXPECT_EQ(registrations[0].types, (std::vector<DataType>{DataType::UINT8}));
  EXPECT_EQ(registrations[0].since_version, 1);
  EXPECT_EQ(registrations[0].until_version, 1);
}

} // namespace
