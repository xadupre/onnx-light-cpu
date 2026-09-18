// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/math/tanh_kernel.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

namespace rt = ONNX_LIGHT_NAMESPACE::core::runtime;

TEST(OnnxLightTanhKernel, PreservesShapeAndTypeIncludingScalarAndEmpty) {
  const rt::KernelContext context{rt::DefaultOpset(13)};
  const onnx_light_cpu::TanhKernel kernel{context};
  for (const auto type : {rt::DataType::FLOAT, rt::DataType::FLOAT16, rt::DataType::BFLOAT16}) {
    for (const rt::Shape shape : {rt::Shape{}, rt::Shape{2, 3}, rt::Shape{2, 0, 3}}) {
      const auto count = static_cast<std::size_t>(shape.product(0, shape.size(), "Tanh test"));
      auto input =
          rt::MakeOutputTensor(static_cast<std::int32_t>(type), shape,
                               count * rt::ElementSize(static_cast<std::int32_t>(type)), nullptr);
      if (count != 0) {
        std::memset(input.mutable_bytes(), 0, input.size_bytes());
      }
      const auto output = kernel(input);
      EXPECT_EQ(output.shape, shape);
      EXPECT_EQ(output.data_type, static_cast<std::int32_t>(type));
      EXPECT_EQ(output.element_count(), count);
      EXPECT_EQ(output.size_bytes(), input.size_bytes());
    }
  }
}

TEST(OnnxLightTanhKernel, RejectsWrongOutputMetadataAndUnsupportedTypes) {
  const rt::KernelContext context{rt::DefaultOpset(13)};
  const onnx_light_cpu::TanhKernel kernel{context};
  const auto input = rt::Tensor::FromFloat("", {2}, {0, 1});
  auto wrong_shape = rt::Tensor::FromFloat("", {1, 2}, {0, 0});
  auto wrong_type = rt::Tensor::FromDouble("", {2}, {0, 0});
  EXPECT_THROW(kernel(input, wrong_shape), std::invalid_argument);
  EXPECT_THROW(kernel(input, wrong_type), std::invalid_argument);
  EXPECT_THROW(kernel(wrong_type), std::invalid_argument);
}

TEST(OnnxLightTanhKernel, RejectsUndersizedBuffers) {
  const rt::KernelContext context{rt::DefaultOpset(13)};
  const onnx_light_cpu::TanhKernel kernel{context};
  const std::vector<float> values{0, 1};
  const auto *bytes = reinterpret_cast<const std::uint8_t *>(values.data());
  const auto undersized =
      rt::Tensor::Borrow("", rt::DataType::FLOAT, {3}, bytes, 2 * sizeof(float));
  EXPECT_THROW(kernel(undersized), std::invalid_argument);
  const auto input = rt::Tensor::FromFloat("", {2}, values);
  auto output = rt::Tensor::Borrow("", rt::DataType::FLOAT, {2}, bytes, sizeof(float));
  EXPECT_THROW(kernel(input, output), std::invalid_argument);
}

} // namespace
