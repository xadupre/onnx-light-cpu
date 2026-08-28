// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/data_type.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <type_traits>

namespace {

using onnx_light_cpu::DataType;

TEST(DataType, UsesOnnxTensorElementTypeValues) {
  EXPECT_TRUE((std::is_same_v<std::underlying_type_t<DataType>, std::int32_t>));
  EXPECT_EQ(static_cast<std::int32_t>(DataType::UNDEFINED), 0);
  EXPECT_EQ(static_cast<std::int32_t>(DataType::FLOAT), 1);
  EXPECT_EQ(static_cast<std::int32_t>(DataType::INT8), 3);
  EXPECT_EQ(static_cast<std::int32_t>(DataType::FLOAT16), 10);
  EXPECT_EQ(static_cast<std::int32_t>(DataType::BFLOAT16), 16);
  EXPECT_EQ(static_cast<std::int32_t>(DataType::FLOAT8E4M3FN), 17);
  EXPECT_EQ(static_cast<std::int32_t>(DataType::FLOAT8E8M0), 24);
}

} // namespace
