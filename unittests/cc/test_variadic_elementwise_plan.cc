// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/binary/variadic_elementwise_plan.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

using BinaryDataType = onnx_light_cpu::DataType;
using onnx_light_cpu::VariadicElementwisePlan;
using onnx_light_cpu::VariadicOperator;

template <typename T>
std::vector<T> Execute(VariadicOperator op, BinaryDataType type,
                       const std::vector<std::vector<std::int64_t>> &shapes,
                       const std::vector<std::vector<T>> &values) {
  std::vector<BinaryDataType> types(shapes.size(), type);
  VariadicElementwisePlan plan(op, types, shapes);
  std::vector<const void *> inputs;
  for (const auto &value : values) {
    inputs.push_back(value.data());
  }
  std::vector<T> output(plan.element_count());
  plan.Execute(inputs, output.data());
  return output;
}

TEST(VariadicElementwisePlan, ValidatesEveryInput) {
  const std::vector<BinaryDataType> no_types;
  const std::vector<std::vector<std::int64_t>> no_shapes;
  EXPECT_THROW(VariadicElementwisePlan(VariadicOperator::kSum, no_types, no_shapes),
               std::invalid_argument);

  const std::vector<BinaryDataType> mixed_types = {BinaryDataType::FLOAT, BinaryDataType::DOUBLE};
  const std::vector<std::vector<std::int64_t>> shapes = {{2}, {2}};
  EXPECT_THROW(VariadicElementwisePlan(VariadicOperator::kSum, mixed_types, shapes),
               std::invalid_argument);

  const std::vector<BinaryDataType> types(2, BinaryDataType::FLOAT);
  const std::vector<std::vector<std::int64_t>> incompatible = {{2, 3}, {2, 2}};
  EXPECT_THROW(VariadicElementwisePlan(VariadicOperator::kSum, types, incompatible),
               std::invalid_argument);
}

TEST(VariadicElementwisePlan, TraversesOneCommonMultidirectionalBroadcastSpace) {
  const std::vector<std::vector<std::int64_t>> shapes = {{2, 1}, {1, 3}, {}};
  const std::vector<std::vector<float>> values = {{1.0f, 2.0f}, {10.0f, 20.0f, 30.0f}, {100.0f}};
  EXPECT_EQ(Execute<float>(VariadicOperator::kSum, BinaryDataType::FLOAT, shapes, values),
            (std::vector<float>{111.0f, 121.0f, 131.0f, 112.0f, 122.0f, 132.0f}));
}

TEST(VariadicElementwisePlan, PreservesOneInputAndEmptyTensorsWithoutWorkspace) {
  const std::vector<BinaryDataType> types = {BinaryDataType::FLOAT};
  const std::vector<std::vector<std::int64_t>> shape = {{0, 3}};
  VariadicElementwisePlan empty(VariadicOperator::kSum, types, shape);
  EXPECT_EQ(empty.element_count(), 0u);
  EXPECT_EQ(empty.workspace_bytes(), 0u);
  const std::array<const void *, 1> null_input = {nullptr};
  EXPECT_NO_THROW(empty.Execute(null_input, nullptr));

  const float nan = std::bit_cast<float>(std::uint32_t{0x7fc12345u});
  const std::vector<std::vector<std::int64_t>> scalar_shape = {{}};
  const std::vector<std::vector<float>> scalar_value = {{nan}};
  const auto output =
      Execute<float>(VariadicOperator::kMax, BinaryDataType::FLOAT, scalar_shape, scalar_value);
  EXPECT_EQ(std::bit_cast<std::uint32_t>(output[0]), std::bit_cast<std::uint32_t>(nan));
}

TEST(VariadicElementwisePlan, PreservesFloatingEvaluationOrderAndMeanScaling) {
  const std::vector<std::vector<std::int64_t>> shapes(3);
  const std::vector<std::vector<float>> values = {{1.0e20f}, {-1.0e20f}, {3.0f}};
  EXPECT_EQ(Execute<float>(VariadicOperator::kSum, BinaryDataType::FLOAT, shapes, values)[0], 3.0f);

  const std::vector<std::vector<double>> mean_values = {{1.0}, {2.0}, {8.0}};
  EXPECT_DOUBLE_EQ(
      Execute<double>(VariadicOperator::kMean, BinaryDataType::DOUBLE, shapes, mean_values)[0],
      (1.0 + 2.0 + 8.0) * (1.0 / 3.0));
}

TEST(VariadicElementwisePlan, WrapsSignedIntegerSumWithoutUndefinedBehavior) {
  const std::vector<std::vector<std::int64_t>> shapes(2);
  const std::vector<std::vector<std::int32_t>> values = {{std::numeric_limits<std::int32_t>::max()},
                                                         {1}};
  EXPECT_EQ(Execute<std::int32_t>(VariadicOperator::kSum, BinaryDataType::INT32, shapes, values)[0],
            std::numeric_limits<std::int32_t>::min());
}

TEST(VariadicElementwisePlan, MatchesPortableMinMaxNanAndSignedZeroSemantics) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const std::vector<std::vector<std::int64_t>> shapes(2);
  const std::vector<std::vector<float>> leading_nan = {{nan}, {2.0f}};
  const std::vector<std::vector<float>> trailing_nan = {{2.0f}, {nan}};
  EXPECT_FLOAT_EQ(
      Execute<float>(VariadicOperator::kMin, BinaryDataType::FLOAT, shapes, leading_nan)[0], 2.0f);
  EXPECT_FLOAT_EQ(
      Execute<float>(VariadicOperator::kMax, BinaryDataType::FLOAT, shapes, leading_nan)[0], 2.0f);
  EXPECT_TRUE(std::isnan(
      Execute<float>(VariadicOperator::kMin, BinaryDataType::FLOAT, shapes, trailing_nan)[0]));
  EXPECT_TRUE(std::isnan(
      Execute<float>(VariadicOperator::kMax, BinaryDataType::FLOAT, shapes, trailing_nan)[0]));

  const std::vector<std::vector<float>> zeros = {{0.0f}, {-0.0f}};
  const float minimum =
      Execute<float>(VariadicOperator::kMin, BinaryDataType::FLOAT, shapes, zeros)[0];
  const float maximum =
      Execute<float>(VariadicOperator::kMax, BinaryDataType::FLOAT, shapes, zeros)[0];
  EXPECT_TRUE(std::signbit(minimum));
  EXPECT_TRUE(std::signbit(maximum));
}

} // namespace
