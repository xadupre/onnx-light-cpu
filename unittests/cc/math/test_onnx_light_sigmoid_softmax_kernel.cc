// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/impl/math/math_kernels.h"
#include "onnx_light_cpu/kernels/math/sigmoid_softmax_kernel.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace {

namespace rt = ONNX_LIGHT_NAMESPACE::core::runtime;

rt::KernelContext Context() { return rt::KernelContext(rt::OpsetId(std::string(), 13)); }

rt::Tensor ActivationTensor(rt::DataType type, const rt::Shape &shape,
                            const std::vector<double> &values) {
  if (type == rt::DataType::DOUBLE) {
    return rt::Tensor::FromDouble("", shape, values);
  }
  if (type == rt::DataType::FLOAT) {
    std::vector<float> converted(values.size());
    std::transform(values.begin(), values.end(), converted.begin(),
                   [](double value) { return static_cast<float>(value); });
    return rt::Tensor::FromFloat("", shape, converted);
  }
  auto tensor = rt::MakeOutputTensor(static_cast<std::int32_t>(type), shape,
                                     values.size() * sizeof(std::uint16_t), nullptr);
  auto *bits = reinterpret_cast<std::uint16_t *>(tensor.mutable_bytes());
  for (std::size_t index = 0; index < values.size(); ++index) {
    const float value = static_cast<float>(values[index]);
    bits[index] = type == rt::DataType::FLOAT16
                      ? onnx_light_cpu::detail::FloatToFloat16Bits(value)
                      : onnx_light_cpu::detail::FloatToBFloat16Bits(value);
  }
  return tensor;
}

double ActivationValue(const rt::Tensor &tensor, std::size_t index) {
  const auto type = static_cast<rt::DataType>(tensor.data_type);
  if (type == rt::DataType::DOUBLE) {
    return tensor.AsDouble()[index];
  }
  if (type == rt::DataType::FLOAT) {
    return tensor.AsFloat()[index];
  }
  const auto bits = reinterpret_cast<const std::uint16_t *>(tensor.bytes())[index];
  return type == rt::DataType::FLOAT16 ? onnx_light_cpu::detail::Float16BitsToFloat(bits)
                                       : onnx_light_cpu::detail::Bfloat16BitsToFloat(bits);
}

void ExpectActivationNear(double actual, long double expected, rt::DataType type) {
  if (std::isnan(expected)) {
    EXPECT_TRUE(std::isnan(actual));
    return;
  }
  double relative = 3e-14;
  double minimum = std::numeric_limits<double>::denorm_min();
  if (type == rt::DataType::FLOAT) {
    relative = 2e-6;
    minimum = std::numeric_limits<float>::denorm_min();
  } else if (type == rt::DataType::FLOAT16) {
    relative = 6e-4;
    minimum = onnx_light_cpu::detail::Float16BitsToFloat(1);
  } else if (type == rt::DataType::BFLOAT16) {
    relative = 4e-3;
    minimum = onnx_light_cpu::detail::Bfloat16BitsToFloat(1);
  }
  EXPECT_NEAR(actual, static_cast<double>(expected),
              std::max(std::abs(static_cast<double>(expected)) * relative, minimum));
  if (expected == 0 || expected == 1) {
    EXPECT_EQ(actual, static_cast<double>(expected));
  }
}

std::vector<long double> SoftmaxReference(const rt::Tensor &input, std::int64_t axis,
                                          std::int64_t opset) {
  const auto rank = static_cast<std::int64_t>(input.shape.size());
  if (axis < 0) {
    axis += rank;
  }
  std::int64_t outer = 1;
  std::int64_t inner = 1;
  for (std::int64_t dimension = 0; dimension < axis; ++dimension) {
    outer *= input.shape[dimension];
  }
  for (std::int64_t dimension = axis + 1; dimension < rank; ++dimension) {
    inner *= input.shape[dimension];
  }
  std::int64_t columns = input.shape[axis];
  if (opset < 13) {
    columns *= inner;
    inner = 1;
  }
  std::vector<long double> expected(input.element_count());
  for (std::int64_t row = 0; row < outer; ++row) {
    for (std::int64_t lane = 0; lane < inner; ++lane) {
      long double maximum = -std::numeric_limits<long double>::infinity();
      for (std::int64_t column = 0; column < columns; ++column) {
        maximum = std::max(maximum, static_cast<long double>(ActivationValue(
                                        input, (row * columns + column) * inner + lane)));
      }
      long double sum = 0;
      for (std::int64_t column = 0; column < columns; ++column) {
        const auto index = (row * columns + column) * inner + lane;
        expected[index] =
            std::exp(static_cast<long double>(ActivationValue(input, index)) - maximum);
        sum += expected[index];
      }
      for (std::int64_t column = 0; column < columns; ++column) {
        expected[(row * columns + column) * inner + lane] /= sum;
      }
    }
  }
  return expected;
}

void CheckSoftmax(rt::DataType type, const rt::Shape &shape, const std::vector<double> &values,
                  std::int64_t axis, std::int64_t opset) {
  SCOPED_TRACE("dtype=" + std::to_string(static_cast<int>(type)) + ",axis=" + std::to_string(axis) +
               ",opset=" + std::to_string(opset));
  auto input = ActivationTensor(type, shape, values);
  const auto expected = SoftmaxReference(input, axis, opset);
  const onnx_light_cpu::SoftmaxKernel kernel(rt::KernelContext(rt::OpsetId(std::string(), opset)));
  const auto output = kernel(input, axis);
  EXPECT_EQ(output.shape, input.shape);
  EXPECT_EQ(output.data_type, input.data_type);
  ASSERT_EQ(output.element_count(), expected.size());
  kernel(input, axis, input);
  for (std::size_t index = 0; index < expected.size(); ++index) {
    SCOPED_TRACE(index);
    ExpectActivationNear(ActivationValue(output, index), expected[index], type);
    ExpectActivationNear(ActivationValue(input, index), expected[index], type);
  }
}

std::vector<float> Values(std::size_t count) {
  std::vector<float> values(count);
  for (std::size_t index = 0; index < count; ++index) {
    values[index] = static_cast<float>(static_cast<int>(index * 41 % 10001) - 5000) * 0.002f;
  }
  return values;
}

struct InlineExecutor {
  std::int64_t dispatches = 0;
  std::int64_t blocks = 0;
  bool nested = false;

  static void Run(void *context, std::int64_t blocks, void *task_context,
                  onnx_light_cpu::ExecutionBlockFn task) {
    auto &self = *static_cast<InlineExecutor *>(context);
    self.nested |= onnx_light_cpu::ExecutionInParallelRegion();
    ++self.dispatches;
    self.blocks = blocks;
    for (std::int64_t block = blocks; block > 0; --block) {
      task(task_context, block - 1);
    }
  }
};

TEST(OnnxLightSigmoidSoftmaxKernel, SigmoidMatchesReferenceIncludingTailsAndAliasing) {
  const onnx_light_cpu::SigmoidKernel kernel(Context());
  for (std::int64_t count : {0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1023, 1024, 1025}) {
    const auto values = Values(static_cast<std::size_t>(count));
    auto input = rt::Tensor::FromFloat("", {count}, values);
    kernel(input, input);
    for (std::int64_t index = 0; index < count; ++index) {
      const double expected = 1.0 / (1.0 + std::exp(-static_cast<double>(values[index])));
      EXPECT_NEAR(input.AsFloat()[index], expected, 2e-6) << count << "," << index;
    }
  }
}

TEST(OnnxLightSigmoidSoftmaxKernel, SoftmaxMatchesReferenceIncludingTailsAndAliasing) {
  const onnx_light_cpu::SoftmaxKernel kernel(Context());
  for (std::int64_t columns : {1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1023, 1025}) {
    constexpr std::int64_t rows = 3;
    const auto values = Values(static_cast<std::size_t>(rows * columns));
    auto input = rt::Tensor::FromFloat("", {rows, columns}, values);
    kernel(input, -1, input);
    for (std::int64_t row = 0; row < rows; ++row) {
      const auto begin = values.begin() + row * columns;
      const float maximum = *std::max_element(begin, begin + columns);
      double sum = 0;
      for (std::int64_t column = 0; column < columns; ++column) {
        sum += std::exp(static_cast<double>(begin[column]) - maximum);
      }
      for (std::int64_t column = 0; column < columns; ++column) {
        const double expected = std::exp(static_cast<double>(begin[column]) - maximum) / sum;
        EXPECT_NEAR(input.AsFloat()[row * columns + column], expected, 2e-6)
            << columns << "," << row << "," << column;
      }
    }
  }
}

TEST(OnnxLightSigmoidSoftmaxKernel, SpecialValues) {
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  auto input = rt::Tensor::FromFloat("", {9}, {-inf, inf, nan, -0.0f, 0, -100, 100, 1, -1});
  const auto sigmoid = onnx_light_cpu::SigmoidKernel(Context())(input);
  EXPECT_EQ(sigmoid.AsFloat()[0], 0.0f);
  EXPECT_EQ(sigmoid.AsFloat()[1], 1.0f);
  EXPECT_TRUE(std::isnan(sigmoid.AsFloat()[2]));
  EXPECT_NEAR(sigmoid.AsFloat()[3], 0.5f, 1e-7f);
  EXPECT_NEAR(sigmoid.AsFloat()[4], 0.5f, 1e-7f);

  input = rt::Tensor::FromFloat("", {3, 3}, {0, -inf, 0, -inf, -inf, -inf, 0, nan, 1});
  const auto softmax = onnx_light_cpu::SoftmaxKernel(Context())(input, -1);
  EXPECT_EQ(softmax.AsFloat()[0], 0.5f);
  EXPECT_EQ(softmax.AsFloat()[1], 0.0f);
  EXPECT_EQ(softmax.AsFloat()[2], 0.5f);
  for (std::size_t index = 3; index < 9; ++index) {
    EXPECT_TRUE(std::isnan(softmax.AsFloat()[index])) << index;
  }
}

TEST(OnnxLightSigmoidSoftmaxKernel, SigmoidExtremeValuesInEveryTail) {
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float maximum = std::numeric_limits<float>::max();
  const std::vector<float> extremes = {-inf,    -maximum, -104, -100,    -90, -87.34f,
                                       -87.33f, -20,      -1,   -0.0f,   0,   1,
                                       20,      87.33f,   100,  maximum, inf, nan};
  const onnx_light_cpu::SigmoidKernel kernel(Context());
  for (std::int64_t count = 1; count <= 65; ++count) {
    std::vector<float> values(static_cast<std::size_t>(count));
    for (std::size_t index = 0; index < values.size(); ++index) {
      values[index] = extremes[(index + count) % extremes.size()];
    }
    auto input = rt::Tensor::FromFloat("", {count}, values);
    kernel(input, input);
    for (std::size_t index = 0; index < values.size(); ++index) {
      SCOPED_TRACE(std::to_string(count) + "," + std::to_string(index));
      if (std::isnan(values[index])) {
        EXPECT_TRUE(std::isnan(input.AsFloat()[index]));
        continue;
      }
      const double exponent = std::exp(-std::abs(static_cast<double>(values[index])));
      const double expected = values[index] < 0 ? exponent / (1 + exponent) : 1 / (1 + exponent);
      EXPECT_NEAR(input.AsFloat()[index], expected,
                  std::max(expected * 3e-7, 2.0 * std::numeric_limits<float>::denorm_min()));
      if (std::isinf(values[index])) {
        EXPECT_EQ(input.AsFloat()[index], values[index] < 0 ? 0.0f : 1.0f);
      }
    }
  }
}

TEST(OnnxLightSigmoidSoftmaxKernel, SigmoidNearZeroSweepBoundariesAndEveryTail) {
  const onnx_light_cpu::SigmoidKernel kernel(Context());
  const auto check = [&kernel](const std::vector<float> &values) {
    auto input = rt::Tensor::FromFloat("", {static_cast<std::int64_t>(values.size())}, values);
    const auto output = kernel(input);
    kernel(input, input);
    for (std::size_t index = 0; index < values.size(); ++index) {
      SCOPED_TRACE("count=" + std::to_string(values.size()) + ",index=" + std::to_string(index));
      if (std::isnan(values[index])) {
        EXPECT_TRUE(std::isnan(output.AsFloat()[index]));
        EXPECT_TRUE(std::isnan(input.AsFloat()[index]));
        continue;
      }
      const double exponent = std::exp(-std::abs(static_cast<double>(values[index])));
      const double expected = values[index] < 0 ? exponent / (1 + exponent) : 1 / (1 + exponent);
      const double tolerance =
          std::max(expected * 3e-7, 2.0 * std::numeric_limits<float>::denorm_min());
      EXPECT_NEAR(output.AsFloat()[index], expected, tolerance) << values[index];
      EXPECT_NEAR(input.AsFloat()[index], expected, tolerance) << values[index];
    }
  };

  std::vector<float> sweep(28673);
  for (std::size_t index = 0; index < sweep.size(); ++index) {
    sweep[index] = -3.5f + static_cast<float>(index) / 4096.0f;
  }
  check(sweep);

  const float inf = std::numeric_limits<float>::infinity();
  const float tiny = std::numeric_limits<float>::denorm_min();
  const std::vector<float> near = {
      -2.0f, std::nextafter(-2.0f, 0.0f), -1.5f, -0.5f, -tiny, -0.0f, 0.0f, tiny, 0.5f,
      1.5f,  std::nextafter(2.0f, 0.0f),  2.0f};
  auto boundaries = near;
  boundaries.insert(boundaries.end(),
                    {-3.5f, std::nextafter(-3.5f, 0.0f), std::nextafter(3.5f, 0.0f), 3.5f});
  auto mixed = boundaries;
  mixed.insert(mixed.end(), {std::nextafter(-3.5f, -inf), std::nextafter(3.5f, inf)});
  mixed.insert(mixed.end(),
               {std::nextafter(-2.0f, -inf), std::nextafter(2.0f, inf), -100.0f, -90.0f, -20.0f,
                20.0f, 100.0f, -inf, inf, std::numeric_limits<float>::quiet_NaN()});
  for (bool mixed_lanes : {false, true}) {
    SCOPED_TRACE(mixed_lanes ? "mixed fallback lanes" : "near-only lanes");
    const auto &cases = mixed_lanes ? mixed : boundaries;
    for (std::size_t count = 1; count <= 65; ++count) {
      std::vector<float> values(count);
      for (std::size_t index = 0; index < count; ++index) {
        values[index] = cases[(index + count) % cases.size()];
      }
      check(values);
    }
  }
}

TEST(OnnxLightSigmoidSoftmaxKernel, SoftmaxExtremeRowsIncludingTails) {
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const onnx_light_cpu::SoftmaxKernel kernel(Context());
  for (std::int64_t columns : {7, 8, 9, 15, 16, 17, 31, 32, 33, 1023, 1024, 1025}) {
    constexpr std::int64_t rows = 5;
    auto values = Values(static_cast<std::size_t>(rows * columns));
    for (std::int64_t column = 0; column < columns; ++column) {
      values[column] = -86.0f - static_cast<float>(column % 20);
      values[3 * columns + column] = -inf;
      values[4 * columns + column] = -std::numeric_limits<float>::max();
    }
    values[columns - 1] = 0;
    values[2 * columns - 1] = nan;
    values[3 * columns - 1] = inf;
    values[5 * columns - 1] = std::numeric_limits<float>::max();
    auto input = rt::Tensor::FromFloat("", {rows, columns}, values);
    kernel(input, -1, input);
    for (std::int64_t row = 0; row < rows; ++row) {
      const auto begin = values.begin() + row * columns;
      const double maximum = *std::max_element(begin, begin + columns);
      double sum = 0;
      for (std::int64_t column = 0; column < columns; ++column) {
        sum += std::exp(static_cast<double>(begin[column]) - maximum);
      }
      for (std::int64_t column = 0; column < columns; ++column) {
        SCOPED_TRACE(std::to_string(columns) + "," + std::to_string(row) + "," +
                     std::to_string(column));
        const double expected = std::exp(static_cast<double>(begin[column]) - maximum) / sum;
        const float actual = input.AsFloat()[row * columns + column];
        if (std::isnan(expected)) {
          EXPECT_TRUE(std::isnan(actual));
        } else {
          EXPECT_NEAR(actual, expected,
                      std::max(expected * 3e-7, 2.0 * std::numeric_limits<float>::denorm_min()));
        }
      }
    }
  }
}

TEST(OnnxLightSigmoidSoftmaxKernel, SoftmaxRowSpanBoundaryIncludingTails) {
  const float inf = std::numeric_limits<float>::infinity();
  const std::vector<float> spans = {std::nextafter(87.0f, 0.0f), 87.0f, std::nextafter(87.0f, inf)};
  const onnx_light_cpu::SoftmaxKernel kernel(Context());
  for (std::int64_t columns : {7, 8, 9, 15, 16, 17, 31, 32, 33}) {
    SCOPED_TRACE(columns);
    std::vector<float> values(spans.size() * static_cast<std::size_t>(columns));
    for (std::size_t row = 0; row < spans.size(); ++row) {
      for (std::int64_t column = 0; column < columns; ++column) {
        values[row * columns + column] =
            -spans[row] * (static_cast<float>(column) / static_cast<float>(columns - 1));
      }
    }
    auto input =
        rt::Tensor::FromFloat("", {static_cast<std::int64_t>(spans.size()), columns}, values);
    const auto expected = SoftmaxReference(input, -1, 13);
    const auto output = kernel(input, -1);
    kernel(input, -1, input);
    for (std::size_t index = 0; index < values.size(); ++index) {
      SCOPED_TRACE(index);
      const double reference = static_cast<double>(expected[index]);
      const double tolerance =
          std::max(reference * 3e-7, 2.0 * std::numeric_limits<float>::denorm_min());
      EXPECT_NEAR(output.AsFloat()[index], reference, tolerance);
      EXPECT_NEAR(input.AsFloat()[index], reference, tolerance);
    }
  }
}

TEST(OnnxLightSigmoidSoftmaxKernel, OtherTypesSigmoidTailsExtremesAndAliasing) {
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<double> extremes = {-inf,
                                        -std::numeric_limits<double>::max(),
                                        -65504,
                                        -1000,
                                        -745,
                                        -744,
                                        -740,
                                        -710,
                                        -709,
                                        -708,
                                        -100,
                                        -90,
                                        -20,
                                        -17,
                                        -10,
                                        -1.234567890123,
                                        -1e-8,
                                        -0.0,
                                        0,
                                        1e-8,
                                        0.1234567890123,
                                        1,
                                        10,
                                        20,
                                        100,
                                        710,
                                        1000,
                                        65504,
                                        std::numeric_limits<double>::max(),
                                        inf,
                                        nan};
  const onnx_light_cpu::SigmoidKernel kernel(Context());
  for (auto type : {rt::DataType::DOUBLE, rt::DataType::FLOAT16, rt::DataType::BFLOAT16}) {
    for (std::int64_t count : {0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 255, 256, 257}) {
      SCOPED_TRACE("dtype=" + std::to_string(static_cast<int>(type)) +
                   ",count=" + std::to_string(count));
      std::vector<double> values(static_cast<std::size_t>(count));
      for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] = extremes[(index + count) % extremes.size()];
      }
      auto input = ActivationTensor(type, {count}, values);
      std::vector<long double> expected(values.size());
      for (std::size_t index = 0; index < values.size(); ++index) {
        const long double value = ActivationValue(input, index);
        const long double exponent = std::exp(-std::abs(value));
        expected[index] = value < 0 ? exponent / (1 + exponent) : 1 / (1 + exponent);
      }
      const auto output = kernel(input);
      EXPECT_EQ(output.shape, input.shape);
      EXPECT_EQ(output.data_type, input.data_type);
      ASSERT_EQ(output.element_count(), expected.size());
      kernel(input, input);
      for (std::size_t index = 0; index < values.size(); ++index) {
        SCOPED_TRACE(index);
        ExpectActivationNear(ActivationValue(output, index), expected[index], type);
        ExpectActivationNear(ActivationValue(input, index), expected[index], type);
      }
    }
  }
}

TEST(OnnxLightSigmoidSoftmaxKernel, OtherTypesSoftmaxTailsExtremesAndAliasing) {
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (auto type : {rt::DataType::DOUBLE, rt::DataType::FLOAT16, rt::DataType::BFLOAT16}) {
    for (std::int64_t columns : {1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 255, 257, 1025}) {
      SCOPED_TRACE(columns);
      constexpr std::int64_t rows = 6;
      std::vector<double> values(static_cast<std::size_t>(rows * columns));
      for (std::int64_t column = 0; column < columns; ++column) {
        // A float intermediate loses the double row's small differences.
        values[column] = (type == rt::DataType::DOUBLE ? 1e8 : 0) +
                         static_cast<double>(column % 13) * 0.1234567890123;
        values[columns + column] =
            type == rt::DataType::DOUBLE ? -708.0 - column % 39 : -8.0 - column % 100;
        values[2 * columns + column] = -inf;
        values[3 * columns + column] = column % 7;
        values[4 * columns + column] = column % 5;
        values[5 * columns + column] = -60000;
      }
      values[2 * columns - 1] = 0;
      values[4 * columns - 1] = nan;
      values[5 * columns - 1] = inf;
      values[6 * columns - 1] = 60000;
      CheckSoftmax(type, {rows, columns}, values, -1, 13);
    }
  }
}

TEST(OnnxLightSigmoidSoftmaxKernel, SoftmaxNonLastAxisAndOpset12Flattening) {
  for (auto type :
       {rt::DataType::FLOAT, rt::DataType::DOUBLE, rt::DataType::FLOAT16, rt::DataType::BFLOAT16}) {
    std::vector<double> values(2 * 5 * 3);
    for (std::size_t index = 0; index < values.size(); ++index) {
      values[index] = static_cast<double>(static_cast<int>(index * 17 % 31) - 15) * 0.1234567890123;
    }
    for (std::int64_t opset : {12, 13}) {
      for (std::int64_t axis : {0, 1, -2, -1}) {
        CheckSoftmax(type, {2, 5, 3}, values, axis, opset);
      }
      const double inf = std::numeric_limits<double>::infinity();
      const double nan = std::numeric_limits<double>::quiet_NaN();
      CheckSoftmax(type, {2, 3, 3},
                   {0, -inf, 0, -inf, -inf, nan, 0, -inf, 1, inf, -60000, 1, 0, 0, 2, 1, 60000, 3},
                   1, opset);
    }
  }
}

TEST(OnnxLightSigmoidSoftmaxKernel, SoftmaxEmptyAxesPreserveShapeAndType) {
  for (auto type :
       {rt::DataType::FLOAT, rt::DataType::DOUBLE, rt::DataType::FLOAT16, rt::DataType::BFLOAT16}) {
    for (const rt::Shape &shape : {rt::Shape{0, 3, 5}, rt::Shape{2, 0, 5}, rt::Shape{2, 3, 0}}) {
      for (std::int64_t opset : {12, 13}) {
        for (std::int64_t axis : {1, -1}) {
          CheckSoftmax(type, shape, {}, axis, opset);
        }
      }
    }
  }
}

TEST(OnnxLightSigmoidSoftmaxKernel, UsesRuntimeExecutorForLargeInputs) {
  InlineExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run};
  onnx_light_cpu::ExecutionExecutorScope scope(&view);
  const auto small = rt::Tensor::FromFloat("", {1, 17}, Values(17));
  const onnx_light_cpu::SigmoidKernel sigmoid(Context());
  const onnx_light_cpu::SoftmaxKernel softmax(Context());
  sigmoid(small);
  softmax(small, -1);
  EXPECT_EQ(executor.dispatches, 0);
  const auto large = rt::Tensor::FromFloat("", {256, 1025}, Values(256 * 1025));
  view.effective_threads = 1;
  const auto expected_sigmoid = sigmoid(large);
  const auto expected_softmax = softmax(large, -1);
  EXPECT_EQ(executor.dispatches, 0);
  view.effective_threads = 4;
  const auto actual_sigmoid = sigmoid(large);
  EXPECT_TRUE(std::equal(expected_sigmoid.AsFloat(),
                         expected_sigmoid.AsFloat() + large.element_count(),
                         actual_sigmoid.AsFloat()));
  EXPECT_EQ(executor.dispatches, 1);
  EXPECT_GT(executor.blocks, 1);
  EXPECT_LE(executor.blocks, view.effective_threads);
  EXPECT_FALSE(executor.nested);
  executor = {};
  const auto actual_softmax = softmax(large, -1);
  EXPECT_TRUE(std::equal(expected_softmax.AsFloat(),
                         expected_softmax.AsFloat() + large.element_count(),
                         actual_softmax.AsFloat()));
  EXPECT_EQ(executor.dispatches, 1);
  EXPECT_GT(executor.blocks, 1);
  EXPECT_LE(executor.blocks, view.effective_threads);
  EXPECT_FALSE(executor.nested);

  executor = {};
  {
    onnx_light_cpu::detail::ExecutionRegionScope region;
    sigmoid(large);
    softmax(large, -1);
  }
  EXPECT_EQ(executor.dispatches, 0);
}

#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
TEST(OnnxLightSigmoidSoftmaxKernel, Avx2SmallSigmoidUsesBoundedTeams) {
  if (onnx_light_cpu::DetectSimdLevel() != onnx_light_cpu::SimdLevel::kAVX2 ||
      !onnx_light_cpu::CpuSupportsFma()) {
    GTEST_SKIP() << "AVX2-specific scheduling is unavailable";
  }
  InlineExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 32, &InlineExecutor::Run};
  onnx_light_cpu::ExecutionExecutorScope scope(&view);
  const onnx_light_cpu::SigmoidKernel sigmoid(Context());
  for (std::int64_t count : {32767, 32768, 65535, 65536, 131072}) {
    executor = {};
    const auto input = rt::Tensor::FromFloat("", {count}, Values(count));
    sigmoid(input);
    if (count < 32768) {
      EXPECT_EQ(executor.dispatches, 0);
    } else {
      EXPECT_EQ(executor.dispatches, 1);
      EXPECT_GT(executor.blocks, 1);
      EXPECT_LE(executor.blocks, count < 96 * 1024 ? 2 : 3);
    }
    EXPECT_FALSE(executor.nested);
  }
}

TEST(OnnxLightSigmoidSoftmaxKernel, RegisteredFloat64UsesFusedAvx2Implementation) {
  if (onnx_light_cpu::DetectSimdLevel() < onnx_light_cpu::SimdLevel::kAVX2 ||
      !onnx_light_cpu::CpuSupportsFma()) {
    GTEST_SKIP() << "AVX2/FMA is unavailable";
  }
  for (std::int64_t columns : {1, 3, 4, 5, 7, 8, 9, 1023, 1024, 1025}) {
    const auto floats = Values(3 * columns);
    const std::vector<double> values(floats.begin(), floats.end());
    const auto input = rt::Tensor::FromDouble("", {3, columns}, values);
    std::vector<double> expected(values.size());
    onnx_light_cpu::SigmoidFloat64_AVX2_FMA(values.data(), expected.data(), values.size());
    const auto sigmoid = onnx_light_cpu::SigmoidKernel(Context())(input);
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), sigmoid.AsDouble()));
    onnx_light_cpu::SoftmaxFloat64_AVX2_FMA(values.data(), expected.data(), 3, columns);
    const auto softmax = onnx_light_cpu::SoftmaxKernel(Context())(input, -1);
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), softmax.AsDouble()));
  }
}

#ifdef ONNX_LIGHT_CPU_HAVE_F16C
TEST(OnnxLightSigmoidSoftmaxKernel, FusedFloat16SigmoidCoversEveryBitPattern) {
  if (onnx_light_cpu::DetectSimdLevel() != onnx_light_cpu::SimdLevel::kAVX2 ||
      !onnx_light_cpu::CpuSupportsFma() || !onnx_light_cpu::CpuSupportsF16C()) {
    GTEST_SKIP() << "AVX2/FMA/F16C is unavailable";
  }
  constexpr std::size_t count = 65536;
  auto input = rt::MakeOutputTensor(static_cast<int>(rt::DataType::FLOAT16), {count},
                                    count * sizeof(std::uint16_t), nullptr);
  auto *bits = reinterpret_cast<std::uint16_t *>(input.mutable_bytes());
  for (std::size_t index = 0; index < count; ++index) {
    bits[index] = static_cast<std::uint16_t>(index);
  }
  std::vector<std::uint16_t> expected(count);
  onnx_light_cpu::SigmoidFloat16_AVX2_FMA(bits, expected.data(), count);
  onnx_light_cpu::SigmoidKernel{Context()}(input, input);
  for (std::size_t index = 0; index < count; ++index) {
    SCOPED_TRACE(index);
    const double value =
        onnx_light_cpu::detail::Float16BitsToFloat(static_cast<std::uint16_t>(index));
    const double exponent = std::exp(-std::abs(value));
    ExpectActivationNear(ActivationValue(input, index),
                         value < 0 ? exponent / (1 + exponent) : 1 / (1 + exponent),
                         rt::DataType::FLOAT16);
    EXPECT_EQ(bits[index], expected[index]);
  }
}
#endif

TEST(OnnxLightSigmoidSoftmaxKernel, SmallAvx2SoftmaxAvoidsParallelDispatch) {
  if (onnx_light_cpu::DetectSimdLevel() != onnx_light_cpu::SimdLevel::kAVX2 ||
      !onnx_light_cpu::CpuSupportsFma()) {
    GTEST_SKIP() << "AVX2-specific scheduling is unavailable";
  }
  InlineExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 32, &InlineExecutor::Run};
  onnx_light_cpu::ExecutionExecutorScope scope(&view);
  const onnx_light_cpu::SoftmaxKernel softmax(Context());
  for (std::int64_t rows : {1, 32, 64, 128, 255}) {
    const auto input = rt::Tensor::FromFloat("", {rows, 1024}, Values(rows * 1024));
    softmax(input, -1);
    EXPECT_EQ(executor.dispatches, 0) << rows;
  }
}

TEST(OnnxLightSigmoidSoftmaxKernel, RegisteredFloat32UsesFusedAvx2Implementation) {
  if (onnx_light_cpu::DetectSimdLevel() < onnx_light_cpu::SimdLevel::kAVX2 ||
      !onnx_light_cpu::CpuSupportsFma()) {
    GTEST_SKIP() << "AVX2/FMA is unavailable";
  }
  const onnx_light_cpu::SigmoidKernel sigmoid(Context());
  const onnx_light_cpu::SoftmaxKernel softmax(Context());
  // Exact equality checks dispatch, not just the scalar fallback's accuracy.
  for (std::int64_t columns : {7, 8, 9, 15, 16, 17, 1023, 1024, 1025}) {
    constexpr std::int64_t rows = 3;
    const auto values = Values(static_cast<std::size_t>(rows * columns));
    const auto input = rt::Tensor::FromFloat("", {rows, columns}, values);
    std::vector<float> expected(values.size());
    onnx_light_cpu::SigmoidFloat32_AVX2_FMA(values.data(), expected.data(), expected.size());
    const auto sigmoid_output = sigmoid(input);
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), sigmoid_output.AsFloat()))
        << "Sigmoid columns=" << columns;
    onnx_light_cpu::SoftmaxFloat32_AVX2_FMA(values.data(), expected.data(), rows, columns);
    const auto softmax_output = softmax(input, -1);
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), softmax_output.AsFloat()))
        << "Softmax columns=" << columns;
  }
}
#endif

} // namespace
