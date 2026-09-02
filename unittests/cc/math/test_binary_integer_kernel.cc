// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/binary/binary_broadcast_plan.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

using onnx_light_cpu::BinaryBroadcastPlan;
using BinaryDataType = onnx_light_cpu::DataType;
using onnx_light_cpu::BinaryKernelDescriptor;

template <typename T> constexpr BinaryDataType DataTypeOf();
template <> constexpr BinaryDataType DataTypeOf<std::int8_t>() { return BinaryDataType::INT8; }
template <> constexpr BinaryDataType DataTypeOf<std::int16_t>() { return BinaryDataType::INT16; }
template <> constexpr BinaryDataType DataTypeOf<std::int32_t>() { return BinaryDataType::INT32; }
template <> constexpr BinaryDataType DataTypeOf<std::int64_t>() { return BinaryDataType::INT64; }
template <> constexpr BinaryDataType DataTypeOf<std::uint8_t>() { return BinaryDataType::UINT8; }
template <> constexpr BinaryDataType DataTypeOf<std::uint16_t>() { return BinaryDataType::UINT16; }
template <> constexpr BinaryDataType DataTypeOf<std::uint32_t>() { return BinaryDataType::UINT32; }
template <> constexpr BinaryDataType DataTypeOf<std::uint64_t>() { return BinaryDataType::UINT64; }

std::int64_t Opset(std::string_view op) {
  if (op == "And" || op == "Or" || op == "Xor")
    return 7;
  if (op == "BitShift")
    return 11;
  if (op == "Mod" || op == "Greater" || op == "Less")
    return 13;
  if (op == "GreaterOrEqual" || op == "LessOrEqual")
    return 16;
  if (op == "BitwiseAnd" || op == "BitwiseOr" || op == "BitwiseXor")
    return 18;
  if (op == "Equal")
    return 19;
  return 14;
}

template <typename T, typename Out = T>
std::vector<Out> Execute(std::string_view op, const std::vector<T> &left,
                         const std::vector<T> &right,
                         const BinaryKernelDescriptor::Attributes &attributes = {}) {
  const BinaryKernelDescriptor descriptor(std::string(op), Opset(op), attributes);
  const std::vector<std::int64_t> shape{static_cast<std::int64_t>(left.size())};
  const bool logical = op == "And" || op == "Or" || op == "Xor";
  const bool comparison = op == "Equal" || op == "Greater" || op == "GreaterOrEqual" ||
                          op == "Less" || op == "LessOrEqual";
  const BinaryDataType input_type = logical ? BinaryDataType::BOOL : DataTypeOf<T>();
  const BinaryDataType output_type =
      logical || comparison ? BinaryDataType::BOOL : DataTypeOf<Out>();
  const BinaryBroadcastPlan plan(descriptor, input_type, input_type, output_type, shape, shape);
  std::vector<Out> output(left.size());
  plan.Execute(left.data(), right.data(), output.data());
  return output;
}

template <typename T, typename Op> T Wrapped(T left, T right, Op op) {
  using U = std::make_unsigned_t<T>;
  return std::bit_cast<T>(static_cast<U>(op(static_cast<U>(left), static_cast<U>(right))));
}

template <typename T> T PythonMod(T left, T right) {
  T value = static_cast<T>(left % right);
  if constexpr (std::is_signed_v<T>) {
    if (value != 0 && ((value < 0) != (right < 0)))
      value = static_cast<T>(value + right);
  }
  return value;
}

template <typename T> void CheckIntegerKernels() {
  constexpr std::size_t count = 17;
  std::vector<T> left(count);
  std::vector<T> right(count);
  for (std::size_t i = 0; i < count; ++i) {
    if constexpr (std::is_signed_v<T>) {
      left[i] = static_cast<T>(static_cast<int>(i % 9) - 4);
      if (left[i] == 0)
        left[i] = T(-2);
      right[i] = static_cast<T>(static_cast<int>(i % 5) + 1);
    } else {
      left[i] = static_cast<T>(i * 3 + 1);
      right[i] = static_cast<T>(i % 5 + 1);
    }
  }

  const auto check_arithmetic = [&](std::string_view op, auto expected) {
    EXPECT_EQ(Execute<T>(op, left, right), expected(left, right));
    EXPECT_EQ(Execute<T>(op, right, left), expected(right, left));
  };
  check_arithmetic("Add", [](const auto &a, const auto &b) {
    std::vector<T> result(a.size());
    for (std::size_t i = 0; i < a.size(); ++i)
      result[i] = Wrapped<T>(a[i], b[i], [](auto x, auto y) { return x + y; });
    return result;
  });
  check_arithmetic("Sub", [](const auto &a, const auto &b) {
    std::vector<T> result(a.size());
    for (std::size_t i = 0; i < a.size(); ++i)
      result[i] = Wrapped<T>(a[i], b[i], [](auto x, auto y) { return x - y; });
    return result;
  });
  check_arithmetic("Mul", [](const auto &a, const auto &b) {
    std::vector<T> result(a.size());
    for (std::size_t i = 0; i < a.size(); ++i)
      result[i] = Wrapped<T>(a[i], b[i], [](auto x, auto y) { return x * y; });
    return result;
  });

  for (const auto &[a, b] :
       std::array<std::pair<const std::vector<T> *, const std::vector<T> *>, 2>{
           std::pair{&left, &right}, std::pair{&right, &left}}) {
    std::vector<T> div_expected(count);
    std::vector<T> mod_expected(count);
    for (std::size_t i = 0; i < count; ++i) {
      div_expected[i] = static_cast<T>((*a)[i] / (*b)[i]);
      mod_expected[i] = PythonMod((*a)[i], (*b)[i]);
    }
    EXPECT_EQ(Execute<T>("Div", *a, *b), div_expected);
    EXPECT_EQ(Execute<T>("Mod", *a, *b), mod_expected);
  }

  for (std::string_view op : {"Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual"}) {
    const std::vector<std::uint8_t> actual = Execute<T, std::uint8_t>(op, left, right);
    for (std::size_t i = 0; i < count; ++i) {
      bool expected = left[i] == right[i];
      if (op == "Greater")
        expected = left[i] > right[i];
      else if (op == "GreaterOrEqual")
        expected = left[i] >= right[i];
      else if (op == "Less")
        expected = left[i] < right[i];
      else if (op == "LessOrEqual")
        expected = left[i] <= right[i];
      EXPECT_EQ(actual[i], expected ? 1U : 0U) << op << " index=" << i;
    }
  }

  for (std::string_view op : {"BitwiseAnd", "BitwiseOr", "BitwiseXor"}) {
    const std::vector<T> actual = Execute<T>(op, left, right);
    for (std::size_t i = 0; i < count; ++i) {
      const T expected = op == "BitwiseAnd"  ? static_cast<T>(left[i] & right[i])
                         : op == "BitwiseOr" ? static_cast<T>(left[i] | right[i])
                                             : static_cast<T>(left[i] ^ right[i]);
      EXPECT_EQ(actual[i], expected) << op << " index=" << i;
    }
  }

  if constexpr (std::is_unsigned_v<T>) {
    std::vector<T> shifts(count);
    for (std::size_t i = 0; i < count; ++i)
      shifts[i] = static_cast<T>(i % (sizeof(T) * 8));
    for (auto direction : {BinaryKernelDescriptor::Attributes::BitShiftDirection::kLeft,
                           BinaryKernelDescriptor::Attributes::BitShiftDirection::kRight}) {
      BinaryKernelDescriptor::Attributes attributes;
      attributes.bitshift_direction = direction;
      const std::vector<T> actual = Execute<T>("BitShift", left, shifts, attributes);
      for (std::size_t i = 0; i < count; ++i) {
        const T expected = direction == BinaryKernelDescriptor::Attributes::BitShiftDirection::kLeft
                               ? static_cast<T>(left[i] << shifts[i])
                               : static_cast<T>(left[i] >> shifts[i]);
        EXPECT_EQ(actual[i], expected) << "index=" << i;
      }
    }
  }
}

template <typename T> void CheckValidationFailures() {
  const BinaryDataType type = DataTypeOf<T>();
  const std::vector<std::int64_t> left_shape{2, 1};
  const std::vector<std::int64_t> right_shape{1, 3};
  const auto expect_unchanged = [&](std::string_view op, const std::vector<T> &left,
                                    const std::vector<T> &right,
                                    const BinaryKernelDescriptor::Attributes &attributes = {}) {
    const BinaryKernelDescriptor descriptor(std::string(op), Opset(op), attributes);
    const BinaryBroadcastPlan plan(descriptor, type, type, type, left_shape, right_shape);
    std::vector<T> output(6, static_cast<T>(0x5A));
    const std::vector<T> before = output;
    EXPECT_THROW(plan.Execute(left.data(), right.data(), output.data()), std::invalid_argument);
    EXPECT_EQ(output, before);
  };

  expect_unchanged("Div", {T(1), T(2)}, {T(1), T(1), T(0)});
  expect_unchanged("Mod", {T(1), T(2)}, {T(1), T(1), T(0)});
  if constexpr (std::is_signed_v<T>) {
    expect_unchanged("Div", {T(1), std::numeric_limits<T>::min()}, {T(1), T(1), T(-1)});
    expect_unchanged("Mod", {T(1), std::numeric_limits<T>::min()}, {T(1), T(1), T(-1)});
  } else {
    BinaryKernelDescriptor::Attributes attributes;
    attributes.bitshift_direction = BinaryKernelDescriptor::Attributes::BitShiftDirection::kLeft;
    expect_unchanged("BitShift", {T(1), T(2)}, {T(0), T(1), static_cast<T>(sizeof(T) * 8)},
                     attributes);
  }
}

template <typename T> void CheckPhysicalShiftValidation() {
  BinaryKernelDescriptor::Attributes attributes;
  attributes.bitshift_direction = BinaryKernelDescriptor::Attributes::BitShiftDirection::kLeft;
  const BinaryKernelDescriptor descriptor("BitShift", 11, attributes);
  const BinaryDataType type = DataTypeOf<T>();
  const auto &adapter = descriptor.ResolveAdapter(type, type, type);
  ASSERT_NE(adapter.validate_right_bulk, nullptr);

  const std::vector<T> valid{T(0), static_cast<T>(sizeof(T) * 8 - 1)};
  EXPECT_FALSE(adapter.validate_right_bulk(valid.data(), valid.size()));
  const std::vector<T> invalid{T(0), static_cast<T>(sizeof(T) * 8)};
  EXPECT_THROW(adapter.validate_right_bulk(invalid.data(), invalid.size()), std::invalid_argument);
}

TEST(BinaryIntegerKernel, CoversEverySignedAndUnsignedWidthWithTail) {
  CheckIntegerKernels<std::int8_t>();
  CheckIntegerKernels<std::int16_t>();
  CheckIntegerKernels<std::int32_t>();
  CheckIntegerKernels<std::int64_t>();
  CheckIntegerKernels<std::uint8_t>();
  CheckIntegerKernels<std::uint16_t>();
  CheckIntegerKernels<std::uint32_t>();
  CheckIntegerKernels<std::uint64_t>();
}

TEST(BinaryIntegerKernel, WrapsSignedOverflowByBitPattern) {
  const auto check = []<typename T>() {
    const std::vector<T> left{std::numeric_limits<T>::max(), std::numeric_limits<T>::min(),
                              std::numeric_limits<T>::max()};
    const std::vector<T> right{T(1), T(1), T(2)};
    EXPECT_EQ(Execute<T>("Add", left, right),
              (std::vector<T>{std::numeric_limits<T>::min(),
                              static_cast<T>(std::numeric_limits<T>::min() + 1),
                              static_cast<T>(std::numeric_limits<T>::min() + 1)}));
    EXPECT_EQ(Execute<T>("Sub", left, right),
              (std::vector<T>{static_cast<T>(std::numeric_limits<T>::max() - 1),
                              std::numeric_limits<T>::max(),
                              static_cast<T>(std::numeric_limits<T>::max() - 2)}));
    EXPECT_EQ(
        Execute<T>("Mul", left, right),
        (std::vector<T>{std::numeric_limits<T>::max(), std::numeric_limits<T>::min(), T(-2)}));
  };
  check.template operator()<std::int8_t>();
  check.template operator()<std::int16_t>();
  check.template operator()<std::int32_t>();
  check.template operator()<std::int64_t>();
}

TEST(BinaryIntegerKernel, RejectsUndefinedInputsBeforeWritingOutput) {
  CheckValidationFailures<std::int8_t>();
  CheckValidationFailures<std::int16_t>();
  CheckValidationFailures<std::int32_t>();
  CheckValidationFailures<std::int64_t>();
  CheckValidationFailures<std::uint8_t>();
  CheckValidationFailures<std::uint16_t>();
  CheckValidationFailures<std::uint32_t>();
  CheckValidationFailures<std::uint64_t>();
}

TEST(BinaryIntegerKernel, ValidatesPhysicalShiftTensorWithoutExpandedPairScan) {
  CheckPhysicalShiftValidation<std::uint8_t>();
  CheckPhysicalShiftValidation<std::uint16_t>();
  CheckPhysicalShiftValidation<std::uint32_t>();
  CheckPhysicalShiftValidation<std::uint64_t>();
}

TEST(BinaryIntegerKernel, ValidatesSignedDivisionOverflowForActualPairsOnly) {
  const BinaryKernelDescriptor descriptor("Div", 14, {});
  const std::vector<std::int64_t> left_shape{2, 2};
  const std::vector<std::int64_t> right_shape{2, 1};
  const BinaryBroadcastPlan plan(descriptor, BinaryDataType::INT32, BinaryDataType::INT32,
                                 BinaryDataType::INT32, left_shape, right_shape);
  const std::vector<std::int32_t> left{std::numeric_limits<std::int32_t>::min(), 4, 1, 2};
  const std::vector<std::int32_t> right{1, -1};
  std::vector<std::int32_t> output(left.size());
  EXPECT_NO_THROW(plan.Execute(left.data(), right.data(), output.data()));
  EXPECT_EQ(output,
            (std::vector<std::int32_t>{std::numeric_limits<std::int32_t>::min(), 4, -1, -2}));
}

TEST(BinaryLogicalKernel, EmitsCanonicalByteBool) {
  const std::vector<std::uint8_t> left{0, 0, 1, 1, 1};
  const std::vector<std::uint8_t> right{0, 1, 0, 1, 1};
  EXPECT_EQ(Execute<std::uint8_t>("And", left, right), (std::vector<std::uint8_t>{0, 0, 0, 1, 1}));
  EXPECT_EQ(Execute<std::uint8_t>("Or", left, right), (std::vector<std::uint8_t>{0, 1, 1, 1, 1}));
  EXPECT_EQ(Execute<std::uint8_t>("Xor", left, right), (std::vector<std::uint8_t>{0, 1, 1, 0, 0}));
}

TEST(BinaryComparisonKernel, BroadcastsBothOperandOrdersToByteBool) {
  const BinaryKernelDescriptor descriptor("Less", 13, {});
  const std::vector<std::int64_t> left_shape{2, 1};
  const std::vector<std::int64_t> right_shape{1, 3};
  const BinaryBroadcastPlan plan(descriptor, BinaryDataType::INT32, BinaryDataType::INT32,
                                 BinaryDataType::BOOL, left_shape, right_shape);
  const std::vector<std::int32_t> left{1, 4};
  const std::vector<std::int32_t> right{2, 4, 0};
  std::vector<std::uint8_t> output(6, 0xFF);
  plan.Execute(left.data(), right.data(), output.data());
  EXPECT_EQ(output, (std::vector<std::uint8_t>{1, 1, 0, 0, 0, 0}));

  const BinaryBroadcastPlan swapped(descriptor, BinaryDataType::INT32, BinaryDataType::INT32,
                                    BinaryDataType::BOOL, right_shape, left_shape);
  swapped.Execute(right.data(), left.data(), output.data());
  EXPECT_EQ(output, (std::vector<std::uint8_t>{0, 0, 1, 1, 0, 1}));
}

TEST(BinaryEqualKernel, BroadcastsStringsToByteBool) {
  const BinaryKernelDescriptor descriptor("Equal", 19, {});
  EXPECT_EQ(descriptor.ResolveOutputType(BinaryDataType::STRING, BinaryDataType::STRING),
            BinaryDataType::BOOL);
  const std::vector<std::int64_t> left_shape{2, 1};
  const std::vector<std::int64_t> right_shape{1, 3};
  const BinaryBroadcastPlan plan(descriptor, BinaryDataType::STRING, BinaryDataType::STRING,
                                 BinaryDataType::BOOL, left_shape, right_shape);
  const std::vector<std::string> left{"a", "b"};
  const std::vector<std::string> right{"a", "b", "c"};
  std::vector<std::uint8_t> output(6);
  plan.Execute(left.data(), right.data(), output.data());
  EXPECT_EQ(output, (std::vector<std::uint8_t>{1, 0, 0, 0, 1, 0}));
}

} // namespace
