// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "ml_cpu_compute/execution.h"
#include "ml_cpu_compute/logical/logical.h"
#include "ml_cpu_compute/math/binary/binary_broadcast_plan.h"
#include "ml_cpu_compute/math/binary/binary_comparison.h"
#include "ml_cpu_compute/math/half_conversion.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ml_cpu_compute::BinaryBroadcastPlan;
using ml_cpu_compute::BinaryBroadcastPlanCache;
using BinaryDataType = ml_cpu_compute::DataType;
using ml_cpu_compute::BinaryOperationDescriptor;
using LoopFamily = ml_cpu_compute::BinaryBroadcastPlan::LoopFamily;

TEST(BinaryBroadcastPlan, FloatGreaterContiguousAndRows) {
  const std::array<float, 12> values{-std::numeric_limits<float>::infinity(),
                                     -17.0f,
                                     -1.0f,
                                     -0.0f,
                                     0.0f,
                                     1.0f,
                                     17.0f,
                                     std::numeric_limits<float>::infinity(),
                                     std::numeric_limits<float>::quiet_NaN(),
                                     std::numeric_limits<float>::denorm_min(),
                                     -std::numeric_limits<float>::denorm_min(),
                                     2.0f};
  BinaryOperationDescriptor descriptor("Greater", 13, {});
  for (std::size_t count = 0; count <= 257; ++count) {
    for (bool rows : {false, true}) {
      const std::size_t row_count = rows ? 3 : 1;
      const std::vector<std::int64_t> shape{static_cast<std::int64_t>(row_count),
                                            static_cast<std::int64_t>(count)};
      const std::vector<std::int64_t> right_shape{static_cast<std::int64_t>(count)};
      // Offset the inputs and output to exercise unaligned loads and stores.
      std::vector<float> left(row_count * count + 1), right(count + 1);
      for (std::size_t i = 0; i < row_count * count; ++i) {
        left[i + 1] = values[i % values.size()];
      }
      for (std::size_t i = 0; i < count; ++i) {
        right[i + 1] = values[(i / values.size() + i) % values.size()];
      }
      for (bool swapped : {false, true}) {
        SCOPED_TRACE(::testing::Message() << count << " rows=" << rows << " swapped=" << swapped);
        BinaryBroadcastPlan plan(descriptor, BinaryDataType::FLOAT, BinaryDataType::FLOAT,
                                 BinaryDataType::BOOL, swapped ? right_shape : shape,
                                 swapped ? shape : right_shape);
        std::vector<std::uint8_t> output(row_count * count + 2, 0xA5);
        plan.Execute(swapped ? right.data() + 1 : left.data() + 1,
                     swapped ? left.data() + 1 : right.data() + 1, output.data() + 1);
        EXPECT_EQ(output.front(), 0xA5);
        EXPECT_EQ(output.back(), 0xA5);
        for (std::size_t i = 0; i < row_count * count; ++i) {
          const float a = left[i + 1], b = right[i % count + 1];
          EXPECT_EQ(output[i + 1], static_cast<std::uint8_t>(swapped ? b > a : a > b));
        }
      }
    }
  }
}

TEST(BinaryBroadcastPlan, IntegerImplementationDiagnostics) {
  const std::vector<std::int64_t> vector{33}, scalar{}, channel{3, 1}, tensor{2, 3, 33}, empty{0};
  for (BinaryDataType type : {BinaryDataType::INT32, BinaryDataType::INT64}) {
    ml_cpu_compute::BinaryOperationDescriptor pow("Pow", 15, {});
    EXPECT_EQ(ml_cpu_compute::BinaryBroadcastPlan(pow, type, type, type, vector, scalar)
                  .implementation_path(),
              "integer_pow.scalar_exponent");
    EXPECT_EQ(ml_cpu_compute::BinaryBroadcastPlan(pow, type, type, type, tensor, channel)
                  .implementation_path(),
              "integer_pow.scalar_exponent");
    EXPECT_EQ(ml_cpu_compute::BinaryBroadcastPlan(pow, type, type, type, vector, vector)
                  .implementation_path(),
              "integer_pow.checked");
    EXPECT_EQ(ml_cpu_compute::BinaryBroadcastPlan(pow, type, type, type, empty, scalar)
                  .implementation_path(),
              "empty");
    for (const char *op : {"Div", "Mod"}) {
      ml_cpu_compute::BinaryOperationDescriptor descriptor(op, 14, {});
      EXPECT_EQ(ml_cpu_compute::BinaryBroadcastPlan(descriptor, type, type, type, tensor, channel)
                    .implementation_path(),
                "integer_divmod.invariant_divisor");
      EXPECT_EQ(ml_cpu_compute::BinaryBroadcastPlan(descriptor, type, type, type, vector, vector)
                    .implementation_path(),
                "integer_divmod.typed");
    }
  }
  for (BinaryDataType type : {BinaryDataType::INT64, BinaryDataType::UINT64}) {
    for (const char *op : {"Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual"}) {
      ml_cpu_compute::BinaryOperationDescriptor descriptor(op, 16, {});
      const auto path = ml_cpu_compute::BinaryBroadcastPlan(descriptor, type, type,
                                                            BinaryDataType::BOOL, tensor, channel)
                            .implementation_path();
      EXPECT_EQ(path,
                std::string("compare64.") + ml_cpu_compute::BinaryCompareInt64Implementation(33));
      const std::vector<std::int64_t> short_vector{3};
      EXPECT_EQ(ml_cpu_compute::BinaryBroadcastPlan(descriptor, type, type, BinaryDataType::BOOL,
                                                    short_vector, scalar)
                    .implementation_path(),
                "compare64.scalar");
    }
  }
}

std::size_t ElementSize(BinaryDataType type) {
  switch (type) {
  case BinaryDataType::BOOL:
  case BinaryDataType::INT8:
  case BinaryDataType::UINT8:
    return 1;
  case BinaryDataType::INT16:
  case BinaryDataType::UINT16:
  case BinaryDataType::FLOAT16:
  case BinaryDataType::BFLOAT16:
    return 2;
  case BinaryDataType::INT32:
  case BinaryDataType::UINT32:
  case BinaryDataType::FLOAT:
    return 4;
  case BinaryDataType::INT64:
  case BinaryDataType::UINT64:
  case BinaryDataType::DOUBLE:
    return 8;
  default:
    throw std::invalid_argument("unsupported test type");
  }
}

std::size_t ElementCount(std::span<const std::int64_t> shape) {
  std::size_t count = 1;
  for (std::int64_t dim : shape) {
    count *= static_cast<std::size_t>(dim);
  }
  return count;
}

std::vector<std::int64_t> BroadcastShape(std::span<const std::int64_t> left,
                                         std::span<const std::int64_t> right) {
  const std::size_t rank = std::max(left.size(), right.size());
  std::vector<std::int64_t> out(rank, 1);
  const std::size_t left_offset = rank - left.size();
  const std::size_t right_offset = rank - right.size();
  for (std::size_t i = 0; i < rank; ++i) {
    const std::int64_t l = i < left_offset ? 1 : left[i - left_offset];
    const std::int64_t r = i < right_offset ? 1 : right[i - right_offset];
    if (l == r || l == 1) {
      out[i] = r;
    } else if (r == 1) {
      out[i] = l;
    } else {
      throw std::invalid_argument("non-broadcastable");
    }
  }
  return out;
}

std::size_t OffsetFor(std::span<const std::int64_t> shape, std::span<const std::int64_t> coords) {
  const std::size_t rank = coords.size();
  const std::size_t shape_offset = rank - shape.size();
  std::size_t offset = 0;
  std::size_t stride = 1;
  for (std::size_t i = rank; i-- > 0;) {
    const std::int64_t dim = i < shape_offset ? 1 : shape[i - shape_offset];
    const std::size_t coord = dim == 1 ? 0u : static_cast<std::size_t>(coords[i]);
    offset += coord * stride;
    stride *= static_cast<std::size_t>(dim);
  }
  return offset;
}

void StoreScalar(BinaryDataType type, std::byte *dst, std::size_t index, std::size_t count,
                 bool is_left, std::string_view op_type) {
  const auto write = [&](auto value) {
    using T = decltype(value);
    std::memcpy(dst + index * sizeof(T), &value, sizeof(T));
  };
  const int base = static_cast<int>((index % 5) + 1);
  const int signed_value =
      (op_type == "PRelu" && is_left) ? (base - 3) : (is_left ? base - 2 : base);
  switch (type) {
  case BinaryDataType::BOOL:
    write(static_cast<std::uint8_t>(((index + (is_left ? 0 : 1)) & 1U) != 0U));
    break;
  case BinaryDataType::INT8:
    write(static_cast<std::int8_t>(signed_value));
    break;
  case BinaryDataType::INT16:
    write(static_cast<std::int16_t>(signed_value));
    break;
  case BinaryDataType::INT32:
    if (op_type == "Pow" && !is_left) {
      write(static_cast<std::int32_t>(index % 3));
    } else {
      write(static_cast<std::int32_t>(signed_value));
    }
    break;
  case BinaryDataType::INT64:
    if (op_type == "Pow" && !is_left) {
      write(static_cast<std::int64_t>(index % 3));
    } else {
      write(static_cast<std::int64_t>(signed_value));
    }
    break;
  case BinaryDataType::UINT8:
    write(static_cast<std::uint8_t>(op_type == "BitShift" && !is_left ? (index % 3) : base));
    break;
  case BinaryDataType::UINT16:
    write(static_cast<std::uint16_t>(op_type == "BitShift" && !is_left ? (index % 3) : base));
    break;
  case BinaryDataType::UINT32:
    write(static_cast<std::uint32_t>(
        op_type == "Pow" && !is_left ? (index % 3)
                                     : (op_type == "BitShift" && !is_left ? (index % 5) : base)));
    break;
  case BinaryDataType::UINT64:
    write(static_cast<std::uint64_t>(
        op_type == "Pow" && !is_left ? (index % 3)
                                     : (op_type == "BitShift" && !is_left ? (index % 6) : base)));
    break;
  case BinaryDataType::FLOAT:
    if (op_type == "Pow" && !is_left) {
      write(static_cast<float>(index % 3));
    } else {
      write(static_cast<float>(signed_value));
    }
    break;
  case BinaryDataType::DOUBLE:
    write(static_cast<double>(op_type == "Pow" && !is_left ? (index % 3) : signed_value));
    break;
  case BinaryDataType::FLOAT16: {
    const float v = static_cast<float>(op_type == "Pow" && !is_left ? (index % 3) : signed_value);
    const std::uint16_t bits = ml_cpu_compute::detail::FloatToFloat16Bits(v);
    write(bits);
    break;
  }
  case BinaryDataType::BFLOAT16: {
    const float v = static_cast<float>(op_type == "Pow" && !is_left ? (index % 3) : signed_value);
    const std::uint16_t bits = ml_cpu_compute::detail::FloatToBFloat16Bits(v);
    write(bits);
    break;
  }
  default:
    throw std::invalid_argument("unsupported test type");
  }
}

std::vector<std::byte> MakeBuffer(BinaryDataType type, std::span<const std::int64_t> shape,
                                  bool is_left, std::string_view op_type) {
  const std::size_t count = ElementCount(shape);
  std::vector<std::byte> data(count * ElementSize(type));
  for (std::size_t i = 0; i < count; ++i) {
    StoreScalar(type, data.data(), i, count, is_left, op_type);
  }
  return data;
}

std::vector<std::byte>
ReferenceExecute(const ml_cpu_compute::BinaryOperationDescriptor::Adapter &adapter,
                 std::span<const std::int64_t> left_shape,
                 std::span<const std::int64_t> right_shape,
                 std::span<const std::int64_t> output_shape, const std::vector<std::byte> &left,
                 const std::vector<std::byte> &right) {
  const std::size_t output_count = ElementCount(output_shape);
  std::vector<std::byte> output(output_count * adapter.output_size);
  std::vector<std::int64_t> coords(output_shape.size(), 0);
  for (std::size_t index = 0; index < output_count; ++index) {
    std::size_t left_offset = OffsetFor(left_shape, coords);
    std::size_t right_offset = OffsetFor(right_shape, coords);
    adapter.scalar(left.data() + left_offset * adapter.left_size,
                   right.data() + right_offset * adapter.right_size,
                   output.data() + index * adapter.output_size);
    for (std::size_t axis = output_shape.size(); axis-- > 0;) {
      if (++coords[axis] < output_shape[axis]) {
        break;
      }
      coords[axis] = 0;
    }
  }
  return output;
}

ml_cpu_compute::BinaryOperationDescriptor::Attributes DefaultAttributes(std::string_view op_type,
                                                                        BinaryDataType left) {
  ml_cpu_compute::BinaryOperationDescriptor::Attributes attrs;
  if (op_type == "Mod") {
    attrs.mod_fmod = (left == BinaryDataType::FLOAT || left == BinaryDataType::DOUBLE ||
                      left == BinaryDataType::FLOAT16 || left == BinaryDataType::BFLOAT16)
                         ? 1
                         : 0;
  }
  return attrs;
}

template <typename T>
std::vector<T> NaiveAdd(std::span<const T> left, std::span<const std::int64_t> left_shape,
                        std::span<const T> right, std::span<const std::int64_t> right_shape) {
  const auto output_shape = BroadcastShape(left_shape, right_shape);
  const std::size_t output_count = ElementCount(output_shape);
  std::vector<T> output(output_count);
  std::vector<std::int64_t> coords(output_shape.size(), 0);
  for (std::size_t index = 0; index < output_count; ++index) {
    output[index] = left[OffsetFor(left_shape, coords)] + right[OffsetFor(right_shape, coords)];
    for (std::size_t axis = output_shape.size(); axis-- > 0;) {
      if (++coords[axis] < output_shape[axis]) {
        break;
      }
      coords[axis] = 0;
    }
  }
  return output;
}

TEST(BinaryBroadcastPlan, ClassifiesLoopFamiliesAndAliasSafety) {
  const ml_cpu_compute::BinaryOperationDescriptor descriptor("Add", 14, {});
  const std::array cases = {
      std::pair{std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>{{2, 3}, {2, 3}},
                LoopFamily::kContiguous},
      std::pair{std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>{{}, {2, 3}},
                LoopFamily::kLeftScalar},
      std::pair{std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>{{2, 3}, {}},
                LoopFamily::kRightScalar},
      std::pair{
          std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>{{2, 3, 4, 5}, {3, 1, 1}},
          LoopFamily::kRepeatedContiguousBlock},
      std::pair{std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>{{2, 3, 4, 5},
                                                                                {1, 3, 1, 5}},
                LoopFamily::kOuterBroadcast},
      std::pair{std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>{{2, 1, 4, 5},
                                                                                {1, 3, 4, 5}},
                LoopFamily::kOuterBroadcast},
      std::pair{std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>{{2, 1, 4, 1, 8, 1},
                                                                                {1, 3, 1, 5, 1, 7}},
                LoopFamily::kInnerVectorBroadcast},
  };

  for (const auto &[shapes, family] : cases) {
    const ml_cpu_compute::BinaryBroadcastPlan plan(descriptor, BinaryDataType::FLOAT,
                                                   BinaryDataType::FLOAT, BinaryDataType::FLOAT,
                                                   shapes.first, shapes.second);
    EXPECT_EQ(plan.loop_family(), family);
  }

  const ml_cpu_compute::BinaryBroadcastPlan exact(
      descriptor, BinaryDataType::FLOAT, BinaryDataType::FLOAT, BinaryDataType::FLOAT,
      std::array<std::int64_t, 2>{2, 3}, std::array<std::int64_t, 2>{2, 3});
  EXPECT_TRUE(exact.left_output_alias_safe());
  EXPECT_TRUE(exact.right_output_alias_safe());

  const ml_cpu_compute::BinaryBroadcastPlan broadcast_right(
      descriptor, BinaryDataType::FLOAT, BinaryDataType::FLOAT, BinaryDataType::FLOAT,
      std::array<std::int64_t, 2>{2, 3}, std::array<std::int64_t, 0>{});
  EXPECT_TRUE(broadcast_right.left_output_alias_safe());
  EXPECT_FALSE(broadcast_right.right_output_alias_safe());
}

TEST(BinaryBroadcastPlan, ExecutesRepresentativeBroadcastLoops) {
  const std::array families = {
      std::pair{std::vector<std::int64_t>{2, 3}, std::vector<std::int64_t>{2, 3}},
      std::pair{std::vector<std::int64_t>{}, std::vector<std::int64_t>{2, 3}},
      std::pair{std::vector<std::int64_t>{2, 3}, std::vector<std::int64_t>{}},
  };

  const ml_cpu_compute::BinaryOperationDescriptor add_descriptor("Add", 14, {});
  for (const auto &[left_shape, right_shape] : families) {
    const ml_cpu_compute::BinaryBroadcastPlan plan(add_descriptor, BinaryDataType::FLOAT,
                                                   BinaryDataType::FLOAT, BinaryDataType::FLOAT,
                                                   left_shape, right_shape);
    std::vector<float> left(ElementCount(left_shape));
    std::vector<float> right(ElementCount(right_shape));
    for (std::size_t i = 0; i < left.size(); ++i)
      left[i] = static_cast<float>(static_cast<int>(i % 5) - 2);
    for (std::size_t i = 0; i < right.size(); ++i)
      right[i] = static_cast<float>(static_cast<int>(i % 7) + 1);
    std::vector<float> actual(ElementCount(plan.output_shape()), -1.0f);
    plan.Execute(left.data(), right.data(), actual.data());
    const auto expected = NaiveAdd<float>(left, left_shape, right, right_shape);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
      EXPECT_FLOAT_EQ(actual[i], expected[i]) << "index=" << i;
    }
  }
}

// Binary PR02: BinaryBroadcastPlan::Execute now special-cases kContiguous,
// kLeftScalar and kRightScalar to invoke a bulk SIMD kernel instead of the
// per-element scalar loop. This test drives that fast path directly for
// every arithmetic op, FP32 and FP64, across SIMD tail sizes, and checks the
// bulk result against the same adapter's per-element scalar reference
// (ReferenceExecute), which never uses a bulk kernel.
template <typename T> BinaryDataType TypeOf();
template <> BinaryDataType TypeOf<float>() { return BinaryDataType::FLOAT; }
template <> BinaryDataType TypeOf<double>() { return BinaryDataType::DOUBLE; }

template <typename T>
void CheckArithmeticFastPathMatchesScalarReference(std::string_view op_type, std::size_t count) {
  const BinaryDataType type = TypeOf<T>();
  const ml_cpu_compute::BinaryOperationDescriptor descriptor(std::string(op_type), 14, {});
  const std::array<std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>, 3> shapes{{
      {std::vector<std::int64_t>{static_cast<std::int64_t>(count)},
       std::vector<std::int64_t>{static_cast<std::int64_t>(count)}},
      {std::vector<std::int64_t>{}, std::vector<std::int64_t>{static_cast<std::int64_t>(count)}},
      {std::vector<std::int64_t>{static_cast<std::int64_t>(count)}, std::vector<std::int64_t>{}},
  }};
  for (const auto &[left_shape, right_shape] : shapes) {
    const ml_cpu_compute::BinaryBroadcastPlan plan(descriptor, type, type, type, left_shape,
                                                   right_shape);
    const std::vector<std::byte> left = MakeBuffer(type, left_shape, /*is_left=*/true, op_type);
    const std::vector<std::byte> right = MakeBuffer(type, right_shape, /*is_left=*/false, op_type);
    const std::vector<std::byte> expected =
        ReferenceExecute(descriptor.ResolveAdapter(type, type, type), left_shape, right_shape,
                         plan.output_shape(), left, right);
    std::vector<std::byte> actual(expected.size(), std::byte{0xCD});
    plan.Execute(left.data(), right.data(), actual.data());
    ASSERT_EQ(actual.size(), expected.size());
    const std::size_t output_count = ElementCount(plan.output_shape());
    for (std::size_t i = 0; i < output_count; ++i) {
      T actual_value;
      T expected_value;
      std::memcpy(&actual_value, actual.data() + i * sizeof(T), sizeof(T));
      std::memcpy(&expected_value, expected.data() + i * sizeof(T), sizeof(T));
      if (std::isnan(expected_value)) {
        EXPECT_TRUE(std::isnan(actual_value))
            << "op=" << op_type << " count=" << count << " index=" << i;
      } else {
        EXPECT_EQ(actual_value, expected_value)
            << "op=" << op_type << " count=" << count << " index=" << i;
      }
    }
  }
}

TEST(BinaryBroadcastPlan, PreparesFixedRankTraversalAndKeepsArbitraryRankFallback) {
  const ml_cpu_compute::BinaryOperationDescriptor descriptor("Add", 14, {});
  const std::array cases = {
      std::pair{std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>{{64, 1}, {1, 64}},
                std::size_t{1}},
      std::pair{
          std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>{{2, 64, 32}, {64, 1}},
          std::size_t{2}},
      std::pair{std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>{{2, 1, 8, 1},
                                                                                {1, 4, 1, 64}},
                std::size_t{3}},
      std::pair{std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>{{2, 1, 4, 1, 8, 1},
                                                                                {1, 3, 1, 5, 1, 7}},
                std::size_t{0}},
  };

  for (const auto &[shapes, expected_rank] : cases) {
    const ml_cpu_compute::BinaryBroadcastPlan plan(descriptor, BinaryDataType::FLOAT,
                                                   BinaryDataType::FLOAT, BinaryDataType::FLOAT,
                                                   shapes.first, shapes.second);
    EXPECT_EQ(plan.prepared_outer_rank(), expected_rank);
  }
}

TEST(BinaryBroadcastPlan, ArithmeticFastPathMatchesScalarReferenceAcrossOpsTypesAndTailSizes) {
  const std::array<std::string_view, 4> ops = {"Add", "Sub", "Mul", "Div"};
  const std::array<std::size_t, 13> sizes = {0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32};
  for (std::string_view op : ops) {
    for (std::size_t count : sizes) {
      CheckArithmeticFastPathMatchesScalarReference<float>(op, count);
      CheckArithmeticFastPathMatchesScalarReference<double>(op, count);
    }
  }
}

TEST(BinaryBroadcastPlan, IntegerAndHalfArithmeticAdaptersProvideBulkPaths) {
  const std::array<std::string_view, 3> ops = {"Add", "Sub", "Mul"};
  const std::array<BinaryDataType, 10> types = {
      BinaryDataType::FLOAT16, BinaryDataType::BFLOAT16, BinaryDataType::INT8,
      BinaryDataType::INT16,   BinaryDataType::INT32,    BinaryDataType::INT64,
      BinaryDataType::UINT8,   BinaryDataType::UINT16,   BinaryDataType::UINT32,
      BinaryDataType::UINT64,
  };
  const std::array<std::int64_t, 1> shape{263};
  for (std::string_view op : ops) {
    for (BinaryDataType type : types) {
      const ml_cpu_compute::BinaryOperationDescriptor descriptor(std::string(op), 14, {});
      const auto &adapter = descriptor.ResolveAdapter(type, type, type);
      ASSERT_NE(adapter.bulk_contiguous, nullptr) << op;
      ASSERT_NE(adapter.bulk_left_scalar, nullptr) << op;
      ASSERT_NE(adapter.bulk_right_scalar, nullptr) << op;

      const auto left = MakeBuffer(type, shape, true, op);
      const auto right = MakeBuffer(type, shape, false, op);
      const auto expected = ReferenceExecute(adapter, shape, shape, shape, left, right);
      std::vector<std::byte> actual(expected.size());
      adapter.bulk_contiguous(left.data(), right.data(), actual.data(), shape[0]);
      EXPECT_EQ(actual, expected) << "op=" << op << " type=" << static_cast<int>(type);
    }
  }
}

TEST(BinaryBroadcastPlan, HalfPrecisionAdaptersUseBulkConversionAndMatchScalarReference) {
  const std::array<std::string_view, 3> ops = {"Mod", "Pow", "PRelu"};
  const std::array<BinaryDataType, 2> types = {BinaryDataType::FLOAT16, BinaryDataType::BFLOAT16};
  const std::array<std::size_t, 7> sizes = {0, 1, 7, 8, 15, 256, 263};
  for (std::string_view op : ops) {
    for (BinaryDataType type : types) {
      const ml_cpu_compute::BinaryOperationDescriptor descriptor(
          std::string(op), op == "Pow" ? 15 : 16, DefaultAttributes(op, type));
      const auto &adapter = descriptor.ResolveAdapter(type, type, type);
      ASSERT_NE(adapter.bulk_contiguous, nullptr) << op;
      for (std::size_t count : sizes) {
        const std::array<std::int64_t, 1> shape{static_cast<std::int64_t>(count)};
        const ml_cpu_compute::BinaryBroadcastPlan plan(descriptor, type, type, type, shape, shape);
        const auto left = MakeBuffer(type, shape, true, op);
        const auto right = MakeBuffer(type, shape, false, op);
        const auto expected = ReferenceExecute(adapter, shape, shape, shape, left, right);
        std::vector<std::byte> actual(expected.size());
        plan.Execute(left.data(), right.data(), actual.data());
        EXPECT_EQ(actual, expected) << "op=" << op << " count=" << count;
      }
    }
  }
}

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

// Binary PR03: BinaryBroadcastPlan::Execute now vectorizes the repeated
// block, inner-vector broadcast, outer broadcast and general strided
// families by dispatching a bulk kernel over each outer block's inner
// extent (or the per-element scalar fallback when no bulk kernel applies),
// and may submit independent outer-block ranges to the session executor.
// This checks every non-trivial family against the naive per-element
// reference, both serially and while forced through an executor, across
// sizes that straddle the parallel byte thresholds.
TEST(BinaryBroadcastPlan, MultiDimensionalFamiliesMatchNaiveReferenceSerialAndParallel) {
  const ml_cpu_compute::BinaryOperationDescriptor descriptor("Add", 14, {});
  const std::array<std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>, 5> shapes{{
      // kRepeatedContiguousBlock: row broadcast, large enough to cross the
      // 1 MiB block-parallel threshold (400*8192*4 bytes >> 1 MiB).
      {std::vector<std::int64_t>{400, 8192}, std::vector<std::int64_t>{8192}},
      // kOuterBroadcast: both inner strides contiguous, broadcast outer dim.
      {std::vector<std::int64_t>{300, 2, 4096}, std::vector<std::int64_t>{1, 2, 4096}},
      // kInnerVectorBroadcast: alternating singleton dims.
      {std::vector<std::int64_t>{2, 1, 4, 1, 8, 1}, std::vector<std::int64_t>{1, 3, 1, 5, 1, 7}},
      // kGeneralStrided: column broadcast, no contiguous inner run on the
      // scalar side, large enough to cross the 16 KiB scalar threshold.
      {std::vector<std::int64_t>{4096, 64}, std::vector<std::int64_t>{4096, 1}},
      // Small case that should stay below every parallel threshold.
      {std::vector<std::int64_t>{3, 1, 5}, std::vector<std::int64_t>{1, 4, 1}},
  }};

  for (const auto &[left_shape, right_shape] : shapes) {
    const ml_cpu_compute::BinaryBroadcastPlan plan(descriptor, BinaryDataType::FLOAT,
                                                   BinaryDataType::FLOAT, BinaryDataType::FLOAT,
                                                   left_shape, right_shape);
    ASSERT_GT(plan.dimensions().size(), 1u) << "expected a multi-dimensional plan";
    std::vector<float> left(ElementCount(left_shape));
    std::vector<float> right(ElementCount(right_shape));
    for (std::size_t i = 0; i < left.size(); ++i)
      left[i] = static_cast<float>(static_cast<int>(i % 13) - 6);
    for (std::size_t i = 0; i < right.size(); ++i)
      right[i] = static_cast<float>(static_cast<int>(i % 11) - 5);
    const auto expected = NaiveAdd<float>(left, left_shape, right, right_shape);

    std::vector<float> serial(expected.size(), -1.0f);
    plan.Execute(left.data(), right.data(), serial.data());
    ASSERT_EQ(serial.size(), expected.size());
    for (std::size_t i = 0; i < serial.size(); ++i) {
      EXPECT_FLOAT_EQ(serial[i], expected[i]) << "serial index=" << i;
    }

    std::vector<float> parallel(expected.size(), -1.0f);
    InlineExecutor executor;
    ml_cpu_compute::ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run};
    ml_cpu_compute::ExecutionExecutorScope scope(&view);
    plan.Execute(left.data(), right.data(), parallel.data());
    ASSERT_EQ(parallel.size(), expected.size());
    for (std::size_t i = 0; i < parallel.size(); ++i) {
      EXPECT_FLOAT_EQ(parallel[i], expected[i]) << "parallel index=" << i;
    }
  }
}

// Binary PR03: the flat contiguous/left-scalar/right-scalar path submits
// independent element ranges to the session executor once a plan's byte
// count crosses the calibrated threshold, and stays single-dispatch below
// it, mirroring ExpLogParallel.OperatorSpecificParticipantPolicy.
TEST(BinaryBroadcastPlan, FlatPathDispatchesToExecutorAboveThreshold) {
  const ml_cpu_compute::BinaryOperationDescriptor descriptor("Add", 14, {});
  InlineExecutor executor;
  ml_cpu_compute::ExecutionExecutorView view{&executor, 8, &InlineExecutor::Run};
  ml_cpu_compute::ExecutionExecutorScope scope(&view);

  // The 1 MiB calibrated contiguous bulk threshold (see
  // binary_execution_schedule.h) covers left + right + output traffic
  // (4 + 4 + 4 bytes per FP32 element), i.e. ceil(1048576 / 12) = 87382
  // elements.
  constexpr std::size_t kThresholdElements = 87382;
  const std::vector<std::int64_t> below_shape{static_cast<std::int64_t>(kThresholdElements - 1)};
  const std::vector<std::int64_t> threshold_shape{static_cast<std::int64_t>(kThresholdElements)};
  const std::vector<std::int64_t> split_shape{static_cast<std::int64_t>(kThresholdElements * 2)};
  const ml_cpu_compute::BinaryBroadcastPlan below(descriptor, BinaryDataType::FLOAT,
                                                  BinaryDataType::FLOAT, BinaryDataType::FLOAT,
                                                  below_shape, below_shape);
  const ml_cpu_compute::BinaryBroadcastPlan threshold(descriptor, BinaryDataType::FLOAT,
                                                      BinaryDataType::FLOAT, BinaryDataType::FLOAT,
                                                      threshold_shape, threshold_shape);
  const ml_cpu_compute::BinaryBroadcastPlan split(descriptor, BinaryDataType::FLOAT,
                                                  BinaryDataType::FLOAT, BinaryDataType::FLOAT,
                                                  split_shape, split_shape);
  std::vector<float> left(kThresholdElements * 2, 1.0f);
  std::vector<float> right(kThresholdElements * 2, 2.0f);
  std::vector<float> output(kThresholdElements * 2);

  below.Execute(left.data(), right.data(), output.data());
  EXPECT_EQ(executor.dispatches, 0);

  threshold.Execute(left.data(), right.data(), output.data());
  EXPECT_EQ(executor.dispatches, 0);

  split.Execute(left.data(), right.data(), output.data());
  EXPECT_EQ(executor.dispatches, 1);
  EXPECT_EQ(executor.blocks, 2);

  const std::vector<std::int64_t> large_shape{static_cast<std::int64_t>(kThresholdElements * 8)};
  const ml_cpu_compute::BinaryBroadcastPlan large(descriptor, BinaryDataType::FLOAT,
                                                  BinaryDataType::FLOAT, BinaryDataType::FLOAT,
                                                  large_shape, large_shape);
  left.resize(kThresholdElements * 8, 1.0f);
  right.resize(kThresholdElements * 8, 2.0f);
  output.resize(kThresholdElements * 8);
  large.Execute(left.data(), right.data(), output.data());
  EXPECT_EQ(executor.dispatches, 2);
  EXPECT_EQ(executor.blocks, 8);
  for (float value : output) {
    EXPECT_FLOAT_EQ(value, 3.0f);
  }
}

TEST(BinaryBroadcastPlan, LargeFloat32DivParallelNestedAndInPlace) {
  const ml_cpu_compute::BinaryOperationDescriptor descriptor("Div", 14, {});
  InlineExecutor executor;
  ml_cpu_compute::ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run};
  ml_cpu_compute::ExecutionExecutorScope scope(&view);
  for (std::size_t count : {65u, 4194304u + 17u}) {
    const std::vector<std::int64_t> shape{static_cast<std::int64_t>(count)};
    const BinaryBroadcastPlan plan(descriptor, BinaryDataType::FLOAT, BinaryDataType::FLOAT,
                                   BinaryDataType::FLOAT, shape, shape);
    std::vector<float> left(count), right(count), expected(count), output(count);
    for (std::size_t i = 0; i < count; ++i) {
      left[i] = static_cast<float>(static_cast<int>(i % 23) - 11) / 7.0f;
      right[i] = static_cast<float>(1 + i % 13) / 3.0f;
      expected[i] = left[i] / right[i];
    }
    const auto check = [&] {
      for (std::size_t i = 0; i < count; ++i) {
        ASSERT_EQ(std::bit_cast<std::uint32_t>(output[i]),
                  std::bit_cast<std::uint32_t>(expected[i]))
            << "index=" << i;
      }
    };
    const auto before = executor.dispatches;
    {
      ml_cpu_compute::ExecutionExecutorScope serial_scope(nullptr);
      plan.Execute(left.data(), right.data(), output.data());
    }
    check();
    EXPECT_EQ(executor.dispatches, before);
    plan.Execute(left.data(), right.data(), output.data());
    check();
    if (count == 65) {
      EXPECT_EQ(executor.dispatches, before);
    } else {
      EXPECT_GT(executor.dispatches, before);
      EXPECT_GT(executor.blocks, 1);
    }
    const auto parallel_dispatches = executor.dispatches;
    {
      ml_cpu_compute::detail::ExecutionRegionScope region;
      plan.Execute(left.data(), right.data(), output.data());
    }
    check();
    EXPECT_EQ(executor.dispatches, parallel_dispatches);
    output = left;
    plan.Execute(output.data(), right.data(), output.data());
    check();
    output = right;
    plan.Execute(left.data(), output.data(), output.data());
    check();
  }
}

TEST(BinaryBroadcastPlan, LargeFloat32AddParallelNestedAndInPlace) {
  const BinaryOperationDescriptor descriptor("Add", 14, {});
  InlineExecutor executor;
  ml_cpu_compute::ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run};
  ml_cpu_compute::ExecutionExecutorScope scope(&view);
  for (std::size_t count : {65u, 4194304u + 17u}) {
    const std::vector<std::int64_t> shape{static_cast<std::int64_t>(count)}, scalar{};
    for (int mode = 0; mode < 3; ++mode) {
      SCOPED_TRACE(::testing::Message() << "count=" << count << " mode=" << mode);
      const BinaryBroadcastPlan plan(descriptor, BinaryDataType::FLOAT, BinaryDataType::FLOAT,
                                     BinaryDataType::FLOAT, mode == 1 ? scalar : shape,
                                     mode == 2 ? scalar : shape);
      std::vector<float> left(mode == 1 ? 1 : count), right(mode == 2 ? 1 : count);
      std::vector<float> expected(count), output(count);
      for (std::size_t i = 0; i < left.size(); ++i) {
        left[i] = static_cast<float>(static_cast<int>(i % 23) - 11) / 7.0f;
      }
      for (std::size_t i = 0; i < right.size(); ++i) {
        right[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 3.0f;
      }
      for (std::size_t i = 0; i < count; ++i) {
        expected[i] = left[mode == 1 ? 0 : i] + right[mode == 2 ? 0 : i];
      }
      const auto check = [&] {
        for (std::size_t i = 0; i < count; ++i) {
          ASSERT_EQ(std::bit_cast<std::uint32_t>(output[i]),
                    std::bit_cast<std::uint32_t>(expected[i]))
              << "index=" << i;
        }
      };
      const auto before = executor.dispatches;
      {
        ml_cpu_compute::ExecutionExecutorScope serial_scope(nullptr);
        plan.Execute(left.data(), right.data(), output.data());
      }
      check();
      EXPECT_EQ(executor.dispatches, before);
      plan.Execute(left.data(), right.data(), output.data());
      check();
      EXPECT_EQ(executor.dispatches, before + (count > 65 ? 1 : 0));
      if (count > 65) {
        EXPECT_GT(executor.blocks, 1);
      }
      const auto parallel_dispatches = executor.dispatches;
      {
        ml_cpu_compute::detail::ExecutionRegionScope region;
        plan.Execute(left.data(), right.data(), output.data());
      }
      check();
      EXPECT_EQ(executor.dispatches, parallel_dispatches);
      for (bool alias_left : {false, true}) {
        if ((alias_left && mode == 1) || (!alias_left && mode == 2)) {
          continue;
        }
        output = alias_left ? left : right;
        ml_cpu_compute::ExecutionExecutorScope serial_scope(nullptr);
        plan.Execute(alias_left ? output.data() : left.data(),
                     alias_left ? right.data() : output.data(), output.data());
        check();
      }
    }
  }
}

TEST(BinaryBroadcastPlan, ScalarPowUsesFineGrainedBoundedScheduling) {
  const ml_cpu_compute::BinaryOperationDescriptor descriptor("Pow", 15, {});
  InlineExecutor executor;
  ml_cpu_compute::ExecutionExecutorView view{&executor, 64, &InlineExecutor::Run};
  ml_cpu_compute::ExecutionExecutorScope scope(&view);
  const std::int32_t exponent = 1;

  const ml_cpu_compute::BinaryBroadcastPlan small(
      descriptor, BinaryDataType::FLOAT, BinaryDataType::INT32, BinaryDataType::FLOAT,
      std::array<std::int64_t, 1>{4096}, std::array<std::int64_t, 0>{});
  std::vector<float> small_input(4096, 2.0f);
  std::vector<float> small_output(small_input.size());
  small.Execute(small_input.data(), &exponent, small_output.data());
  EXPECT_EQ(executor.dispatches, 0);

  constexpr std::size_t kCount = 2 * 1024 * 1024;
  const ml_cpu_compute::BinaryBroadcastPlan large(
      descriptor, BinaryDataType::FLOAT, BinaryDataType::INT32, BinaryDataType::FLOAT,
      std::array<std::int64_t, 1>{kCount}, std::array<std::int64_t, 0>{});
  std::vector<float> large_input(kCount, 3.0f);
  std::vector<float> large_output(kCount);
  large.Execute(large_input.data(), &exponent, large_output.data());
  EXPECT_EQ(executor.dispatches, 1);
  EXPECT_EQ(executor.blocks, 32);
  EXPECT_EQ(large_output, large_input);

  ml_cpu_compute::BinaryExecutionTuning configured = ml_cpu_compute::kDefaultBinaryExecutionTuning;
  configured.bulk_parallel_threshold_bytes = 64 * 1024 * 1024;
  large.Execute(large_input.data(), &exponent, large_output.data(), configured);
  EXPECT_EQ(executor.dispatches, 1);

  const float float_exponent = 1.0F;
  const ml_cpu_compute::BinaryBroadcastPlan float_pow(
      descriptor, BinaryDataType::FLOAT, BinaryDataType::FLOAT, BinaryDataType::FLOAT,
      std::array<std::int64_t, 1>{kCount}, std::array<std::int64_t, 0>{});
  float_pow.Execute(large_input.data(), &float_exponent, large_output.data());
  EXPECT_EQ(executor.dispatches, 2);
  EXPECT_EQ(executor.blocks, 32);
  EXPECT_EQ(large_output, large_input);

  {
    ml_cpu_compute::detail::ExecutionRegionScope region;
    float_pow.Execute(large_input.data(), &float_exponent, large_output.data());
  }
  EXPECT_EQ(executor.dispatches, 2);
  EXPECT_EQ(large_output, large_input);
}

TEST(BinaryBroadcastPlan, MultiDimensionalBulkPathUsesAllAvailableParticipants) {
  const ml_cpu_compute::BinaryOperationDescriptor descriptor("Add", 14, {});
  const std::vector<std::int64_t> left_shape{88, 8192};
  const std::vector<std::int64_t> right_shape{8192};
  const ml_cpu_compute::BinaryBroadcastPlan plan(descriptor, BinaryDataType::FLOAT,
                                                 BinaryDataType::FLOAT, BinaryDataType::FLOAT,
                                                 left_shape, right_shape);
  std::vector<float> left(ElementCount(left_shape), 1.0f);
  std::vector<float> right(ElementCount(right_shape), 2.0f);
  std::vector<float> output(ElementCount(plan.output_shape()));

  InlineExecutor executor;
  ml_cpu_compute::ExecutionExecutorView view{&executor, 8, &InlineExecutor::Run};
  ml_cpu_compute::ExecutionExecutorScope scope(&view);
  plan.Execute(left.data(), right.data(), output.data());

  EXPECT_EQ(executor.dispatches, 1);
  EXPECT_EQ(executor.blocks, 8);
  for (float value : output) {
    EXPECT_FLOAT_EQ(value, 3.0f);
  }
}

void CheckLogicalBulk(const char *op_type) {
  const BinaryOperationDescriptor descriptor(op_type, 7, {});
  const auto &adapter =
      descriptor.ResolveAdapter(BinaryDataType::BOOL, BinaryDataType::BOOL, BinaryDataType::BOOL);
#ifdef ML_CPU_COMPUTE_HAVE_AVX2
  if (ml_cpu_compute::DetectSimdLevel() >= ml_cpu_compute::SimdLevel::kAVX2) {
    if (descriptor.op() == ml_cpu_compute::BinaryOperator::kOr) {
      EXPECT_EQ(adapter.bulk_contiguous, &ml_cpu_compute::BulkOrBool_AVX2);
      EXPECT_EQ(adapter.bulk_left_scalar, &ml_cpu_compute::BulkOrBoolLeft_AVX2);
      EXPECT_EQ(adapter.bulk_right_scalar, &ml_cpu_compute::BulkOrBoolRight_AVX2);
    } else if (descriptor.op() == ml_cpu_compute::BinaryOperator::kXor) {
      EXPECT_EQ(adapter.bulk_contiguous, &ml_cpu_compute::BulkXorBool_AVX2);
      EXPECT_EQ(adapter.bulk_left_scalar, &ml_cpu_compute::BulkXorBoolLeft_AVX2);
      EXPECT_EQ(adapter.bulk_right_scalar, &ml_cpu_compute::BulkXorBoolRight_AVX2);
    } else {
      EXPECT_EQ(adapter.bulk_contiguous, &ml_cpu_compute::BulkAndBool_AVX2);
      EXPECT_EQ(adapter.bulk_left_scalar, &ml_cpu_compute::BulkAndBoolLeft_AVX2);
      EXPECT_EQ(adapter.bulk_right_scalar, &ml_cpu_compute::BulkAndBoolRight_AVX2);
    }
    adapter.bulk_contiguous(nullptr, nullptr, nullptr, 0);
    adapter.bulk_left_scalar(nullptr, nullptr, nullptr, 0);
    adapter.bulk_right_scalar(nullptr, nullptr, nullptr, 0);
  }
#endif
  const std::array<std::uint8_t, 5> values{0, 1, 2, 128, 255};
  for (std::size_t count = 0; count <= 257; ++count) {
    SCOPED_TRACE(count);
    // The offset inputs end exactly at the allocation boundary.
    std::vector<std::uint8_t> left(count + 1), right(count + 1);
    for (std::size_t i = 0; i < count; ++i) {
      left[i + 1] = values[i % values.size()];
      right[i + 1] = values[(i / values.size()) % values.size()];
    }
    for (int mode = 0; mode < 3; ++mode) {
      for (std::uint8_t scalar : values) {
        const auto *a = mode == 1 ? &scalar : left.data() + 1;
        const auto *b = mode == 2 ? &scalar : right.data() + 1;
        const auto bulk = mode == 1   ? adapter.bulk_left_scalar
                          : mode == 2 ? adapter.bulk_right_scalar
                                      : adapter.bulk_contiguous;
        std::vector<std::uint8_t> expected(count + 2, 0xA5), output = expected;
        for (std::size_t i = 0; i < count; ++i) {
          adapter.scalar(a + (mode == 1 ? 0 : i), b + (mode == 2 ? 0 : i), &expected[i + 1]);
        }
        bulk(a, b, output.data() + 1, count);
        EXPECT_EQ(output, expected);
        if (mode != 1) {
          std::vector<std::uint8_t> aliased = left;
          bulk(aliased.data() + 1, b, aliased.data() + 1, count);
          EXPECT_TRUE(std::equal(aliased.begin() + 1, aliased.end(), expected.begin() + 1));
        }
        if (mode != 2) {
          std::vector<std::uint8_t> aliased = right;
          bulk(a, aliased.data() + 1, aliased.data() + 1, count);
          EXPECT_TRUE(std::equal(aliased.begin() + 1, aliased.end(), expected.begin() + 1));
        }
      }
    }
  }
}

TEST(BinaryBroadcastPlan, OrBulkHandlesNoncanonicalBytesTailsAndAliasing) {
  CheckLogicalBulk("Or");
}

TEST(BinaryBroadcastPlan, AndBulkHandlesNoncanonicalBytesTailsAndAliasing) {
  CheckLogicalBulk("And");
}

TEST(BinaryBroadcastPlan, XorBulkHandlesNoncanonicalBytesTailsAndAliasing) {
  CheckLogicalBulk("Xor");
}

void CheckLogicalBroadcasts(const char *op_type) {
  const BinaryOperationDescriptor descriptor(op_type, 7, {});
  const auto &adapter =
      descriptor.ResolveAdapter(BinaryDataType::BOOL, BinaryDataType::BOOL, BinaryDataType::BOOL);
  const std::vector<std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>> shapes = {
      {{2, 3, 5, 65}, {3, 1, 1}},
      {{2, 3, 5, 65}, {1, 3, 1, 65}},
      {{2, 1, 3, 1, 65}, {1, 4, 1, 5, 1}},
      {{2, 1, 3, 1, 4, 1}, {1, 3, 1, 5, 1, 65}},
      {{0, 3, 65}, {3, 1}},
  };
  InlineExecutor executor;
  ml_cpu_compute::ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run};
  ml_cpu_compute::ExecutionExecutorScope scope(&view);
  auto tuning = ml_cpu_compute::kDefaultBinaryExecutionTuning;
  tuning.block_parallel_threshold_bytes = 1;
  tuning.target_block_bytes = 512;
  for (const auto &[left_shape, right_shape] : shapes) {
    for (bool swapped : {false, true}) {
      const auto &a_shape = swapped ? right_shape : left_shape;
      const auto &b_shape = swapped ? left_shape : right_shape;
      const BinaryBroadcastPlan plan(descriptor, BinaryDataType::BOOL, BinaryDataType::BOOL,
                                     BinaryDataType::BOOL, a_shape, b_shape);
      auto left = MakeBuffer(BinaryDataType::BOOL, a_shape, true, op_type);
      auto right = MakeBuffer(BinaryDataType::BOOL, b_shape, false, op_type);
      for (std::size_t i = 0; i < left.size(); ++i)
        left[i] = static_cast<std::byte>(i % 3 == 0 ? 0 : 128);
      for (std::size_t i = 0; i < right.size(); ++i)
        right[i] = static_cast<std::byte>(i % 5 == 0 ? 255 : 0);
      const auto expected =
          ReferenceExecute(adapter, a_shape, b_shape, plan.output_shape(), left, right);
      std::vector<std::byte> output(expected.size());
      plan.Execute(left.data(), right.data(), output.data());
      EXPECT_EQ(output, expected);
      const auto before = executor.dispatches;
      plan.Execute(left.data(), right.data(), output.data(), tuning);
      EXPECT_EQ(output, expected);
      if (!output.empty())
        EXPECT_GT(executor.dispatches, before);
      const auto parallel_dispatches = executor.dispatches;
      {
        ml_cpu_compute::detail::ExecutionRegionScope region;
        plan.Execute(left.data(), right.data(), output.data(), tuning);
      }
      EXPECT_EQ(output, expected);
      EXPECT_EQ(executor.dispatches, parallel_dispatches);
    }
  }
}

TEST(BinaryBroadcastPlan, OrBroadcastsMatchScalarWithExecutorAndNestedFallback) {
  CheckLogicalBroadcasts("Or");
}

TEST(BinaryBroadcastPlan, AndBroadcastsMatchScalarWithExecutorAndNestedFallback) {
  CheckLogicalBroadcasts("And");
}

TEST(BinaryBroadcastPlan, XorBroadcastsMatchScalarWithExecutorAndNestedFallback) {
  CheckLogicalBroadcasts("Xor");
}

TEST(BinaryBroadcastPlanCache, HitsEvictsAndKeepsInUsePlansAlive) {
  const ml_cpu_compute::BinaryOperationDescriptor descriptor("Add", 14, {});
  ml_cpu_compute::BinaryBroadcastPlanCache cache;
  const auto base = cache.GetOrCreate(descriptor, BinaryDataType::FLOAT, BinaryDataType::FLOAT,
                                      BinaryDataType::FLOAT, std::array<std::int64_t, 2>{2, 3},
                                      std::array<std::int64_t, 2>{2, 3});
  const auto hit = cache.GetOrCreate(descriptor, BinaryDataType::FLOAT, BinaryDataType::FLOAT,
                                     BinaryDataType::FLOAT, std::array<std::int64_t, 2>{2, 3},
                                     std::array<std::int64_t, 2>{2, 3});
  EXPECT_EQ(base.get(), hit.get());

  EXPECT_THROW((cache.GetOrCreate(descriptor, BinaryDataType::FLOAT, BinaryDataType::FLOAT,
                                  BinaryDataType::FLOAT, std::array<std::int64_t, 1>{2},
                                  std::array<std::int64_t, 1>{3})),
               std::invalid_argument);
  const auto hit_after_throw = cache.GetOrCreate(
      descriptor, BinaryDataType::FLOAT, BinaryDataType::FLOAT, BinaryDataType::FLOAT,
      std::array<std::int64_t, 2>{2, 3}, std::array<std::int64_t, 2>{2, 3});
  EXPECT_EQ(base.get(), hit_after_throw.get());

  std::weak_ptr<const ml_cpu_compute::BinaryBroadcastPlan> weak = base;
  for (int i = 0; i < 8; ++i) {
    const std::array<std::int64_t, 2> shape{2, 4 + i};
    (void)cache.GetOrCreate(descriptor, BinaryDataType::FLOAT, BinaryDataType::FLOAT,
                            BinaryDataType::FLOAT, shape, shape);
  }
  EXPECT_EQ(cache.size(), 8u);
  EXPECT_FALSE(weak.expired());
  EXPECT_NE(cache
                .GetOrCreate(descriptor, BinaryDataType::FLOAT, BinaryDataType::FLOAT,
                             BinaryDataType::FLOAT, std::array<std::int64_t, 2>{2, 3},
                             std::array<std::int64_t, 2>{2, 3})
                .get(),
            base.get());
}

} // namespace
