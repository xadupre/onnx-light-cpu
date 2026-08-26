// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/binary/binary_broadcast_plan.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace {

using onnx_light_cpu::BinaryBroadcastPlan;
using onnx_light_cpu::BinaryBroadcastPlanCache;
using BinaryDataType = onnx_light_cpu::DataType;
using onnx_light_cpu::BinaryKernelDescriptor;
using LoopFamily = BinaryBroadcastPlan::LoopFamily;

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
    const std::uint16_t bits = onnx_light_cpu::detail::FloatToFloat16Bits(v);
    write(bits);
    break;
  }
  case BinaryDataType::BFLOAT16: {
    const float v = static_cast<float>(op_type == "Pow" && !is_left ? (index % 3) : signed_value);
    const std::uint16_t bits = onnx_light_cpu::detail::FloatToBFloat16Bits(v);
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

std::vector<std::byte> ReferenceExecute(const BinaryKernelDescriptor::Adapter &adapter,
                                        std::span<const std::int64_t> left_shape,
                                        std::span<const std::int64_t> right_shape,
                                        std::span<const std::int64_t> output_shape,
                                        const std::vector<std::byte> &left,
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

BinaryKernelDescriptor::Attributes DefaultAttributes(std::string_view op_type,
                                                     BinaryDataType left) {
  BinaryKernelDescriptor::Attributes attrs;
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
  const BinaryKernelDescriptor descriptor("Add", 14, {});
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
    const BinaryBroadcastPlan plan(descriptor, BinaryDataType::FLOAT, BinaryDataType::FLOAT,
                                   BinaryDataType::FLOAT, shapes.first, shapes.second);
    EXPECT_EQ(plan.loop_family(), family);
  }

  const BinaryBroadcastPlan exact(descriptor, BinaryDataType::FLOAT, BinaryDataType::FLOAT,
                                  BinaryDataType::FLOAT, std::array<std::int64_t, 2>{2, 3},
                                  std::array<std::int64_t, 2>{2, 3});
  EXPECT_TRUE(exact.left_output_alias_safe());
  EXPECT_TRUE(exact.right_output_alias_safe());

  const BinaryBroadcastPlan broadcast_right(
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

  const BinaryKernelDescriptor add_descriptor("Add", 14, {});
  for (const auto &[left_shape, right_shape] : families) {
    const BinaryBroadcastPlan plan(add_descriptor, BinaryDataType::FLOAT, BinaryDataType::FLOAT,
                                   BinaryDataType::FLOAT, left_shape, right_shape);
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
  const BinaryKernelDescriptor descriptor(std::string(op_type), 14, {});
  const std::array<std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>, 3> shapes{{
      {std::vector<std::int64_t>{static_cast<std::int64_t>(count)},
       std::vector<std::int64_t>{static_cast<std::int64_t>(count)}},
      {std::vector<std::int64_t>{}, std::vector<std::int64_t>{static_cast<std::int64_t>(count)}},
      {std::vector<std::int64_t>{static_cast<std::int64_t>(count)}, std::vector<std::int64_t>{}},
  }};
  for (const auto &[left_shape, right_shape] : shapes) {
    const BinaryBroadcastPlan plan(descriptor, type, type, type, left_shape, right_shape);
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
  const BinaryKernelDescriptor descriptor("Add", 14, {});
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
    const BinaryBroadcastPlan plan(descriptor, BinaryDataType::FLOAT, BinaryDataType::FLOAT,
                                   BinaryDataType::FLOAT, shapes.first, shapes.second);
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
      const BinaryKernelDescriptor descriptor(std::string(op), 14, {});
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
      const BinaryKernelDescriptor descriptor(std::string(op), op == "Pow" ? 15 : 16,
                                              DefaultAttributes(op, type));
      const auto &adapter = descriptor.ResolveAdapter(type, type, type);
      ASSERT_NE(adapter.bulk_contiguous, nullptr) << op;
      for (std::size_t count : sizes) {
        const std::array<std::int64_t, 1> shape{static_cast<std::int64_t>(count)};
        const BinaryBroadcastPlan plan(descriptor, type, type, type, shape, shape);
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
                  onnx_light_cpu::ExecutionBlockFn task) {
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
  const BinaryKernelDescriptor descriptor("Add", 14, {});
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
    const BinaryBroadcastPlan plan(descriptor, BinaryDataType::FLOAT, BinaryDataType::FLOAT,
                                   BinaryDataType::FLOAT, left_shape, right_shape);
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

    InlineExecutor executor;
    onnx_light_cpu::ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run};
    onnx_light_cpu::ExecutionExecutorScope scope(&view);
    std::vector<float> parallel(expected.size(), -1.0f);
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
  const BinaryKernelDescriptor descriptor("Add", 14, {});
  InlineExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 8, &InlineExecutor::Run};
  onnx_light_cpu::ExecutionExecutorScope scope(&view);

  // The 1 MiB calibrated contiguous bulk threshold (see
  // binary_execution_schedule.h) covers left + right + output traffic
  // (4 + 4 + 4 bytes per FP32 element), i.e. ceil(1048576 / 12) = 87382
  // elements.
  constexpr std::size_t kThresholdElements = 87382;
  const std::vector<std::int64_t> below_shape{static_cast<std::int64_t>(kThresholdElements - 1)};
  const std::vector<std::int64_t> threshold_shape{static_cast<std::int64_t>(kThresholdElements)};
  const std::vector<std::int64_t> split_shape{static_cast<std::int64_t>(kThresholdElements * 2)};
  const BinaryBroadcastPlan below(descriptor, BinaryDataType::FLOAT, BinaryDataType::FLOAT,
                                  BinaryDataType::FLOAT, below_shape, below_shape);
  const BinaryBroadcastPlan threshold(descriptor, BinaryDataType::FLOAT, BinaryDataType::FLOAT,
                                      BinaryDataType::FLOAT, threshold_shape, threshold_shape);
  const BinaryBroadcastPlan split(descriptor, BinaryDataType::FLOAT, BinaryDataType::FLOAT,
                                  BinaryDataType::FLOAT, split_shape, split_shape);
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
  const BinaryBroadcastPlan large(descriptor, BinaryDataType::FLOAT, BinaryDataType::FLOAT,
                                  BinaryDataType::FLOAT, large_shape, large_shape);
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

TEST(BinaryBroadcastPlan, MultiDimensionalBulkPathUsesAllAvailableParticipants) {
  const BinaryKernelDescriptor descriptor("Add", 14, {});
  const std::vector<std::int64_t> left_shape{88, 8192};
  const std::vector<std::int64_t> right_shape{8192};
  const BinaryBroadcastPlan plan(descriptor, BinaryDataType::FLOAT, BinaryDataType::FLOAT,
                                 BinaryDataType::FLOAT, left_shape, right_shape);
  std::vector<float> left(ElementCount(left_shape), 1.0f);
  std::vector<float> right(ElementCount(right_shape), 2.0f);
  std::vector<float> output(ElementCount(plan.output_shape()));

  InlineExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 8, &InlineExecutor::Run};
  onnx_light_cpu::ExecutionExecutorScope scope(&view);
  plan.Execute(left.data(), right.data(), output.data());

  EXPECT_EQ(executor.dispatches, 1);
  EXPECT_EQ(executor.blocks, 8);
  for (float value : output) {
    EXPECT_FLOAT_EQ(value, 3.0f);
  }
}

TEST(BinaryBroadcastPlanCache, HitsEvictsAndKeepsInUsePlansAlive) {
  const BinaryKernelDescriptor descriptor("Add", 14, {});
  BinaryBroadcastPlanCache cache;
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

  std::weak_ptr<const BinaryBroadcastPlan> weak = base;
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
