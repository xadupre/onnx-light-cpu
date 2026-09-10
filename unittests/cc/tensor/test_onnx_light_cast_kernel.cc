// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/tensor/cast_kernel.h"

#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/tensor/cast_kernel.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
using rt_ns::DataType;
using rt_ns::Shape;
using rt_ns::Tensor;
using BuiltinCast = ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::Cast;
using CpuType = onnx_light_cpu::DataType;

rt_ns::KernelContext MakeCtx() { return rt_ns::KernelContext(rt_ns::OpsetId("", 27)); }

constexpr std::array<DataType, 11> kBuiltinNumeric = {
    DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16,
    DataType::INT8,  DataType::UINT8,  DataType::INT16,   DataType::UINT16,
    DataType::INT32, DataType::INT64,  DataType::BOOL};

constexpr std::array<DataType, 12> kExtended = {
    DataType::FLOAT8E4M3FN,   DataType::FLOAT8E4M3FNUZ, DataType::FLOAT8E5M2,
    DataType::FLOAT8E5M2FNUZ, DataType::FLOAT8E8M0,     DataType::INT4,
    DataType::UINT4,          DataType::INT2,           DataType::UINT2,
    DataType::FLOAT4E2M1,     DataType::FLOAT6E2M3,     DataType::FLOAT6E3M2};

void Equal(const Tensor &actual, const Tensor &expected) {
  ASSERT_EQ(actual.data_type, expected.data_type);
  ASSERT_EQ(actual.shape, expected.shape);
  ASSERT_EQ(actual.size_bytes(), expected.size_bytes());
  if (actual.data_type == DataType::STRING) {
    EXPECT_EQ(actual.AsStrings(), expected.AsStrings());
  } else if (actual.size_bytes() != 0) {
    EXPECT_EQ(std::memcmp(actual.bytes(), expected.bytes(), actual.size_bytes()), 0);
  }
}

Tensor Output(int32_t type, const Shape &shape, std::size_t bytes) {
  Tensor result = rt_ns::MakeOutputTensor(type, shape, bytes, nullptr);
  if (type == DataType::STRING) {
    result.string_data.resize(static_cast<std::size_t>(shape.product()));
  }
  return result;
}

void Compare(const Tensor &data, int32_t to, bool saturate = true) {
  SCOPED_TRACE(to);
  onnx_light_cpu::CastKernel kernel(MakeCtx());
  const Tensor expected = BuiltinCast(MakeCtx())(data, to, saturate);
  const Tensor actual = kernel(data, to, saturate);
  Equal(actual, expected);
  if (data.size_bytes() != 0 && actual.size_bytes() != 0) {
    EXPECT_NE(actual.bytes(), data.bytes());
  }
  Tensor output = Output(to, data.shape, expected.size_bytes());
  kernel(data, to, saturate, output);
  Equal(output, expected);
}

TEST(OnnxLightCastKernel, AllBuiltinNumericPairsScalarsEmptyAndTails) {
  BuiltinCast builtin(MakeCtx());
  for (int64_t count : {0, 1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65, 257}) {
    std::vector<double> values(static_cast<std::size_t>(count));
    for (std::size_t i = 0; i < values.size(); ++i) {
      values[i] = static_cast<double>(i % 128);
    }
    for (const Shape &shape : {Shape{count}, Shape{1, count, 1}}) {
      const Tensor seed = Tensor::From<double>("data", shape, values);
      for (DataType from : kBuiltinNumeric) {
        SCOPED_TRACE(static_cast<int>(from));
        const Tensor input = builtin(seed, from);
        for (DataType to : kBuiltinNumeric) {
          Compare(input, to);
        }
      }
    }
  }
  for (DataType from : kBuiltinNumeric) {
    const Tensor input = builtin(Tensor::From<double>("data", {}, {17}), from);
    for (DataType to : kBuiltinNumeric) {
      Compare(input, to);
    }
  }
}

TEST(OnnxLightCastKernel, SignedFractionalAndModularNarrowingMatchesBuiltin) {
  const Tensor input = Tensor::From<double>("data", {13},
                                            {-65537.5, -32769, -257.5, -129, -127.75, -1.5, -0.0,
                                             0.0, 1.5, 127.75, 255.75, 32768.5, 65537.5});
  for (DataType to : kBuiltinNumeric) {
    Compare(input, to);
  }
  const Tensor small = Tensor::From<float>("data", {7}, {-127.5f, -1.5f, -0.0f, 0, 1.5f, 7, 127});
  BuiltinCast builtin(MakeCtx());
  for (DataType from : {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16}) {
    for (DataType to : kBuiltinNumeric) {
      Compare(builtin(small, from), to);
    }
  }
}

TEST(OnnxLightCastKernel, IntegerExtremaNeverPassThroughDouble) {
  onnx_light_cpu::CastKernel kernel(MakeCtx());
  const Tensor input = Tensor::From<int64_t>("data", {8},
                                             {INT64_MIN, INT64_MIN + 1, -9007199254740993LL, -1, 0,
                                              9007199254740993LL, INT64_MAX - 1, INT64_MAX});
  Equal(kernel(input, DataType::INT32),
        Tensor::From<int32_t>("expected", {8}, {0, 1, -1, -1, 0, 1, -2, -1}));
  Equal(kernel(input, DataType::UINT64),
        Tensor::From<uint64_t>("expected", {8},
                               {0x8000000000000000ULL, 0x8000000000000001ULL, 0xffdfffffffffffffULL,
                                UINT64_MAX, 0, 9007199254740993ULL, 0x7ffffffffffffffeULL,
                                0x7fffffffffffffffULL}));
  Equal(kernel(input, DataType::INT8),
        Tensor::From<int8_t>("expected", {8}, {0, 1, -1, -1, 0, 1, -2, -1}));
  Equal(kernel(input, DataType::UINT32),
        Tensor::From<uint32_t>("expected", {8},
                               {0, 1, UINT32_MAX, UINT32_MAX, 0, 1, UINT32_MAX - 1, UINT32_MAX}));
  Equal(kernel(input, DataType::INT64), input);
  const Tensor unsigned_input =
      Tensor::From<uint64_t>("data", {6},
                             {0, 9007199254740993ULL, 0x7fffffffffffffffULL, 0x8000000000000000ULL,
                              UINT64_MAX - 1, UINT64_MAX});
  Equal(kernel(unsigned_input, DataType::INT64),
        Tensor::From<int64_t>("expected", {6},
                              {0, 9007199254740993LL, INT64_MAX, INT64_MIN, -2, -1}));
  Equal(kernel(unsigned_input, DataType::UINT32),
        Tensor::From<uint32_t>("expected", {6}, {0, 1, UINT32_MAX, 0, UINT32_MAX - 1, UINT32_MAX}));
  Equal(kernel(unsigned_input, DataType::UINT64), unsigned_input);
}

TEST(OnnxLightCastKernel, Unsigned32And64NumericPairs) {
  onnx_light_cpu::CastKernel kernel(MakeCtx());
  const Tensor seed = Tensor::From<int32_t>("data", {7}, {0, 1, 3, 7, 16, 63, 127});
  for (DataType wide : {DataType::UINT32, DataType::UINT64}) {
    const Tensor wide_input = kernel(seed, wide);
    for (DataType other : kBuiltinNumeric) {
      const Tensor expected = BuiltinCast(MakeCtx())(seed, other);
      Equal(kernel(wide_input, other), expected);
      const Tensor back = kernel(expected, wide);
      Equal(back, kernel(BuiltinCast(MakeCtx())(expected, DataType::INT64), wide));
    }
    Equal(kernel(wide_input, DataType::UINT32), kernel(seed, DataType::UINT32));
    Equal(kernel(wide_input, DataType::UINT64), kernel(seed, DataType::UINT64));
  }
}

TEST(OnnxLightCastKernel, SafeFloatingIntegerEdgesAndNonFiniteValues) {
  onnx_light_cpu::CastKernel kernel(MakeCtx());
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const Tensor input = Tensor::From<double>("data", {10},
                                            {nan, -inf, inf, -0x1p63, std::nextafter(-0x1p63, -inf),
                                             std::nextafter(0x1p63, 0.0), 0x1p63,
                                             std::nextafter(0x1p64, 0.0), 0x1p64, -1.75});
  Equal(kernel(input, DataType::INT64),
        Tensor::From<int64_t>("expected", {10},
                              {0, INT64_MIN, INT64_MAX, INT64_MIN, INT64_MIN, INT64_MAX - 1023,
                               INT64_MAX, INT64_MAX, INT64_MAX, -1}));
  Equal(kernel(input, DataType::UINT64),
        Tensor::From<uint64_t>("expected", {10},
                               {0, 0, UINT64_MAX, 0, 0, 0x7ffffffffffffc00ULL,
                                0x8000000000000000ULL, 0xfffffffffffff800ULL, UINT64_MAX, 0}));
  Equal(kernel(input, DataType::INT8),
        Tensor::From<int8_t>("expected", {10}, {0, -128, 127, 0, -128, 0, 127, 127, 127, -1}));
  Equal(kernel(input, DataType::UINT8),
        Tensor::From<uint8_t>("expected", {10}, {0, 0, 255, 0, 0, 0, 255, 255, 255, 255}));
  Tensor expected_bool = Tensor::From<uint8_t>("expected", {10}, std::vector<uint8_t>(10, 1));
  expected_bool.data_type = DataType::BOOL;
  Equal(kernel(input, DataType::BOOL), expected_bool);
  Compare(Tensor::From<double>("data", {4}, {-0.0, 0.0, nan, inf}), DataType::BOOL);
  const Tensor floats = Tensor::From<float>(
      "data", {5},
      {-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
       std::numeric_limits<float>::quiet_NaN(), -0x1p63f, 0x1p63f});
  Equal(kernel(floats, DataType::INT64),
        Tensor::From<int64_t>("expected", {5}, {INT64_MIN, INT64_MAX, 0, INT64_MIN, INT64_MAX}));
}

TEST(OnnxLightCastKernel, HalfRoundingSignedZeroInfinitiesAndIdentityPayloads) {
  const Tensor input = Tensor::From<double>(
      "data", {12},
      {-0.0, 0.0, 1.00048828125, 1.00146484375, 1.00390625, 1.01171875, 0x1p-24, 0x1p-25, 65504,
       65520, -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()});
  for (DataType to : {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16}) {
    Compare(input, to);
  }
  onnx_light_cpu::CastKernel kernel(MakeCtx());
  const Tensor rounded = kernel(input, DataType::FLOAT16);
  std::array<uint16_t, 12> rounded_bits{};
  std::memcpy(rounded_bits.data(), rounded.bytes(), rounded.size_bytes());
  EXPECT_EQ(rounded_bits[0], 0x8000);
  EXPECT_EQ(rounded_bits[2], 0x3c00);
  EXPECT_EQ(rounded_bits[3], 0x3c02);
  EXPECT_EQ(rounded_bits[6], 0x0001);
  EXPECT_EQ(rounded_bits[7], 0x0000);
  for (DataType type : {DataType::FLOAT16, DataType::BFLOAT16}) {
    Tensor bits =
        Tensor::From<uint16_t>("data", {8}, {0, 0x8000, 1, 0x7c00, 0x7e01, 0xfeff, 0x7fc1, 0xffc3});
    bits.data_type = type;
    Compare(bits, type);
    const Tensor widened = kernel(bits, DataType::FLOAT);
    for (std::size_t i = 0; i < 8; ++i) {
      const Tensor one = Tensor::Borrow("data", type, {}, bits.bytes() + i * 2, 2);
      const Tensor expected = BuiltinCast(MakeCtx())(one, DataType::FLOAT);
      const float actual = widened.As<float>()[i];
      if (std::isnan(expected.As<float>()[0])) {
        EXPECT_TRUE(std::isnan(actual));
      } else {
        EXPECT_EQ(actual, expected.As<float>()[0]);
      }
    }
  }
  Tensor boolean = Tensor::From<uint8_t>("data", {4}, {0, 1, 127, 255});
  boolean.data_type = DataType::BOOL;
  Compare(boolean, DataType::BOOL);
  Compare(boolean, DataType::INT64);
}

TEST(OnnxLightCastKernel, ExtendedCompatibilitySaturationAndPackedTails) {
  BuiltinCast builtin(MakeCtx());
  for (int64_t count : {0, 1, 3, 4, 5, 7, 9, 17}) {
    std::vector<float> values(static_cast<std::size_t>(count));
    for (std::size_t i = 0; i < values.size(); ++i) {
      values[i] = static_cast<float>(i) - 3.5f;
    }
    const Tensor seed = Tensor::From<float>("data", {count}, values);
    for (DataType extended : kExtended) {
      SCOPED_TRACE(static_cast<int>(extended));
      for (bool saturate : {false, true}) {
        for (DataType partner : {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16}) {
          const Tensor input = builtin(seed, partner);
          Compare(input, extended, saturate);
          const Tensor packed = builtin(input, extended, saturate);
          Compare(packed, partner, saturate);
        }
      }
    }
  }
  const Tensor overflow = Tensor::From<float>("data", {5}, {-1e20f, -500, 0, 500, 1e20f});
  for (DataType extended : {DataType::FLOAT8E4M3FN, DataType::FLOAT8E4M3FNUZ, DataType::FLOAT8E5M2,
                            DataType::FLOAT8E5M2FNUZ}) {
    Compare(overflow, extended, false);
    Compare(overflow, extended, true);
    const Tensor saturated = onnx_light_cpu::CastKernel(MakeCtx())(overflow, extended, true);
    const Tensor unsaturated = onnx_light_cpu::CastKernel(MakeCtx())(overflow, extended, false);
    EXPECT_NE(std::memcmp(saturated.bytes(), unsaturated.bytes(), 5), 0);
  }
  for (DataType packed : {DataType::INT4, DataType::UINT4, DataType::INT2, DataType::UINT2}) {
    const DataType partner =
        packed == DataType::INT4 || packed == DataType::INT2 ? DataType::INT8 : DataType::UINT8;
    const Tensor input = builtin(Tensor::From<float>("data", {5}, {-8, -1, 0, 7, 16}), partner);
    Compare(input, packed);
    Compare(builtin(input, packed), partner);
  }
}

TEST(OnnxLightCastKernel, StringsPreserveParserFormattingAndBorrowedInputs) {
  const Tensor strings =
      Tensor::FromStrings("data", {6}, {" 1.5tail", "-2", "0x1p2", "-0", "7", "63"});
  for (DataType to : kBuiltinNumeric) {
    Compare(strings, to);
  }
  Compare(strings, DataType::STRING);
  const Tensor borrowed = Tensor::BorrowStrings("data", strings.shape, strings.AsStrings());
  onnx_light_cpu::CastKernel kernel(MakeCtx());
  Equal(kernel(borrowed, DataType::DOUBLE), kernel(strings, DataType::DOUBLE));
  Equal(kernel(borrowed, DataType::STRING), strings);
  const Tensor seed = Tensor::From<double>("data", {5}, {-1.5, -0.0, 0, 7, 63});
  for (DataType from : kBuiltinNumeric) {
    Compare(BuiltinCast(MakeCtx())(seed, from), DataType::STRING);
  }
  Compare(Tensor::FromStrings("data", {0}, {}), DataType::FLOAT);
  Compare(Tensor::FromStrings("data", {0}, {}), DataType::STRING);
  const Tensor special = Tensor::FromStrings("data", {3}, {"NaN", "INF", "-INF"});
  Equal(kernel(special, DataType::INT64),
        Tensor::From<int64_t>("expected", {3}, {0, INT64_MAX, INT64_MIN}));
}

TEST(OnnxLightCastKernel, UnalignedInputOutputAndSentinels) {
  onnx_light_cpu::CastKernel kernel(MakeCtx());
  BuiltinCast builtin(MakeCtx());
  const Tensor seed = Tensor::From<double>("data", {9}, {-7, -3.5, -1, -0.0, 0, 1, 3.5, 7, 63});
  for (DataType from : kBuiltinNumeric) {
    const Tensor aligned = builtin(seed, from);
    std::vector<uint8_t> source(aligned.size_bytes() + 2, 0xa5);
    std::memcpy(source.data() + 1, aligned.bytes(), aligned.size_bytes());
    const Tensor borrowed =
        Tensor::Borrow("data", from, seed.shape, source.data() + 1, aligned.size_bytes());
    for (DataType to : kBuiltinNumeric) {
      const Tensor expected = kernel(aligned, to);
      std::vector<uint8_t> destination(expected.size_bytes() + 2, 0x5a);
      Tensor output =
          Tensor::Borrow("out", to, seed.shape, destination.data() + 1, expected.size_bytes());
      kernel(borrowed, to, output);
      Equal(output, expected);
      EXPECT_EQ(destination.front(), 0x5a);
      EXPECT_EQ(destination.back(), 0x5a);
    }
    EXPECT_EQ(source.front(), 0xa5);
    EXPECT_EQ(source.back(), 0xa5);
  }
  const Tensor half = builtin(seed, DataType::FLOAT16);
  std::vector<uint8_t> source(half.size_bytes() + 1);
  std::memcpy(source.data() + 1, half.bytes(), half.size_bytes());
  const Tensor borrowed =
      Tensor::Borrow("data", DataType::FLOAT16, half.shape, source.data() + 1, half.size_bytes());
  Equal(kernel(borrowed, DataType::FLOAT8E5M2), builtin(half, DataType::FLOAT8E5M2));
}

TEST(OnnxLightCastKernel, ValidationErrorsDoNotWritePreallocatedOutput) {
  onnx_light_cpu::CastKernel kernel(MakeCtx());
  const Tensor input = Tensor::From<int32_t>("data", {2}, {1, 2});
  const Tensor original = Tensor::From<int32_t>("out", {2}, {123, 456});
  Tensor output = original;
  for (DataType type : {DataType::UNDEFINED, DataType::COMPLEX64, DataType::COMPLEX128}) {
    EXPECT_THROW(kernel(input, type, output), std::invalid_argument);
    Equal(output, original);
    Tensor malformed = input;
    malformed.data_type = type;
    EXPECT_THROW(kernel(malformed, DataType::INT32, output), std::invalid_argument);
  }
  for (const Shape &shape : {Shape{-1}, Shape{3}, Shape{INT64_MAX, 2}, Shape{INT64_MAX / 2}}) {
    Tensor malformed = input;
    malformed.shape = shape;
    EXPECT_THROW(kernel(malformed, DataType::INT32, output), std::invalid_argument);
    Equal(output, original);
  }
  output.data_type = DataType::FLOAT;
  EXPECT_THROW(kernel(input, DataType::INT32, output), std::invalid_argument);
  output = original;
  output.shape = {1, 2};
  EXPECT_THROW(kernel(input, DataType::INT32, output), std::invalid_argument);
  output = Tensor::From<int32_t>("out", {1}, {123});
  output.shape = {2};
  EXPECT_THROW(kernel(input, DataType::INT32, output), std::invalid_argument);
  output = original;
  Tensor alias = Tensor::Borrow("out", DataType::INT32, {2}, input.bytes(), input.size_bytes());
  EXPECT_THROW(kernel(input, DataType::INT32, alias), std::invalid_argument);
  const Tensor null_input = Tensor::Borrow("data", DataType::INT32, {2}, nullptr, 8);
  EXPECT_THROW(kernel(null_input, DataType::INT32, output), std::invalid_argument);
  Tensor null_output = Tensor::Borrow("out", DataType::INT32, {2}, nullptr, 8);
  EXPECT_THROW(kernel(input, DataType::INT32, null_output), std::invalid_argument);
  std::array<uint8_t, 20> shared{};
  Tensor overlapping_input = Tensor::Borrow("data", DataType::INT32, {2}, shared.data(), 8);
  Tensor overlapping_output = Tensor::Borrow("out", DataType::INT32, {2}, shared.data() + 4, 8);
  EXPECT_THROW(kernel(overlapping_input, DataType::INT32, overlapping_output),
               std::invalid_argument);
  EXPECT_THROW(kernel(overlapping_output, DataType::INT32, overlapping_input),
               std::invalid_argument);
  const Tensor invalid_strings = Tensor::FromStrings("data", {2}, {"1", "not-a-number"});
  EXPECT_THROW(kernel(invalid_strings, DataType::INT32, output), std::invalid_argument);
  Equal(output, original);
  Tensor bad_strings = Tensor::FromStrings("data", {1}, {"1"});
  bad_strings.shape = {2};
  EXPECT_THROW(kernel(bad_strings, DataType::INT32, output), std::invalid_argument);
  Tensor strings = Tensor::FromStrings("data", {2}, {"1", "2"});
  EXPECT_THROW(kernel(strings, DataType::STRING, strings), std::invalid_argument);
  const Tensor &const_strings = strings;
  Tensor borrowed_strings = Tensor::BorrowStrings("out", strings.shape, const_strings.AsStrings());
  EXPECT_THROW(kernel(strings, DataType::STRING, borrowed_strings), std::invalid_argument);
  const Tensor packed =
      BuiltinCast(MakeCtx())(Tensor::From<float>("data", {2}, {1, 2}), DataType::INT4);
  EXPECT_THROW(kernel(packed, DataType::INT32, output), std::invalid_argument);
  Equal(output, original);
  Tensor malformed_packed = packed;
  malformed_packed.shape = {3};
  EXPECT_THROW(kernel(malformed_packed, DataType::FLOAT), std::invalid_argument);
}

ONNX_LIGHT_NAMESPACE::NodeProto Node(int64_t to, int64_t saturate = 1) {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("Cast");
  node.add_input("data");
  node.add_output("output");
  for (const auto &[name, value] :
       {std::pair<const char *, int64_t>{"to", to}, {"saturate", saturate}}) {
    auto *attribute = node.add_attribute();
    attribute->set_name(name);
    attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::INT);
    attribute->set_i(value);
  }
  return node;
}

TEST(OnnxLightCastKernel, NodeAttributesRuntimeUsageAndArity) {
  const Tensor data = Tensor::From<float>("data", {3}, {-1000, 0, 1000});
  for (int64_t saturate : {0, 1}) {
    const auto node = Node(DataType::FLOAT8E5M2, saturate);
    onnx_light_cpu::CastKernel kernel(node, MakeCtx());
    rt_ns::RuntimeContext runtime(MakeCtx());
    runtime.Set("data", data);
    onnx_light_cpu::ClearUsedKernelNames();
    onnx_light_cpu::SetKernelUsageRecording(true);
    EXPECT_NO_THROW(kernel.Run(runtime));
    onnx_light_cpu::SetKernelUsageRecording(false);
    Equal(runtime.Get("output"), BuiltinCast(MakeCtx())(data, DataType::FLOAT8E5M2, saturate != 0));
    EXPECT_EQ(onnx_light_cpu::UsedKernelNames(),
              std::vector<std::string>{onnx_light_cpu::CastKernel::kName});
  }
  for (int64_t to : {int64_t{-1}, int64_t{0}, int64_t{14}, INT64_MAX}) {
    EXPECT_THROW(onnx_light_cpu::CastKernel(Node(to), MakeCtx()), std::invalid_argument);
  }
  EXPECT_THROW(onnx_light_cpu::CastKernel(Node(DataType::FLOAT, 2), MakeCtx()),
               std::invalid_argument);
  auto node = Node(DataType::FLOAT);
  node.mutable_attribute(0)->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::FLOAT);
  EXPECT_THROW(onnx_light_cpu::CastKernel(node, MakeCtx()), std::invalid_argument);
  node = Node(DataType::FLOAT);
  node.mutable_attribute(1)->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::FLOAT);
  EXPECT_THROW(onnx_light_cpu::CastKernel(node, MakeCtx()), std::invalid_argument);
  ONNX_LIGHT_NAMESPACE::NodeProto missing;
  missing.set_op_type("Cast");
  EXPECT_THROW(onnx_light_cpu::CastKernel(missing, MakeCtx()), std::invalid_argument);
  node = Node(DataType::FLOAT);
  const auto duplicate = node.attribute(0);
  *node.add_attribute() = duplicate;
  EXPECT_THROW(onnx_light_cpu::CastKernel(node, MakeCtx()), std::invalid_argument);
  node = Node(DataType::FLOAT);
  node.add_output("extra");
  onnx_light_cpu::CastKernel bad_arity(node, MakeCtx());
  rt_ns::RuntimeContext runtime(MakeCtx());
  runtime.Set("data", data);
  EXPECT_THROW(bad_arity.Run(runtime), std::invalid_argument);
}

struct InlineExecutor {
  int64_t dispatches = 0;
  int64_t blocks = 0;

  static void Run(void *context, int64_t count, void *task_context,
                  onnx_light_cpu::ExecutionBlockFn task) {
    auto &self = *static_cast<InlineExecutor *>(context);
    ++self.dispatches;
    self.blocks = count;
    for (int64_t i = 0; i < count; ++i) {
      task(task_context, i);
    }
  }
};

TEST(OnnxLightCastKernel, ExecutorSmallLargeIdentityAndNestedBypass) {
  InlineExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 8, &InlineExecutor::Run};
  onnx_light_cpu::ExecutionExecutorScope scope(&view);
  Compare(Tensor::From<int64_t>("data", {3}, {1, 5, 7}), DataType::INT32);
  EXPECT_EQ(executor.dispatches, 0);
  const Tensor large = Tensor::From<int64_t>("data", {262145}, std::vector<int64_t>(262145, 17));
  Compare(large, DataType::INT32);
  EXPECT_EQ(executor.dispatches, 2);
  EXPECT_GT(executor.blocks, 1);
  EXPECT_LE(executor.blocks, view.effective_threads);
  Compare(large, DataType::INT64);
  EXPECT_EQ(executor.dispatches, 4);
  {
    onnx_light_cpu::detail::ExecutionRegionScope region;
    Compare(large, DataType::INT32);
    EXPECT_EQ(onnx_light_cpu::detail::ExecutionRegionDepth(), 1);
  }
  EXPECT_EQ(executor.dispatches, 4);
  {
    onnx_light_cpu::ExecutionExecutorScope no_executor(nullptr);
    Compare(large, DataType::INT32);
  }
  EXPECT_EQ(executor.dispatches, 4);
  const Tensor floating = BuiltinCast(MakeCtx())(large, DataType::FLOAT);
  Compare(floating, DataType::FLOAT16);
  Compare(BuiltinCast(MakeCtx())(floating, DataType::BFLOAT16), DataType::DOUBLE);
  EXPECT_EQ(executor.dispatches, 8);
  view.run_blocks = nullptr;
  Compare(large, DataType::INT32);
  EXPECT_EQ(executor.dispatches, 8);
}

TEST(OnnxLightCastKernel, StandaloneContractValidation) {
  using onnx_light_cpu::CastConvert;
  std::array<int64_t, 4> data{1, 2, 3, 4};
  std::array<int32_t, 4> output{};
  EXPECT_NO_THROW(CastConvert(nullptr, CpuType::FLOAT, nullptr, CpuType::INT32, 0));
  EXPECT_THROW(CastConvert(nullptr, CpuType::FLOAT, output.data(), CpuType::INT32, 1),
               std::invalid_argument);
  EXPECT_THROW(CastConvert(data.data(), CpuType::INT64, nullptr, CpuType::INT32, 1),
               std::invalid_argument);
  EXPECT_THROW(CastConvert(data.data(), CpuType::STRING, output.data(), CpuType::INT32, 0),
               std::invalid_argument);
  EXPECT_THROW(CastConvert(data.data(), CpuType::INT64, data.data(), CpuType::INT64, 4),
               std::invalid_argument);
  EXPECT_THROW(CastConvert(data.data(), CpuType::INT64, output.data(), CpuType::INT32,
                           std::numeric_limits<std::size_t>::max()),
               std::invalid_argument);
  CastConvert(data.data(), CpuType::INT64, output.data(), CpuType::INT32, 4);
  EXPECT_EQ(output, (std::array<int32_t, 4>{1, 2, 3, 4}));
}

} // namespace
