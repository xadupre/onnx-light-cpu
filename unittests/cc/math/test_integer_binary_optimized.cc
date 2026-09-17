// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/binary/binary_broadcast_plan.h"
#include "onnx_light_cpu/impl/math/binary/binary_integer_arithmetic.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace onnx_light_cpu;
using Shape = std::vector<std::int64_t>;

template <typename T> constexpr DataType Type() {
  if constexpr (std::is_same_v<T, float>) {
    return DataType::FLOAT;
  } else if constexpr (std::is_signed_v<T>) {
    if constexpr (sizeof(T) == 1)
      return DataType::INT8;
    if constexpr (sizeof(T) == 2)
      return DataType::INT16;
    if constexpr (sizeof(T) == 4)
      return DataType::INT32;
    return DataType::INT64;
  } else {
    if constexpr (sizeof(T) == 1)
      return DataType::UINT8;
    if constexpr (sizeof(T) == 2)
      return DataType::UINT16;
    if constexpr (sizeof(T) == 4)
      return DataType::UINT32;
    return DataType::UINT64;
  }
}

template <typename U> void CheckDivider() {
  std::mt19937_64 random(722);
  std::vector<U> divisors{1, 2, 3, 5, 7, 10, 13, 31, 63, 127};
  for (unsigned bit = 1; bit < sizeof(U) * 8; ++bit) {
    const U power = static_cast<U>(U{1} << bit);
    divisors.insert(divisors.end(), {static_cast<U>(power - 1), power, static_cast<U>(power + 1)});
  }
  divisors.push_back(std::numeric_limits<U>::max());
  for (int i = 0; i < 2000; ++i) {
    const U divisor = static_cast<U>(random());
    divisors.push_back(divisor == 0 ? U{1} : divisor);
  }
  for (U divisor : divisors) {
    detail::IntegerDivider<U> divider(divisor);
    const std::array<U, 6> boundaries{0,
                                      1,
                                      static_cast<U>(divisor - 1),
                                      divisor,
                                      static_cast<U>(divisor + U{1}),
                                      std::numeric_limits<U>::max()};
    for (U numerator : boundaries)
      ASSERT_EQ(divider.Divide(numerator), numerator / divisor) << +numerator << "/" << +divisor;
    for (int i = 0; i < 128; ++i) {
      const U numerator = static_cast<U>(random());
      ASSERT_EQ(divider.Divide(numerator), numerator / divisor) << +numerator << "/" << +divisor;
    }
  }
}

TEST(IntegerBinaryOptimized, ReciprocalDivisionRandomAndBoundaries) {
  CheckDivider<std::uint8_t>();
  CheckDivider<std::uint16_t>();
  CheckDivider<std::uint32_t>();
  CheckDivider<std::uint64_t>();
  for (unsigned divisor = 1; divisor < 256; ++divisor) {
    const detail::IntegerDivider<std::uint8_t> divider(static_cast<std::uint8_t>(divisor));
    for (unsigned numerator = 0; numerator < 256; ++numerator)
      ASSERT_EQ(divider.Divide(static_cast<std::uint8_t>(numerator)), numerator / divisor);
  }
}

template <typename T> void CheckDivModPlans() {
  std::mt19937_64 random(723);
  const T minimum = std::numeric_limits<T>::min(), maximum = std::numeric_limits<T>::max();
  std::vector<T> divisors{1, 2, 3, 5, 7, 10, 31, maximum};
  if constexpr (std::is_signed_v<T>)
    divisors.insert(divisors.end(), {-1, -2, -3, -7, minimum, static_cast<T>(minimum + 1)});
  for (int i = 0; i < 100; ++i) {
    const T divisor = std::bit_cast<T>(static_cast<std::make_unsigned_t<T>>(random()));
    if (divisor != 0)
      divisors.push_back(divisor);
  }
  for (const std::string op : {"Div", "Mod"}) {
    for (int fmod : {0, 1}) {
      BinaryKernelDescriptor::Attributes attributes;
      attributes.mod_fmod = fmod;
      const BinaryKernelDescriptor descriptor(op, 14, attributes);
      const auto &adapter = descriptor.ResolveAdapter(Type<T>(), Type<T>(), Type<T>());
      ASSERT_NE(adapter.bulk_right_scalar, nullptr);
      ASSERT_NE(adapter.validate_right_bulk, nullptr);
      for (T divisor : divisors) {
        using U = std::make_unsigned_t<T>;
        const T previous = std::bit_cast<T>(static_cast<U>(static_cast<U>(divisor) - U{1}));
        std::vector<T> input{minimum, maximum, 0, 1, previous, divisor};
        for (int i = 0; i < 123; ++i)
          input.push_back(std::bit_cast<T>(static_cast<std::make_unsigned_t<T>>(random())));
        if constexpr (std::is_signed_v<T>) {
          if (divisor == -1)
            for (auto &value : input)
              if (value == minimum)
                value = minimum + 1;
        }
        for (std::size_t count : {std::size_t{1}, std::size_t{3}, std::size_t{4}, input.size()}) {
          const BinaryBroadcastPlan plan(descriptor, Type<T>(), Type<T>(), Type<T>(),
                                         Shape{static_cast<std::int64_t>(count)}, {});
          std::vector<T> output(count);
          plan.Execute(input.data(), &divisor, output.data());
          std::vector<T> in_place(input.begin(), input.begin() + count);
          plan.Execute(in_place.data(), &divisor, in_place.data());
          ASSERT_EQ(in_place, output);
          for (std::size_t i = 0; i < count; ++i) {
            T expected = op == "Div" ? static_cast<T>(input[i] / divisor)
                                     : static_cast<T>(input[i] % divisor);
            if constexpr (std::is_signed_v<T>) {
              if (op == "Mod" && fmod == 0 && expected != 0 && ((expected < 0) != (divisor < 0)))
                expected = static_cast<T>(expected + divisor);
            }
            ASSERT_EQ(output[i], expected) << op << " divisor=" << +divisor << " i=" << i;
          }
        }
      }
      T input[4]{minimum, maximum, 0, 1}, output[4]{42, 42, 42, 42}, zero = 0;
      const BinaryBroadcastPlan plan(descriptor, Type<T>(), Type<T>(), Type<T>(), Shape{4}, {});
      EXPECT_THROW(plan.Execute(input, &zero, output), std::invalid_argument);
      EXPECT_EQ(output[0], 42);
      if constexpr (std::is_signed_v<T>) {
        T minus_one = -1;
        EXPECT_THROW(plan.Execute(input, &minus_one, output), std::invalid_argument);
        EXPECT_EQ(output[0], 42);
      }
      const T varying_divisors[4]{1, 3, 0, 7};
      const BinaryBroadcastPlan contiguous(descriptor, Type<T>(), Type<T>(), Type<T>(), Shape{4},
                                           Shape{4});
      EXPECT_THROW(contiguous.Execute(input, varying_divisors, output), std::invalid_argument);
      EXPECT_EQ(output[0], 42);
      const T row_divisors[3]{3, 7, 11};
      const BinaryBroadcastPlan rows(descriptor, Type<T>(), Type<T>(), Type<T>(), Shape{1, 4},
                                     Shape{3, 1});
      T row_output[12]{};
      rows.Execute(input, row_divisors, row_output);
      for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 4; ++col) {
          T expected;
          adapter.scalar(input + col, row_divisors + row, &expected);
          EXPECT_EQ(row_output[row * 4 + col], expected);
        }
      EXPECT_NO_THROW(adapter.bulk_right_scalar(nullptr, nullptr, nullptr, 0));
    }
  }
}

TEST(IntegerBinaryOptimized, DivModAllIntegerTypes) {
  CheckDivModPlans<std::int8_t>();
  CheckDivModPlans<std::int16_t>();
  CheckDivModPlans<std::int32_t>();
  CheckDivModPlans<std::int64_t>();
  CheckDivModPlans<std::uint8_t>();
  CheckDivModPlans<std::uint16_t>();
  CheckDivModPlans<std::uint32_t>();
  CheckDivModPlans<std::uint64_t>();
}

template <typename T, typename E> void CheckPowPlans() {
  const BinaryKernelDescriptor descriptor("Pow", 15, {});
  const auto &adapter = descriptor.ResolveAdapter(Type<T>(), Type<E>(), Type<T>());
  ASSERT_NE(adapter.bulk_contiguous, nullptr);
  ASSERT_NE(adapter.bulk_left_scalar, nullptr);
  ASSERT_NE(adapter.bulk_right_scalar, nullptr);
  const std::vector<T> bases{-3, -2, -1, 0, 1, 2, 3, 7, 11};
  const std::vector<E> exponents{0, 1, 2, 3};
  std::vector<T> output(bases.size() * exponents.size());
  const BinaryBroadcastPlan inner(descriptor, Type<T>(), Type<E>(), Type<T>(), Shape{1, 9},
                                  Shape{4, 1});
  inner.Execute(bases.data(), exponents.data(), output.data());
  for (std::size_t row = 0; row < exponents.size(); ++row)
    for (std::size_t col = 0; col < bases.size(); ++col) {
      T expected = 1;
      for (std::size_t i = 0; i < row; ++i)
        expected *= bases[col];
      EXPECT_EQ(output[row * bases.size() + col], expected);
    }
  T base = -3;
  T actual[4]{};
  const BinaryBroadcastPlan left(descriptor, Type<T>(), Type<E>(), Type<T>(), {}, Shape{4});
  left.Execute(&base, exponents.data(), actual);
  EXPECT_EQ((std::vector<T>(actual, actual + 4)), (std::vector<T>{1, -3, 9, -27}));
  const T contiguous_bases[4]{-3, -3, -3, -3};
  const BinaryBroadcastPlan contiguous(descriptor, Type<T>(), Type<E>(), Type<T>(), Shape{4},
                                       Shape{4});
  contiguous.Execute(contiguous_bases, exponents.data(), actual);
  EXPECT_EQ((std::vector<T>(actual, actual + 4)), (std::vector<T>{1, -3, 9, -27}));
  base = std::numeric_limits<T>::max();
  E exponent = 1;
  adapter.scalar(&base, &exponent, actual);
  EXPECT_EQ(actual[0], base);
  exponent = 2;
  EXPECT_THROW(adapter.scalar(&base, &exponent, actual), std::invalid_argument);
  base = -2;
  exponent = static_cast<E>(sizeof(T) * 8 - 1);
  adapter.scalar(&base, &exponent, actual);
  EXPECT_EQ(actual[0], std::numeric_limits<T>::min());
  base = 2;
  EXPECT_THROW(adapter.scalar(&base, &exponent, actual), std::invalid_argument);
  if constexpr (std::is_signed_v<E>) {
    exponent = -1;
    base = 1;
    EXPECT_THROW(adapter.scalar(&base, &exponent, actual), std::invalid_argument);
    EXPECT_THROW(adapter.bulk_left_scalar(&base, &exponent, actual, 1), std::invalid_argument);
  }
  if constexpr (std::is_floating_point_v<E>) {
    for (E invalid : {E{0.5}, std::numeric_limits<E>::infinity(),
                      std::numeric_limits<E>::quiet_NaN(), E{9223372036854775808.0}}) {
      EXPECT_THROW(adapter.scalar(&base, &invalid, actual), std::invalid_argument);
      EXPECT_THROW(adapter.bulk_right_scalar(&base, &invalid, actual, 1), std::invalid_argument);
    }
  }
  for (E small : {E{0}, E{1}, E{2}}) {
    std::vector<T> in_place = bases;
    adapter.bulk_right_scalar(in_place.data(), &small, in_place.data(), in_place.size());
    for (std::size_t i = 0; i < bases.size(); ++i) {
      T expected = 1;
      for (unsigned power = 0; power < static_cast<unsigned>(small); ++power)
        expected *= bases[i];
      EXPECT_EQ(in_place[i], expected);
    }
  }
  EXPECT_NO_THROW(adapter.bulk_right_scalar(nullptr, nullptr, nullptr, 0));
}

TEST(IntegerBinaryOptimized, PowAllSignaturesAndBroadcasts) {
  CheckPowPlans<std::int32_t, float>();
  CheckPowPlans<std::int32_t, std::int32_t>();
  CheckPowPlans<std::int32_t, std::int64_t>();
  CheckPowPlans<std::int32_t, std::uint32_t>();
  CheckPowPlans<std::int32_t, std::uint64_t>();
  CheckPowPlans<std::int64_t, float>();
  CheckPowPlans<std::int64_t, std::int32_t>();
  CheckPowPlans<std::int64_t, std::int64_t>();
  CheckPowPlans<std::int64_t, std::uint32_t>();
  CheckPowPlans<std::int64_t, std::uint64_t>();
}

TEST(IntegerBinaryOptimized, PortableMultiplyMatchesWideReference) {
  std::mt19937_64 random(724);
  const std::array<std::uint64_t, 6> boundaries{
      0, 1, UINT32_MAX, UINT64_C(1) << 32, UINT64_C(1) << 63, UINT64_MAX};
  for (auto a : boundaries)
    for (auto b : boundaries)
      EXPECT_EQ(detail::MultiplyHigh64Portable(a, b), detail::MultiplyHigh64(a, b));
  for (int i = 0; i < 100000; ++i) {
    const auto a = random(), b = random();
    ASSERT_EQ(detail::MultiplyHigh64Portable(a, b), detail::MultiplyHigh64(a, b));
#if defined(__SIZEOF_INT128__)
    const std::int64_t sa = std::bit_cast<std::int64_t>(a);
    const std::int64_t sb = std::bit_cast<std::int64_t>(b);
    const __int128 product = static_cast<__int128>(sa) * sb;
    const bool overflow = product < INT64_MIN || product > INT64_MAX;
    std::int64_t actual;
    ASSERT_EQ(detail::MultiplyOverflowPortable(sa, sb, actual), overflow);
    if (!overflow)
      ASSERT_EQ(actual, product);
#endif
  }
  std::int64_t result;
  EXPECT_FALSE(detail::MultiplyOverflowPortable(INT64_MIN, INT64_C(1), result));
  EXPECT_EQ(result, INT64_MIN);
  EXPECT_TRUE(detail::MultiplyOverflowPortable(INT64_MIN, INT64_C(-1), result));
  EXPECT_FALSE(detail::MultiplyOverflowPortable(INT64_C(-2097152), INT64_C(4398046511104), result));
  EXPECT_EQ(result, INT64_MIN);
  EXPECT_FALSE(detail::MultiplyOverflowPortable(INT64_C(3037000499), INT64_C(3037000499), result));
  EXPECT_EQ(result, INT64_C(9223372030926249001));
  EXPECT_TRUE(detail::MultiplyOverflowPortable(INT64_C(3037000500), INT64_C(3037000500), result));
}

TEST(IntegerBinaryOptimized, ExactPowScalarContracts) {
  using detail::CheckedIntegerPow;
  EXPECT_EQ(CheckedIntegerPow(INT64_C(3037000499), 2, "Pow"), INT64_C(9223372030926249001));
  EXPECT_EQ(CheckedIntegerPow(INT64_C(2097151), 3, "Pow"), INT64_C(9223358842721533951));
  EXPECT_EQ(CheckedIntegerPow(INT64_C(-2097152), 3, "Pow"), INT64_MIN);
  EXPECT_THROW(CheckedIntegerPow(INT64_C(2097152), 3, "Pow"), std::invalid_argument);
  EXPECT_EQ(CheckedIntegerPow(INT64_MIN, 1, "Pow"), INT64_MIN);
  EXPECT_EQ(CheckedIntegerPow(INT64_MIN, 0, "Pow"), 1);
  EXPECT_THROW(CheckedIntegerPow(INT64_MIN, 2, "Pow"), std::invalid_argument);
  for (std::int64_t base : {INT64_C(-1), INT64_C(0), INT64_C(1)}) {
    EXPECT_EQ(CheckedIntegerPow(base, UINT64_MAX, "Pow"), base);
    EXPECT_THROW(CheckedIntegerPow(base, INT64_MIN, "Pow"), std::invalid_argument);
    EXPECT_THROW(CheckedIntegerPow(base, -1.0f, "Pow"), std::invalid_argument);
    EXPECT_THROW(CheckedIntegerPow(base, -0.5f, "Pow"), std::invalid_argument);
    EXPECT_THROW(CheckedIntegerPow(base, -9223372036854775808.0f, "Pow"), std::invalid_argument);
    EXPECT_EQ(CheckedIntegerPow(base, -0.0f, "Pow"), 1);
    EXPECT_EQ(CheckedIntegerPow(base, std::nextafter(9223372036854775808.0f, 0.0f), "Pow"),
              base == 0 ? 0 : 1);
  }
#if defined(__SIZEOF_INT128__)
  std::mt19937_64 random(725);
  for (int i = 0; i < 10000; ++i) {
    const std::int64_t base = static_cast<std::int64_t>(random() % 10000000) - 5000000;
    const unsigned exponent = random() % 12;
    __int128 expected = 1;
    bool overflow = false;
    for (unsigned j = 0; j < exponent; ++j) {
      expected *= base;
      if (expected < INT64_MIN || expected > INT64_MAX) {
        overflow = true;
        break;
      }
    }
    if (overflow) {
      EXPECT_THROW(CheckedIntegerPow(base, exponent, "Pow"), std::invalid_argument);
    } else {
      EXPECT_EQ(CheckedIntegerPow(base, exponent, "Pow"), expected);
    }
  }
#endif
}

template <typename T> void CheckRegisteredComparisons() {
  const std::array<T, 8> boundaries{std::numeric_limits<T>::min(),
                                    std::numeric_limits<T>::max(),
                                    T{0},
                                    T{1},
                                    std::bit_cast<T>(UINT64_C(0x8000000000000000)),
                                    std::bit_cast<T>(UINT64_C(0x7fffffffffffffff)),
                                    std::bit_cast<T>(UINT64_C(0x00000000ffffffff)),
                                    std::bit_cast<T>(UINT64_C(0xffffffff00000000))};
  for (const std::string op : {"Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual"}) {
    const BinaryKernelDescriptor descriptor(op, 19, {});
    const auto &adapter = descriptor.ResolveAdapter(Type<T>(), Type<T>(), DataType::BOOL);
    ASSERT_EQ(adapter.bulk_implementation,
              BinaryKernelDescriptor::Adapter::BulkImplementation::kCompare64);
    ASSERT_NE(adapter.bulk_contiguous, nullptr);
    ASSERT_NE(adapter.bulk_left_scalar, nullptr);
    ASSERT_NE(adapter.bulk_right_scalar, nullptr);
    for (int layout = 0; layout < 3; ++layout) {
      const auto kernel = layout == 0   ? adapter.bulk_contiguous
                          : layout == 1 ? adapter.bulk_left_scalar
                                        : adapter.bulk_right_scalar;
      EXPECT_NO_THROW(kernel(nullptr, nullptr, nullptr, 0));
      for (std::size_t count : {1, 3, 4, 7, 8, 9, 17, 129}) {
        for (std::size_t shift = 0; shift < boundaries.size(); ++shift) {
          std::vector<T> left(layout == 1 ? 1 : count), right(layout == 2 ? 1 : count);
          for (std::size_t i = 0; i < left.size(); ++i)
            left[i] = boundaries[(i + shift) % boundaries.size()];
          for (std::size_t i = 0; i < right.size(); ++i)
            right[i] = boundaries[(2 * i + 3 * shift) % boundaries.size()];
          std::vector<std::uint8_t> output(count + 2, 0xa5);
          kernel(left.data(), right.data(), output.data() + 1, count);
          EXPECT_EQ(output.front(), 0xa5);
          EXPECT_EQ(output.back(), 0xa5);
          for (std::size_t i = 0; i < count; ++i) {
            const T a = left[layout == 1 ? 0 : i], b = right[layout == 2 ? 0 : i];
            const bool expected = op == "Equal"            ? a == b
                                  : op == "Greater"        ? a > b
                                  : op == "GreaterOrEqual" ? a >= b
                                  : op == "Less"           ? a < b
                                                           : a <= b;
            ASSERT_EQ(output[i + 1], static_cast<std::uint8_t>(expected))
                << op << " layout=" << layout << " count=" << count << " i=" << i;
          }
        }
      }
    }
  }
}

TEST(IntegerBinaryOptimized, RegisteredSignedComparisonBulkBindings) {
  CheckRegisteredComparisons<std::int64_t>();
}

TEST(IntegerBinaryOptimized, RegisteredUnsignedComparisonBulkBindings) {
  CheckRegisteredComparisons<std::uint64_t>();
}

} // namespace
