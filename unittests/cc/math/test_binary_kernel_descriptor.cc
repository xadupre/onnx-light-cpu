// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/binary/binary_comparison_kernel.h"
#include "onnx_light_cpu/impl/math/binary/binary_kernel_descriptor.h"
#include "onnx_light_cpu/impl/math/binary/binary_manifest.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace {

using BinaryDataType = onnx_light_cpu::DataType;
using onnx_light_cpu::BinaryComparisonKind;
using onnx_light_cpu::BinaryKernelDescriptor;
using onnx_light_cpu::BinaryOperator;
using onnx_light_cpu::GetBinaryManifest;
using onnx_light_cpu::GetBinaryManifestEntry;

bool ExpectedComparison(float left, float right, BinaryComparisonKind kind) {
  switch (kind) {
  case BinaryComparisonKind::kEqual:
    return left == right;
  case BinaryComparisonKind::kGreater:
    return left > right;
  case BinaryComparisonKind::kGreaterOrEqual:
    return left >= right;
  case BinaryComparisonKind::kLess:
    return left < right;
  case BinaryComparisonKind::kLessOrEqual:
    return left <= right;
  }
  return false;
}

TEST(BinaryManifest, CoversRoadmapScopeExactlyOnce) {
  const auto manifest = GetBinaryManifest();
  ASSERT_EQ(manifest.size(), 19u);
  std::vector<std::string> names;
  for (const auto &entry : manifest) {
    EXPECT_FALSE(entry.op_type.empty());
    EXPECT_GT(entry.since_version, 0);
    EXPECT_GT(entry.minimum_version, 0);
    EXPECT_LE(entry.minimum_version, entry.since_version);
    EXPECT_FALSE(entry.signatures.empty());
    names.emplace_back(entry.op_type);
  }
  std::sort(names.begin(), names.end());
  EXPECT_EQ(std::unique(names.begin(), names.end()), names.end());
  EXPECT_EQ(GetBinaryManifestEntry("Add").op, BinaryOperator::kAdd);
  EXPECT_EQ(GetBinaryManifestEntry("PRelu").op, BinaryOperator::kPRelu);
}

TEST(BinaryKernelDescriptor, ValidatesOpsetAndAttributes) {
  EXPECT_THROW((BinaryKernelDescriptor{"Add", 6, {}}), std::invalid_argument);
  EXPECT_NO_THROW((BinaryKernelDescriptor{"Add", 7, {}}));
  EXPECT_NO_THROW((BinaryKernelDescriptor{"Pow", 14, {}}));
  EXPECT_NO_THROW((BinaryKernelDescriptor{"GreaterOrEqual", 13, {}}));
  EXPECT_THROW((BinaryKernelDescriptor{"Equal", 7, {}}.ResolveOutputType(BinaryDataType::STRING,
                                                                         BinaryDataType::STRING)),
               std::invalid_argument);
  EXPECT_EQ(BinaryKernelDescriptor("Equal", 19, {})
                .ResolveOutputType(BinaryDataType::STRING, BinaryDataType::STRING),
            BinaryDataType::BOOL);

  BinaryKernelDescriptor::Attributes invalid_mod;
  invalid_mod.mod_fmod = 2;
  EXPECT_THROW((BinaryKernelDescriptor{"Mod", 13, invalid_mod}), std::invalid_argument);

  BinaryKernelDescriptor::Attributes right_shift;
  right_shift.bitshift_direction = BinaryKernelDescriptor::Attributes::BitShiftDirection::kRight;
  EXPECT_NO_THROW((BinaryKernelDescriptor{"BitShift", 11, right_shift}));
}

TEST(BinaryKernelDescriptor, FiltersAndResolvesSupportedTypeSignatures) {
  BinaryKernelDescriptor::Attributes mod_int;
  mod_int.mod_fmod = 0;
  const BinaryKernelDescriptor int_mod("Mod", 13, mod_int);
  EXPECT_EQ(int_mod.ResolveOutputType(BinaryDataType::INT32, BinaryDataType::INT32),
            BinaryDataType::INT32);
  EXPECT_THROW((int_mod.ResolveOutputType(BinaryDataType::FLOAT, BinaryDataType::FLOAT)),
               std::invalid_argument);

  BinaryKernelDescriptor::Attributes mod_float;
  mod_float.mod_fmod = 1;
  const BinaryKernelDescriptor float_mod("Mod", 13, mod_float);
  EXPECT_EQ(float_mod.ResolveOutputType(BinaryDataType::FLOAT, BinaryDataType::FLOAT),
            BinaryDataType::FLOAT);

  const BinaryKernelDescriptor pow("Pow", 15, {});
  EXPECT_EQ(pow.ResolveOutputType(BinaryDataType::FLOAT, BinaryDataType::FLOAT),
            BinaryDataType::FLOAT);
  EXPECT_EQ(
      pow.ResolveAdapter(BinaryDataType::FLOAT16, BinaryDataType::FLOAT16, BinaryDataType::FLOAT16)
          .output_size,
      2u);
  EXPECT_EQ(pow.ResolveOutputType(BinaryDataType::FLOAT, BinaryDataType::INT64),
            BinaryDataType::FLOAT);
  EXPECT_EQ(pow.ResolveOutputType(BinaryDataType::BFLOAT16, BinaryDataType::UINT32),
            BinaryDataType::BFLOAT16);
  EXPECT_EQ(pow.ResolveOutputType(BinaryDataType::INT64, BinaryDataType::UINT32),
            BinaryDataType::INT64);
  EXPECT_THROW((pow.ResolveOutputType(BinaryDataType::INT64, BinaryDataType::FLOAT16)),
               std::invalid_argument);

  const BinaryKernelDescriptor prelu("PRelu", 16, {});
  EXPECT_THROW((prelu.ResolveOutputType(BinaryDataType::UINT16, BinaryDataType::UINT16)),
               std::invalid_argument);
}

TEST(BinaryKernelDescriptor, AssignsStableDistinctCacheIdentities) {
  const BinaryKernelDescriptor first("Add", 14, {});
  const BinaryKernelDescriptor second("Add", 14, {});
  EXPECT_NE(first.cache_identity(), 0u);
  EXPECT_NE(second.cache_identity(), 0u);
  EXPECT_NE(first.cache_identity(), second.cache_identity());
}

TEST(BinaryKernelDescriptor, Mod28EnablesFloatingRemainderWithoutChangingOlderOpsets) {
  const auto &entry = GetBinaryManifestEntry("Mod");
  EXPECT_EQ(entry.since_version, 28);
  EXPECT_EQ(entry.minimum_version, 10);
  for (const auto type : {BinaryDataType::FLOAT, BinaryDataType::DOUBLE, BinaryDataType::FLOAT16,
                          BinaryDataType::BFLOAT16}) {
    for (const int version : {10, 13, 27}) {
      EXPECT_THROW((BinaryKernelDescriptor("Mod", version, {}).ResolveOutputType(type, type)),
                   std::invalid_argument);
      BinaryKernelDescriptor::Attributes attrs;
      attrs.mod_fmod = 1;
      EXPECT_EQ(BinaryKernelDescriptor("Mod", version, attrs).ResolveOutputType(type, type), type);
    }
    EXPECT_EQ(BinaryKernelDescriptor("Mod", 28, {}).ResolveOutputType(type, type), type);
  }
}

template <typename T, typename Encode, typename Decode>
void CheckFloatingModAdapters(BinaryDataType type, Encode encode, Decode decode) {
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  struct Case {
    double left, right, remainder, fmod;
  };
  const Case cases[] = {
      {5.5, 2, 1.5, 1.5},
      {-5.5, 2, 0.5, -1.5},
      {5.5, -2, -0.5, 1.5},
      {-5.5, -2, -1.5, -1.5},
      {4, -2, -0.0, 0.0},
      {-4, 2, 0.0, -0.0},
      {0.0, -2, -0.0, 0.0},
      {-0.0, 2, 0.0, -0.0},
      {0.0, 2, 0.0, 0.0},
      {-0.0, -2, -0.0, -0.0},
      {-3, inf, inf, -3},
      {3, inf, 3, 3},
      {-1, -inf, -1, -1},
      {1, -inf, -inf, 1},
      {inf, 2, nan, nan},
      {-inf, 2, nan, nan},
      {inf, -2, nan, nan},
      {-inf, -2, nan, nan},
      {1, 0.0, nan, nan},
      {1, -0.0, nan, nan},
      {nan, 2, nan, nan},
      {nan, -2, nan, nan},
      {1, nan, nan, nan},
      {inf, inf, nan, nan},
      {0.0, -inf, -0.0, 0.0},
      {-0.0, inf, 0.0, -0.0},
      // The quotient overflows FLOAT16; a quotient-based implementation loses this remainder.
      {65504, 0.25, 0.0, 0.0},
      {-65504, 0.25, 0.0, -0.0},
  };
  for (const int fmod : {0, 1}) {
    BinaryKernelDescriptor::Attributes attributes;
    attributes.mod_fmod = fmod;
    const BinaryKernelDescriptor descriptor("Mod", 28, attributes);
    const auto &adapter = descriptor.ResolveAdapter(type, type, type);
    ASSERT_NE(adapter.bulk_contiguous, nullptr);
    ASSERT_NE(adapter.bulk_left_scalar, nullptr);
    ASSERT_NE(adapter.bulk_right_scalar, nullptr);
    for (const auto &test : cases) {
      SCOPED_TRACE(::testing::Message() << "type=" << static_cast<int>(type) << " fmod=" << fmod
                                        << " left=" << test.left << " right=" << test.right);
      const double expected = decode(encode(fmod == 0 ? test.remainder : test.fmod));
      const auto check = [&](T value) {
        const double actual = decode(value);
        if (std::isnan(expected)) {
          EXPECT_TRUE(std::isnan(actual));
        } else {
          EXPECT_EQ(actual, expected);
          EXPECT_EQ(std::signbit(actual), std::signbit(expected));
        }
      };
      T left = encode(test.left), right = encode(test.right), output{};
      adapter.scalar(&left, &right, &output);
      check(output);
      for (const std::size_t count : {0, 1, 7, 8, 9, 15, 16, 17, 1023, 1024, 1025}) {
        SCOPED_TRACE(count);
        std::vector<T> a(count + 1, left), b(count + 1, right);
        const T sentinel = encode(42);
        std::vector<T> result(count + 1, sentinel);
        for (auto bulk :
             {adapter.bulk_contiguous, adapter.bulk_left_scalar, adapter.bulk_right_scalar}) {
          bulk(a.data(), b.data(), result.data(), count);
          for (std::size_t i = 0; i < count; ++i) {
            check(result[i]);
          }
          EXPECT_EQ(result[count], sentinel);
        }
        adapter.bulk_contiguous(a.data(), b.data(), a.data(), count);
        for (std::size_t i = 0; i < count; ++i) {
          check(a[i]);
        }
      }
    }
  }
}

TEST(BinaryKernelDescriptor, ModFloatingScalarBulkBroadcastAndTailsPreserveSpecialValues) {
  CheckFloatingModAdapters<float>(
      BinaryDataType::FLOAT, [](double value) { return static_cast<float>(value); },
      [](float value) { return value; });
  CheckFloatingModAdapters<double>(
      BinaryDataType::DOUBLE, [](double value) { return value; },
      [](double value) { return value; });
  CheckFloatingModAdapters<std::uint16_t>(
      BinaryDataType::FLOAT16,
      [](double value) {
        return onnx_light_cpu::detail::FloatToFloat16Bits(static_cast<float>(value));
      },
      onnx_light_cpu::detail::Float16BitsToFloat);
  CheckFloatingModAdapters<std::uint16_t>(
      BinaryDataType::BFLOAT16,
      [](double value) {
        return onnx_light_cpu::detail::FloatToBFloat16Bits(static_cast<float>(value));
      },
      onnx_light_cpu::detail::Bfloat16BitsToFloat);
}

TEST(BinaryKernelDescriptor, ModFloatingRemainderAvoidsOverflowingQuotients) {
  const auto check = []<typename T>(BinaryDataType type) {
    const BinaryKernelDescriptor descriptor("Mod", 28, {});
    const auto &adapter = descriptor.ResolveAdapter(type, type, type);
    const T large = std::ldexp(T{1}, std::numeric_limits<T>::max_exponent - 1);
    const T small = std::numeric_limits<T>::min();
    for (const T left : {large, -large}) {
      for (const T right : {small, -small}) {
        T output{};
        adapter.scalar(&left, &right, &output);
        EXPECT_EQ(output, T{0});
        EXPECT_EQ(std::signbit(output), std::signbit(right));
        adapter.bulk_contiguous(&left, &right, &output, 1);
        EXPECT_EQ(output, T{0});
        EXPECT_EQ(std::signbit(output), std::signbit(right));
      }
    }
  };
  check.operator()<float>(BinaryDataType::FLOAT);
  check.operator()<double>(BinaryDataType::DOUBLE);
}

TEST(BinaryKernelDescriptor, SameTypeSignaturesProvideAllBulkLoopFamilies) {
  for (const auto &entry : GetBinaryManifest()) {
    BinaryKernelDescriptor::Attributes attributes;
    if (entry.op == BinaryOperator::kMod) {
      attributes.mod_fmod = 1;
    }
    const BinaryKernelDescriptor descriptor(std::string(entry.op_type), entry.since_version,
                                            attributes);
    for (const auto &adapter : descriptor.adapters()) {
      if (adapter.signature.left != adapter.signature.right ||
          adapter.signature.left == BinaryDataType::STRING) {
        continue;
      }
      EXPECT_NE(adapter.bulk_contiguous, nullptr) << entry.op_type;
      EXPECT_NE(adapter.bulk_left_scalar, nullptr) << entry.op_type;
      EXPECT_NE(adapter.bulk_right_scalar, nullptr) << entry.op_type;
    }
  }
}

TEST(BinaryKernelDescriptor, Float16ComparisonBulkPreservesIeeeSemanticsAndTails) {
  struct ComparisonCase {
    const char *name;
    int version;
    BinaryComparisonKind kind;
  };
  constexpr std::array<ComparisonCase, 5> cases = {
      ComparisonCase{"Equal", 19, BinaryComparisonKind::kEqual},
      ComparisonCase{"Greater", 13, BinaryComparisonKind::kGreater},
      ComparisonCase{"GreaterOrEqual", 16, BinaryComparisonKind::kGreaterOrEqual},
      ComparisonCase{"Less", 13, BinaryComparisonKind::kLess},
      ComparisonCase{"LessOrEqual", 16, BinaryComparisonKind::kLessOrEqual},
  };
  constexpr std::array<float, 19> left_values = {
      0.0f,
      -0.0f,
      1.0f,
      -2.0f,
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(),
      0.5f,
      -0.5f,
      3.0f,
      -4.0f,
      65504.0f,
      -65504.0f,
      0.00006103515625f,
      -0.00006103515625f,
      7.0f,
      7.0f,
      -8.0f,
      9.0f,
  };
  constexpr std::array<float, 19> right_values = {
      -0.0f,
      0.0f,
      2.0f,
      -2.0f,
      std::numeric_limits<float>::infinity(),
      0.0f,
      1.0f,
      std::numeric_limits<float>::quiet_NaN(),
      -1.0f,
      2.0f,
      -5.0f,
      65504.0f,
      65504.0f,
      0.00006103515625f,
      0.0f,
      8.0f,
      6.0f,
      -8.0f,
      10.0f,
  };
  std::array<std::uint16_t, left_values.size()> left{};
  std::array<std::uint16_t, right_values.size()> right{};
  for (std::size_t i = 0; i < left.size(); ++i) {
    left[i] = onnx_light_cpu::detail::FloatToFloat16Bits(left_values[i]);
    right[i] = onnx_light_cpu::detail::FloatToFloat16Bits(right_values[i]);
  }

  for (const auto &comparison : cases) {
    const BinaryKernelDescriptor descriptor(comparison.name, comparison.version, {});
    const auto &adapter = descriptor.ResolveAdapter(BinaryDataType::FLOAT16,
                                                    BinaryDataType::FLOAT16, BinaryDataType::BOOL);
    std::array<std::uint8_t, left.size()> output{};

    adapter.bulk_contiguous(left.data(), right.data(), output.data(), output.size());
    for (std::size_t i = 0; i < output.size(); ++i) {
      EXPECT_EQ(output[i],
                ExpectedComparison(left_values[i], right_values[i], comparison.kind) ? 1U : 0U)
          << comparison.name << " contiguous lane " << i;
    }

    adapter.bulk_left_scalar(left.data() + 3, right.data(), output.data(), output.size());
    for (std::size_t i = 0; i < output.size(); ++i) {
      EXPECT_EQ(output[i],
                ExpectedComparison(left_values[3], right_values[i], comparison.kind) ? 1U : 0U)
          << comparison.name << " left-scalar lane " << i;
    }

    adapter.bulk_right_scalar(left.data(), right.data() + 5, output.data(), output.size());
    for (std::size_t i = 0; i < output.size(); ++i) {
      EXPECT_EQ(output[i],
                ExpectedComparison(left_values[i], right_values[5], comparison.kind) ? 1U : 0U)
          << comparison.name << " right-scalar lane " << i;
    }
  }
}

TEST(BinaryKernelDescriptor, PowMixedTypesExecuteWithBaseOutputType) {
  const BinaryKernelDescriptor pow("Pow", 15, {});

  const float float_base = 2.0f;
  const std::int64_t integer_exponent = 3;
  float float_output = 0.0f;
  pow.ResolveAdapter(BinaryDataType::FLOAT, BinaryDataType::INT64, BinaryDataType::FLOAT)
      .scalar(&float_base, &integer_exponent, &float_output);
  EXPECT_FLOAT_EQ(float_output, 8.0f);

  const std::int32_t integer_base = -3;
  const float integral_float_exponent = 2.0f;
  std::int32_t integer_output = 0;
  pow.ResolveAdapter(BinaryDataType::INT32, BinaryDataType::FLOAT, BinaryDataType::INT32)
      .scalar(&integer_base, &integral_float_exponent, &integer_output);
  EXPECT_EQ(integer_output, 9);

  const std::uint16_t half_base = onnx_light_cpu::detail::FloatToFloat16Bits(4.0f);
  const std::uint16_t bfloat_exponent = onnx_light_cpu::detail::FloatToBFloat16Bits(0.5f);
  std::uint16_t half_output = 0;
  pow.ResolveAdapter(BinaryDataType::FLOAT16, BinaryDataType::BFLOAT16, BinaryDataType::FLOAT16)
      .scalar(&half_base, &bfloat_exponent, &half_output);
  EXPECT_FLOAT_EQ(onnx_light_cpu::detail::Float16BitsToFloat(half_output), 2.0f);

  const auto &float_mixed =
      pow.ResolveAdapter(BinaryDataType::FLOAT, BinaryDataType::INT64, BinaryDataType::FLOAT);
  EXPECT_NE(float_mixed.bulk_contiguous, nullptr);
  EXPECT_NE(float_mixed.bulk_left_scalar, nullptr);
  EXPECT_NE(float_mixed.bulk_right_scalar, nullptr);
  const std::array<float, 8> mixed_bases = {-2.0f,
                                            -1.0f,
                                            0.0f,
                                            0.5f,
                                            2.0f,
                                            3.0f,
                                            std::numeric_limits<float>::infinity(),
                                            std::numeric_limits<float>::quiet_NaN()};
  const std::array<std::int64_t, 8> mixed_exponents = {0, 1, 2, 3, 4, 5, -2, 7};
  std::array<float, 8> mixed_outputs = {};
  float_mixed.bulk_contiguous(mixed_bases.data(), mixed_exponents.data(), mixed_outputs.data(),
                              mixed_outputs.size());
  for (std::size_t i = 0; i < mixed_outputs.size(); ++i) {
    const float expected = std::pow(mixed_bases[i], static_cast<float>(mixed_exponents[i]));
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(mixed_outputs[i])) << i;
    } else {
      EXPECT_EQ(mixed_outputs[i], expected) << i;
    }
  }
  const auto check_scalar_paths = [&]<typename TExp>(BinaryDataType exponent_type) {
    const auto &adapter =
        pow.ResolveAdapter(BinaryDataType::FLOAT, exponent_type, BinaryDataType::FLOAT);
    const TExp identity = 1;
    adapter.bulk_right_scalar(mixed_bases.data(), &identity, mixed_outputs.data(),
                              mixed_outputs.size());
    for (std::size_t i = 0; i < mixed_outputs.size(); ++i) {
      if (std::isnan(mixed_bases[i])) {
        EXPECT_TRUE(std::isnan(mixed_outputs[i])) << i;
      } else {
        EXPECT_EQ(mixed_outputs[i], mixed_bases[i]) << i;
      }
    }
    const std::array<TExp, 6> exponents = {0, 1, 2, 3, 4, 5};
    const float scalar_base = 2.0f;
    adapter.bulk_left_scalar(&scalar_base, exponents.data(), mixed_outputs.data(),
                             exponents.size());
    for (std::size_t i = 0; i < exponents.size(); ++i) {
      EXPECT_EQ(mixed_outputs[i], std::pow(scalar_base, static_cast<float>(exponents[i]))) << i;
    }
  };
  check_scalar_paths.template operator()<std::int32_t>(BinaryDataType::INT32);
  check_scalar_paths.template operator()<std::int64_t>(BinaryDataType::INT64);
  check_scalar_paths.template operator()<std::uint32_t>(BinaryDataType::UINT32);
  check_scalar_paths.template operator()<std::uint64_t>(BinaryDataType::UINT64);

  const auto &integer_mixed =
      pow.ResolveAdapter(BinaryDataType::INT64, BinaryDataType::FLOAT, BinaryDataType::INT64);
  ASSERT_NE(integer_mixed.bulk_right_scalar, nullptr);
  const std::array<std::int64_t, 5> integer_bases = {-4, -1, 0, 2, 3037000499};
  const float square_exponent = 2.0f;
  std::array<std::int64_t, 5> integer_outputs = {};
  integer_mixed.bulk_right_scalar(integer_bases.data(), &square_exponent, integer_outputs.data(),
                                  integer_outputs.size());
  EXPECT_EQ(integer_outputs, (std::array<std::int64_t, 5>{16, 1, 0, 4, 9223372030926249001LL}));
  const std::int64_t overflowing_base = 3037000500;
  EXPECT_THROW(integer_mixed.bulk_right_scalar(&overflowing_base, &square_exponent,
                                               integer_outputs.data(), 1),
               std::invalid_argument);
  const float fractional_exponent = 1.5f;
  EXPECT_THROW(integer_mixed.bulk_right_scalar(integer_bases.data(), &fractional_exponent,
                                               integer_outputs.data(), integer_outputs.size()),
               std::invalid_argument);
  const float out_of_range_exponent = 1.0e20f;
  EXPECT_THROW(integer_mixed.bulk_right_scalar(integer_bases.data(), &out_of_range_exponent,
                                               integer_outputs.data(), integer_outputs.size()),
               std::invalid_argument);
  const float identity_exponent = 1.0f;
  auto in_place_bases = integer_bases;
  integer_mixed.bulk_right_scalar(in_place_bases.data(), &identity_exponent, in_place_bases.data(),
                                  in_place_bases.size());
  EXPECT_EQ(in_place_bases, integer_bases);

  const float negative_one = -1.0f;
  const std::int32_t odd_exponent = 16777217;
  float parity_output = 0.0f;
  pow.ResolveAdapter(BinaryDataType::FLOAT, BinaryDataType::INT32, BinaryDataType::FLOAT)
      .scalar(&negative_one, &odd_exponent, &parity_output);
  EXPECT_FLOAT_EQ(parity_output, -1.0f);

  const std::uint16_t half_negative_one = onnx_light_cpu::detail::FloatToFloat16Bits(-1.0f);
  std::uint16_t half_parity_output = 0;
  pow.ResolveAdapter(BinaryDataType::FLOAT16, BinaryDataType::INT32, BinaryDataType::FLOAT16)
      .scalar(&half_negative_one, &odd_exponent, &half_parity_output);
  EXPECT_FLOAT_EQ(onnx_light_cpu::detail::Float16BitsToFloat(half_parity_output), -1.0f);

  const std::int64_t large_odd_exponent = (std::int64_t{1} << 53) + 1;
  pow.ResolveAdapter(BinaryDataType::FLOAT16, BinaryDataType::INT64, BinaryDataType::FLOAT16)
      .scalar(&half_negative_one, &large_odd_exponent, &half_parity_output);
  EXPECT_FLOAT_EQ(onnx_light_cpu::detail::Float16BitsToFloat(half_parity_output), -1.0f);

  const auto &half_integer =
      pow.ResolveAdapter(BinaryDataType::FLOAT16, BinaryDataType::INT32, BinaryDataType::FLOAT16);
  ASSERT_NE(half_integer.bulk_contiguous, nullptr);
  ASSERT_NE(half_integer.bulk_left_scalar, nullptr);
  ASSERT_NE(half_integer.bulk_right_scalar, nullptr);
  const std::array<std::uint16_t, 5> half_bases = {
      onnx_light_cpu::detail::FloatToFloat16Bits(2.0f),
      onnx_light_cpu::detail::FloatToFloat16Bits(-2.0f),
      onnx_light_cpu::detail::FloatToFloat16Bits(3.0f),
      onnx_light_cpu::detail::FloatToFloat16Bits(0.5f), half_negative_one};
  const std::array<std::int32_t, 5> integer_exponents = {2, 3, 4, -2, odd_exponent};
  std::array<std::uint16_t, 5> half_outputs = {};
  half_integer.bulk_contiguous(half_bases.data(), integer_exponents.data(), half_outputs.data(),
                               half_outputs.size());
  const std::array<float, 5> expected_half_outputs = {4.0f, -8.0f, 81.0f, 4.0f, -1.0f};
  for (std::size_t i = 0; i < half_outputs.size(); ++i) {
    EXPECT_FLOAT_EQ(onnx_light_cpu::detail::Float16BitsToFloat(half_outputs[i]),
                    expected_half_outputs[i]);
  }

  for (const BinaryDataType exponent_type :
       {BinaryDataType::FLOAT, BinaryDataType::FLOAT16, BinaryDataType::INT32,
        BinaryDataType::INT64, BinaryDataType::UINT32, BinaryDataType::UINT64}) {
    const auto &bfloat_mixed =
        pow.ResolveAdapter(BinaryDataType::BFLOAT16, exponent_type, BinaryDataType::BFLOAT16);
    EXPECT_NE(bfloat_mixed.bulk_contiguous, nullptr);
    EXPECT_NE(bfloat_mixed.bulk_left_scalar, nullptr);
    EXPECT_NE(bfloat_mixed.bulk_right_scalar, nullptr);
  }
}

TEST(BinaryKernelDescriptor, PowIntegerFastPathPreservesFiniteBoundaryResults) {
  const BinaryKernelDescriptor pow("Pow", 15, {});
  const auto &adapter =
      pow.ResolveAdapter(BinaryDataType::FLOAT, BinaryDataType::FLOAT, BinaryDataType::FLOAT);
  const float input = 6981463572480.0f;
  const float exponent = 3.0f;
  float output = 0.0f;
  adapter.bulk_right_scalar(&input, &exponent, &output, 1);
  EXPECT_EQ(output, std::pow(input, exponent));
}

TEST(BinaryKernelDescriptor, PowBulkMatchesPositiveFractionalAndSpecialValues) {
  const BinaryKernelDescriptor pow("Pow", 15, {});
  const auto &adapter =
      pow.ResolveAdapter(BinaryDataType::FLOAT, BinaryDataType::FLOAT, BinaryDataType::FLOAT);
  std::vector<float> bases(35);
  std::vector<float> exponents(35);
  for (std::size_t i = 0; i < bases.size(); ++i) {
    bases[i] = 0.125f + static_cast<float>(i) * 0.25f;
    exponents[i] = -2.75f + static_cast<float>(i % 13) * 0.375f;
  }
  bases[0] = -2.0f;
  exponents[0] = 3.0f;
  bases[1] = 0.0f;
  exponents[1] = -1.0f;
  bases[2] = std::numeric_limits<float>::infinity();
  exponents[2] = 0.0f;
  std::vector<float> output(bases.size());

  adapter.bulk_contiguous(bases.data(), exponents.data(), output.data(), output.size());

  for (std::size_t i = 0; i < output.size(); ++i) {
    const float expected = std::pow(bases[i], exponents[i]);
    if (std::isfinite(expected)) {
      EXPECT_NEAR(output[i], expected, std::max(std::fabs(expected) * 3e-5f, 1e-6f)) << i;
    } else if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(output[i])) << i;
    } else {
      EXPECT_EQ(output[i], expected) << i;
    }
  }
}

TEST(BinaryKernelDescriptor, PowLeftScalarBulkMatchesFractionalAndSpecialValues) {
  const BinaryKernelDescriptor pow("Pow", 15, {});
  const auto &adapter =
      pow.ResolveAdapter(BinaryDataType::FLOAT, BinaryDataType::FLOAT, BinaryDataType::FLOAT);
  const float base = 1.75f;
  std::vector<float> exponents(35);
  for (std::size_t i = 0; i < exponents.size(); ++i) {
    exponents[i] = -2.75f + static_cast<float>(i % 13) * 0.375f;
  }
  exponents[0] = std::numeric_limits<float>::infinity();
  exponents[1] = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> output(exponents.size());

  adapter.bulk_left_scalar(&base, exponents.data(), output.data(), output.size());

  for (std::size_t i = 0; i < output.size(); ++i) {
    const float expected = std::pow(base, exponents[i]);
    if (std::isfinite(expected)) {
      EXPECT_NEAR(output[i], expected, std::max(std::fabs(expected) * 3e-5f, 1e-6f)) << i;
    } else if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(output[i])) << i;
    } else {
      EXPECT_EQ(output[i], expected) << i;
    }
  }
}

TEST(BinaryKernelDescriptor, PowRightScalarBulkMatchesFractionalAndSpecialValues) {
  const BinaryKernelDescriptor pow("Pow", 15, {});
  const auto &adapter =
      pow.ResolveAdapter(BinaryDataType::FLOAT, BinaryDataType::FLOAT, BinaryDataType::FLOAT);
  const float exponent = 1.375f;
  std::vector<float> bases(35);
  for (std::size_t i = 0; i < bases.size(); ++i) {
    bases[i] = 0.125f + static_cast<float>(i) * 0.25f;
  }
  bases[0] = -2.0f;
  bases[1] = 0.0f;
  bases[2] = std::numeric_limits<float>::infinity();
  bases[3] = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> output(bases.size());

  adapter.bulk_right_scalar(bases.data(), &exponent, output.data(), output.size());

  for (std::size_t i = 0; i < output.size(); ++i) {
    const float expected = std::pow(bases[i], exponent);
    if (std::isfinite(expected)) {
      EXPECT_NEAR(output[i], expected, std::max(std::fabs(expected) * 3e-5f, 1e-6f)) << i;
    } else if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(output[i])) << i;
    } else {
      EXPECT_EQ(output[i], expected) << i;
    }
  }
}

TEST(BinaryKernelDescriptor, IntegerPReluWrapsSignedMultiplication) {
  const BinaryKernelDescriptor prelu("PRelu", 16, {});
  const auto &adapter =
      prelu.ResolveAdapter(BinaryDataType::INT32, BinaryDataType::INT32, BinaryDataType::INT32);
  const std::int32_t input = std::numeric_limits<std::int32_t>::min();
  const std::int32_t slope = -1;
  std::int32_t output = 0;
  adapter.bulk_contiguous(&input, &slope, &output, 1);
  EXPECT_EQ(std::bit_cast<std::uint32_t>(output), std::bit_cast<std::uint32_t>(input));
}

} // namespace
