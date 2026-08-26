// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/binary/binary_kernel_descriptor.h"
#include "onnx_light_cpu/impl/math/binary/binary_manifest.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace {

namespace BinaryDataType = onnx_light_cpu::BinaryDataType;
using onnx_light_cpu::BinaryKernelDescriptor;
using onnx_light_cpu::BinaryOperator;
using onnx_light_cpu::BinaryTypeSignature;
using onnx_light_cpu::GetBinaryManifest;
using onnx_light_cpu::GetBinaryManifestEntry;

TEST(BinaryManifest, CoversRoadmapScopeExactlyOnce) {
  const auto manifest = GetBinaryManifest();
  ASSERT_EQ(manifest.size(), 19u);
  std::vector<std::string> names;
  for (const auto &entry : manifest) {
    EXPECT_FALSE(entry.op_type.empty());
    EXPECT_GT(entry.since_version, 0);
    EXPECT_FALSE(entry.signatures.empty());
    names.emplace_back(entry.op_type);
  }

  std::sort(names.begin(), names.end());
  EXPECT_EQ(std::unique(names.begin(), names.end()), names.end());
  EXPECT_EQ(GetBinaryManifestEntry("Add").op, BinaryOperator::kAdd);
  EXPECT_EQ(GetBinaryManifestEntry("PRelu").op, BinaryOperator::kPRelu);
}

TEST(BinaryManifest, UsesStandardIntegerTypeIds) {
  EXPECT_TRUE((std::is_same_v<decltype(BinaryTypeSignature{}.left), std::int32_t>));
  EXPECT_TRUE((std::is_same_v<decltype(BinaryDataType::FLOAT), const std::int32_t>));
}

TEST(BinaryKernelDescriptor, ValidatesOpsetAndAttributes) {
  EXPECT_THROW((BinaryKernelDescriptor{"Add", 13, {}}), std::invalid_argument);

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

TEST(BinaryKernelDescriptor, SameTypeSignaturesProvideAllBulkLoopFamilies) {
  for (const auto &entry : GetBinaryManifest()) {
    BinaryKernelDescriptor::Attributes attributes;
    if (entry.op == BinaryOperator::kMod) {
      attributes.mod_fmod = 1;
    }
    const BinaryKernelDescriptor descriptor(std::string(entry.op_type), entry.since_version,
                                            attributes);
    for (const auto &adapter : descriptor.adapters()) {
      if (adapter.signature.left != adapter.signature.right) {
        continue;
      }
      EXPECT_NE(adapter.bulk_contiguous, nullptr) << entry.op_type;
      EXPECT_NE(adapter.bulk_left_scalar, nullptr) << entry.op_type;
      EXPECT_NE(adapter.bulk_right_scalar, nullptr) << entry.op_type;
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
