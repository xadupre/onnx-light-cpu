// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/binary/binary_kernel_descriptor.h"
#include "onnx_light_cpu/impl/math/binary/binary_manifest.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

namespace {

using onnx_light_cpu::BinaryDataType;
using onnx_light_cpu::BinaryKernelDescriptor;
using onnx_light_cpu::BinaryOperator;
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

  const BinaryKernelDescriptor pow("Pow", 7, {});
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

} // namespace
