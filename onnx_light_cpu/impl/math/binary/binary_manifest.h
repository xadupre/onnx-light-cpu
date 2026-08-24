// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <span>
#include <string_view>

namespace onnx_light_cpu {

enum class BinaryDataType : std::int32_t {
  UNDEFINED = 0,
  FLOAT = 1,
  UINT8 = 2,
  INT8 = 3,
  UINT16 = 4,
  INT16 = 5,
  INT32 = 6,
  INT64 = 7,
  STRING = 8,
  BOOL = 9,
  FLOAT16 = 10,
  DOUBLE = 11,
  UINT32 = 12,
  UINT64 = 13,
  BFLOAT16 = 16,
};

/// In-scope binary elementwise operators covered by Binary PR01.
enum class BinaryOperator : std::uint8_t {
  kAdd,
  kSub,
  kMul,
  kDiv,
  kMod,
  kPow,
  kEqual,
  kGreater,
  kGreaterOrEqual,
  kLess,
  kLessOrEqual,
  kAnd,
  kOr,
  kXor,
  kBitwiseAnd,
  kBitwiseOr,
  kBitwiseXor,
  kBitShift,
  kPRelu,
};

/// One concrete input/output type triple advertised by the prepared binary engine.
struct BinaryTypeSignature {
  BinaryDataType left = BinaryDataType::UNDEFINED;
  BinaryDataType right = BinaryDataType::UNDEFINED;
  BinaryDataType output = BinaryDataType::UNDEFINED;
};

/// Explicit manifest entry for one optimized binary operator.
///
/// The table is intentionally handwritten rather than inferred at runtime so the
/// CPU registrations remain auditable: each row documents the latest ONNX opset
/// version validated for this optimized path together with the exact type
/// signatures that onnx-light's portable kernel implementation already supports.
struct BinaryManifestEntry {
  BinaryOperator op = BinaryOperator::kAdd;
  std::string_view op_type;
  std::int64_t since_version = 0;
  std::span<const BinaryTypeSignature> signatures;
};

std::span<const BinaryManifestEntry> GetBinaryManifest() noexcept;
const BinaryManifestEntry &GetBinaryManifestEntry(BinaryOperator op);
const BinaryManifestEntry &GetBinaryManifestEntry(std::string_view op_type);
std::string_view BinaryOperatorName(BinaryOperator op) noexcept;

} // namespace onnx_light_cpu
