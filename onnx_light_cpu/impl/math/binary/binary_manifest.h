// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <span>
#include <string_view>

namespace onnx_light_cpu {

namespace BinaryDataType {
inline constexpr std::int32_t UNDEFINED = 0;
inline constexpr std::int32_t FLOAT = 1;
inline constexpr std::int32_t UINT8 = 2;
inline constexpr std::int32_t INT8 = 3;
inline constexpr std::int32_t UINT16 = 4;
inline constexpr std::int32_t INT16 = 5;
inline constexpr std::int32_t INT32 = 6;
inline constexpr std::int32_t INT64 = 7;
inline constexpr std::int32_t STRING = 8;
inline constexpr std::int32_t BOOL = 9;
inline constexpr std::int32_t FLOAT16 = 10;
inline constexpr std::int32_t DOUBLE = 11;
inline constexpr std::int32_t UINT32 = 12;
inline constexpr std::int32_t UINT64 = 13;
inline constexpr std::int32_t BFLOAT16 = 16;
} // namespace BinaryDataType

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
  std::int32_t left = BinaryDataType::UNDEFINED;
  std::int32_t right = BinaryDataType::UNDEFINED;
  std::int32_t output = BinaryDataType::UNDEFINED;
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
