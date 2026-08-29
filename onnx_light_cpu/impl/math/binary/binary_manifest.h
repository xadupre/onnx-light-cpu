// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu/impl/data_type.h"

#include <cstdint>
#include <span>
#include <string_view>

namespace onnx_light_cpu {

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
  DataType left = DataType::UNDEFINED;
  DataType right = DataType::UNDEFINED;
  DataType output = DataType::UNDEFINED;
  /// Earliest operator schema version that permits this type signature.
  /// Zero inherits :cpp:member:`BinaryManifestEntry::minimum_version`.
  std::int64_t minimum_version = 0;
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
  /// Latest ONNX schema version whose type constraints this manifest mirrors.
  std::int64_t since_version = 0;
  /// Earliest schema version with semantics compatible with this kernel.
  std::int64_t minimum_version = 0;
  std::span<const BinaryTypeSignature> signatures;
};

std::span<const BinaryManifestEntry> GetBinaryManifest() noexcept;
const BinaryManifestEntry &GetBinaryManifestEntry(BinaryOperator op);
const BinaryManifestEntry &GetBinaryManifestEntry(std::string_view op_type);
std::string_view BinaryOperatorName(BinaryOperator op) noexcept;

} // namespace onnx_light_cpu
