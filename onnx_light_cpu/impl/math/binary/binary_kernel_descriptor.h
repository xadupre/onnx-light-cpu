// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu/impl/math/binary/binary_manifest.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace onnx_light_cpu {

/// Immutable node-level description of one binary operator.
class BinaryKernelDescriptor {
public:
  struct Attributes {
    std::int64_t mod_fmod = 0;
    enum class BitShiftDirection : std::uint8_t {
      kLeft,
      kRight
    } bitshift_direction = BitShiftDirection::kLeft;
  };

  struct Adapter {
    BinaryTypeSignature signature;
    using ScalarFn = void (*)(const void *, const void *, void *);
    ScalarFn scalar = nullptr;
    std::size_t left_size = 0;
    std::size_t right_size = 0;
    std::size_t output_size = 0;

    // Optional bulk SIMD entry points, populated only for the FP32/FP64
    // Add/Sub/Mul/Div signatures (Binary PR02). ``BinaryBroadcastPlan::Execute``
    // uses these instead of the per-element ``scalar`` callback whenever the
    // loop family is ``kContiguous``/``kLeftScalar``/``kRightScalar`` and a
    // bulk function is available; other operators/types leave these null and
    // fall back to the element-by-element ``scalar`` path.
    using BulkContiguousFn = void (*)(const void *left, const void *right, void *out,
                                      std::size_t count);
    using BulkLeftScalarFn = void (*)(const void *left, const void *right, void *out,
                                      std::size_t count);
    using BulkRightScalarFn = void (*)(const void *left, const void *right, void *out,
                                       std::size_t count);
    BulkContiguousFn bulk_contiguous = nullptr;
    BulkLeftScalarFn bulk_left_scalar = nullptr;
    BulkRightScalarFn bulk_right_scalar = nullptr;
  };

  BinaryKernelDescriptor(std::string op_type, std::int64_t opset_version,
                         const Attributes &attributes);

  BinaryOperator op() const noexcept { return manifest_.op; }
  std::string_view op_type() const noexcept { return manifest_.op_type; }
  std::int64_t opset_version() const noexcept { return opset_version_; }
  const Attributes &attributes() const noexcept { return attributes_; }
  std::uint64_t cache_identity() const noexcept { return cache_identity_; }
  const BinaryManifestEntry &manifest_entry() const noexcept { return manifest_; }
  std::span<const Adapter> adapters() const noexcept { return adapters_; }

  BinaryDataType ResolveOutputType(BinaryDataType left, BinaryDataType right) const;
  const Adapter &ResolveAdapter(BinaryDataType left, BinaryDataType right,
                                BinaryDataType output) const;

private:
  static std::uint64_t NextCacheIdentity();

  const BinaryManifestEntry &manifest_;
  std::int64_t opset_version_;
  Attributes attributes_;
  std::uint64_t cache_identity_;
  std::vector<Adapter> adapters_;
};
} // namespace onnx_light_cpu
