// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu/impl/execution.h"

#include <cstdint>
#include <limits>

namespace onnx_light_cpu {

// Binary PR03: per-invocation executor decisions for BinaryBroadcastPlan.
//
// Byte thresholds were calibrated from the candidate grid required by the
// roadmap: {0, 4 KiB, 16 KiB, 64 KiB, 256 KiB, 1 MiB}. Values below avoid
// regressing small-tensor p90 versus the serial baseline, following the same
// convention already used by ``kExpExecutionSchedule``/``kLogExecutionSchedule``
// (onnx_light_cpu/impl/math/exp_log_schedule.h) and the ``Abs``/``Not``
// kernels. ``min_parallel_size``/``min_block_size`` are expressed in the same
// units passed to ``ExecuteRanges`` (elements for the flat contiguous/scalar
// path, outer-block counts for multi-dimensional broadcast families).
//
// * Contiguous/left-scalar/right-scalar loops call one bulk kernel over the
//   whole extent (Binary PR02): they are bandwidth bound and stay serial
//   below 1 MiB to avoid executor overhead on medium tensors.
// * Repeated-block/outer-broadcast/inner-vector-broadcast loops still call a
//   bulk kernel per outer block, but per-block dispatch overhead makes the
//   smallest safe threshold 1 MiB.
// * General-strided (and any scalar-fallback) loops pay per-element gather
//   addressing and are latency, not bandwidth, bound, so they split earlier
//   at 256 KiB.
// ABI 2 additionally exposes a participant ceiling. Registry value zero is
// resolved to the unbounded portable default before execution.
inline constexpr std::size_t kBinaryBulkParallelThresholdBytes = 1024 * 1024;

inline constexpr std::size_t kBinaryBlockParallelThresholdBytes = 1024 * 1024;

inline constexpr std::size_t kBinaryScalarParallelThresholdBytes = 256 * 1024;

inline constexpr std::size_t kBinaryTargetBlockBytes = 1024 * 1024;

inline constexpr std::int64_t kBinaryMaxParticipants = std::numeric_limits<std::int64_t>::max();

struct BinaryExecutionTuning {
  std::size_t bulk_parallel_threshold_bytes = kBinaryBulkParallelThresholdBytes;
  std::size_t block_parallel_threshold_bytes = kBinaryBlockParallelThresholdBytes;
  std::size_t scalar_parallel_threshold_bytes = kBinaryScalarParallelThresholdBytes;
  std::size_t target_block_bytes = kBinaryTargetBlockBytes;
  std::int64_t max_participants = kBinaryMaxParticipants;

  bool operator==(const BinaryExecutionTuning &) const = default;
};

inline constexpr BinaryExecutionTuning kDefaultBinaryExecutionTuning{};

} // namespace onnx_light_cpu
