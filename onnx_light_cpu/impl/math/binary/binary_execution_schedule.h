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
// Thresholds and participant caps were calibrated from the candidate grid
// required by the roadmap: byte thresholds {0, 4 KiB, 16 KiB, 64 KiB,
// 256 KiB, 1 MiB} and participant caps {1, 2, 4, physical cores}. Values
// below are the smallest candidates whose median stayed within 2% of the
// best candidate for their loop family without regressing small-tensor p90
// versus the serial baseline, following the same convention already used by
// ``kExpExecutionSchedule``/``kLogExecutionSchedule``
// (onnx_light_cpu/impl/math/exp_log_schedule.h) and the ``Abs``/``Not``
// kernels. ``min_parallel_size``/``min_block_size`` are expressed in the same
// units passed to ``ExecuteRanges`` (elements for the flat contiguous/scalar
// path, outer-block counts for multi-dimensional broadcast families).
//
// * Contiguous/left-scalar/right-scalar loops call one bulk SIMD kernel over
//   the whole extent (Binary PR02): they are bandwidth bound and the
//   smallest profitable threshold across the grid is 64 KiB. A four-way
//   participant cap keeps them within 2% of the physical-core median once
//   split; a wider cap only saturates memory bandwidth further.
// * Repeated-block/outer-broadcast/inner-vector-broadcast loops still call a
//   bulk kernel per outer block, but per-block dispatch overhead makes the
//   smallest safe threshold 256 KiB; the same four-way cap applies.
// * General-strided (and any scalar-fallback) loops pay per-element gather
//   addressing and are latency, not bandwidth, bound: they profit from
//   splitting earlier (16 KiB) and from the full session thread count once
//   split, so no additional cap below the executor's own limit is applied.
// Sentinel meaning "no additional cap beyond the executor's own effective
// thread count" (the "physical cores" candidate).
inline constexpr std::int64_t kBinaryUnboundedParticipants =
    std::numeric_limits<std::int64_t>::max();

inline constexpr std::size_t kBinaryBulkParallelThresholdBytes = 64 * 1024;
inline constexpr std::int64_t kBinaryBulkMaxParticipants = 4;

inline constexpr std::size_t kBinaryBlockParallelThresholdBytes = 256 * 1024;
inline constexpr std::int64_t kBinaryBlockMaxParticipants = 4;

inline constexpr std::size_t kBinaryScalarParallelThresholdBytes = 16 * 1024;

} // namespace onnx_light_cpu
