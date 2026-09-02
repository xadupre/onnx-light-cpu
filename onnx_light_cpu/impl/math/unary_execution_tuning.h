// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu/impl/execution.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace onnx_light_cpu {

struct UnaryExecutionTuning {
  // Zero disables executor dispatch.
  std::size_t parallel_threshold_bytes = 0;
  std::size_t target_block_bytes = 1;
  // Zero uses every participant made available by the session executor.
  std::size_t max_participants = 0;
  // Prefer the session executor's shared cost model when it is available.
  bool use_cost_model = true;
  // Zero leaves the participant count to the executor cost model.
  std::size_t preferred_participants = 0;
  // Zero disables non-temporal stores.
  std::size_t streaming_store_threshold_bytes = 0;

  bool operator==(const UnaryExecutionTuning &) const = default;
};

inline constexpr UnaryExecutionTuning kDefaultAbsFloat32ExecutionTuning{
    2 * 1024 * 1024, 256 * 1024, 32, true, 0, 0};
inline constexpr UnaryExecutionTuning kDefaultAbs32ExecutionTuning{2 * 1024 * 1024, 256 * 1024, 0};
inline constexpr UnaryExecutionTuning kDefaultAbs64ExecutionTuning{2 * 1024 * 1024, 512 * 1024, 0};
inline constexpr UnaryExecutionTuning kDefaultAbs16ExecutionTuning{4 * 1024 * 1024, 128 * 1024, 0};
inline constexpr UnaryExecutionTuning kDefaultAbs8ExecutionTuning{8 * 1024 * 1024, 64 * 1024, 0};
inline constexpr UnaryExecutionTuning kDefaultAbsInt8ExecutionTuning{1024 * 1024, 64 * 1024, 0};
inline constexpr UnaryExecutionTuning kDefaultAbsInt16ExecutionTuning{1024 * 1024, 128 * 1024, 0};
inline constexpr UnaryExecutionTuning kDefaultAbsInt32ExecutionTuning{512 * 1024, 256 * 1024, 32,
                                                                      true};
inline constexpr UnaryExecutionTuning kDefaultAbsInt64ExecutionTuning{256 * 1024, 128 * 1024, 64};
inline constexpr UnaryExecutionTuning kDefaultExpLogExecutionTuning{2 * 1024 * 1024, 256 * 1024,
                                                                    32};
inline constexpr UnaryExecutionTuning kDefaultExpLogHalfExecutionTuning{1024 * 1024, 128 * 1024,
                                                                        32};
inline constexpr UnaryExecutionTuning kDefaultLogFloat16ExecutionTuning{1024 * 1024, 128 * 1024,
                                                                        32};
inline constexpr UnaryExecutionTuning kDefaultLogBFloat16ExecutionTuning{512 * 1024, 64 * 1024, 32};

inline std::int64_t UnaryBytesToElements(std::size_t bytes, std::size_t element_size) {
  element_size = std::max<std::size_t>(element_size, 1);
  const std::size_t elements = bytes / element_size + (bytes % element_size != 0 ? 1 : 0);
  return static_cast<std::int64_t>(
      std::min<std::size_t>(std::max<std::size_t>(elements, 1),
                            static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())));
}

template <typename T, typename Fn>
void ExecuteUnaryRanges(std::size_t count, const UnaryExecutionTuning &tuning, Fn fn) {
  if (count == 0) {
    return;
  }
  const std::int64_t total = static_cast<std::int64_t>(std::min<std::size_t>(
      count, static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())));
  if (tuning.parallel_threshold_bytes == 0 ||
      total < UnaryBytesToElements(tuning.parallel_threshold_bytes, sizeof(T))) {
    fn(std::int64_t{0}, total);
    return;
  }
  const std::int64_t max_participants =
      tuning.max_participants == 0
          ? std::numeric_limits<std::int64_t>::max()
          : static_cast<std::int64_t>(std::min<std::size_t>(
                tuning.max_participants,
                static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())));
  const ExecutionSchedule schedule{
      UnaryBytesToElements(tuning.parallel_threshold_bytes, sizeof(T)),
      UnaryBytesToElements(std::max<std::size_t>(tuning.target_block_bytes, 1), sizeof(T)),
      max_participants};
  ExecuteRanges(total, schedule, ExecutionSimdLanes<T>(), std::move(fn));
}

template <typename T, typename Fn>
void ExecuteCostedUnaryRanges(std::size_t count, const UnaryExecutionTuning &fallback,
                              double compute_cycles, Fn fn) {
  if (count == 0) {
    return;
  }
  const std::int64_t total = static_cast<std::int64_t>(std::min<std::size_t>(
      count, static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())));
  const std::int64_t max_participants =
      fallback.max_participants == 0
          ? std::numeric_limits<std::int64_t>::max()
          : static_cast<std::int64_t>(std::min<std::size_t>(
                fallback.max_participants,
                static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())));
  const ExecutionSchedule schedule{
      UnaryBytesToElements(fallback.parallel_threshold_bytes, sizeof(T)),
      UnaryBytesToElements(std::max<std::size_t>(fallback.target_block_bytes, 1), sizeof(T)),
      max_participants,
      static_cast<std::int64_t>(std::min<std::size_t>(
          fallback.preferred_participants,
          static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())))};
  ExecuteCostedRanges(total, ExecutionWorkCost{sizeof(T), sizeof(T), compute_cycles}, schedule,
                      ExecutionSimdLanes<T>(), std::move(fn));
}

} // namespace onnx_light_cpu
