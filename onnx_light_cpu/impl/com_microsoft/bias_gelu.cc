// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/com_microsoft/bias_gelu.h"

#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/impl/math/unary_execution_tuning.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace onnx_light_cpu {
namespace {

template <typename T> T GeluExact(T z) {
  constexpr T kInvSqrtTwo = static_cast<T>(0.70710678118654752440084436210485L);
  return static_cast<T>(0.5) * z * (static_cast<T>(1) + std::erf(z * kInvSqrtTwo));
}

float GeluExactFloat(float z) { return GeluExact<float>(z); }

// Builds the ``ExecutionSchedule`` shared by every dtype: thresholds are
// expressed in bytes of one broadcast row (``inner`` elements, matching the
// bias broadcast boundary) via ``UnaryBytesToElements``.
ExecutionSchedule MakeRowSchedule(const BiasGeluExecutionTuning &tuning, std::size_t row_bytes) {
  const std::int64_t max_participants =
      tuning.max_participants == 0
          ? std::numeric_limits<std::int64_t>::max()
          : static_cast<std::int64_t>(std::min<std::size_t>(
                tuning.max_participants,
                static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())));
  return ExecutionSchedule{
      UnaryBytesToElements(tuning.parallel_threshold_bytes, row_bytes),
      UnaryBytesToElements(std::max<std::size_t>(tuning.target_block_bytes, 1), row_bytes),
      max_participants,
      static_cast<std::int64_t>(std::min<std::size_t>(
          tuning.preferred_participants,
          static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())))};
}

template <typename Fn>
void DispatchRows(std::size_t outer, std::size_t row_bytes, const BiasGeluExecutionTuning &tuning,
                  Fn fn) {
  if (outer == 0) {
    return;
  }
  const std::int64_t total = static_cast<std::int64_t>(outer);
  if (tuning.parallel_threshold_bytes == 0) {
    fn(0, total);
    return;
  }
  const ExecutionSchedule schedule = MakeRowSchedule(tuning, row_bytes);
  if (tuning.use_cost_model) {
    // Every row reads its ``A`` slice and the shared ``bias`` vector (the
    // latter is small and stays cache-resident across rows, so only the
    // per-row ``A``/output traffic is charged) and applies one ``erf`` per
    // element.
    const ExecutionWorkCost cost{static_cast<double>(row_bytes), static_cast<double>(row_bytes),
                                 static_cast<double>(row_bytes)};
    ExecuteCostedRanges(total, cost, schedule, std::int64_t{1}, std::move(fn));
  } else {
    ExecuteRanges(total, schedule, std::int64_t{1}, std::move(fn));
  }
}

template <typename T>
void BiasGeluDispatch(const T *a, const T *bias, T *output, std::size_t outer, std::size_t inner,
                      const BiasGeluExecutionTuning &tuning) {
  if (outer == 0 || inner == 0) {
    return;
  }
  const std::size_t row_bytes = inner * sizeof(T);
  DispatchRows(outer, row_bytes, tuning, [=](std::int64_t begin, std::int64_t end) {
    for (std::int64_t row = begin; row < end; ++row) {
      const std::size_t offset = static_cast<std::size_t>(row) * inner;
      for (std::size_t column = 0; column < inner; ++column) {
        output[offset + column] = GeluExact<T>(a[offset + column] + bias[column]);
      }
    }
  });
}

template <bool BFloat16>
void BiasGeluHalfDispatch(const std::uint16_t *a, const std::uint16_t *bias, std::uint16_t *output,
                          std::size_t outer, std::size_t inner,
                          const BiasGeluExecutionTuning &tuning) {
  if (outer == 0 || inner == 0) {
    return;
  }
  const std::size_t row_bytes = inner * sizeof(std::uint16_t);
  DispatchRows(outer, row_bytes, tuning, [=](std::int64_t begin, std::int64_t end) {
    for (std::int64_t row = begin; row < end; ++row) {
      const std::size_t offset = static_cast<std::size_t>(row) * inner;
      for (std::size_t column = 0; column < inner; ++column) {
        const float a_value = BFloat16 ? detail::Bfloat16BitsToFloat(a[offset + column])
                                       : detail::Float16BitsToFloat(a[offset + column]);
        const float bias_value = BFloat16 ? detail::Bfloat16BitsToFloat(bias[column])
                                          : detail::Float16BitsToFloat(bias[column]);
        const float value = GeluExactFloat(a_value + bias_value);
        output[offset + column] =
            BFloat16 ? detail::FloatToBFloat16Bits(value) : detail::FloatToFloat16Bits(value);
      }
    }
  });
}

} // namespace

void BiasGeluFloat32(const float *a, const float *bias, float *output, std::size_t outer,
                     std::size_t inner) {
  BiasGeluFloat32WithTuning(a, bias, output, outer, inner, kDefaultBiasGeluFloat32ExecutionTuning);
}

void BiasGeluFloat32WithTuning(const float *a, const float *bias, float *output, std::size_t outer,
                               std::size_t inner, const BiasGeluExecutionTuning &tuning) {
  BiasGeluDispatch<float>(a, bias, output, outer, inner, tuning);
}

void BiasGeluFloat64(const double *a, const double *bias, double *output, std::size_t outer,
                     std::size_t inner) {
  BiasGeluFloat64WithTuning(a, bias, output, outer, inner, kDefaultBiasGeluFloat64ExecutionTuning);
}

void BiasGeluFloat64WithTuning(const double *a, const double *bias, double *output,
                               std::size_t outer, std::size_t inner,
                               const BiasGeluExecutionTuning &tuning) {
  BiasGeluDispatch<double>(a, bias, output, outer, inner, tuning);
}

void BiasGeluFloat16(const std::uint16_t *a, const std::uint16_t *bias, std::uint16_t *output,
                     std::size_t outer, std::size_t inner) {
  BiasGeluFloat16WithTuning(a, bias, output, outer, inner, kDefaultBiasGeluHalfExecutionTuning);
}

void BiasGeluFloat16WithTuning(const std::uint16_t *a, const std::uint16_t *bias,
                               std::uint16_t *output, std::size_t outer, std::size_t inner,
                               const BiasGeluExecutionTuning &tuning) {
  BiasGeluHalfDispatch<false>(a, bias, output, outer, inner, tuning);
}

void BiasGeluBFloat16(const std::uint16_t *a, const std::uint16_t *bias, std::uint16_t *output,
                      std::size_t outer, std::size_t inner) {
  BiasGeluBFloat16WithTuning(a, bias, output, outer, inner, kDefaultBiasGeluHalfExecutionTuning);
}

void BiasGeluBFloat16WithTuning(const std::uint16_t *a, const std::uint16_t *bias,
                                std::uint16_t *output, std::size_t outer, std::size_t inner,
                                const BiasGeluExecutionTuning &tuning) {
  BiasGeluHalfDispatch<true>(a, bias, output, outer, inner, tuning);
}

} // namespace onnx_light_cpu
