// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/com_microsoft/bias_gelu.h"

#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/impl/math/unary_execution_tuning.h"
#include "onnx_light_cpu/impl/simd_level.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <type_traits>
#include <vector>

namespace onnx_light_cpu {

#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
void BiasGeluFloat32_AVX2_FMA(const float *a, const float *bias, float *output, std::size_t count);
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
void BiasGeluFloat32_AVX512(const float *a, const float *bias, float *output, std::size_t count);
#endif

namespace {

template <typename T> T GeluExact(T z) {
  constexpr T kInvSqrtTwo = static_cast<T>(0.70710678118654752440084436210485L);
  return static_cast<T>(0.5) * z * (static_cast<T>(1) + std::erf(z * kInvSqrtTwo));
}

float GeluFloat32Approx(float z) {
  if (z == 0.0f) {
    return z;
  }
  if (z >= 6.0f) {
    return z;
  }
  if (z <= -6.0f) {
    return z == -std::numeric_limits<float>::infinity() ? std::numeric_limits<float>::quiet_NaN()
                                                        : 0.0f;
  }

  // On [-6, 6], approximate the even part of GELU as a degree-14 Chebyshev
  // polynomial in z^2. The maximum float32 error is 9e-6, while the
  // branch-free recurrence lets compilers vectorize the inner broadcast loop.
  constexpr float coefficients[] = {
      1.8827368f,       1.32283127f,     -0.29206121f,   0.131107315f,    -0.0683164895f,
      0.0358457267f,    -0.0179936364f,  0.00847151037f, -0.00371518102f, 0.0015147439f,
      -0.000575297687f, 0.000203221469f, -6.7434863e-5f, 2.01439125e-5f,  -5.92767856e-6f,
  };
  const float x = z * z / 18.0f - 1.0f;
  float next = 0.0f;
  float next_next = 0.0f;
  for (std::size_t index = std::size(coefficients) - 1; index > 0; --index) {
    const float current = 2.0f * x * next - next_next + coefficients[index];
    next_next = next;
    next = current;
  }
  return 0.5f * z + (x * next - next_next + coefficients[0]);
}

using BiasGeluFloat32RangeFn = void (*)(const float *, const float *, float *, std::size_t);

void BiasGeluFloat32Scalar(const float *a, const float *bias, float *output, std::size_t count) {
  for (std::size_t column = 0; column < count; ++column) {
    output[column] = GeluFloat32Approx(a[column] + bias[column]);
  }
}

BiasGeluFloat32RangeFn SelectBiasGeluFloat32Range() {
  static const BiasGeluFloat32RangeFn selected = []() -> BiasGeluFloat32RangeFn {
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
    if (DetectSimdLevel() >= SimdLevel::kAVX512) {
      return &BiasGeluFloat32_AVX512;
    }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
    if (DetectSimdLevel() >= SimdLevel::kAVX2 && CpuSupportsFma()) {
      return &BiasGeluFloat32_AVX2_FMA;
    }
#endif
    return &BiasGeluFloat32Scalar;
  }();
  return selected;
}

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
  const BiasGeluFloat32RangeFn float32_range =
      std::is_same_v<T, float> ? SelectBiasGeluFloat32Range() : nullptr;
  DispatchRows(outer, row_bytes, tuning, [=](std::int64_t begin, std::int64_t end) {
    for (std::int64_t row = begin; row < end; ++row) {
      const std::size_t offset = static_cast<std::size_t>(row) * inner;
      if constexpr (std::is_same_v<T, float>) {
        float32_range(a + offset, bias, output + offset, inner);
      } else {
        for (std::size_t column = 0; column < inner; ++column) {
          output[offset + column] = GeluExact<T>(a[offset + column] + bias[column]);
        }
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
  std::vector<float> bias_float(inner);
  if constexpr (BFloat16) {
    detail::ConvertBFloat16ToFloat32(bias, bias_float.data(), inner);
  } else {
    detail::ConvertFloat16ToFloat32(bias, bias_float.data(), inner);
  }
  const std::size_t row_bytes = inner * sizeof(std::uint16_t);
  const BiasGeluFloat32RangeFn float32_range = SelectBiasGeluFloat32Range();
  const float *bias_values = bias_float.data();
  DispatchRows(outer, row_bytes, tuning, [=](std::int64_t begin, std::int64_t end) {
    constexpr std::size_t block_size = 256;
    std::array<float, block_size> a_buffer{};
    std::array<float, block_size> output_buffer{};
    for (std::int64_t row = begin; row < end; ++row) {
      const std::size_t offset = static_cast<std::size_t>(row) * inner;
      for (std::size_t column = 0; column < inner; column += block_size) {
        const std::size_t block = std::min(block_size, inner - column);
        if constexpr (BFloat16) {
          detail::ConvertBFloat16ToFloat32(a + offset + column, a_buffer.data(), block);
        } else {
          detail::ConvertFloat16ToFloat32(a + offset + column, a_buffer.data(), block);
        }
        float32_range(a_buffer.data(), bias_values + column, output_buffer.data(), block);
        if constexpr (BFloat16) {
          detail::ConvertFloat32ToBFloat16(output_buffer.data(), output + offset + column, block);
        } else {
          detail::ConvertFloat32ToFloat16(output_buffer.data(), output + offset + column, block);
        }
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
