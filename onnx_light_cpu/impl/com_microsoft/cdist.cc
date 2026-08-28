// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/com_microsoft/cdist.h"

#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/unary_execution_tuning.h"
#include "onnx_light_cpu/impl/simd_level.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace onnx_light_cpu {

#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
void CDistFloat32Rows_AVX2_FMA(const float *a, const float *b, float *c, std::size_t k,
                               std::size_t n, CDistMetric metric, std::size_t row_begin,
                               std::size_t row_end);
void CDistFloat64Rows_AVX2_FMA(const double *a, const double *b, double *c, std::size_t k,
                               std::size_t n, CDistMetric metric, std::size_t row_begin,
                               std::size_t row_end);
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
void CDistFloat32Rows_AVX512(const float *a, const float *b, float *c, std::size_t k, std::size_t n,
                             CDistMetric metric, std::size_t row_begin, std::size_t row_end);
void CDistFloat64Rows_AVX512(const double *a, const double *b, double *c, std::size_t k,
                             std::size_t n, CDistMetric metric, std::size_t row_begin,
                             std::size_t row_end);
#endif

namespace {

// Computes one contiguous slice of output rows ``[row_begin, row_end)``. Each
// output row reuses the same ``A`` row across every ``B`` row it is paired
// with (kept in the innermost loop's addressing, not re-fetched), and both
// ``A`` and ``B`` rows are read with unit stride, so the scalar loop nest is
// already cache-friendly without any explicit blocking or SIMD.
template <typename T>
void CDistRowsScalar(const T *a, const T *b, T *c, std::size_t k, std::size_t n, CDistMetric metric,
                     std::size_t row_begin, std::size_t row_end) {
  for (std::size_t row = row_begin; row < row_end; ++row) {
    const T *a_row = a + row * n;
    T *c_row = c + row * k;
    for (std::size_t col = 0; col < k; ++col) {
      const T *b_row = b + col * n;
      T sum_squares = T(0);
      for (std::size_t feature = 0; feature < n; ++feature) {
        const T difference = a_row[feature] - b_row[feature];
        sum_squares += difference * difference;
      }
      c_row[col] = metric == CDistMetric::kEuclidean ? std::sqrt(sum_squares) : sum_squares;
    }
  }
}

template <typename T>
using CDistRowsFn = void (*)(const T *, const T *, T *, std::size_t, std::size_t, CDistMetric,
                             std::size_t, std::size_t);

template <typename T> CDistRowsFn<T> SelectCDistRows(std::size_t n) {
#if defined(ONNX_LIGHT_CPU_HAVE_AVX512) || defined(ONNX_LIGHT_CPU_HAVE_AVX2_FMA)
  static const SimdLevel simd_level = DetectSimdLevel();
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
  constexpr std::size_t avx512_lanes = std::is_same_v<T, float> ? 16 : 8;
  if (simd_level >= SimdLevel::kAVX512 && n >= avx512_lanes) {
    if constexpr (std::is_same_v<T, float>) {
      return &CDistFloat32Rows_AVX512;
    } else {
      return &CDistFloat64Rows_AVX512;
    }
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
  static const bool supports_fma = CpuSupportsFma();
  constexpr std::size_t avx2_lanes = std::is_same_v<T, float> ? 8 : 4;
  if (simd_level >= SimdLevel::kAVX2 && supports_fma && n >= avx2_lanes) {
    if constexpr (std::is_same_v<T, float>) {
      return &CDistFloat32Rows_AVX2_FMA;
    } else {
      return &CDistFloat64Rows_AVX2_FMA;
    }
  }
#endif
  return &CDistRowsScalar<T>;
}

template <typename T>
void CDistDispatch(const T *a, const T *b, T *c, std::size_t m, std::size_t k, std::size_t n,
                   CDistMetric metric, const CDistExecutionTuning &tuning) {
  if (m == 0) {
    return;
  }
  const std::size_t row_bytes = std::max<std::size_t>(n, 1) * sizeof(T);
  const std::int64_t total = static_cast<std::int64_t>(m);
  const CDistRowsFn<T> rows = SelectCDistRows<T>(n);
  auto execute = [=](std::int64_t begin, std::int64_t end) {
    rows(a, b, c, k, n, metric, static_cast<std::size_t>(begin), static_cast<std::size_t>(end));
  };
  if (tuning.parallel_threshold_bytes == 0) {
    execute(0, total);
    return;
  }
  const std::int64_t max_participants =
      tuning.max_participants == 0
          ? std::numeric_limits<std::int64_t>::max()
          : static_cast<std::int64_t>(std::min<std::size_t>(
                tuning.max_participants,
                static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())));
  const ExecutionSchedule schedule{
      UnaryBytesToElements(tuning.parallel_threshold_bytes, row_bytes),
      UnaryBytesToElements(std::max<std::size_t>(tuning.target_block_bytes, 1), row_bytes),
      max_participants,
      static_cast<std::int64_t>(std::min<std::size_t>(
          tuning.preferred_participants,
          static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())))};
  if (tuning.use_cost_model) {
    // One output row reads its ``A`` row once but rereads the whole ``B``
    // matrix (``k * n`` elements) and performs ``k * n`` subtract+multiply+add
    // triples; ``B`` typically stays resident in cache across rows, so only
    // the ``A`` row and the ``k`` outputs are charged as this row's own
    // traffic while the compute cost captures the full ``k * n`` reduction.
    const double compute_cycles = static_cast<double>(k) * static_cast<double>(n);
    const ExecutionWorkCost cost{static_cast<double>(row_bytes), static_cast<double>(k * sizeof(T)),
                                 compute_cycles};
    ExecuteCostedRanges(total, cost, schedule, std::int64_t{1}, std::move(execute));
  } else {
    ExecuteRanges(total, schedule, std::int64_t{1}, std::move(execute));
  }
}

} // namespace

void CDistFloat32(const float *a, const float *b, float *c, std::size_t m, std::size_t k,
                  std::size_t n, CDistMetric metric) {
  CDistFloat32WithTuning(a, b, c, m, k, n, metric, kDefaultCDistFloat32ExecutionTuning);
}

void CDistFloat32WithTuning(const float *a, const float *b, float *c, std::size_t m, std::size_t k,
                            std::size_t n, CDistMetric metric, const CDistExecutionTuning &tuning) {
  CDistDispatch<float>(a, b, c, m, k, n, metric, tuning);
}

void CDistFloat64(const double *a, const double *b, double *c, std::size_t m, std::size_t k,
                  std::size_t n, CDistMetric metric) {
  CDistFloat64WithTuning(a, b, c, m, k, n, metric, kDefaultCDistFloat64ExecutionTuning);
}

void CDistFloat64WithTuning(const double *a, const double *b, double *c, std::size_t m,
                            std::size_t k, std::size_t n, CDistMetric metric,
                            const CDistExecutionTuning &tuning) {
  CDistDispatch<double>(a, b, c, m, k, n, metric, tuning);
}

} // namespace onnx_light_cpu
