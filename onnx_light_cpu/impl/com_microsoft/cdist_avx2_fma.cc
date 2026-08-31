// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/com_microsoft/cdist.h"

#include <cmath>
#include <cstddef>
#include <immintrin.h>
#include <type_traits>

namespace onnx_light_cpu {
namespace {

float Reduce(__m256 value) {
  const __m128 low = _mm256_castps256_ps128(value);
  const __m128 high = _mm256_extractf128_ps(value, 1);
  __m128 sum = _mm_add_ps(low, high);
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  return _mm_cvtss_f32(sum);
}

double Reduce(__m256d value) {
  const __m128d low = _mm256_castpd256_pd128(value);
  const __m128d high = _mm256_extractf128_pd(value, 1);
  const __m128d sum = _mm_add_pd(low, high);
  return _mm_cvtsd_f64(_mm_hadd_pd(sum, sum));
}

template <typename T, typename Vector, std::size_t Lanes, typename Load, typename Sub,
          typename MultiplyAdd>
void CDistRows(const T *a, const T *b, T *c, std::size_t k, std::size_t n, CDistMetric metric,
               std::size_t row_begin, std::size_t row_end, Vector zero, Load load, Sub subtract,
               MultiplyAdd multiply_add) {
  for (std::size_t row = row_begin; row < row_end; ++row) {
    const T *a_row = a + row * n;
    T *c_row = c + row * k;
    std::size_t col = 0;
    if constexpr (std::is_same_v<T, float>) {
      for (; col + 8 <= k; col += 8) {
        Vector sums[8] = {zero, zero, zero, zero, zero, zero, zero, zero};
        std::size_t feature = 0;
        for (; feature + Lanes <= n; feature += Lanes) {
          const Vector a_values = load(a_row + feature);
          for (std::size_t block_col = 0; block_col < 8; ++block_col) {
            const Vector difference = subtract(a_values, load(b + (col + block_col) * n + feature));
            sums[block_col] = multiply_add(difference, difference, sums[block_col]);
          }
        }
        for (std::size_t block_col = 0; block_col < 8; ++block_col) {
          T sum_squares = Reduce(sums[block_col]);
          const T *b_row = b + (col + block_col) * n;
          for (std::size_t tail = feature; tail < n; ++tail) {
            const T difference = a_row[tail] - b_row[tail];
            sum_squares += difference * difference;
          }
          c_row[col + block_col] =
              metric == CDistMetric::kEuclidean ? std::sqrt(sum_squares) : sum_squares;
        }
      }
    }
    for (; col + 4 <= k; col += 4) {
      Vector sums[4] = {zero, zero, zero, zero};
      std::size_t feature = 0;
      for (; feature + Lanes <= n; feature += Lanes) {
        const Vector a_values = load(a_row + feature);
        for (std::size_t block_col = 0; block_col < 4; ++block_col) {
          const Vector difference = subtract(a_values, load(b + (col + block_col) * n + feature));
          sums[block_col] = multiply_add(difference, difference, sums[block_col]);
        }
      }
      for (std::size_t block_col = 0; block_col < 4; ++block_col) {
        T sum_squares = Reduce(sums[block_col]);
        const T *b_row = b + (col + block_col) * n;
        for (std::size_t tail = feature; tail < n; ++tail) {
          const T difference = a_row[tail] - b_row[tail];
          sum_squares += difference * difference;
        }
        c_row[col + block_col] =
            metric == CDistMetric::kEuclidean ? std::sqrt(sum_squares) : sum_squares;
      }
    }
    for (; col < k; ++col) {
      const T *b_row = b + col * n;
      Vector sum_vector = zero;
      std::size_t feature = 0;
      for (; feature + Lanes <= n; feature += Lanes) {
        const Vector difference = subtract(load(a_row + feature), load(b_row + feature));
        sum_vector = multiply_add(difference, difference, sum_vector);
      }
      T sum_squares = Reduce(sum_vector);
      for (; feature < n; ++feature) {
        const T difference = a_row[feature] - b_row[feature];
        sum_squares += difference * difference;
      }
      c_row[col] = metric == CDistMetric::kEuclidean ? std::sqrt(sum_squares) : sum_squares;
    }
  }
}

} // namespace

void CDistFloat32Rows_AVX2_FMA(const float *a, const float *b, float *c, std::size_t k,
                               std::size_t n, CDistMetric metric, std::size_t row_begin,
                               std::size_t row_end) {
  CDistRows<float, __m256, 8>(
      a, b, c, k, n, metric, row_begin, row_end, _mm256_setzero_ps(),
      [](const float *value) { return _mm256_loadu_ps(value); },
      [](__m256 left, __m256 right) { return _mm256_sub_ps(left, right); },
      [](__m256 left, __m256 right, __m256 sum) { return _mm256_fmadd_ps(left, right, sum); });
}

void CDistFloat64Rows_AVX2_FMA(const double *a, const double *b, double *c, std::size_t k,
                               std::size_t n, CDistMetric metric, std::size_t row_begin,
                               std::size_t row_end) {
  CDistRows<double, __m256d, 4>(
      a, b, c, k, n, metric, row_begin, row_end, _mm256_setzero_pd(),
      [](const double *value) { return _mm256_loadu_pd(value); },
      [](__m256d left, __m256d right) { return _mm256_sub_pd(left, right); },
      [](__m256d left, __m256d right, __m256d sum) { return _mm256_fmadd_pd(left, right, sum); });
}

} // namespace onnx_light_cpu
