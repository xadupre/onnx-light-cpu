// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/com_microsoft/cdist.h"
#include "onnx_light_cpu/impl/com_microsoft/cdist_simd.h"

#include <cmath>
#include <cstddef>
#include <immintrin.h>

namespace onnx_light_cpu {

void CDistFloat32Rows_AVX512(const float *a, const float *b, float *c, std::size_t k, std::size_t n,
                             CDistMetric metric, std::size_t row_begin, std::size_t row_end) {
  for (std::size_t row = row_begin; row < row_end; ++row) {
    const float *a_row = a + row * n;
    float *c_row = c + row * k;
    std::size_t col = 0;
    for (; col + 8 <= k; col += 8) {
      __m512 sums[8] = {_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                        _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                        _mm512_setzero_ps(), _mm512_setzero_ps()};
      std::size_t feature = 0;
      for (; feature + 16 <= n; feature += 16) {
        const __m512 a_values = _mm512_loadu_ps(a_row + feature);
        for (std::size_t block_col = 0; block_col < 8; ++block_col) {
          const __m512 difference =
              _mm512_sub_ps(a_values, _mm512_loadu_ps(b + (col + block_col) * n + feature));
          sums[block_col] = _mm512_fmadd_ps(difference, difference, sums[block_col]);
        }
      }
      for (std::size_t block_col = 0; block_col < 8; ++block_col) {
        float sum_squares = _mm512_reduce_add_ps(sums[block_col]);
        const float *b_row = b + (col + block_col) * n;
        for (std::size_t tail = feature; tail < n; ++tail) {
          const float difference = a_row[tail] - b_row[tail];
          sum_squares += difference * difference;
        }
        c_row[col + block_col] =
            metric == CDistMetric::kEuclidean ? std::sqrt(sum_squares) : sum_squares;
      }
    }
    for (; col + 4 <= k; col += 4) {
      __m512 sums[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                        _mm512_setzero_ps()};
      std::size_t feature = 0;
      for (; feature + 16 <= n; feature += 16) {
        const __m512 a_values = _mm512_loadu_ps(a_row + feature);
        for (std::size_t block_col = 0; block_col < 4; ++block_col) {
          const __m512 difference =
              _mm512_sub_ps(a_values, _mm512_loadu_ps(b + (col + block_col) * n + feature));
          sums[block_col] = _mm512_fmadd_ps(difference, difference, sums[block_col]);
        }
      }
      for (std::size_t block_col = 0; block_col < 4; ++block_col) {
        float sum_squares = _mm512_reduce_add_ps(sums[block_col]);
        const float *b_row = b + (col + block_col) * n;
        for (std::size_t tail = feature; tail < n; ++tail) {
          const float difference = a_row[tail] - b_row[tail];
          sum_squares += difference * difference;
        }
        c_row[col + block_col] =
            metric == CDistMetric::kEuclidean ? std::sqrt(sum_squares) : sum_squares;
      }
    }
    for (; col < k; ++col) {
      const float *b_row = b + col * n;
      __m512 sum_vector = _mm512_setzero_ps();
      std::size_t feature = 0;
      for (; feature + 16 <= n; feature += 16) {
        const __m512 difference =
            _mm512_sub_ps(_mm512_loadu_ps(a_row + feature), _mm512_loadu_ps(b_row + feature));
        sum_vector = _mm512_fmadd_ps(difference, difference, sum_vector);
      }
      float sum_squares = _mm512_reduce_add_ps(sum_vector);
      for (; feature < n; ++feature) {
        const float difference = a_row[feature] - b_row[feature];
        sum_squares += difference * difference;
      }
      c_row[col] = metric == CDistMetric::kEuclidean ? std::sqrt(sum_squares) : sum_squares;
    }
  }
}

void CDistFloat64Rows_AVX512(const double *a, const double *b, double *c, std::size_t k,
                             std::size_t n, CDistMetric metric, std::size_t row_begin,
                             std::size_t row_end) {
  for (std::size_t row = row_begin; row < row_end; ++row) {
    const double *a_row = a + row * n;
    double *c_row = c + row * k;
    std::size_t col = 0;
    for (; col + 4 <= k; col += 4) {
      __m512d sums[4] = {_mm512_setzero_pd(), _mm512_setzero_pd(), _mm512_setzero_pd(),
                         _mm512_setzero_pd()};
      std::size_t feature = 0;
      for (; feature + 8 <= n; feature += 8) {
        const __m512d a_values = _mm512_loadu_pd(a_row + feature);
        for (std::size_t block_col = 0; block_col < 4; ++block_col) {
          const __m512d difference =
              _mm512_sub_pd(a_values, _mm512_loadu_pd(b + (col + block_col) * n + feature));
          sums[block_col] = _mm512_fmadd_pd(difference, difference, sums[block_col]);
        }
      }
      for (std::size_t block_col = 0; block_col < 4; ++block_col) {
        double sum_squares = _mm512_reduce_add_pd(sums[block_col]);
        const double *b_row = b + (col + block_col) * n;
        for (std::size_t tail = feature; tail < n; ++tail) {
          const double difference = a_row[tail] - b_row[tail];
          sum_squares += difference * difference;
        }
        c_row[col + block_col] =
            metric == CDistMetric::kEuclidean ? std::sqrt(sum_squares) : sum_squares;
      }
    }
    for (; col < k; ++col) {
      const double *b_row = b + col * n;
      __m512d sum_vector = _mm512_setzero_pd();
      std::size_t feature = 0;
      for (; feature + 8 <= n; feature += 8) {
        const __m512d difference =
            _mm512_sub_pd(_mm512_loadu_pd(a_row + feature), _mm512_loadu_pd(b_row + feature));
        sum_vector = _mm512_fmadd_pd(difference, difference, sum_vector);
      }
      double sum_squares = _mm512_reduce_add_pd(sum_vector);
      for (; feature < n; ++feature) {
        const double difference = a_row[feature] - b_row[feature];
        sum_squares += difference * difference;
      }
      c_row[col] = metric == CDistMetric::kEuclidean ? std::sqrt(sum_squares) : sum_squares;
    }
  }
}

void CDistFloat32PackedRows_AVX512(const float *a, const float *b, float *c, std::size_t k,
                                   std::size_t n, CDistMetric metric, std::size_t row_begin,
                                   std::size_t row_end) {
  if (CDistPackedRows<float, __m512, 16>(
          a, b, c, k, n, metric, row_begin, row_end, _mm512_setzero_ps(),
          [](const float *p) { return _mm512_loadu_ps(p); },
          [](float *p, __m512 v) { _mm512_storeu_ps(p, v); },
          [](float v) { return _mm512_set1_ps(v); },
          [](__m512 x, __m512 y) { return _mm512_sub_ps(x, y); },
          [](__m512 x, __m512 y, __m512 sum) { return _mm512_fmadd_ps(x, y, sum); },
          [](__m512 v) { return _mm512_sqrt_ps(v); })) {
    return;
  }
  CDistFloat32Rows_AVX512(a, b, c, k, n, metric, row_begin, row_end);
}

void CDistFloat64PackedRows_AVX512(const double *a, const double *b, double *c, std::size_t k,
                                   std::size_t n, CDistMetric metric, std::size_t row_begin,
                                   std::size_t row_end) {
  if (CDistPackedRows<double, __m512d, 8>(
          a, b, c, k, n, metric, row_begin, row_end, _mm512_setzero_pd(),
          [](const double *p) { return _mm512_loadu_pd(p); },
          [](double *p, __m512d v) { _mm512_storeu_pd(p, v); },
          [](double v) { return _mm512_set1_pd(v); },
          [](__m512d x, __m512d y) { return _mm512_sub_pd(x, y); },
          [](__m512d x, __m512d y, __m512d sum) { return _mm512_fmadd_pd(x, y, sum); },
          [](__m512d v) { return _mm512_sqrt_pd(v); })) {
    return;
  }
  CDistFloat64Rows_AVX512(a, b, c, k, n, metric, row_begin, row_end);
}

} // namespace onnx_light_cpu
