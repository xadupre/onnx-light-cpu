// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/math_kernels.h"

#include <immintrin.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace onnx_light_cpu {
namespace {

constexpr double kActivationExpFloor64 = -708.0;
constexpr double kLog2e64 = 1.4426950408889634073599;
constexpr double kExpC1_64 = 6.93145751953125e-1;
constexpr double kExpC2_64 = 1.42860682030941723212e-6;
constexpr double kExpP0_64 = 1.26177193074810590878e-4;
constexpr double kExpP1_64 = 3.02994407707441961300e-2;
constexpr double kExpP2_64 = 9.99999999999999999910e-1;
constexpr double kExpQ0_64 = 3.00198505138664455042e-6;
constexpr double kExpQ1_64 = 2.52448340349684104192e-3;
constexpr double kExpQ2_64 = 2.27265548208155028766e-1;
constexpr double kExpQ3_64 = 2.00000000000000000005e0;

#if defined(_MSC_VER)
#define ONNX_LIGHT_CPU_FORCE_INLINE __forceinline
#else
#define ONNX_LIGHT_CPU_FORCE_INLINE inline __attribute__((always_inline))
#endif

ONNX_LIGHT_CPU_FORCE_INLINE __m256i TailMask(std::size_t count) {
  const __m256i lanes = _mm256_setr_epi64x(0, 1, 2, 3);
  return _mm256_cmpgt_epi64(_mm256_set1_epi64x(static_cast<long long>(count)), lanes);
}

ONNX_LIGHT_CPU_FORCE_INLINE double HorizontalMax(__m256d value) {
  const __m128d upper = _mm256_extractf128_pd(value, 1);
  __m128d reduced = _mm_max_pd(_mm256_castpd256_pd128(value), upper);
  reduced = _mm_max_sd(reduced, _mm_unpackhi_pd(reduced, reduced));
  return _mm_cvtsd_f64(reduced);
}

ONNX_LIGHT_CPU_FORCE_INLINE double HorizontalSum(__m256d value) {
  const __m128d upper = _mm256_extractf128_pd(value, 1);
  __m128d reduced = _mm_add_pd(_mm256_castpd256_pd128(value), upper);
  reduced = _mm_add_sd(reduced, _mm_unpackhi_pd(reduced, reduced));
  return _mm_cvtsd_f64(reduced);
}

ONNX_LIGHT_CPU_FORCE_INLINE __m256d ExpNegativeNormalPd256Fma(__m256d x) {
  const __m256d one = _mm256_set1_pd(1.0);
  const __m256d exponent =
      _mm256_floor_pd(_mm256_fmadd_pd(x, _mm256_set1_pd(kLog2e64), _mm256_set1_pd(0.5)));

  __m256d reduced = _mm256_fnmadd_pd(exponent, _mm256_set1_pd(kExpC1_64), x);
  reduced = _mm256_fnmadd_pd(exponent, _mm256_set1_pd(kExpC2_64), reduced);
  const __m256d squared = _mm256_mul_pd(reduced, reduced);

  __m256d numerator = _mm256_set1_pd(kExpP0_64);
  numerator = _mm256_fmadd_pd(numerator, squared, _mm256_set1_pd(kExpP1_64));
  numerator = _mm256_fmadd_pd(numerator, squared, _mm256_set1_pd(kExpP2_64));
  numerator = _mm256_mul_pd(numerator, reduced);

  __m256d denominator = _mm256_set1_pd(kExpQ0_64);
  denominator = _mm256_fmadd_pd(denominator, squared, _mm256_set1_pd(kExpQ1_64));
  denominator = _mm256_fmadd_pd(denominator, squared, _mm256_set1_pd(kExpQ2_64));
  denominator = _mm256_fmadd_pd(denominator, squared, _mm256_set1_pd(kExpQ3_64));

  __m256d result = _mm256_div_pd(numerator, _mm256_sub_pd(denominator, numerator));
  result = _mm256_add_pd(one, _mm256_add_pd(result, result));

  const __m128i exponent_int32 = _mm256_cvttpd_epi32(exponent);
  __m256i exponent_int64 = _mm256_cvtepi32_epi64(exponent_int32);
  exponent_int64 = _mm256_add_epi64(exponent_int64, _mm256_set1_epi64x(1023));
  exponent_int64 = _mm256_slli_epi64(exponent_int64, 52);
  return _mm256_mul_pd(result, _mm256_castsi256_pd(exponent_int64));
}

ONNX_LIGHT_CPU_FORCE_INLINE __m256d ExpNegativePd256Fma(__m256d x) {
  const __m256d exp_floor = _mm256_set1_pd(kActivationExpFloor64);
  // Both callers produce non-positive arguments; NaNs also take the scalar path.
  const __m256d fast_mask = _mm256_cmp_pd(x, exp_floor, _CMP_GE_OQ);
  if (_mm256_movemask_pd(fast_mask) == 0xf) {
    return ExpNegativeNormalPd256Fma(x);
  }

  alignas(32) std::array<double, 4> input_lanes;
  alignas(32) std::array<double, 4> output_lanes;
  _mm256_store_pd(input_lanes.data(), x);
  for (std::size_t lane = 0; lane < input_lanes.size(); ++lane) {
    output_lanes[lane] = std::exp(input_lanes[lane]);
  }
  return _mm256_load_pd(output_lanes.data());
}

ONNX_LIGHT_CPU_FORCE_INLINE __m256d SigmoidPd256Fma(__m256d value) {
  const __m256d magnitude = _mm256_andnot_pd(_mm256_set1_pd(-0.0), value);
  const __m256d exponent = ExpNegativePd256Fma(_mm256_sub_pd(_mm256_setzero_pd(), magnitude));
  const __m256d denominator = _mm256_add_pd(_mm256_set1_pd(1.0), exponent);
  const __m256d numerator = _mm256_blendv_pd(_mm256_set1_pd(1.0), exponent, value);
  return _mm256_div_pd(numerator, denominator);
}

template <bool normal_range>
void SoftmaxNormalizeRow(const double *input, double *output, std::size_t columns,
                         __m256d maximum) {
  const std::size_t vector_columns = columns - columns % 4;
  const __m256i tail_mask = TailMask(columns - vector_columns);
  const auto exponential = [](const __m256d value) {
    if constexpr (normal_range) {
      return ExpNegativeNormalPd256Fma(value);
    } else {
      return ExpNegativePd256Fma(value);
    }
  };

  __m256d sum0 = _mm256_setzero_pd();
  __m256d sum1 = _mm256_setzero_pd();
  std::size_t column = 0;
  for (; column + 8 <= vector_columns; column += 8) {
    const __m256d exponent0 = exponential(_mm256_sub_pd(_mm256_loadu_pd(input + column), maximum));
    const __m256d exponent1 =
        exponential(_mm256_sub_pd(_mm256_loadu_pd(input + column + 4), maximum));
    _mm256_storeu_pd(output + column, exponent0);
    _mm256_storeu_pd(output + column + 4, exponent1);
    sum0 = _mm256_add_pd(sum0, exponent0);
    sum1 = _mm256_add_pd(sum1, exponent1);
  }
  for (; column < vector_columns; column += 4) {
    const __m256d exponent = exponential(_mm256_sub_pd(_mm256_loadu_pd(input + column), maximum));
    _mm256_storeu_pd(output + column, exponent);
    sum0 = _mm256_add_pd(sum0, exponent);
  }
  if (column < columns) {
    const __m256d tail = _mm256_blendv_pd(maximum, _mm256_maskload_pd(input + column, tail_mask),
                                          _mm256_castsi256_pd(tail_mask));
    const __m256d exponent = exponential(_mm256_sub_pd(tail, maximum));
    _mm256_maskstore_pd(output + column, tail_mask, exponent);
    sum0 = _mm256_add_pd(sum0, _mm256_and_pd(exponent, _mm256_castsi256_pd(tail_mask)));
  }

  const __m256d inverse_sum = _mm256_set1_pd(1.0 / HorizontalSum(_mm256_add_pd(sum0, sum1)));
  for (column = 0; column < vector_columns; column += 4) {
    _mm256_storeu_pd(output + column, _mm256_mul_pd(_mm256_loadu_pd(output + column), inverse_sum));
  }
  if (column < columns) {
    _mm256_maskstore_pd(output + column, tail_mask,
                        _mm256_mul_pd(_mm256_maskload_pd(output + column, tail_mask), inverse_sum));
  }
}

void FillNaNRow(double *output, std::size_t columns) {
  const __m256d nan = _mm256_set1_pd(std::numeric_limits<double>::quiet_NaN());
  std::size_t column = 0;
  for (; column + 4 <= columns; column += 4) {
    _mm256_storeu_pd(output + column, nan);
  }
  if (column < columns) {
    _mm256_maskstore_pd(output + column, TailMask(columns - column), nan);
  }
}

#undef ONNX_LIGHT_CPU_FORCE_INLINE

} // namespace

void SigmoidFloat64_AVX2_FMA(const double *input, double *output, std::size_t count) {
  std::size_t index = 0;
  for (; index + 8 <= count; index += 8) {
    const __m256d result0 = SigmoidPd256Fma(_mm256_loadu_pd(input + index));
    const __m256d result1 = SigmoidPd256Fma(_mm256_loadu_pd(input + index + 4));
    _mm256_storeu_pd(output + index, result0);
    _mm256_storeu_pd(output + index + 4, result1);
  }
  for (; index + 4 <= count; index += 4) {
    _mm256_storeu_pd(output + index, SigmoidPd256Fma(_mm256_loadu_pd(input + index)));
  }
  if (index < count) {
    const __m256i mask = TailMask(count - index);
    _mm256_maskstore_pd(output + index, mask,
                        SigmoidPd256Fma(_mm256_maskload_pd(input + index, mask)));
  }
}

void SoftmaxFloat64_AVX2_FMA(const double *input, double *output, std::size_t rows,
                             std::size_t columns) {
  const __m256d negative_infinity = _mm256_set1_pd(-std::numeric_limits<double>::infinity());
  const __m256d positive_infinity = _mm256_set1_pd(std::numeric_limits<double>::infinity());
  const std::size_t vector_columns = columns - columns % 4;
  const __m256i tail_mask = TailMask(columns - vector_columns);

  for (std::size_t row = 0; row < rows; ++row) {
    const double *row_input = input + row * columns;
    double *row_output = output + row * columns;

    __m256d maximum_vector0 = negative_infinity;
    __m256d maximum_vector1 = negative_infinity;
    __m256d minimum_vector = positive_infinity;
    bool has_nan = false;
    std::size_t column = 0;
    for (; column + 8 <= vector_columns; column += 8) {
      const __m256d value0 = _mm256_loadu_pd(row_input + column);
      const __m256d value1 = _mm256_loadu_pd(row_input + column + 4);
      const __m256d nan0 = _mm256_cmp_pd(value0, value0, _CMP_UNORD_Q);
      const __m256d nan1 = _mm256_cmp_pd(value1, value1, _CMP_UNORD_Q);
      has_nan = has_nan || _mm256_movemask_pd(_mm256_or_pd(nan0, nan1)) != 0;
      const __m256d clean0 = _mm256_blendv_pd(value0, negative_infinity, nan0);
      const __m256d clean1 = _mm256_blendv_pd(value1, negative_infinity, nan1);
      maximum_vector0 = _mm256_max_pd(clean0, maximum_vector0);
      maximum_vector1 = _mm256_max_pd(clean1, maximum_vector1);
      minimum_vector = _mm256_min_pd(_mm256_min_pd(clean0, clean1), minimum_vector);
    }
    for (; column < vector_columns; column += 4) {
      const __m256d value = _mm256_loadu_pd(row_input + column);
      const __m256d nan = _mm256_cmp_pd(value, value, _CMP_UNORD_Q);
      has_nan = has_nan || _mm256_movemask_pd(nan) != 0;
      const __m256d clean = _mm256_blendv_pd(value, negative_infinity, nan);
      maximum_vector0 = _mm256_max_pd(clean, maximum_vector0);
      minimum_vector = _mm256_min_pd(clean, minimum_vector);
    }
    if (column < columns) {
      const __m256d tail =
          _mm256_blendv_pd(negative_infinity, _mm256_maskload_pd(row_input + column, tail_mask),
                           _mm256_castsi256_pd(tail_mask));
      const __m256d nan = _mm256_cmp_pd(tail, tail, _CMP_UNORD_Q);
      has_nan =
          has_nan || _mm256_movemask_pd(_mm256_and_pd(nan, _mm256_castsi256_pd(tail_mask))) != 0;
      const __m256d clean = _mm256_blendv_pd(tail, negative_infinity, nan);
      maximum_vector0 = _mm256_max_pd(clean, maximum_vector0);
      const __m256d minimum_tail =
          _mm256_blendv_pd(positive_infinity, clean, _mm256_castsi256_pd(tail_mask));
      minimum_vector = _mm256_min_pd(minimum_tail, minimum_vector);
    }

    const double maximum_scalar = HorizontalMax(_mm256_max_pd(maximum_vector0, maximum_vector1));
    if (has_nan || !std::isfinite(maximum_scalar)) {
      FillNaNRow(row_output, columns);
      continue;
    }

    const __m256d maximum = _mm256_set1_pd(maximum_scalar);
    const double minimum_scalar = HorizontalMax(_mm256_sub_pd(_mm256_setzero_pd(), minimum_vector));
    if (maximum_scalar + minimum_scalar <= -kActivationExpFloor64) {
      SoftmaxNormalizeRow<true>(row_input, row_output, columns, maximum);
    } else {
      SoftmaxNormalizeRow<false>(row_input, row_output, columns, maximum);
    }
  }
}

} // namespace onnx_light_cpu
