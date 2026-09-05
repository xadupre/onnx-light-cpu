// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/math_kernels.h"

#include <immintrin.h>

#include <cmath>
#include <cstddef>
#include <limits>

namespace onnx_light_cpu {
namespace {

constexpr float kExpHi = 88.7762626647950f;
constexpr float kExpLo = -103.9720840454f;
constexpr float kLog2ef = 1.44269504088896341f;
constexpr float kExpLog2High = -6.93145752e-1f;
constexpr float kExpLog2Low = -1.42860677e-6f;
constexpr float kExpP0 = 0x1.694000p-10f;
constexpr float kExpP1 = 0x1.125edcp-7f;
constexpr float kExpP2 = 0x1.555b5ap-5f;
constexpr float kExpP3 = 0x1.555450p-3f;
constexpr float kExpP4 = 0x1.fffff6p-2f;

constexpr float kSmallestNormal = 1.17549435e-38f;
constexpr float kSubnormalScale = 8388608.0f;
constexpr float kSqrtHalf = 0.707106781186547524f;
constexpr float kLogP0 = 7.0376836292e-2f;
constexpr float kLogP1 = -1.1514610310e-1f;
constexpr float kLogP2 = 1.1676998740e-1f;
constexpr float kLogP3 = -1.2420140846e-1f;
constexpr float kLogP4 = 1.4249322787e-1f;
constexpr float kLogP5 = -1.6668057665e-1f;
constexpr float kLogP6 = 2.0000714765e-1f;
constexpr float kLogP7 = -2.4999993993e-1f;
constexpr float kLogP8 = 3.3333331174e-1f;
constexpr float kLogQ1 = -2.12194440e-4f;
constexpr float kLogQ2 = 0.693359375f;

inline __m256 Select(__m256 mask, __m256 selected, __m256 fallback) {
  return _mm256_blendv_ps(fallback, selected, mask);
}

__m256 ExpPs256Fma(__m256 x) {
  // Keeping the rounded exponent in its biased floating-point representation
  // avoids a float-to-integer conversion on the polynomial dependency chain.
  __m256 reduced = _mm256_min_ps(_mm256_set1_ps(kExpHi), _mm256_max_ps(_mm256_set1_ps(kExpLo), x));
  const __m256 magic = _mm256_set1_ps(12582912.0f);
  const __m256 biased = _mm256_fmadd_ps(reduced, _mm256_set1_ps(kLog2ef), magic);
  const __m256 exponent = _mm256_sub_ps(biased, magic);
  reduced = _mm256_fmadd_ps(exponent, _mm256_set1_ps(kExpLog2High), reduced);
  reduced = _mm256_fmadd_ps(exponent, _mm256_set1_ps(kExpLog2Low), reduced);

  __m256i exponent_bits = _mm256_slli_epi32(_mm256_castps_si256(biased), 23);
  __m256i normal_bits =
      _mm256_min_epi32(exponent_bits, _mm256_set1_epi32(static_cast<int>(0x3f800000u)));
  normal_bits = _mm256_max_epi32(normal_bits, _mm256_set1_epi32(static_cast<int>(0xc1000000u)));
  __m256i overflow_bits = _mm256_sub_epi32(exponent_bits, normal_bits);
  normal_bits = _mm256_add_epi32(normal_bits, _mm256_set1_epi32(0x3f800000));
  overflow_bits = _mm256_add_epi32(overflow_bits, _mm256_set1_epi32(0x3f800000));
  const __m256 normal = _mm256_castsi256_ps(normal_bits);
  const __m256 overflow = _mm256_castsi256_ps(overflow_bits);

  __m256 polynomial = _mm256_set1_ps(kExpP0);
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(kExpP1));
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(kExpP2));
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(kExpP3));
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(kExpP4));
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(1.0f));
  const __m256 scaled_reduced = _mm256_mul_ps(reduced, overflow);
  polynomial = _mm256_fmadd_ps(polynomial, scaled_reduced, overflow);
  return _mm256_mul_ps(polynomial, normal);
}

__m256 ExpNegativePs256Fma(__m256 x) {
  constexpr float kLogSmallestNormal = -87.3365447505531f;
  const __m256 in_fast_range =
      _mm256_and_ps(_mm256_cmp_ps(x, _mm256_set1_ps(kLogSmallestNormal), _CMP_GE_OQ),
                    _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OQ));
  if (_mm256_movemask_ps(in_fast_range) != 0xff) {
    return ExpPs256Fma(x);
  }

  const __m256 magic = _mm256_set1_ps(12582912.0f);
  const __m256 biased = _mm256_fmadd_ps(x, _mm256_set1_ps(kLog2ef), magic);
  const __m256 exponent = _mm256_sub_ps(biased, magic);
  __m256 reduced = _mm256_fmadd_ps(exponent, _mm256_set1_ps(kExpLog2High), x);
  reduced = _mm256_fmadd_ps(exponent, _mm256_set1_ps(kExpLog2Low), reduced);

  const __m256 scale = _mm256_castsi256_ps(_mm256_add_epi32(
      _mm256_slli_epi32(_mm256_castps_si256(biased), 23), _mm256_set1_epi32(0x3f800000)));
  __m256 polynomial = _mm256_set1_ps(kExpP0);
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(kExpP1));
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(kExpP2));
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(kExpP3));
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(kExpP4));
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(1.0f));
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(1.0f));
  return _mm256_mul_ps(polynomial, scale);
}

#if defined(_MSC_VER)
#define ONNX_LIGHT_CPU_FORCE_INLINE __forceinline
#else
#define ONNX_LIGHT_CPU_FORCE_INLINE inline __attribute__((always_inline))
#endif

ONNX_LIGHT_CPU_FORCE_INLINE __m256 LogPs256Fma(__m256 x) {
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 zero = _mm256_setzero_ps();
  const __m256 positive_infinity = _mm256_set1_ps(std::numeric_limits<float>::infinity());
  const __m256 is_nan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
  const __m256 is_negative = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);
  const __m256 is_zero = _mm256_cmp_ps(x, zero, _CMP_EQ_OQ);
  const __m256 is_infinite = _mm256_cmp_ps(x, positive_infinity, _CMP_EQ_OQ);

  const __m256 is_subnormal =
      _mm256_and_ps(_mm256_cmp_ps(x, zero, _CMP_GT_OQ),
                    _mm256_cmp_ps(x, _mm256_set1_ps(kSmallestNormal), _CMP_LT_OQ));
  const __m256 scaled = _mm256_mul_ps(x, _mm256_set1_ps(kSubnormalScale));
  __m256 reduced = Select(is_subnormal, scaled, x);
  __m256i exponent_int = _mm256_srli_epi32(_mm256_castps_si256(reduced), 23);
  reduced = _mm256_and_ps(reduced,
                          _mm256_castsi256_ps(_mm256_set1_epi32(static_cast<int>(~0x7f800000u))));
  reduced = _mm256_or_ps(reduced, _mm256_set1_ps(0.5f));
  exponent_int = _mm256_sub_epi32(exponent_int, _mm256_set1_epi32(0x7f));
  __m256 exponent = _mm256_add_ps(_mm256_cvtepi32_ps(exponent_int), one);
  exponent = _mm256_sub_ps(exponent, _mm256_and_ps(is_subnormal, _mm256_set1_ps(23.0f)));

  const __m256 normalize = _mm256_cmp_ps(reduced, _mm256_set1_ps(kSqrtHalf), _CMP_LT_OQ);
  const __m256 normalized_part = _mm256_and_ps(reduced, normalize);
  reduced = _mm256_sub_ps(reduced, one);
  exponent = _mm256_sub_ps(exponent, _mm256_and_ps(one, normalize));
  reduced = _mm256_add_ps(reduced, normalized_part);
  const __m256 squared = _mm256_mul_ps(reduced, reduced);

  __m256 polynomial = _mm256_set1_ps(kLogP0);
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(kLogP1));
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(kLogP2));
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(kLogP3));
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(kLogP4));
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(kLogP5));
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(kLogP6));
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(kLogP7));
  polynomial = _mm256_fmadd_ps(polynomial, reduced, _mm256_set1_ps(kLogP8));
  polynomial = _mm256_mul_ps(polynomial, reduced);
  polynomial = _mm256_mul_ps(polynomial, squared);
  polynomial = _mm256_fmadd_ps(exponent, _mm256_set1_ps(kLogQ1), polynomial);
  polynomial = _mm256_fnmadd_ps(squared, _mm256_set1_ps(0.5f), polynomial);
  __m256 result = _mm256_add_ps(reduced, polynomial);
  result = _mm256_fmadd_ps(exponent, _mm256_set1_ps(kLogQ2), result);

  result = Select(is_negative, _mm256_set1_ps(std::numeric_limits<float>::quiet_NaN()), result);
  result = Select(is_zero, _mm256_set1_ps(-std::numeric_limits<float>::infinity()), result);
  result = Select(is_infinite, positive_infinity, result);
  return Select(is_nan, _mm256_set1_ps(std::numeric_limits<float>::quiet_NaN()), result);
}

ONNX_LIGHT_CPU_FORCE_INLINE float HorizontalMax(__m256 value) {
  __m128 reduced = _mm_max_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1));
  reduced = _mm_max_ps(reduced, _mm_movehl_ps(reduced, reduced));
  reduced = _mm_max_ss(reduced, _mm_shuffle_ps(reduced, reduced, 0x55));
  return _mm_cvtss_f32(reduced);
}

ONNX_LIGHT_CPU_FORCE_INLINE float HorizontalSum(__m256 value) {
  __m128 reduced = _mm_add_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1));
  reduced = _mm_hadd_ps(reduced, reduced);
  reduced = _mm_hadd_ps(reduced, reduced);
  return _mm_cvtss_f32(reduced);
}

ONNX_LIGHT_CPU_FORCE_INLINE __m256i TailMask(std::size_t count) {
  const __m256i lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  return _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(count)), lanes);
}

#undef ONNX_LIGHT_CPU_FORCE_INLINE

} // namespace

void ExpFloat32_AVX2_FMA(const float *input, float *output, std::size_t count) {
  std::size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    _mm256_storeu_ps(output + i, ExpPs256Fma(_mm256_loadu_ps(input + i)));
  }
  if (i < count) {
    const __m256i lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i tail = _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(count - i)), lanes);
    const __m256 result = ExpPs256Fma(_mm256_maskload_ps(input + i, tail));
    _mm256_maskstore_ps(output + i, tail, result);
  }
}

void SigmoidFloat32_AVX2_FMA(const float *input, float *output, std::size_t count) {
  const __m256 zero = _mm256_setzero_ps();
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 sign_bit = _mm256_set1_ps(-0.0f);
  std::size_t index = 0;
  for (; index + 8 <= count; index += 8) {
    const __m256 value = _mm256_loadu_ps(input + index);
    const __m256 exponent =
        ExpNegativePs256Fma(_mm256_sub_ps(zero, _mm256_andnot_ps(sign_bit, value)));
    const __m256 denominator = _mm256_add_ps(one, exponent);
    __m256 positive = _mm256_rcp_ps(denominator);
    positive =
        _mm256_mul_ps(positive, _mm256_fnmadd_ps(denominator, positive, _mm256_set1_ps(2.0f)));
    const __m256 negative = _mm256_mul_ps(exponent, positive);
    _mm256_storeu_ps(output + index, _mm256_blendv_ps(positive, negative, value));
  }
  if (index < count) {
    const __m256i mask = TailMask(count - index);
    const __m256 value = _mm256_maskload_ps(input + index, mask);
    const __m256 exponent =
        ExpNegativePs256Fma(_mm256_sub_ps(zero, _mm256_andnot_ps(sign_bit, value)));
    const __m256 denominator = _mm256_add_ps(one, exponent);
    __m256 positive = _mm256_rcp_ps(denominator);
    positive =
        _mm256_mul_ps(positive, _mm256_fnmadd_ps(denominator, positive, _mm256_set1_ps(2.0f)));
    const __m256 negative = _mm256_mul_ps(exponent, positive);
    _mm256_maskstore_ps(output + index, mask, _mm256_blendv_ps(positive, negative, value));
  }
}

void SoftmaxFloat32_AVX2_FMA(const float *input, float *output, std::size_t rows,
                             std::size_t columns) {
  const __m256 negative_infinity = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
  const __m256 zero = _mm256_setzero_ps();
  const __m256 one = _mm256_set1_ps(1.0f);
  const std::size_t vector_columns = columns - columns % 8;
  const __m256i tail_mask = TailMask(columns - vector_columns);

  for (std::size_t row = 0; row < rows; ++row) {
    const float *row_input = input + row * columns;
    float *row_output = output + row * columns;

    __m256 maximum_vector0 = negative_infinity;
    __m256 maximum_vector1 = negative_infinity;
    std::size_t column = 0;
    for (; column + 16 <= vector_columns; column += 16) {
      maximum_vector0 = _mm256_max_ps(_mm256_loadu_ps(row_input + column), maximum_vector0);
      maximum_vector1 = _mm256_max_ps(_mm256_loadu_ps(row_input + column + 8), maximum_vector1);
    }
    for (; column < vector_columns; column += 8) {
      maximum_vector0 = _mm256_max_ps(_mm256_loadu_ps(row_input + column), maximum_vector0);
    }
    if (column < columns) {
      const __m256 tail =
          _mm256_blendv_ps(negative_infinity, _mm256_maskload_ps(row_input + column, tail_mask),
                           _mm256_castsi256_ps(tail_mask));
      maximum_vector0 = _mm256_max_ps(tail, maximum_vector0);
    }
    const __m256 maximum =
        _mm256_set1_ps(HorizontalMax(_mm256_max_ps(maximum_vector0, maximum_vector1)));

    __m256 sum_vector0 = zero;
    __m256 sum_vector1 = zero;
    for (column = 0; column + 16 <= vector_columns; column += 16) {
      const __m256 exponent0 =
          ExpNegativePs256Fma(_mm256_sub_ps(_mm256_loadu_ps(row_input + column), maximum));
      const __m256 exponent1 =
          ExpNegativePs256Fma(_mm256_sub_ps(_mm256_loadu_ps(row_input + column + 8), maximum));
      _mm256_storeu_ps(row_output + column, exponent0);
      _mm256_storeu_ps(row_output + column + 8, exponent1);
      sum_vector0 = _mm256_add_ps(sum_vector0, exponent0);
      sum_vector1 = _mm256_add_ps(sum_vector1, exponent1);
    }
    for (; column < vector_columns; column += 8) {
      const __m256 exponent =
          ExpNegativePs256Fma(_mm256_sub_ps(_mm256_loadu_ps(row_input + column), maximum));
      _mm256_storeu_ps(row_output + column, exponent);
      sum_vector0 = _mm256_add_ps(sum_vector0, exponent);
    }
    if (column < columns) {
      const __m256 tail =
          _mm256_blendv_ps(maximum, _mm256_maskload_ps(row_input + column, tail_mask),
                           _mm256_castsi256_ps(tail_mask));
      const __m256 exponent = ExpNegativePs256Fma(_mm256_sub_ps(tail, maximum));
      _mm256_maskstore_ps(row_output + column, tail_mask, exponent);
      sum_vector0 =
          _mm256_add_ps(sum_vector0, _mm256_and_ps(exponent, _mm256_castsi256_ps(tail_mask)));
    }

    const __m256 inverse_sum =
        _mm256_div_ps(one, _mm256_set1_ps(HorizontalSum(_mm256_add_ps(sum_vector0, sum_vector1))));
    for (column = 0; column + 16 <= vector_columns; column += 16) {
      _mm256_storeu_ps(row_output + column,
                       _mm256_mul_ps(_mm256_loadu_ps(row_output + column), inverse_sum));
      _mm256_storeu_ps(row_output + column + 8,
                       _mm256_mul_ps(_mm256_loadu_ps(row_output + column + 8), inverse_sum));
    }
    for (; column < vector_columns; column += 8) {
      _mm256_storeu_ps(row_output + column,
                       _mm256_mul_ps(_mm256_loadu_ps(row_output + column), inverse_sum));
    }
    if (column < columns) {
      _mm256_maskstore_ps(
          row_output + column, tail_mask,
          _mm256_mul_ps(_mm256_maskload_ps(row_output + column, tail_mask), inverse_sum));
    }
  }
}

void LogFloat32_AVX2_FMA(const float *input, float *output, std::size_t count) {
  std::size_t i = 0;
  for (; i + 16 <= count; i += 16) {
    _mm256_storeu_ps(output + i, LogPs256Fma(_mm256_loadu_ps(input + i)));
    _mm256_storeu_ps(output + i + 8, LogPs256Fma(_mm256_loadu_ps(input + i + 8)));
  }
  for (; i + 8 <= count; i += 8) {
    _mm256_storeu_ps(output + i, LogPs256Fma(_mm256_loadu_ps(input + i)));
  }
  if (i < count) {
    const __m256i lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i tail = _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(count - i)), lanes);
    const __m256 result = LogPs256Fma(_mm256_maskload_ps(input + i, tail));
    _mm256_maskstore_ps(output + i, tail, result);
  }
}

namespace {

// Shared eight-lane ``x**y`` core: small non-negative integer exponents use
// exact repeated multiplication (matching ``FastFloatPower``'s scalar
// contract), while every other positive, finite base/exponent pair uses
// ``exp(y * log(x))``. Lanes outside that fast domain (negative or
// non-finite base, non-finite exponent) are left in ``fallback_lanes`` for
// the caller's scalar ``std::pow`` cleanup, matching the AVX-512 kernel's
// mixed vector/scalar contract. ``log_x``, when non-null, supplies a
// precomputed ``log(x)`` (used by the left-scalar entry point, where the
// base is invariant across the whole call and ``log`` must not be
// recomputed every eight-lane block).
__m256 PowPs256Fma(__m256 x, __m256 y, __m256i &fallback_lanes, const __m256 *log_x = nullptr) {
  const __m256 zero = _mm256_setzero_ps();
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 maximum = _mm256_set1_ps(std::numeric_limits<float>::max());
  const __m256 x2 = _mm256_mul_ps(x, x);
  const __m256 x3 = _mm256_mul_ps(x2, x);
  const __m256 x4 = _mm256_mul_ps(x2, x2);
  const __m256 x5 = _mm256_mul_ps(x4, x);
  const __m256 exponent0 = _mm256_cmp_ps(y, zero, _CMP_EQ_OQ);
  const __m256 exponent1 = _mm256_cmp_ps(y, one, _CMP_EQ_OQ);
  const __m256 exponent2 = _mm256_cmp_ps(y, _mm256_set1_ps(2.0f), _CMP_EQ_OQ);
  const __m256 exponent3 = _mm256_cmp_ps(y, _mm256_set1_ps(3.0f), _CMP_EQ_OQ);
  const __m256 exponent4 = _mm256_cmp_ps(y, _mm256_set1_ps(4.0f), _CMP_EQ_OQ);
  const __m256 exponent5 = _mm256_cmp_ps(y, _mm256_set1_ps(5.0f), _CMP_EQ_OQ);
  const __m256 small_integer = _mm256_or_ps(
      _mm256_or_ps(_mm256_or_ps(exponent0, exponent1), _mm256_or_ps(exponent2, exponent3)),
      _mm256_or_ps(exponent4, exponent5));

  __m256 result = Select(exponent0, one, zero);
  result = Select(exponent1, x, result);
  result = Select(exponent2, x2, result);
  result = Select(exponent3, x3, result);
  result = Select(exponent4, x4, result);
  result = Select(exponent5, x5, result);

  const __m256 abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);
  const __m256 abs_y = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), y);
  const __m256 positive_finite = _mm256_and_ps(
      _mm256_cmp_ps(x, zero, _CMP_GT_OQ), _mm256_and_ps(_mm256_cmp_ps(abs_x, maximum, _CMP_LE_OQ),
                                                        _mm256_cmp_ps(abs_y, maximum, _CMP_LE_OQ)));
  const __m256 approximate = _mm256_andnot_ps(small_integer, positive_finite);
  if (_mm256_movemask_ps(approximate) != 0) {
    const __m256 log_value = log_x != nullptr ? *log_x : LogPs256Fma(x);
    result = Select(approximate, ExpPs256Fma(_mm256_mul_ps(log_value, y)), result);
  }

  fallback_lanes = _mm256_castps_si256(
      _mm256_andnot_ps(_mm256_or_ps(small_integer, approximate), _mm256_set1_ps(-1.0f)));
  return result;
}

void ApplyPowFallback(const float *base, const float *exponent, float *output, std::size_t offset,
                      __m256i fallback_lanes) {
  const int mask = _mm256_movemask_ps(_mm256_castsi256_ps(fallback_lanes));
  if (mask == 0) {
    return;
  }
  for (std::size_t lane = 0; lane < 8; ++lane) {
    if ((mask & (1 << lane)) != 0) {
      output[offset + lane] = std::pow(base[offset + lane], exponent[offset + lane]);
    }
  }
}

void ApplyPowLeftScalarFallback(float base, const float *exponent, float *output,
                                std::size_t offset, __m256i fallback_lanes) {
  const int mask = _mm256_movemask_ps(_mm256_castsi256_ps(fallback_lanes));
  if (mask == 0) {
    return;
  }
  for (std::size_t lane = 0; lane < 8; ++lane) {
    if ((mask & (1 << lane)) != 0) {
      output[offset + lane] = std::pow(base, exponent[offset + lane]);
    }
  }
}

void ApplyPowRightScalarFallback(const float *base, float exponent, float *output,
                                 std::size_t offset, __m256i fallback_lanes) {
  const int mask = _mm256_movemask_ps(_mm256_castsi256_ps(fallback_lanes));
  if (mask == 0) {
    return;
  }
  for (std::size_t lane = 0; lane < 8; ++lane) {
    if ((mask & (1 << lane)) != 0) {
      output[offset + lane] = std::pow(base[offset + lane], exponent);
    }
  }
}

} // namespace

void PowFloat32_AVX2_FMA(const float *base, const float *exponent, float *output,
                         std::size_t count) {
  std::size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    const __m256 x = _mm256_loadu_ps(base + i);
    const __m256 y = _mm256_loadu_ps(exponent + i);
    __m256i fallback_lanes;
    const __m256 result = PowPs256Fma(x, y, fallback_lanes);
    _mm256_storeu_ps(output + i, result);
    ApplyPowFallback(base, exponent, output, i, fallback_lanes);
  }
  for (; i < count; ++i) {
    output[i] = std::pow(base[i], exponent[i]);
  }
}

void PowFloat32LeftScalar_AVX2_FMA(float base, const float *exponent, float *output,
                                   std::size_t count) {
  const __m256 x = _mm256_set1_ps(base);
  const bool positive_finite = base > 0.0f && std::isfinite(base);
  // When the base is not positive-finite, PowPs256Fma() never treats any lane as
  // "approximate" for this scalar base, so log_x is never read: the zero placeholder
  // below is only ever a dummy value and is not a meaningful log(base).
  const __m256 log_x = positive_finite ? LogPs256Fma(x) : _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    const __m256 y = _mm256_loadu_ps(exponent + i);
    __m256i fallback_lanes;
    const __m256 result = PowPs256Fma(x, y, fallback_lanes, &log_x);
    _mm256_storeu_ps(output + i, result);
    ApplyPowLeftScalarFallback(base, exponent, output, i, fallback_lanes);
  }
  for (; i < count; ++i) {
    output[i] = std::pow(base, exponent[i]);
  }
}

void PowFloat32RightScalar_AVX2_FMA(const float *base, float exponent, float *output,
                                    std::size_t count) {
  const __m256 y = _mm256_set1_ps(exponent);
  std::size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    const __m256 x = _mm256_loadu_ps(base + i);
    __m256i fallback_lanes;
    const __m256 result = PowPs256Fma(x, y, fallback_lanes);
    _mm256_storeu_ps(output + i, result);
    ApplyPowRightScalarFallback(base, exponent, output, i, fallback_lanes);
  }
  for (; i < count; ++i) {
    output[i] = std::pow(base[i], exponent);
  }
}

} // namespace onnx_light_cpu
