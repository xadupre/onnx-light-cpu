// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/attention/avx512/attention_kernel_avx512.h"

#include "onnx_light_cpu/impl/math/math_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <limits>

namespace onnx_light_cpu {

namespace {

__mmask16 TailMask(std::size_t count) {
  return static_cast<__mmask16>((std::uint32_t{1} << count) - 1);
}

__m512 AttentionExp(__m512 value) {
  const __m512 zero = _mm512_setzero_ps();
  const __m512 lower = _mm512_set1_ps(-103.97208f);
  const __mmask16 nan = _mm512_cmp_ps_mask(value, value, _CMP_UNORD_Q);
  const __mmask16 underflow = _mm512_cmp_ps_mask(value, lower, _CMP_LT_OQ);
  value = _mm512_max_ps(value, lower);
  const __m512 exponent =
      _mm512_roundscale_ps(_mm512_mul_ps(value, _mm512_set1_ps(1.44269504088896341f)),
                           _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m512 reduced = _mm512_fnmadd_ps(exponent, _mm512_set1_ps(0.693145751953125f), value);
  reduced = _mm512_fnmadd_ps(exponent, _mm512_set1_ps(1.42860677e-6f), reduced);
  __m512 polynomial =
      _mm512_fmadd_ps(_mm512_set1_ps(1.0f / 24.0f), reduced, _mm512_set1_ps(1.0f / 6.0f));
  polynomial = _mm512_fmadd_ps(polynomial, reduced, _mm512_set1_ps(0.5f));
  polynomial = _mm512_fmadd_ps(polynomial, reduced, _mm512_set1_ps(1.0f));
  polynomial = _mm512_fmadd_ps(polynomial, reduced, _mm512_set1_ps(1.0f));
  __m512 result = _mm512_scalef_ps(polynomial, exponent);
  result = _mm512_mask_mov_ps(result, underflow, zero);
  return _mm512_mask_mov_ps(result, nan, _mm512_set1_ps(std::numeric_limits<float>::quiet_NaN()));
}

} // namespace

bool AttentionApplyAdditiveMaskFloat32_AVX512(float *scores, const float *mask, std::size_t count) {
  const __m512 negative_infinity = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
  __mmask16 valid = 0;
  std::size_t index = 0;
  for (; index + 16 <= count; index += 16) {
    const __m512 bias = _mm512_loadu_ps(mask + index);
    valid |= _mm512_cmp_ps_mask(bias, negative_infinity, _CMP_NEQ_UQ);
    _mm512_storeu_ps(scores + index, _mm512_add_ps(_mm512_loadu_ps(scores + index), bias));
  }
  if (index < count) {
    const __mmask16 tail = TailMask(count - index);
    const __m512 bias = _mm512_maskz_loadu_ps(tail, mask + index);
    valid |= tail & _mm512_cmp_ps_mask(bias, negative_infinity, _CMP_NEQ_UQ);
    _mm512_mask_storeu_ps(scores + index, tail,
                          _mm512_add_ps(_mm512_maskz_loadu_ps(tail, scores + index), bias));
  }
  return valid != 0;
}

bool AttentionApplyBooleanMaskFloat32_AVX512(float *scores, const std::uint8_t *mask,
                                             std::size_t count) {
  const __m512 negative_infinity = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
  const __m128i zero = _mm_setzero_si128();
  __mmask16 valid = 0;
  std::size_t index = 0;
  for (; index + 16 <= count; index += 16) {
    const __m128i values =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(static_cast<const void *>(mask + index)));
    const auto allowed = static_cast<__mmask16>(
        ~static_cast<unsigned>(_mm_movemask_epi8(_mm_cmpeq_epi8(values, zero))));
    valid |= allowed;
    _mm512_storeu_ps(scores + index, _mm512_mask_mov_ps(negative_infinity, allowed,
                                                        _mm512_loadu_ps(scores + index)));
  }
  bool any_valid = valid != 0;
  for (; index < count; ++index) {
    const bool allowed = mask[index] != 0;
    any_valid |= allowed;
    scores[index] = allowed ? scores[index] : -std::numeric_limits<float>::infinity();
  }
  return any_valid;
}

AttentionSoftmaxBlockResult AttentionSoftmaxBlockFloat32_AVX512(float *scores, std::size_t count,
                                                                float previous_maximum,
                                                                float &denominator) {
  const float negative_infinity = -std::numeric_limits<float>::infinity();
  __m512 maximum = _mm512_set1_ps(negative_infinity);
  std::size_t index = 0;
  for (; index + 16 <= count; index += 16) {
    maximum = _mm512_max_ps(maximum, _mm512_loadu_ps(scores + index));
  }
  if (index < count) {
    const __mmask16 tail = TailMask(count - index);
    maximum = _mm512_max_ps(
        maximum, _mm512_mask_loadu_ps(_mm512_set1_ps(negative_infinity), tail, scores + index));
  }
  const float block_maximum = _mm512_reduce_max_ps(maximum);
  const float new_maximum = std::max(previous_maximum, block_maximum);
  if (new_maximum == negative_infinity) {
    std::fill_n(scores, count, 0.0f);
    return {new_maximum, 1.0f};
  }
  const float correction =
      previous_maximum == negative_infinity ? 0.0f : std::exp(previous_maximum - new_maximum);
  const __m512 offset = _mm512_set1_ps(new_maximum);
  index = 0;
  for (; index + 16 <= count; index += 16) {
    _mm512_storeu_ps(scores + index, _mm512_sub_ps(_mm512_loadu_ps(scores + index), offset));
  }
  if (index < count) {
    const __mmask16 tail = TailMask(count - index);
    const __m512 values = _mm512_maskz_loadu_ps(tail, scores + index);
    _mm512_mask_storeu_ps(scores + index, tail, _mm512_sub_ps(values, offset));
  }

  ExpFloat32_AVX512(scores, scores, count);
  __m512 sum = _mm512_setzero_ps();
  index = 0;
  for (; index + 16 <= count; index += 16) {
    sum = _mm512_add_ps(sum, _mm512_loadu_ps(scores + index));
  }
  if (index < count) {
    sum = _mm512_add_ps(sum, _mm512_maskz_loadu_ps(TailMask(count - index), scores + index));
  }
  denominator = denominator * correction + _mm512_reduce_add_ps(sum);
  return {new_maximum, correction};
}

void AttentionScaleFloat32_AVX512(float *values, float factor, std::size_t count) {
  const __m512 scale = _mm512_set1_ps(factor);
  std::size_t index = 0;
  for (; index + 16 <= count; index += 16) {
    _mm512_storeu_ps(values + index, _mm512_mul_ps(_mm512_loadu_ps(values + index), scale));
  }
  if (index < count) {
    const __mmask16 tail = TailMask(count - index);
    _mm512_mask_storeu_ps(values + index, tail,
                          _mm512_mul_ps(_mm512_maskz_loadu_ps(tail, values + index), scale));
  }
}

void AttentionNormalizeRowsFloat32_AVX512(float *values, const float *denominators,
                                          std::size_t rows, std::size_t columns,
                                          std::ptrdiff_t stride) {
  for (std::size_t row = 0; row < rows; ++row) {
    const float factor = denominators[row] != 0.0f ? 1.0f / denominators[row] : 0.0f;
    const __m512 scale = _mm512_set1_ps(factor);
    float *output_row = values + row * stride;
    std::size_t column = 0;
    for (; column + 16 <= columns; column += 16) {
      _mm512_storeu_ps(output_row + column,
                       _mm512_mul_ps(_mm512_loadu_ps(output_row + column), scale));
    }
    if (column < columns) {
      const __mmask16 tail = TailMask(columns - column);
      _mm512_mask_storeu_ps(output_row + column, tail,
                            _mm512_mul_ps(_mm512_maskz_loadu_ps(tail, output_row + column), scale));
    }
  }
}

void AttentionSoftmaxRowsFloat32_AVX512(float *scores, float *denominators, std::size_t rows,
                                        std::size_t columns) {
  const __m512 negative_infinity = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
  for (std::size_t row = 0; row < rows; ++row) {
    float *values = scores + row * columns;
    __m512 maximum = negative_infinity;
    std::size_t column = 0;
    for (; column + 16 <= columns; column += 16) {
      maximum = _mm512_max_ps(maximum, _mm512_loadu_ps(values + column));
    }
    const float row_maximum = _mm512_reduce_max_ps(maximum);
    const __m512 offset = _mm512_set1_ps(row_maximum);
    __m512 sum = _mm512_setzero_ps();
    column = 0;
    for (; column + 16 <= columns; column += 16) {
      const __m512 exponential =
          AttentionExp(_mm512_sub_ps(_mm512_loadu_ps(values + column), offset));
      _mm512_storeu_ps(values + column, exponential);
      sum = _mm512_add_ps(sum, exponential);
    }
    denominators[row] = _mm512_reduce_add_ps(sum);
  }
}

void AttentionPackRowsFloat32_AVX512(const float *source, float *destination, std::size_t rows,
                                     std::size_t dimension, std::ptrdiff_t stride) {
  for (std::size_t row = 0; row < rows; ++row) {
    const float *source_row = source + row * stride;
    float *destination_row = destination + row * dimension;
    std::size_t column = 0;
    for (; column + 16 <= dimension; column += 16) {
      _mm512_storeu_ps(destination_row + column, _mm512_loadu_ps(source_row + column));
    }
    std::copy(source_row + column, source_row + dimension, destination_row + column);
  }
}

void AttentionScatterRowsFloat32_AVX512(const float *source, float *destination, std::size_t rows,
                                        std::size_t dimension, std::ptrdiff_t stride) {
  for (std::size_t row = 0; row < rows; ++row) {
    const float *source_row = source + row * dimension;
    float *destination_row = destination + row * stride;
    std::size_t column = 0;
    for (; column + 16 <= dimension; column += 16) {
      _mm512_storeu_ps(destination_row + column, _mm512_loadu_ps(source_row + column));
    }
    std::copy(source_row + column, source_row + dimension, destination_row + column);
  }
}

void AttentionScoreQ8K128D64Float32_AVX512(const float *q, const float *k, float scale,
                                           float *scores) {
  for (std::size_t query = 0; query < 8; query += 4) {
    __m512 sums[4][4];
    for (auto &row : sums) {
      for (auto &sum : row) {
        sum = _mm512_setzero_ps();
      }
    }
    for (std::size_t key = 0; key < 128; ++key) {
      const float *k_row = k + key * 64;
      for (std::size_t chunk = 0; chunk < 4; ++chunk) {
        const __m512 kv = _mm512_loadu_ps(k_row + chunk * 16);
        for (std::size_t row = 0; row < 4; ++row) {
          sums[row][chunk] = _mm512_fmadd_ps(_mm512_loadu_ps(q + (query + row) * 64 + chunk * 16),
                                             kv, sums[row][chunk]);
        }
      }
      for (std::size_t row = 0; row < 4; ++row) {
        const __m512 sum01 = _mm512_add_ps(sums[row][0], sums[row][1]);
        const __m512 sum23 = _mm512_add_ps(sums[row][2], sums[row][3]);
        scores[(query + row) * 128 + key] =
            scale * _mm512_reduce_add_ps(_mm512_add_ps(sum01, sum23));
        for (auto &sum : sums[row]) {
          sum = _mm512_setzero_ps();
        }
      }
    }
  }
}

void AttentionProbabilityValueQ8K128D64Float32_AVX512(const float *probabilities, const float *v,
                                                      float *output, std::ptrdiff_t output_stride) {
  for (std::size_t chunk = 0; chunk < 4; ++chunk) {
    __m512 sums[8];
    for (auto &sum : sums) {
      sum = _mm512_setzero_ps();
    }
    for (std::size_t key = 0; key < 128; ++key) {
      const __m512 values = _mm512_loadu_ps(v + key * 64 + chunk * 16);
      for (std::size_t query = 0; query < 8; ++query) {
        sums[query] =
            _mm512_fmadd_ps(_mm512_set1_ps(probabilities[query * 128 + key]), values, sums[query]);
      }
    }
    for (std::size_t query = 0; query < 8; ++query) {
      _mm512_storeu_ps(output + query * output_stride + chunk * 16, sums[query]);
    }
  }
}

} // namespace onnx_light_cpu
