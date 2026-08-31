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

} // namespace

bool AttentionApplyAdditiveMaskFloat32_AVX512(float *scores, const float *mask, std::size_t count) {
  const __m512 negative_infinity = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
  const __m512 lowest = _mm512_set1_ps(std::numeric_limits<float>::lowest());
  __mmask16 valid = 0;
  std::size_t index = 0;
  for (; index + 16 <= count; index += 16) {
    const __m512 bias = _mm512_loadu_ps(mask + index);
    valid |= _mm512_cmp_ps_mask(bias, negative_infinity, _CMP_NEQ_UQ) &
             _mm512_cmp_ps_mask(bias, lowest, _CMP_NEQ_UQ);
    _mm512_storeu_ps(scores + index, _mm512_add_ps(_mm512_loadu_ps(scores + index), bias));
  }
  if (index < count) {
    const __mmask16 tail = TailMask(count - index);
    const __m512 bias = _mm512_maskz_loadu_ps(tail, mask + index);
    valid |= tail & _mm512_cmp_ps_mask(bias, negative_infinity, _CMP_NEQ_UQ) &
             _mm512_cmp_ps_mask(bias, lowest, _CMP_NEQ_UQ);
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

} // namespace onnx_light_cpu
