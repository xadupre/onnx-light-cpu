// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/attention/avx2/attention_kernel_avx2_fma.h"

#include "onnx_light_cpu/impl/math/math_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <limits>

namespace onnx_light_cpu {

namespace {

// Lane-index compare mask for the last, partial 8-wide chunk of a `count`
// sized loop: lane `i` is all-ones iff `i < remaining`. Matches the tail
// convention already used by `ExpFloat32_AVX2_FMA`/`LogFloat32_AVX2_FMA`
// (`onnx_light_cpu/impl/math/avx2/exp_log_kernel_avx2_fma.cc`).
inline __m256i TailMask(std::size_t remaining) {
  const __m256i lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  return _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(remaining)), lanes);
}

inline float ReduceAdd(__m256 value) {
  const __m128 low = _mm256_castps256_ps128(value);
  const __m128 high = _mm256_extractf128_ps(value, 1);
  __m128 sum = _mm_add_ps(low, high);
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  return _mm_cvtss_f32(sum);
}

inline float ReduceMax(__m256 value) {
  const __m128 low = _mm256_castps256_ps128(value);
  const __m128 high = _mm256_extractf128_ps(value, 1);
  __m128 maximum = _mm_max_ps(low, high);
  maximum = _mm_max_ps(maximum, _mm_movehl_ps(maximum, maximum));
  maximum = _mm_max_ss(maximum, _mm_shuffle_ps(maximum, maximum, 1));
  return _mm_cvtss_f32(maximum);
}

} // namespace

float AttentionDotFloat32_AVX2_FMA(const float *q, const float *k, std::size_t count) {
  // Two independent accumulators break the single-`sum` FMA dependency chain
  // (every `vfmadd` would otherwise wait ~4-5 cycles for the previous one to
  // retire even though most cores can issue two FMAs per cycle). Measured
  // against the AVX2 `Q == 1` decode benchmark, two accumulators improve the
  // common `head_dim == 64` case without the register-pressure regression a
  // four-accumulator version showed at `head_dim == 128`.
  __m256 sum0 = _mm256_setzero_ps();
  __m256 sum1 = _mm256_setzero_ps();
  std::size_t d = 0;
  for (; d + 16 <= count; d += 16) {
    sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + d), _mm256_loadu_ps(k + d), sum0);
    sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(q + d + 8), _mm256_loadu_ps(k + d + 8), sum1);
  }
  __m256 sum = _mm256_add_ps(sum0, sum1);
  for (; d + 8 <= count; d += 8) {
    sum = _mm256_fmadd_ps(_mm256_loadu_ps(q + d), _mm256_loadu_ps(k + d), sum);
  }
  float total = ReduceAdd(sum);
  if (d < count) {
    const __m256i tail = TailMask(count - d);
    const __m256 q_tail = _mm256_maskload_ps(q + d, tail);
    const __m256 k_tail = _mm256_maskload_ps(k + d, tail);
    total += ReduceAdd(_mm256_mul_ps(q_tail, k_tail));
  }
  return total;
}

void AttentionAccumulateFloat32_AVX2_FMA(float *accumulator, float weight, const float *v,
                                         std::size_t count) {
  const __m256 weight_vector = _mm256_set1_ps(weight);
  std::size_t d = 0;
  for (; d + 8 <= count; d += 8) {
    const __m256 updated =
        _mm256_fmadd_ps(weight_vector, _mm256_loadu_ps(v + d), _mm256_loadu_ps(accumulator + d));
    _mm256_storeu_ps(accumulator + d, updated);
  }
  if (d < count) {
    const __m256i tail = TailMask(count - d);
    const __m256 updated = _mm256_fmadd_ps(weight_vector, _mm256_maskload_ps(v + d, tail),
                                           _mm256_maskload_ps(accumulator + d, tail));
    _mm256_maskstore_ps(accumulator + d, tail, updated);
  }
}

void AttentionScaleFloat32_AVX2_FMA(float *values, float factor, std::size_t count) {
  const __m256 factor_vector = _mm256_set1_ps(factor);
  std::size_t d = 0;
  for (; d + 8 <= count; d += 8) {
    _mm256_storeu_ps(values + d, _mm256_mul_ps(_mm256_loadu_ps(values + d), factor_vector));
  }
  if (d < count) {
    const __m256i tail = TailMask(count - d);
    const __m256 updated = _mm256_mul_ps(_mm256_maskload_ps(values + d, tail), factor_vector);
    _mm256_maskstore_ps(values + d, tail, updated);
  }
}

bool AttentionApplyAdditiveMaskFloat32_AVX2_FMA(float *scores, const float *mask,
                                                std::size_t count) {
  const __m256 negative_infinity = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
  const __m256 lowest = _mm256_set1_ps(std::numeric_limits<float>::lowest());
  bool any_valid = false;
  std::size_t index = 0;
  for (; index + 8 <= count; index += 8) {
    const __m256 bias = _mm256_loadu_ps(mask + index);
    const __m256 not_negative_infinity = _mm256_cmp_ps(bias, negative_infinity, _CMP_NEQ_UQ);
    const __m256 not_lowest = _mm256_cmp_ps(bias, lowest, _CMP_NEQ_UQ);
    const __m256 valid = _mm256_and_ps(not_negative_infinity, not_lowest);
    any_valid = any_valid || (_mm256_movemask_ps(valid) != 0);
    _mm256_storeu_ps(scores + index, _mm256_add_ps(_mm256_loadu_ps(scores + index), bias));
  }
  if (index < count) {
    const __m256i tail = TailMask(count - index);
    const __m256 bias = _mm256_maskload_ps(mask + index, tail);
    const __m256 not_negative_infinity = _mm256_cmp_ps(bias, negative_infinity, _CMP_NEQ_UQ);
    const __m256 not_lowest = _mm256_cmp_ps(bias, lowest, _CMP_NEQ_UQ);
    const __m256 valid =
        _mm256_and_ps(_mm256_castsi256_ps(tail), _mm256_and_ps(not_negative_infinity, not_lowest));
    any_valid = any_valid || (_mm256_movemask_ps(valid) != 0);
    const __m256 updated = _mm256_add_ps(_mm256_maskload_ps(scores + index, tail), bias);
    _mm256_maskstore_ps(scores + index, tail, updated);
  }
  return any_valid;
}

bool AttentionApplyBooleanMaskFloat32_AVX2_FMA(float *scores, const std::uint8_t *mask,
                                               std::size_t count) {
  const __m256 negative_infinity = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
  const __m256i zero = _mm256_setzero_si256();
  bool any_valid = false;
  std::size_t index = 0;
  for (; index + 8 <= count; index += 8) {
    // `_mm256_cvtepi8_epi32` sign-extends the 8 low bytes into 8 int32 lanes
    // (bool tensors only ever store 0/1, but any nonzero byte is treated as
    // "allowed", matching the scalar `mask[index] != 0` reference exactly).
    const __m128i bytes =
        _mm_loadl_epi64(reinterpret_cast<const __m128i *>(static_cast<const void *>(mask + index)));
    const __m256i widened = _mm256_cvtepi8_epi32(bytes);
    const __m256i allowed =
        _mm256_xor_si256(_mm256_cmpeq_epi32(widened, zero), _mm256_set1_epi32(-1));
    any_valid = any_valid || (_mm256_movemask_ps(_mm256_castsi256_ps(allowed)) != 0);
    const __m256 blended = _mm256_blendv_ps(negative_infinity, _mm256_loadu_ps(scores + index),
                                            _mm256_castsi256_ps(allowed));
    _mm256_storeu_ps(scores + index, blended);
  }
  for (; index < count; ++index) {
    const bool allowed = mask[index] != 0;
    any_valid = any_valid || allowed;
    scores[index] = allowed ? scores[index] : -std::numeric_limits<float>::infinity();
  }
  return any_valid;
}

AttentionSoftmaxBlockResultAVX2 AttentionSoftmaxBlockFloat32_AVX2_FMA(float *scores,
                                                                      std::size_t count,
                                                                      float previous_maximum,
                                                                      float &denominator) {
  const float negative_infinity = -std::numeric_limits<float>::infinity();
  // Two independent accumulators for the same reason as
  // `AttentionDotFloat32_AVX2_FMA`: `count` is the KV block size (up to 256),
  // so a single running `max`/`sum` chain pays its reduction latency on every
  // 8-wide chunk instead of once per call.
  __m256 maximum0 = _mm256_set1_ps(negative_infinity);
  __m256 maximum1 = _mm256_set1_ps(negative_infinity);
  std::size_t index = 0;
  for (; index + 16 <= count; index += 16) {
    maximum0 = _mm256_max_ps(maximum0, _mm256_loadu_ps(scores + index));
    maximum1 = _mm256_max_ps(maximum1, _mm256_loadu_ps(scores + index + 8));
  }
  __m256 maximum = _mm256_max_ps(maximum0, maximum1);
  for (; index + 8 <= count; index += 8) {
    maximum = _mm256_max_ps(maximum, _mm256_loadu_ps(scores + index));
  }
  float block_maximum = ReduceMax(maximum);
  for (std::size_t tail = index; tail < count; ++tail) {
    block_maximum = std::max(block_maximum, scores[tail]);
  }

  const float new_maximum = std::max(previous_maximum, block_maximum);
  if (new_maximum == negative_infinity) {
    std::fill_n(scores, count, 0.0f);
    return {new_maximum, 1.0f};
  }
  const float correction =
      previous_maximum == negative_infinity ? 0.0f : std::exp(previous_maximum - new_maximum);

  const __m256 offset = _mm256_set1_ps(new_maximum);
  index = 0;
  for (; index + 8 <= count; index += 8) {
    _mm256_storeu_ps(scores + index, _mm256_sub_ps(_mm256_loadu_ps(scores + index), offset));
  }
  for (std::size_t tail = index; tail < count; ++tail) {
    scores[tail] -= new_maximum;
  }

  ExpFloat32_AVX2_FMA(scores, scores, count);

  __m256 sum0 = _mm256_setzero_ps();
  __m256 sum1 = _mm256_setzero_ps();
  index = 0;
  for (; index + 16 <= count; index += 16) {
    sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(scores + index));
    sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(scores + index + 8));
  }
  __m256 sum = _mm256_add_ps(sum0, sum1);
  for (; index + 8 <= count; index += 8) {
    sum = _mm256_add_ps(sum, _mm256_loadu_ps(scores + index));
  }
  float total = ReduceAdd(sum);
  for (std::size_t tail = index; tail < count; ++tail) {
    total += scores[tail];
  }
  denominator = denominator * correction + total;
  return {new_maximum, correction};
}

} // namespace onnx_light_cpu
