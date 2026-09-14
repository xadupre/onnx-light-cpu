// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/binary/binary_arithmetic_kernel_avx.h"

#include <algorithm>
#include <immintrin.h>

namespace onnx_light_cpu {

#define ONNX_LIGHT_CPU_BIN_AVX(STEM, T, VECTOR, LANES, LOAD, STORE, SET1, INTRIN, OPCH)            \
  void STEM##_AVX(const T *left, const T *right, T *out, std::size_t count) {                      \
    std::size_t i = 0;                                                                             \
    const std::size_t aligned = count - count % LANES;                                             \
    for (; i < aligned; i += LANES) {                                                              \
      STORE(out + i, INTRIN(LOAD(left + i), LOAD(right + i)));                                     \
    }                                                                                              \
    for (; i < count; ++i) {                                                                       \
      out[i] = left[i] OPCH right[i];                                                              \
    }                                                                                              \
  }                                                                                                \
  void STEM##Left_AVX(T left, const T *right, T *out, std::size_t count) {                         \
    std::size_t i = 0;                                                                             \
    const std::size_t aligned = count - count % LANES;                                             \
    const VECTOR left_vec = SET1(left);                                                            \
    for (; i < aligned; i += LANES) {                                                              \
      STORE(out + i, INTRIN(left_vec, LOAD(right + i)));                                           \
    }                                                                                              \
    for (; i < count; ++i) {                                                                       \
      out[i] = left OPCH right[i];                                                                 \
    }                                                                                              \
  }                                                                                                \
  void STEM##Right_AVX(const T *left, T right, T *out, std::size_t count) {                        \
    std::size_t i = 0;                                                                             \
    const std::size_t aligned = count - count % LANES;                                             \
    const VECTOR right_vec = SET1(right);                                                          \
    for (; i < aligned; i += LANES) {                                                              \
      STORE(out + i, INTRIN(LOAD(left + i), right_vec));                                           \
    }                                                                                              \
    for (; i < count; ++i) {                                                                       \
      out[i] = left[i] OPCH right;                                                                 \
    }                                                                                              \
  }

#define ONNX_LIGHT_CPU_BIN_AVX_F32(STEM, INTRIN, OPCH)                                             \
  ONNX_LIGHT_CPU_BIN_AVX(STEM, float, __m256, 8, _mm256_loadu_ps, _mm256_storeu_ps,                \
                         _mm256_set1_ps, INTRIN, OPCH)
#define ONNX_LIGHT_CPU_BIN_AVX_F64(STEM, INTRIN, OPCH)                                             \
  ONNX_LIGHT_CPU_BIN_AVX(STEM, double, __m256d, 4, _mm256_loadu_pd, _mm256_storeu_pd,              \
                         _mm256_set1_pd, INTRIN, OPCH)

ONNX_LIGHT_CPU_BIN_AVX_F32(BinaryAddFloat32, _mm256_add_ps, +)
ONNX_LIGHT_CPU_BIN_AVX_F32(BinarySubFloat32, _mm256_sub_ps, -)
ONNX_LIGHT_CPU_BIN_AVX_F32(BinaryMulFloat32, _mm256_mul_ps, *)
ONNX_LIGHT_CPU_BIN_AVX_F32(BinaryDivFloat32, _mm256_div_ps, /)
ONNX_LIGHT_CPU_BIN_AVX_F64(BinaryAddFloat64, _mm256_add_pd, +)
ONNX_LIGHT_CPU_BIN_AVX_F64(BinarySubFloat64, _mm256_sub_pd, -)
ONNX_LIGHT_CPU_BIN_AVX_F64(BinaryMulFloat64, _mm256_mul_pd, *)
ONNX_LIGHT_CPU_BIN_AVX_F64(BinaryDivFloat64, _mm256_div_pd, /)

#undef ONNX_LIGHT_CPU_BIN_AVX_F32
#undef ONNX_LIGHT_CPU_BIN_AVX_F64
#undef ONNX_LIGHT_CPU_BIN_AVX

void BinaryPReluFloat32_AVX(const float *left, const float *right, float *out, std::size_t count) {
  std::size_t i = 0;
  const std::size_t aligned = count - count % 8;
  const __m256 zero = _mm256_setzero_ps();
  for (; i < aligned; i += 8) {
    const __m256 x = _mm256_loadu_ps(left + i);
    const __m256 mask = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);
    const __m256 scaled = _mm256_mul_ps(x, _mm256_loadu_ps(right + i));
    _mm256_storeu_ps(out + i, _mm256_or_ps(_mm256_and_ps(mask, scaled), _mm256_andnot_ps(mask, x)));
  }
  for (; i < count; ++i) {
    out[i] = left[i] < 0.0f ? left[i] * right[i] : left[i];
  }
}

void BinaryPReluFloat32Left_AVX(float left, const float *right, float *out, std::size_t count) {
  if (!(left < 0.0f)) {
    std::fill_n(out, count, left);
    return;
  }
  std::size_t i = 0;
  const std::size_t aligned = count - count % 8;
  const __m256 x = _mm256_set1_ps(left);
  for (; i < aligned; i += 8) {
    _mm256_storeu_ps(out + i, _mm256_mul_ps(x, _mm256_loadu_ps(right + i)));
  }
  for (; i < count; ++i) {
    out[i] = left * right[i];
  }
}

void BinaryPReluFloat32Right_AVX(const float *left, float right, float *out, std::size_t count) {
  std::size_t i = 0;
  const std::size_t aligned = count - count % 8;
  const __m256 zero = _mm256_setzero_ps();
  const __m256 slope = _mm256_set1_ps(right);
  for (; i < aligned; i += 8) {
    const __m256 x = _mm256_loadu_ps(left + i);
    const __m256 mask = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);
    const __m256 scaled = _mm256_mul_ps(x, slope);
    _mm256_storeu_ps(out + i, _mm256_or_ps(_mm256_and_ps(mask, scaled), _mm256_andnot_ps(mask, x)));
  }
  for (; i < count; ++i) {
    out[i] = left[i] < 0.0f ? left[i] * right : left[i];
  }
}

} // namespace onnx_light_cpu
