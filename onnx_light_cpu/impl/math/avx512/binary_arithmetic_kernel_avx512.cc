// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/binary/binary_arithmetic_kernel.h"

#include <algorithm>
#include <cstddef>
#include <immintrin.h>

namespace onnx_light_cpu {

// ---------------------------------------------------------------------------
// AVX-512F (512-bit) implementations, compiled in this dedicated translation
// unit with ``-mavx512f``/``/arch:AVX512`` so the rest of the library keeps
// its baseline flags and only reaches these once ``DetectSimdLevel()``
// reports ``SimdLevel::kAVX512`` at runtime. The intrinsic names
// (``_mm512_<op>_p[sd]``) are regular across Add/Sub/Mul/Div, so a single
// macro instantiates the whole type x operator matrix.
// ---------------------------------------------------------------------------
#define ONNX_LIGHT_CPU_BIN_AVX512_F32(STEM, INTRIN, OPCH)                                          \
  void STEM##_AVX512(const float *left, const float *right, float *out, std::size_t count) {       \
    std::size_t i = 0;                                                                             \
    const std::size_t unrolled = count - count % 64;                                               \
    for (; i < unrolled; i += 64) {                                                                \
      _mm512_storeu_ps(out + i, INTRIN(_mm512_loadu_ps(left + i), _mm512_loadu_ps(right + i)));    \
      _mm512_storeu_ps(out + i + 16,                                                               \
                       INTRIN(_mm512_loadu_ps(left + i + 16), _mm512_loadu_ps(right + i + 16)));   \
      _mm512_storeu_ps(out + i + 32,                                                               \
                       INTRIN(_mm512_loadu_ps(left + i + 32), _mm512_loadu_ps(right + i + 32)));   \
      _mm512_storeu_ps(out + i + 48,                                                               \
                       INTRIN(_mm512_loadu_ps(left + i + 48), _mm512_loadu_ps(right + i + 48)));   \
    }                                                                                              \
    const std::size_t aligned = count - count % 16;                                                \
    for (; i < aligned; i += 16) {                                                                 \
      _mm512_storeu_ps(out + i, INTRIN(_mm512_loadu_ps(left + i), _mm512_loadu_ps(right + i)));    \
    }                                                                                              \
    for (; i < count; ++i) {                                                                       \
      out[i] = left[i] OPCH right[i];                                                              \
    }                                                                                              \
  }                                                                                                \
  void STEM##Left_AVX512(float left, const float *right, float *out, std::size_t count) {          \
    std::size_t i = 0;                                                                             \
    const __m512 left_vec = _mm512_set1_ps(left);                                                  \
    const std::size_t unrolled = count - count % 64;                                               \
    for (; i < unrolled; i += 64) {                                                                \
      _mm512_storeu_ps(out + i, INTRIN(left_vec, _mm512_loadu_ps(right + i)));                     \
      _mm512_storeu_ps(out + i + 16, INTRIN(left_vec, _mm512_loadu_ps(right + i + 16)));           \
      _mm512_storeu_ps(out + i + 32, INTRIN(left_vec, _mm512_loadu_ps(right + i + 32)));           \
      _mm512_storeu_ps(out + i + 48, INTRIN(left_vec, _mm512_loadu_ps(right + i + 48)));           \
    }                                                                                              \
    const std::size_t aligned = count - count % 16;                                                \
    for (; i < aligned; i += 16) {                                                                 \
      _mm512_storeu_ps(out + i, INTRIN(left_vec, _mm512_loadu_ps(right + i)));                     \
    }                                                                                              \
    for (; i < count; ++i) {                                                                       \
      out[i] = left OPCH right[i];                                                                 \
    }                                                                                              \
  }                                                                                                \
  void STEM##Right_AVX512(const float *left, float right, float *out, std::size_t count) {         \
    std::size_t i = 0;                                                                             \
    const __m512 right_vec = _mm512_set1_ps(right);                                                \
    const std::size_t unrolled = count - count % 64;                                               \
    for (; i < unrolled; i += 64) {                                                                \
      _mm512_storeu_ps(out + i, INTRIN(_mm512_loadu_ps(left + i), right_vec));                     \
      _mm512_storeu_ps(out + i + 16, INTRIN(_mm512_loadu_ps(left + i + 16), right_vec));           \
      _mm512_storeu_ps(out + i + 32, INTRIN(_mm512_loadu_ps(left + i + 32), right_vec));           \
      _mm512_storeu_ps(out + i + 48, INTRIN(_mm512_loadu_ps(left + i + 48), right_vec));           \
    }                                                                                              \
    const std::size_t aligned = count - count % 16;                                                \
    for (; i < aligned; i += 16) {                                                                 \
      _mm512_storeu_ps(out + i, INTRIN(_mm512_loadu_ps(left + i), right_vec));                     \
    }                                                                                              \
    for (; i < count; ++i) {                                                                       \
      out[i] = left[i] OPCH right;                                                                 \
    }                                                                                              \
  }

#define ONNX_LIGHT_CPU_BIN_AVX512_F64(STEM, INTRIN, OPCH)                                          \
  void STEM##_AVX512(const double *left, const double *right, double *out, std::size_t count) {    \
    std::size_t i = 0;                                                                             \
    const std::size_t unrolled = count - count % 32;                                               \
    for (; i < unrolled; i += 32) {                                                                \
      _mm512_storeu_pd(out + i, INTRIN(_mm512_loadu_pd(left + i), _mm512_loadu_pd(right + i)));    \
      _mm512_storeu_pd(out + i + 8,                                                                \
                       INTRIN(_mm512_loadu_pd(left + i + 8), _mm512_loadu_pd(right + i + 8)));     \
      _mm512_storeu_pd(out + i + 16,                                                               \
                       INTRIN(_mm512_loadu_pd(left + i + 16), _mm512_loadu_pd(right + i + 16)));   \
      _mm512_storeu_pd(out + i + 24,                                                               \
                       INTRIN(_mm512_loadu_pd(left + i + 24), _mm512_loadu_pd(right + i + 24)));   \
    }                                                                                              \
    const std::size_t aligned = count - count % 8;                                                 \
    for (; i < aligned; i += 8) {                                                                  \
      _mm512_storeu_pd(out + i, INTRIN(_mm512_loadu_pd(left + i), _mm512_loadu_pd(right + i)));    \
    }                                                                                              \
    for (; i < count; ++i) {                                                                       \
      out[i] = left[i] OPCH right[i];                                                              \
    }                                                                                              \
  }                                                                                                \
  void STEM##Left_AVX512(double left, const double *right, double *out, std::size_t count) {       \
    std::size_t i = 0;                                                                             \
    const __m512d left_vec = _mm512_set1_pd(left);                                                 \
    const std::size_t unrolled = count - count % 32;                                               \
    for (; i < unrolled; i += 32) {                                                                \
      _mm512_storeu_pd(out + i, INTRIN(left_vec, _mm512_loadu_pd(right + i)));                     \
      _mm512_storeu_pd(out + i + 8, INTRIN(left_vec, _mm512_loadu_pd(right + i + 8)));             \
      _mm512_storeu_pd(out + i + 16, INTRIN(left_vec, _mm512_loadu_pd(right + i + 16)));           \
      _mm512_storeu_pd(out + i + 24, INTRIN(left_vec, _mm512_loadu_pd(right + i + 24)));           \
    }                                                                                              \
    const std::size_t aligned = count - count % 8;                                                 \
    for (; i < aligned; i += 8) {                                                                  \
      _mm512_storeu_pd(out + i, INTRIN(left_vec, _mm512_loadu_pd(right + i)));                     \
    }                                                                                              \
    for (; i < count; ++i) {                                                                       \
      out[i] = left OPCH right[i];                                                                 \
    }                                                                                              \
  }                                                                                                \
  void STEM##Right_AVX512(const double *left, double right, double *out, std::size_t count) {      \
    std::size_t i = 0;                                                                             \
    const __m512d right_vec = _mm512_set1_pd(right);                                               \
    const std::size_t unrolled = count - count % 32;                                               \
    for (; i < unrolled; i += 32) {                                                                \
      _mm512_storeu_pd(out + i, INTRIN(_mm512_loadu_pd(left + i), right_vec));                     \
      _mm512_storeu_pd(out + i + 8, INTRIN(_mm512_loadu_pd(left + i + 8), right_vec));             \
      _mm512_storeu_pd(out + i + 16, INTRIN(_mm512_loadu_pd(left + i + 16), right_vec));           \
      _mm512_storeu_pd(out + i + 24, INTRIN(_mm512_loadu_pd(left + i + 24), right_vec));           \
    }                                                                                              \
    const std::size_t aligned = count - count % 8;                                                 \
    for (; i < aligned; i += 8) {                                                                  \
      _mm512_storeu_pd(out + i, INTRIN(_mm512_loadu_pd(left + i), right_vec));                     \
    }                                                                                              \
    for (; i < count; ++i) {                                                                       \
      out[i] = left[i] OPCH right;                                                                 \
    }                                                                                              \
  }

ONNX_LIGHT_CPU_BIN_AVX512_F32(BinaryAddFloat32, _mm512_add_ps, +)
ONNX_LIGHT_CPU_BIN_AVX512_F32(BinarySubFloat32, _mm512_sub_ps, -)
ONNX_LIGHT_CPU_BIN_AVX512_F32(BinaryMulFloat32, _mm512_mul_ps, *)
ONNX_LIGHT_CPU_BIN_AVX512_F32(BinaryDivFloat32, _mm512_div_ps, /)
ONNX_LIGHT_CPU_BIN_AVX512_F64(BinaryAddFloat64, _mm512_add_pd, +)
ONNX_LIGHT_CPU_BIN_AVX512_F64(BinarySubFloat64, _mm512_sub_pd, -)
ONNX_LIGHT_CPU_BIN_AVX512_F64(BinaryMulFloat64, _mm512_mul_pd, *)
ONNX_LIGHT_CPU_BIN_AVX512_F64(BinaryDivFloat64, _mm512_div_pd, /)

void BinaryPReluFloat32_AVX512(const float *left, const float *right, float *out,
                               std::size_t count) {
  std::size_t i = 0;
  const __m512 zero = _mm512_setzero_ps();
  const std::size_t aligned = count - count % 16;
  for (; i < aligned; i += 16) {
    const __m512 x = _mm512_loadu_ps(left + i);
    const __mmask16 negative = _mm512_cmp_ps_mask(x, zero, _CMP_LT_OQ);
    _mm512_storeu_ps(out + i, _mm512_mask_mul_ps(x, negative, x, _mm512_loadu_ps(right + i)));
  }
  for (; i < count; ++i) {
    out[i] = left[i] < 0.0f ? left[i] * right[i] : left[i];
  }
}

void BinaryPReluFloat32Left_AVX512(float left, const float *right, float *out, std::size_t count) {
  if (!(left < 0.0f)) {
    std::fill_n(out, count, left);
    return;
  }
  std::size_t i = 0;
  const __m512 x = _mm512_set1_ps(left);
  const std::size_t aligned = count - count % 16;
  for (; i < aligned; i += 16) {
    _mm512_storeu_ps(out + i, _mm512_mul_ps(x, _mm512_loadu_ps(right + i)));
  }
  for (; i < count; ++i) {
    out[i] = left * right[i];
  }
}

void BinaryPReluFloat32Right_AVX512(const float *left, float right, float *out, std::size_t count) {
  std::size_t i = 0;
  const __m512 zero = _mm512_setzero_ps();
  const __m512 slope = _mm512_set1_ps(right);
  const std::size_t aligned = count - count % 16;
  for (; i < aligned; i += 16) {
    const __m512 x = _mm512_loadu_ps(left + i);
    const __mmask16 negative = _mm512_cmp_ps_mask(x, zero, _CMP_LT_OQ);
    _mm512_storeu_ps(out + i, _mm512_mask_mul_ps(x, negative, x, slope));
  }
  for (; i < count; ++i) {
    out[i] = left[i] < 0.0f ? left[i] * right : left[i];
  }
}

#undef ONNX_LIGHT_CPU_BIN_AVX512_F32
#undef ONNX_LIGHT_CPU_BIN_AVX512_F64

} // namespace onnx_light_cpu
