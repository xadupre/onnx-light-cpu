// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/binary/binary_arithmetic_kernel.h"

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
    const std::size_t aligned = count - count % 16;                                                \
    const __m512 left_vec = _mm512_set1_ps(left);                                                  \
    for (; i < aligned; i += 16) {                                                                 \
      _mm512_storeu_ps(out + i, INTRIN(left_vec, _mm512_loadu_ps(right + i)));                     \
    }                                                                                              \
    for (; i < count; ++i) {                                                                       \
      out[i] = left OPCH right[i];                                                                 \
    }                                                                                              \
  }                                                                                                \
  void STEM##Right_AVX512(const float *left, float right, float *out, std::size_t count) {         \
    std::size_t i = 0;                                                                             \
    const std::size_t aligned = count - count % 16;                                                \
    const __m512 right_vec = _mm512_set1_ps(right);                                                \
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
    const std::size_t aligned = count - count % 8;                                                 \
    const __m512d left_vec = _mm512_set1_pd(left);                                                 \
    for (; i < aligned; i += 8) {                                                                  \
      _mm512_storeu_pd(out + i, INTRIN(left_vec, _mm512_loadu_pd(right + i)));                     \
    }                                                                                              \
    for (; i < count; ++i) {                                                                       \
      out[i] = left OPCH right[i];                                                                 \
    }                                                                                              \
  }                                                                                                \
  void STEM##Right_AVX512(const double *left, double right, double *out, std::size_t count) {      \
    std::size_t i = 0;                                                                             \
    const std::size_t aligned = count - count % 8;                                                 \
    const __m512d right_vec = _mm512_set1_pd(right);                                               \
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

#undef ONNX_LIGHT_CPU_BIN_AVX512_F32
#undef ONNX_LIGHT_CPU_BIN_AVX512_F64

} // namespace onnx_light_cpu
