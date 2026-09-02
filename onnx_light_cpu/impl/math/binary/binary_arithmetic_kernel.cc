// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/binary/binary_arithmetic_kernel.h"

#include "onnx_light_cpu/impl/arm_simd_level.h"
#include "onnx_light_cpu/impl/execution.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define ONNX_LIGHT_CPU_BINARY_X86 1
#include <immintrin.h>
#else
#define ONNX_LIGHT_CPU_BINARY_X86 0
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define ONNX_LIGHT_CPU_BINARY_ARM64 1
#include <arm_neon.h>
#else
#define ONNX_LIGHT_CPU_BINARY_ARM64 0
#endif

namespace onnx_light_cpu {
namespace {

// ---------------------------------------------------------------------------
// Portable scalar reference. Used as the tail of every SIMD loop and as the
// whole loop on an ISA without a wider path.
// ---------------------------------------------------------------------------
#define ONNX_LIGHT_CPU_BIN_SCALAR(STEM, T, OPCH)                                                   \
  void STEM##_Scalar(const T *left, const T *right, T *out, std::size_t count) {                   \
    for (std::size_t i = 0; i < count; ++i) {                                                      \
      out[i] = left[i] OPCH right[i];                                                              \
    }                                                                                              \
  }                                                                                                \
  void STEM##Left_Scalar(T left, const T *right, T *out, std::size_t count) {                      \
    for (std::size_t i = 0; i < count; ++i) {                                                      \
      out[i] = left OPCH right[i];                                                                 \
    }                                                                                              \
  }                                                                                                \
  void STEM##Right_Scalar(const T *left, T right, T *out, std::size_t count) {                     \
    for (std::size_t i = 0; i < count; ++i) {                                                      \
      out[i] = left[i] OPCH right;                                                                 \
    }                                                                                              \
  }

ONNX_LIGHT_CPU_BIN_SCALAR(BinaryAddFloat32, float, +)
ONNX_LIGHT_CPU_BIN_SCALAR(BinarySubFloat32, float, -)
ONNX_LIGHT_CPU_BIN_SCALAR(BinaryMulFloat32, float, *)
ONNX_LIGHT_CPU_BIN_SCALAR(BinaryDivFloat32, float, /)
ONNX_LIGHT_CPU_BIN_SCALAR(BinaryAddFloat64, double, +)
ONNX_LIGHT_CPU_BIN_SCALAR(BinarySubFloat64, double, -)
ONNX_LIGHT_CPU_BIN_SCALAR(BinaryMulFloat64, double, *)
ONNX_LIGHT_CPU_BIN_SCALAR(BinaryDivFloat64, double, /)

#undef ONNX_LIGHT_CPU_BIN_SCALAR

void BinaryPReluFloat32_Scalar(const float *left, const float *right, float *out,
                               std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    out[i] = left[i] < 0.0f ? left[i] * right[i] : left[i];
  }
}

void BinaryPReluFloat32Left_Scalar(float left, const float *right, float *out, std::size_t count) {
  if (!(left < 0.0f)) {
    std::fill_n(out, count, left);
    return;
  }
  for (std::size_t i = 0; i < count; ++i) {
    out[i] = left * right[i];
  }
}

void BinaryPReluFloat32Right_Scalar(const float *left, float right, float *out, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    out[i] = left[i] < 0.0f ? left[i] * right : left[i];
  }
}

#if ONNX_LIGHT_CPU_BINARY_X86

// ---------------------------------------------------------------------------
// SSE2 (128-bit) and AVX2 (256-bit, enabled by the library's default
// ``-mavx2`` compile flags on x86) implementations. The intrinsic names
// (``_mm[256]_<op>_p[sd]``) are regular across Add/Sub/Mul/Div, so a single
// macro instantiates the whole width x type x operator matrix.
// ---------------------------------------------------------------------------
#define ONNX_LIGHT_CPU_BIN_SSE2_F32(STEM, INTRIN, OPCH)                                            \
  void STEM##_SSE2(const float *left, const float *right, float *out, std::size_t count) {         \
    std::size_t i = 0;                                                                             \
    const std::size_t aligned = count - count % 4;                                                 \
    for (; i < aligned; i += 4) {                                                                  \
      _mm_storeu_ps(out + i, INTRIN(_mm_loadu_ps(left + i), _mm_loadu_ps(right + i)));             \
    }                                                                                              \
    for (; i < count; ++i) {                                                                       \
      out[i] = left[i] OPCH right[i];                                                              \
    }                                                                                              \
  }                                                                                                \
  void STEM##Left_SSE2(float left, const float *right, float *out, std::size_t count) {            \
    std::size_t i = 0;                                                                             \
    const std::size_t aligned = count - count % 4;                                                 \
    const __m128 left_vec = _mm_set1_ps(left);                                                     \
    for (; i < aligned; i += 4) {                                                                  \
      _mm_storeu_ps(out + i, INTRIN(left_vec, _mm_loadu_ps(right + i)));                           \
    }                                                                                              \
    for (; i < count; ++i) {                                                                       \
      out[i] = left OPCH right[i];                                                                 \
    }                                                                                              \
  }                                                                                                \
  void STEM##Right_SSE2(const float *left, float right, float *out, std::size_t count) {           \
    std::size_t i = 0;                                                                             \
    const std::size_t aligned = count - count % 4;                                                 \
    const __m128 right_vec = _mm_set1_ps(right);                                                   \
    for (; i < aligned; i += 4) {                                                                  \
      _mm_storeu_ps(out + i, INTRIN(_mm_loadu_ps(left + i), right_vec));                           \
    }                                                                                              \
    for (; i < count; ++i) {                                                                       \
      out[i] = left[i] OPCH right;                                                                 \
    }                                                                                              \
  }

#define ONNX_LIGHT_CPU_BIN_SSE2_F64(STEM, INTRIN, OPCH)                                            \
  void STEM##_SSE2(const double *left, const double *right, double *out, std::size_t count) {      \
    std::size_t i = 0;                                                                             \
    const std::size_t aligned = count - count % 2;                                                 \
    for (; i < aligned; i += 2) {                                                                  \
      _mm_storeu_pd(out + i, INTRIN(_mm_loadu_pd(left + i), _mm_loadu_pd(right + i)));             \
    }                                                                                              \
    for (; i < count; ++i) {                                                                       \
      out[i] = left[i] OPCH right[i];                                                              \
    }                                                                                              \
  }                                                                                                \
  void STEM##Left_SSE2(double left, const double *right, double *out, std::size_t count) {         \
    std::size_t i = 0;                                                                             \
    const std::size_t aligned = count - count % 2;                                                 \
    const __m128d left_vec = _mm_set1_pd(left);                                                    \
    for (; i < aligned; i += 2) {                                                                  \
      _mm_storeu_pd(out + i, INTRIN(left_vec, _mm_loadu_pd(right + i)));                           \
    }                                                                                              \
    for (; i < count; ++i) {                                                                       \
      out[i] = left OPCH right[i];                                                                 \
    }                                                                                              \
  }                                                                                                \
  void STEM##Right_SSE2(const double *left, double right, double *out, std::size_t count) {        \
    std::size_t i = 0;                                                                             \
    const std::size_t aligned = count - count % 2;                                                 \
    const __m128d right_vec = _mm_set1_pd(right);                                                  \
    for (; i < aligned; i += 2) {                                                                  \
      _mm_storeu_pd(out + i, INTRIN(_mm_loadu_pd(left + i), right_vec));                           \
    }                                                                                              \
    for (; i < count; ++i) {                                                                       \
      out[i] = left[i] OPCH right;                                                                 \
    }                                                                                              \
  }

#define ONNX_LIGHT_CPU_BIN_AVX2_F32(STEM, INTRIN, OPCH)                                            \
  void STEM##_AVX2(const float *left, const float *right, float *out, std::size_t count) {         \
    std::size_t i = 0;                                                                             \
    const std::size_t aligned = count - count % 8;                                                 \
    for (; i < aligned; i += 8) {                                                                  \
      _mm256_storeu_ps(out + i, INTRIN(_mm256_loadu_ps(left + i), _mm256_loadu_ps(right + i)));    \
    }                                                                                              \
    for (; i < count; ++i) {                                                                       \
      out[i] = left[i] OPCH right[i];                                                              \
    }                                                                                              \
  }                                                                                                \
  void STEM##Left_AVX2(float left, const float *right, float *out, std::size_t count) {            \
    std::size_t i = 0;                                                                             \
    const std::size_t aligned = count - count % 8;                                                 \
    const __m256 left_vec = _mm256_set1_ps(left);                                                  \
    for (; i < aligned; i += 8) {                                                                  \
      _mm256_storeu_ps(out + i, INTRIN(left_vec, _mm256_loadu_ps(right + i)));                     \
    }                                                                                              \
    for (; i < count; ++i) {                                                                       \
      out[i] = left OPCH right[i];                                                                 \
    }                                                                                              \
  }                                                                                                \
  void STEM##Right_AVX2(const float *left, float right, float *out, std::size_t count) {           \
    std::size_t i = 0;                                                                             \
    const std::size_t aligned = count - count % 8;                                                 \
    const __m256 right_vec = _mm256_set1_ps(right);                                                \
    for (; i < aligned; i += 8) {                                                                  \
      _mm256_storeu_ps(out + i, INTRIN(_mm256_loadu_ps(left + i), right_vec));                     \
    }                                                                                              \
    for (; i < count; ++i) {                                                                       \
      out[i] = left[i] OPCH right;                                                                 \
    }                                                                                              \
  }

#define ONNX_LIGHT_CPU_BIN_AVX2_F64(STEM, INTRIN, OPCH)                                            \
  void STEM##_AVX2(const double *left, const double *right, double *out, std::size_t count) {      \
    std::size_t i = 0;                                                                             \
    const std::size_t aligned = count - count % 4;                                                 \
    for (; i < aligned; i += 4) {                                                                  \
      _mm256_storeu_pd(out + i, INTRIN(_mm256_loadu_pd(left + i), _mm256_loadu_pd(right + i)));    \
    }                                                                                              \
    for (; i < count; ++i) {                                                                       \
      out[i] = left[i] OPCH right[i];                                                              \
    }                                                                                              \
  }                                                                                                \
  void STEM##Left_AVX2(double left, const double *right, double *out, std::size_t count) {         \
    std::size_t i = 0;                                                                             \
    const std::size_t aligned = count - count % 4;                                                 \
    const __m256d left_vec = _mm256_set1_pd(left);                                                 \
    for (; i < aligned; i += 4) {                                                                  \
      _mm256_storeu_pd(out + i, INTRIN(left_vec, _mm256_loadu_pd(right + i)));                     \
    }                                                                                              \
    for (; i < count; ++i) {                                                                       \
      out[i] = left OPCH right[i];                                                                 \
    }                                                                                              \
  }                                                                                                \
  void STEM##Right_AVX2(const double *left, double right, double *out, std::size_t count) {        \
    std::size_t i = 0;                                                                             \
    const std::size_t aligned = count - count % 4;                                                 \
    const __m256d right_vec = _mm256_set1_pd(right);                                               \
    for (; i < aligned; i += 4) {                                                                  \
      _mm256_storeu_pd(out + i, INTRIN(_mm256_loadu_pd(left + i), right_vec));                     \
    }                                                                                              \
    for (; i < count; ++i) {                                                                       \
      out[i] = left[i] OPCH right;                                                                 \
    }                                                                                              \
  }

ONNX_LIGHT_CPU_BIN_SSE2_F32(BinaryAddFloat32, _mm_add_ps, +)
ONNX_LIGHT_CPU_BIN_SSE2_F32(BinarySubFloat32, _mm_sub_ps, -)
ONNX_LIGHT_CPU_BIN_SSE2_F32(BinaryMulFloat32, _mm_mul_ps, *)
ONNX_LIGHT_CPU_BIN_SSE2_F32(BinaryDivFloat32, _mm_div_ps, /)
ONNX_LIGHT_CPU_BIN_SSE2_F64(BinaryAddFloat64, _mm_add_pd, +)
ONNX_LIGHT_CPU_BIN_SSE2_F64(BinarySubFloat64, _mm_sub_pd, -)
ONNX_LIGHT_CPU_BIN_SSE2_F64(BinaryMulFloat64, _mm_mul_pd, *)
ONNX_LIGHT_CPU_BIN_SSE2_F64(BinaryDivFloat64, _mm_div_pd, /)

ONNX_LIGHT_CPU_BIN_AVX2_F32(BinaryAddFloat32, _mm256_add_ps, +)
ONNX_LIGHT_CPU_BIN_AVX2_F32(BinarySubFloat32, _mm256_sub_ps, -)
ONNX_LIGHT_CPU_BIN_AVX2_F32(BinaryMulFloat32, _mm256_mul_ps, *)
ONNX_LIGHT_CPU_BIN_AVX2_F32(BinaryDivFloat32, _mm256_div_ps, /)
ONNX_LIGHT_CPU_BIN_AVX2_F64(BinaryAddFloat64, _mm256_add_pd, +)
ONNX_LIGHT_CPU_BIN_AVX2_F64(BinarySubFloat64, _mm256_sub_pd, -)
ONNX_LIGHT_CPU_BIN_AVX2_F64(BinaryMulFloat64, _mm256_mul_pd, *)
ONNX_LIGHT_CPU_BIN_AVX2_F64(BinaryDivFloat64, _mm256_div_pd, /)

#define ONNX_LIGHT_CPU_PRELU_X86(SUFFIX, VECTOR, LANES, LOAD, STORE, SET1, ZERO, CMP, MUL, AND,    \
                                 ANDNOT, OR)                                                       \
  void BinaryPReluFloat32##SUFFIX(const float *left, const float *right, float *out,               \
                                  std::size_t count) {                                             \
    std::size_t i = 0;                                                                             \
    const std::size_t aligned = count - count % LANES;                                             \
    const VECTOR zero = ZERO();                                                                    \
    for (; i < aligned; i += LANES) {                                                              \
      const VECTOR x = LOAD(left + i);                                                             \
      const VECTOR mask = CMP(x, zero);                                                            \
      const VECTOR scaled = MUL(x, LOAD(right + i));                                               \
      STORE(out + i, OR(AND(mask, scaled), ANDNOT(mask, x)));                                      \
    }                                                                                              \
    BinaryPReluFloat32_Scalar(left + i, right + i, out + i, count - i);                            \
  }                                                                                                \
  void BinaryPReluFloat32Left##SUFFIX(float left, const float *right, float *out,                  \
                                      std::size_t count) {                                         \
    if (!(left < 0.0f)) {                                                                          \
      std::fill_n(out, count, left);                                                               \
      return;                                                                                      \
    }                                                                                              \
    std::size_t i = 0;                                                                             \
    const std::size_t aligned = count - count % LANES;                                             \
    const VECTOR x = SET1(left);                                                                   \
    for (; i < aligned; i += LANES) {                                                              \
      STORE(out + i, MUL(x, LOAD(right + i)));                                                     \
    }                                                                                              \
    BinaryPReluFloat32Left_Scalar(left, right + i, out + i, count - i);                            \
  }                                                                                                \
  void BinaryPReluFloat32Right##SUFFIX(const float *left, float right, float *out,                 \
                                       std::size_t count) {                                        \
    std::size_t i = 0;                                                                             \
    const std::size_t aligned = count - count % LANES;                                             \
    const VECTOR zero = ZERO();                                                                    \
    const VECTOR slope = SET1(right);                                                              \
    for (; i < aligned; i += LANES) {                                                              \
      const VECTOR x = LOAD(left + i);                                                             \
      const VECTOR mask = CMP(x, zero);                                                            \
      const VECTOR scaled = MUL(x, slope);                                                         \
      STORE(out + i, OR(AND(mask, scaled), ANDNOT(mask, x)));                                      \
    }                                                                                              \
    BinaryPReluFloat32Right_Scalar(left + i, right, out + i, count - i);                           \
  }

__m128 CompareLessThan(__m128 left, __m128 right) { return _mm_cmplt_ps(left, right); }

__m256 CompareLessThan(__m256 left, __m256 right) { return _mm256_cmp_ps(left, right, _CMP_LT_OQ); }

ONNX_LIGHT_CPU_PRELU_X86(_SSE2, __m128, 4, _mm_loadu_ps, _mm_storeu_ps, _mm_set1_ps, _mm_setzero_ps,
                         CompareLessThan, _mm_mul_ps, _mm_and_ps, _mm_andnot_ps, _mm_or_ps)
ONNX_LIGHT_CPU_PRELU_X86(_AVX2, __m256, 8, _mm256_loadu_ps, _mm256_storeu_ps, _mm256_set1_ps,
                         _mm256_setzero_ps, CompareLessThan, _mm256_mul_ps, _mm256_and_ps,
                         _mm256_andnot_ps, _mm256_or_ps)

#undef ONNX_LIGHT_CPU_PRELU_X86

#undef ONNX_LIGHT_CPU_BIN_SSE2_F32
#undef ONNX_LIGHT_CPU_BIN_SSE2_F64
#undef ONNX_LIGHT_CPU_BIN_AVX2_F32
#undef ONNX_LIGHT_CPU_BIN_AVX2_F64

#endif // ONNX_LIGHT_CPU_BINARY_X86

#if ONNX_LIGHT_CPU_BINARY_ARM64

// ---------------------------------------------------------------------------
// NEON (Advanced SIMD) implementations. Baseline on AArch64, so this needs no
// dedicated translation unit or extra compile flags.
// ---------------------------------------------------------------------------
#define ONNX_LIGHT_CPU_BIN_NEON_F32(STEM, INTRIN, OPCH)                                            \
  void STEM##_NEON(const float *left, const float *right, float *out, std::size_t count) {         \
    std::size_t i = 0;                                                                             \
    const std::size_t aligned = count - count % 4;                                                 \
    for (; i < aligned; i += 4) {                                                                  \
      vst1q_f32(out + i, INTRIN(vld1q_f32(left + i), vld1q_f32(right + i)));                       \
    }                                                                                              \
    for (; i < count; ++i) {                                                                       \
      out[i] = left[i] OPCH right[i];                                                              \
    }                                                                                              \
  }                                                                                                \
  void STEM##Left_NEON(float left, const float *right, float *out, std::size_t count) {            \
    std::size_t i = 0;                                                                             \
    const std::size_t aligned = count - count % 4;                                                 \
    const float32x4_t left_vec = vdupq_n_f32(left);                                                \
    for (; i < aligned; i += 4) {                                                                  \
      vst1q_f32(out + i, INTRIN(left_vec, vld1q_f32(right + i)));                                  \
    }                                                                                              \
    for (; i < count; ++i) {                                                                       \
      out[i] = left OPCH right[i];                                                                 \
    }                                                                                              \
  }                                                                                                \
  void STEM##Right_NEON(const float *left, float right, float *out, std::size_t count) {           \
    std::size_t i = 0;                                                                             \
    const std::size_t aligned = count - count % 4;                                                 \
    const float32x4_t right_vec = vdupq_n_f32(right);                                              \
    for (; i < aligned; i += 4) {                                                                  \
      vst1q_f32(out + i, INTRIN(vld1q_f32(left + i), right_vec));                                  \
    }                                                                                              \
    for (; i < count; ++i) {                                                                       \
      out[i] = left[i] OPCH right;                                                                 \
    }                                                                                              \
  }

#define ONNX_LIGHT_CPU_BIN_NEON_F64(STEM, INTRIN, OPCH)                                            \
  void STEM##_NEON(const double *left, const double *right, double *out, std::size_t count) {      \
    std::size_t i = 0;                                                                             \
    const std::size_t aligned = count - count % 2;                                                 \
    for (; i < aligned; i += 2) {                                                                  \
      vst1q_f64(out + i, INTRIN(vld1q_f64(left + i), vld1q_f64(right + i)));                       \
    }                                                                                              \
    for (; i < count; ++i) {                                                                       \
      out[i] = left[i] OPCH right[i];                                                              \
    }                                                                                              \
  }                                                                                                \
  void STEM##Left_NEON(double left, const double *right, double *out, std::size_t count) {         \
    std::size_t i = 0;                                                                             \
    const std::size_t aligned = count - count % 2;                                                 \
    const float64x2_t left_vec = vdupq_n_f64(left);                                                \
    for (; i < aligned; i += 2) {                                                                  \
      vst1q_f64(out + i, INTRIN(left_vec, vld1q_f64(right + i)));                                  \
    }                                                                                              \
    for (; i < count; ++i) {                                                                       \
      out[i] = left OPCH right[i];                                                                 \
    }                                                                                              \
  }                                                                                                \
  void STEM##Right_NEON(const double *left, double right, double *out, std::size_t count) {        \
    std::size_t i = 0;                                                                             \
    const std::size_t aligned = count - count % 2;                                                 \
    const float64x2_t right_vec = vdupq_n_f64(right);                                              \
    for (; i < aligned; i += 2) {                                                                  \
      vst1q_f64(out + i, INTRIN(vld1q_f64(left + i), right_vec));                                  \
    }                                                                                              \
    for (; i < count; ++i) {                                                                       \
      out[i] = left[i] OPCH right;                                                                 \
    }                                                                                              \
  }

ONNX_LIGHT_CPU_BIN_NEON_F32(BinaryAddFloat32, vaddq_f32, +)
ONNX_LIGHT_CPU_BIN_NEON_F32(BinarySubFloat32, vsubq_f32, -)
ONNX_LIGHT_CPU_BIN_NEON_F32(BinaryMulFloat32, vmulq_f32, *)
ONNX_LIGHT_CPU_BIN_NEON_F32(BinaryDivFloat32, vdivq_f32, /)
ONNX_LIGHT_CPU_BIN_NEON_F64(BinaryAddFloat64, vaddq_f64, +)
ONNX_LIGHT_CPU_BIN_NEON_F64(BinarySubFloat64, vsubq_f64, -)
ONNX_LIGHT_CPU_BIN_NEON_F64(BinaryMulFloat64, vmulq_f64, *)
ONNX_LIGHT_CPU_BIN_NEON_F64(BinaryDivFloat64, vdivq_f64, /)

#undef ONNX_LIGHT_CPU_BIN_NEON_F32
#undef ONNX_LIGHT_CPU_BIN_NEON_F64

#endif // ONNX_LIGHT_CPU_BINARY_ARM64

// ---------------------------------------------------------------------------
// ISA pointer helpers. Each resolves to the real implementation's address
// when the translation unit was compiled for that ISA family, or to
// ``nullptr`` otherwise, so the platform-neutral selection code below never
// names a symbol that does not exist in this translation unit.
// ---------------------------------------------------------------------------
#if ONNX_LIGHT_CPU_BINARY_X86
#define ONNX_LIGHT_CPU_BIN_SSE2_PTR(STEM) (&STEM##_SSE2)
#define ONNX_LIGHT_CPU_BIN_SSE2_LEFT_PTR(STEM) (&STEM##Left_SSE2)
#define ONNX_LIGHT_CPU_BIN_SSE2_RIGHT_PTR(STEM) (&STEM##Right_SSE2)
#define ONNX_LIGHT_CPU_BIN_AVX2_PTR(STEM) (&STEM##_AVX2)
#define ONNX_LIGHT_CPU_BIN_AVX2_LEFT_PTR(STEM) (&STEM##Left_AVX2)
#define ONNX_LIGHT_CPU_BIN_AVX2_RIGHT_PTR(STEM) (&STEM##Right_AVX2)
#else
#define ONNX_LIGHT_CPU_BIN_SSE2_PTR(STEM) (nullptr)
#define ONNX_LIGHT_CPU_BIN_SSE2_LEFT_PTR(STEM) (nullptr)
#define ONNX_LIGHT_CPU_BIN_SSE2_RIGHT_PTR(STEM) (nullptr)
#define ONNX_LIGHT_CPU_BIN_AVX2_PTR(STEM) (nullptr)
#define ONNX_LIGHT_CPU_BIN_AVX2_LEFT_PTR(STEM) (nullptr)
#define ONNX_LIGHT_CPU_BIN_AVX2_RIGHT_PTR(STEM) (nullptr)
#endif

#if ONNX_LIGHT_CPU_BINARY_ARM64
#define ONNX_LIGHT_CPU_BIN_NEON_PTR(STEM) (&STEM##_NEON)
#define ONNX_LIGHT_CPU_BIN_NEON_LEFT_PTR(STEM) (&STEM##Left_NEON)
#define ONNX_LIGHT_CPU_BIN_NEON_RIGHT_PTR(STEM) (&STEM##Right_NEON)
#else
#define ONNX_LIGHT_CPU_BIN_NEON_PTR(STEM) (nullptr)
#define ONNX_LIGHT_CPU_BIN_NEON_LEFT_PTR(STEM) (nullptr)
#define ONNX_LIGHT_CPU_BIN_NEON_RIGHT_PTR(STEM) (nullptr)
#endif

#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
#define ONNX_LIGHT_CPU_BIN_AVX512_PTR(STEM) (&STEM##_AVX512)
#define ONNX_LIGHT_CPU_BIN_AVX512_LEFT_PTR(STEM) (&STEM##Left_AVX512)
#define ONNX_LIGHT_CPU_BIN_AVX512_RIGHT_PTR(STEM) (&STEM##Right_AVX512)
#else
#define ONNX_LIGHT_CPU_BIN_AVX512_PTR(STEM) (nullptr)
#define ONNX_LIGHT_CPU_BIN_AVX512_LEFT_PTR(STEM) (nullptr)
#define ONNX_LIGHT_CPU_BIN_AVX512_RIGHT_PTR(STEM) (nullptr)
#endif

#ifdef ONNX_LIGHT_CPU_HAVE_SVE
#define ONNX_LIGHT_CPU_BIN_SVE_PTR(STEM) (&STEM##_SVE)
#define ONNX_LIGHT_CPU_BIN_SVE_LEFT_PTR(STEM) (&STEM##Left_SVE)
#define ONNX_LIGHT_CPU_BIN_SVE_RIGHT_PTR(STEM) (&STEM##Right_SVE)
#else
#define ONNX_LIGHT_CPU_BIN_SVE_PTR(STEM) (nullptr)
#define ONNX_LIGHT_CPU_BIN_SVE_LEFT_PTR(STEM) (nullptr)
#define ONNX_LIGHT_CPU_BIN_SVE_RIGHT_PTR(STEM) (nullptr)
#endif

// ---------------------------------------------------------------------------
// Runtime dispatch. Picks the widest available ISA once (cached in a
// function-local static) and forwards every subsequent call directly to it.
// Never touches a symbol from an unsupported ISA (each family's pointer is
// ``nullptr`` outside its own ``#if``/``#ifdef`` block above), so it never
// executes an unsupported instruction.
// ---------------------------------------------------------------------------
template <typename Fn>
inline Fn PickImpl(Fn scalar_fn, Fn sse2_fn, Fn avx2_fn, Fn avx512_fn, Fn neon_fn, Fn sve_fn) {
#if ONNX_LIGHT_CPU_BINARY_X86
  (void)neon_fn;
  (void)sve_fn;
  static const SimdLevel level = DetectSimdLevel();
  if (avx512_fn != nullptr && level >= SimdLevel::kAVX512) {
    return avx512_fn;
  }
  if (level >= SimdLevel::kAVX) {
    return avx2_fn;
  }
  if (level >= SimdLevel::kSSE2) {
    return sse2_fn;
  }
  return scalar_fn;
#elif ONNX_LIGHT_CPU_BINARY_ARM64
  (void)sse2_fn;
  (void)avx2_fn;
  (void)avx512_fn;
  static const ArmSimdLevel level = DetectArmSimdLevel();
  if (sve_fn != nullptr && (level == ArmSimdLevel::kSve || level == ArmSimdLevel::kSve2)) {
    return sve_fn;
  }
  if (level != ArmSimdLevel::kNone) {
    return neon_fn;
  }
  return scalar_fn;
#else
  (void)sse2_fn;
  (void)avx2_fn;
  (void)avx512_fn;
  (void)neon_fn;
  (void)sve_fn;
  return scalar_fn;
#endif
}

#define ONNX_LIGHT_CPU_BIN_DISPATCH_CONTIG(PUBLIC, STEM, T)                                        \
  void PUBLIC##Contiguous(const T *left, const T *right, T *out, std::size_t count) {              \
    if (count == 0) {                                                                              \
      return;                                                                                      \
    }                                                                                              \
    using Fn = void (*)(const T *, const T *, T *, std::size_t);                                   \
    static const Fn fn =                                                                           \
        PickImpl<Fn>(&STEM##_Scalar, ONNX_LIGHT_CPU_BIN_SSE2_PTR(STEM),                            \
                     ONNX_LIGHT_CPU_BIN_AVX2_PTR(STEM), ONNX_LIGHT_CPU_BIN_AVX512_PTR(STEM),       \
                     ONNX_LIGHT_CPU_BIN_NEON_PTR(STEM), ONNX_LIGHT_CPU_BIN_SVE_PTR(STEM));         \
    fn(left, right, out, count);                                                                   \
  }

#define ONNX_LIGHT_CPU_BIN_DISPATCH_LEFT(PUBLIC, STEM, T)                                          \
  void PUBLIC##LeftScalar(T left, const T *right, T *out, std::size_t count) {                     \
    if (count == 0) {                                                                              \
      return;                                                                                      \
    }                                                                                              \
    using Fn = void (*)(T, const T *, T *, std::size_t);                                           \
    static const Fn fn = PickImpl<Fn>(                                                             \
        &STEM##Left_Scalar, ONNX_LIGHT_CPU_BIN_SSE2_LEFT_PTR(STEM),                                \
        ONNX_LIGHT_CPU_BIN_AVX2_LEFT_PTR(STEM), ONNX_LIGHT_CPU_BIN_AVX512_LEFT_PTR(STEM),          \
        ONNX_LIGHT_CPU_BIN_NEON_LEFT_PTR(STEM), ONNX_LIGHT_CPU_BIN_SVE_LEFT_PTR(STEM));            \
    fn(left, right, out, count);                                                                   \
  }

#define ONNX_LIGHT_CPU_BIN_DISPATCH_RIGHT(PUBLIC, STEM, T)                                         \
  void PUBLIC##RightScalar(const T *left, T right, T *out, std::size_t count) {                    \
    if (count == 0) {                                                                              \
      return;                                                                                      \
    }                                                                                              \
    using Fn = void (*)(const T *, T, T *, std::size_t);                                           \
    static const Fn fn = PickImpl<Fn>(                                                             \
        &STEM##Right_Scalar, ONNX_LIGHT_CPU_BIN_SSE2_RIGHT_PTR(STEM),                              \
        ONNX_LIGHT_CPU_BIN_AVX2_RIGHT_PTR(STEM), ONNX_LIGHT_CPU_BIN_AVX512_RIGHT_PTR(STEM),        \
        ONNX_LIGHT_CPU_BIN_NEON_RIGHT_PTR(STEM), ONNX_LIGHT_CPU_BIN_SVE_RIGHT_PTR(STEM));          \
    fn(left, right, out, count);                                                                   \
  }

#define ONNX_LIGHT_CPU_BIN_DISPATCH(PUBLIC, STEM, T)                                               \
  ONNX_LIGHT_CPU_BIN_DISPATCH_CONTIG(PUBLIC, STEM, T)                                              \
  ONNX_LIGHT_CPU_BIN_DISPATCH_LEFT(PUBLIC, STEM, T)                                                \
  ONNX_LIGHT_CPU_BIN_DISPATCH_RIGHT(PUBLIC, STEM, T)

enum class ShiftInputMode : std::uint8_t {
  kContiguous,
  kLeftScalar,
  kRightScalar,
};

template <typename T, bool Left, ShiftInputMode Mode>
void BinaryBitShift_Scalar(const T *left, const T *right, T *out, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    const T lhs = left[Mode == ShiftInputMode::kLeftScalar ? 0 : i];
    const T rhs = right[Mode == ShiftInputMode::kRightScalar ? 0 : i];
    out[i] = Left ? static_cast<T>(lhs << rhs) : static_cast<T>(lhs >> rhs);
  }
}

#if ONNX_LIGHT_CPU_BINARY_X86

template <typename T> __m256i LoadShiftValues_AVX2(const T *values) {
  if constexpr (sizeof(T) == 8 || sizeof(T) == 4) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(values));
  } else if constexpr (sizeof(T) == 2) {
    return _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(values)));
  } else {
    return _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(values)));
  }
}

template <typename T> __m256i BroadcastShiftValue_AVX2(T value) {
  if constexpr (sizeof(T) == 8) {
    return _mm256_set1_epi64x(static_cast<std::int64_t>(value));
  } else {
    return _mm256_set1_epi32(static_cast<std::int32_t>(value));
  }
}

template <typename T> void StoreShiftValues_AVX2(T *output, __m256i values) {
  if constexpr (sizeof(T) == 8 || sizeof(T) == 4) {
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(output), values);
  } else {
    const __m128i low = _mm256_castsi256_si128(values);
    const __m128i high = _mm256_extracti128_si256(values, 1);
    const __m128i packed16 = _mm_packus_epi32(low, high);
    if constexpr (sizeof(T) == 2) {
      _mm_storeu_si128(reinterpret_cast<__m128i *>(output), packed16);
    } else {
      const __m128i packed8 = _mm_packus_epi16(packed16, _mm_setzero_si128());
      _mm_storel_epi64(reinterpret_cast<__m128i *>(output), packed8);
    }
  }
}

template <typename T, bool Left> __m256i ShiftValues_AVX2(__m256i values, __m256i shifts) {
  __m256i shifted;
  if constexpr (sizeof(T) == 8) {
    shifted = Left ? _mm256_sllv_epi64(values, shifts) : _mm256_srlv_epi64(values, shifts);
  } else {
    shifted = Left ? _mm256_sllv_epi32(values, shifts) : _mm256_srlv_epi32(values, shifts);
  }
  if constexpr (sizeof(T) == 2) {
    return _mm256_and_si256(shifted, _mm256_set1_epi32(0xFFFF));
  } else if constexpr (sizeof(T) == 1) {
    return _mm256_and_si256(shifted, _mm256_set1_epi32(0xFF));
  } else {
    return shifted;
  }
}

template <typename T, bool Left, ShiftInputMode Mode>
void BinaryBitShift_AVX2(const T *left, const T *right, T *out, std::size_t count) {
  constexpr std::size_t kLanes = sizeof(T) == 8 ? 4 : 8;
  const __m256i left_scalar = Mode == ShiftInputMode::kLeftScalar
                                  ? BroadcastShiftValue_AVX2(left[0])
                                  : _mm256_setzero_si256();
  const __m256i right_scalar = Mode == ShiftInputMode::kRightScalar
                                   ? BroadcastShiftValue_AVX2(right[0])
                                   : _mm256_setzero_si256();
  std::size_t i = 0;
  for (; i + kLanes <= count; i += kLanes) {
    const __m256i values =
        Mode == ShiftInputMode::kLeftScalar ? left_scalar : LoadShiftValues_AVX2(left + i);
    const __m256i shifts =
        Mode == ShiftInputMode::kRightScalar ? right_scalar : LoadShiftValues_AVX2(right + i);
    StoreShiftValues_AVX2(out + i, ShiftValues_AVX2<T, Left>(values, shifts));
  }
  BinaryBitShift_Scalar<T, Left, Mode>(left + (Mode == ShiftInputMode::kLeftScalar ? 0 : i),
                                       right + (Mode == ShiftInputMode::kRightScalar ? 0 : i),
                                       out + i, count - i);
}

template <typename T> bool BinaryBitShiftHasInvalidAmount_AVX2(const T *shifts, std::size_t count) {
  constexpr std::size_t kLanes = 32 / sizeof(T);
  __m256i maximum = _mm256_setzero_si256();
  std::size_t i = 0;
  for (; i + kLanes <= count; i += kLanes) {
    const __m256i values = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(shifts + i));
    if constexpr (sizeof(T) == 1) {
      maximum = _mm256_max_epu8(maximum, values);
    } else if constexpr (sizeof(T) == 2) {
      maximum = _mm256_max_epu16(maximum, values);
    } else if constexpr (sizeof(T) == 4) {
      maximum = _mm256_max_epu32(maximum, values);
    } else {
      const __m256i sign = _mm256_set1_epi64x(std::numeric_limits<std::int64_t>::min());
      const __m256i greater =
          _mm256_cmpgt_epi64(_mm256_xor_si256(values, sign), _mm256_xor_si256(maximum, sign));
      maximum = _mm256_blendv_epi8(maximum, values, greater);
    }
  }
  alignas(32) std::array<T, kLanes> lane_maxima;
  _mm256_store_si256(reinterpret_cast<__m256i *>(lane_maxima.data()), maximum);
  const T bit_width = static_cast<T>(sizeof(T) * 8);
  for (T value : lane_maxima) {
    if (value >= bit_width) {
      return true;
    }
  }
  for (; i < count; ++i) {
    if (shifts[i] >= bit_width) {
      return true;
    }
  }
  return false;
}

#endif // ONNX_LIGHT_CPU_BINARY_X86

} // namespace

ONNX_LIGHT_CPU_BIN_DISPATCH(BinaryAddFloat32, BinaryAddFloat32, float)
ONNX_LIGHT_CPU_BIN_DISPATCH(BinarySubFloat32, BinarySubFloat32, float)
ONNX_LIGHT_CPU_BIN_DISPATCH(BinaryMulFloat32, BinaryMulFloat32, float)
ONNX_LIGHT_CPU_BIN_DISPATCH(BinaryDivFloat32, BinaryDivFloat32, float)
ONNX_LIGHT_CPU_BIN_DISPATCH(BinaryAddFloat64, BinaryAddFloat64, double)
ONNX_LIGHT_CPU_BIN_DISPATCH(BinarySubFloat64, BinarySubFloat64, double)
ONNX_LIGHT_CPU_BIN_DISPATCH(BinaryMulFloat64, BinaryMulFloat64, double)
ONNX_LIGHT_CPU_BIN_DISPATCH(BinaryDivFloat64, BinaryDivFloat64, double)

void BinaryPReluFloat32Contiguous(const float *left, const float *right, float *out,
                                  std::size_t count) {
  using Fn = void (*)(const float *, const float *, float *, std::size_t);
  static const Fn fn =
      PickImpl<Fn>(&BinaryPReluFloat32_Scalar, ONNX_LIGHT_CPU_BIN_SSE2_PTR(BinaryPReluFloat32),
                   ONNX_LIGHT_CPU_BIN_AVX2_PTR(BinaryPReluFloat32),
                   ONNX_LIGHT_CPU_BIN_AVX512_PTR(BinaryPReluFloat32), &BinaryPReluFloat32_Scalar,
                   &BinaryPReluFloat32_Scalar);
  fn(left, right, out, count);
}

void BinaryPReluFloat32LeftScalar(float left, const float *right, float *out, std::size_t count) {
  using Fn = void (*)(float, const float *, float *, std::size_t);
  static const Fn fn = PickImpl<Fn>(&BinaryPReluFloat32Left_Scalar,
                                    ONNX_LIGHT_CPU_BIN_SSE2_LEFT_PTR(BinaryPReluFloat32),
                                    ONNX_LIGHT_CPU_BIN_AVX2_LEFT_PTR(BinaryPReluFloat32),
                                    ONNX_LIGHT_CPU_BIN_AVX512_LEFT_PTR(BinaryPReluFloat32),
                                    &BinaryPReluFloat32Left_Scalar, &BinaryPReluFloat32Left_Scalar);
  fn(left, right, out, count);
}

void BinaryPReluFloat32RightScalar(const float *left, float right, float *out, std::size_t count) {
  using Fn = void (*)(const float *, float, float *, std::size_t);
  static const Fn fn = PickImpl<Fn>(
      &BinaryPReluFloat32Right_Scalar, ONNX_LIGHT_CPU_BIN_SSE2_RIGHT_PTR(BinaryPReluFloat32),
      ONNX_LIGHT_CPU_BIN_AVX2_RIGHT_PTR(BinaryPReluFloat32),
      ONNX_LIGHT_CPU_BIN_AVX512_RIGHT_PTR(BinaryPReluFloat32), &BinaryPReluFloat32Right_Scalar,
      &BinaryPReluFloat32Right_Scalar);
  fn(left, right, out, count);
}

template <typename T, T MaxRoot>
bool BinarySquare_Scalar(const T *input, T *output, std::size_t count) {
  bool overflow = false;
  for (std::size_t i = 0; i < count; ++i) {
    const bool element_overflows = input[i] > MaxRoot || input[i] < -MaxRoot;
    overflow |= element_overflows;
    if (!element_overflows) {
      output[i] = input[i] * input[i];
    }
  }
  return !overflow;
}

bool BinarySquareInt32(const std::int32_t *input, std::int32_t *output, std::size_t count) {
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
  if (DetectSimdLevel() >= SimdLevel::kAVX512) {
    return BinarySquareInt32_AVX512(input, output, count);
  }
#endif
  return BinarySquare_Scalar<std::int32_t, 46340>(input, output, count);
}

bool BinarySquareInt64(const std::int64_t *input, std::int64_t *output, std::size_t count) {
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
  if (DetectSimdLevel() >= SimdLevel::kAVX512) {
    return BinarySquareInt64_AVX512(input, output, count);
  }
#endif
  return BinarySquare_Scalar<std::int64_t, 3037000499>(input, output, count);
}

template <typename T, bool Left>
void BinaryBitShiftContiguous(const T *left, const T *right, T *out, std::size_t count) {
#if ONNX_LIGHT_CPU_BINARY_X86
  static const bool use_avx2 = DetectSimdLevel() >= SimdLevel::kAVX2;
  if (use_avx2) {
    BinaryBitShift_AVX2<T, Left, ShiftInputMode::kContiguous>(left, right, out, count);
    return;
  }
#endif
  BinaryBitShift_Scalar<T, Left, ShiftInputMode::kContiguous>(left, right, out, count);
}

template <typename T, bool Left>
void BinaryBitShiftLeftScalar(T left, const T *right, T *out, std::size_t count) {
#if ONNX_LIGHT_CPU_BINARY_X86
  static const bool use_avx2 = DetectSimdLevel() >= SimdLevel::kAVX2;
  if (use_avx2) {
    BinaryBitShift_AVX2<T, Left, ShiftInputMode::kLeftScalar>(&left, right, out, count);
    return;
  }
#endif
  BinaryBitShift_Scalar<T, Left, ShiftInputMode::kLeftScalar>(&left, right, out, count);
}

template <typename T, bool Left>
void BinaryBitShiftRightScalar(const T *left, T right, T *out, std::size_t count) {
#if ONNX_LIGHT_CPU_BINARY_X86
  static const bool use_avx2 = DetectSimdLevel() >= SimdLevel::kAVX2;
  if (use_avx2) {
    BinaryBitShift_AVX2<T, Left, ShiftInputMode::kRightScalar>(left, &right, out, count);
    return;
  }
#endif
  BinaryBitShift_Scalar<T, Left, ShiftInputMode::kRightScalar>(left, &right, out, count);
}

template <typename T> bool BinaryBitShiftHasInvalidAmount(const T *shifts, std::size_t count) {
  constexpr std::size_t kMinParallelBytes = 1024 * 1024;
  if (count <= kMinParallelBytes / sizeof(T)) {
#if ONNX_LIGHT_CPU_BINARY_X86
    if (DetectSimdLevel() >= SimdLevel::kAVX2) {
      return BinaryBitShiftHasInvalidAmount_AVX2(shifts, count);
    }
#endif
    const T bit_width = static_cast<T>(sizeof(T) * 8);
    for (std::size_t i = 0; i < count; ++i) {
      if (shifts[i] >= bit_width) {
        return true;
      }
    }
    return false;
  }

  std::atomic<bool> invalid{false};
  ExecuteRanges(static_cast<std::int64_t>(count), static_cast<double>(sizeof(T)),
                [&](std::int64_t begin, std::int64_t end) {
                  const T *range = shifts + begin;
                  const std::size_t range_size = static_cast<std::size_t>(end - begin);
#if ONNX_LIGHT_CPU_BINARY_X86
                  if (DetectSimdLevel() >= SimdLevel::kAVX2) {
                    if (BinaryBitShiftHasInvalidAmount_AVX2(range, range_size)) {
                      invalid.store(true, std::memory_order_relaxed);
                    }
                    return;
                  }
#endif
                  const T bit_width = static_cast<T>(sizeof(T) * 8);
                  for (std::size_t i = 0; i < range_size; ++i) {
                    if (range[i] >= bit_width) {
                      invalid.store(true, std::memory_order_relaxed);
                      return;
                    }
                  }
                });
  return invalid.load(std::memory_order_relaxed);
}

#define ONNX_LIGHT_CPU_INSTANTIATE_BITSHIFT(T)                                                     \
  template void BinaryBitShiftContiguous<T, true>(const T *, const T *, T *, std::size_t);         \
  template void BinaryBitShiftContiguous<T, false>(const T *, const T *, T *, std::size_t);        \
  template void BinaryBitShiftLeftScalar<T, true>(T, const T *, T *, std::size_t);                 \
  template void BinaryBitShiftLeftScalar<T, false>(T, const T *, T *, std::size_t);                \
  template void BinaryBitShiftRightScalar<T, true>(const T *, T, T *, std::size_t);                \
  template void BinaryBitShiftRightScalar<T, false>(const T *, T, T *, std::size_t);               \
  template bool BinaryBitShiftHasInvalidAmount<T>(const T *, std::size_t);

ONNX_LIGHT_CPU_INSTANTIATE_BITSHIFT(std::uint8_t)
ONNX_LIGHT_CPU_INSTANTIATE_BITSHIFT(std::uint16_t)
ONNX_LIGHT_CPU_INSTANTIATE_BITSHIFT(std::uint32_t)
ONNX_LIGHT_CPU_INSTANTIATE_BITSHIFT(std::uint64_t)

#undef ONNX_LIGHT_CPU_INSTANTIATE_BITSHIFT

} // namespace onnx_light_cpu
