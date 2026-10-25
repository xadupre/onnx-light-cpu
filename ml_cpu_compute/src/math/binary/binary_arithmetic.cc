// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "ml_cpu_compute/math/binary/binary_arithmetic.h"
#include "ml_cpu_compute/math/binary/avx/binary_arithmetic_avx.h"
#include "ml_cpu_compute/math/binary/binary_bitshift.h"

#include "ml_cpu_compute/arm_simd_level.h"
#include "ml_cpu_compute/cpu_cache_topology.h"
#include "ml_cpu_compute/execution.h"
#include "ml_cpu_compute/math/half_conversion.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define ML_CPU_COMPUTE_BINARY_X86 1
#include <immintrin.h>
#else
#define ML_CPU_COMPUTE_BINARY_X86 0
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define ML_CPU_COMPUTE_BINARY_ARM64 1
#include <arm_neon.h>
#else
#define ML_CPU_COMPUTE_BINARY_ARM64 0
#endif

namespace ml_cpu_compute {
using detail::BinaryBitShift_Scalar;
using detail::ShiftInputMode;
#ifdef ML_CPU_COMPUTE_HAVE_AVX2
using detail::BinaryBitShift_AVX2;
using detail::BinaryBitShiftHasInvalidAmount_AVX2;
#endif

namespace {

// ---------------------------------------------------------------------------
// Portable scalar reference. Used as the tail of every SIMD loop and as the
// whole loop on an ISA without a wider path.
// ---------------------------------------------------------------------------
#define ML_CPU_COMPUTE_BIN_SCALAR(STEM, T, OPCH)                                                   \
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

ML_CPU_COMPUTE_BIN_SCALAR(BinaryAddFloat32, float, +)
ML_CPU_COMPUTE_BIN_SCALAR(BinarySubFloat32, float, -)
ML_CPU_COMPUTE_BIN_SCALAR(BinaryMulFloat32, float, *)
ML_CPU_COMPUTE_BIN_SCALAR(BinaryDivFloat32, float, /)
ML_CPU_COMPUTE_BIN_SCALAR(BinaryAddFloat64, double, +)
ML_CPU_COMPUTE_BIN_SCALAR(BinarySubFloat64, double, -)
ML_CPU_COMPUTE_BIN_SCALAR(BinaryMulFloat64, double, *)
ML_CPU_COMPUTE_BIN_SCALAR(BinaryDivFloat64, double, /)

#undef ML_CPU_COMPUTE_BIN_SCALAR

#define ML_CPU_COMPUTE_HALF_SCALAR(STEM, OPCH)                                                     \
  void STEM##_Scalar(const std::uint16_t *left, const std::uint16_t *right, std::uint16_t *out,    \
                     std::size_t count) {                                                          \
    for (std::size_t i = 0; i < count; ++i) {                                                      \
      out[i] = detail::FloatToFloat16Bits(detail::Float16BitsToFloat(left[i])                      \
                                              OPCH detail::Float16BitsToFloat(right[i]));          \
    }                                                                                              \
  }                                                                                                \
  void STEM##Left_Scalar(std::uint16_t left, const std::uint16_t *right, std::uint16_t *out,       \
                         std::size_t count) {                                                      \
    const float scalar = detail::Float16BitsToFloat(left);                                         \
    for (std::size_t i = 0; i < count; ++i) {                                                      \
      out[i] = detail::FloatToFloat16Bits(scalar OPCH detail::Float16BitsToFloat(right[i]));       \
    }                                                                                              \
  }                                                                                                \
  void STEM##Right_Scalar(const std::uint16_t *left, std::uint16_t right, std::uint16_t *out,      \
                          std::size_t count) {                                                     \
    const float scalar = detail::Float16BitsToFloat(right);                                        \
    for (std::size_t i = 0; i < count; ++i) {                                                      \
      out[i] = detail::FloatToFloat16Bits(detail::Float16BitsToFloat(left[i]) OPCH scalar);        \
    }                                                                                              \
  }

ML_CPU_COMPUTE_HALF_SCALAR(BinaryAddFloat16, +)
ML_CPU_COMPUTE_HALF_SCALAR(BinarySubFloat16, -)
ML_CPU_COMPUTE_HALF_SCALAR(BinaryMulFloat16, *)
ML_CPU_COMPUTE_HALF_SCALAR(BinaryDivFloat16, /)

#undef ML_CPU_COMPUTE_HALF_SCALAR

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

#if ML_CPU_COMPUTE_BINARY_X86

// ---------------------------------------------------------------------------
// SSE2 (128-bit) implementations. The intrinsic names
// (``_mm_<op>_p[sd]``) are regular across Add/Sub/Mul/Div, so a single
// macro instantiates the whole width x type x operator matrix.
// ---------------------------------------------------------------------------
#define ML_CPU_COMPUTE_BIN_SSE2_F32(STEM, INTRIN, OPCH)                                            \
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

#define ML_CPU_COMPUTE_BIN_SSE2_F64(STEM, INTRIN, OPCH)                                            \
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

ML_CPU_COMPUTE_BIN_SSE2_F32(BinaryAddFloat32, _mm_add_ps, +)
ML_CPU_COMPUTE_BIN_SSE2_F32(BinarySubFloat32, _mm_sub_ps, -)
ML_CPU_COMPUTE_BIN_SSE2_F32(BinaryMulFloat32, _mm_mul_ps, *)
ML_CPU_COMPUTE_BIN_SSE2_F32(BinaryDivFloat32, _mm_div_ps, /)
ML_CPU_COMPUTE_BIN_SSE2_F64(BinaryAddFloat64, _mm_add_pd, +)
ML_CPU_COMPUTE_BIN_SSE2_F64(BinarySubFloat64, _mm_sub_pd, -)
ML_CPU_COMPUTE_BIN_SSE2_F64(BinaryMulFloat64, _mm_mul_pd, *)
ML_CPU_COMPUTE_BIN_SSE2_F64(BinaryDivFloat64, _mm_div_pd, /)

#define ML_CPU_COMPUTE_PRELU_X86(SUFFIX, VECTOR, LANES, LOAD, STORE, SET1, ZERO, CMP, MUL, AND,    \
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

ML_CPU_COMPUTE_PRELU_X86(_SSE2, __m128, 4, _mm_loadu_ps, _mm_storeu_ps, _mm_set1_ps, _mm_setzero_ps,
                         CompareLessThan, _mm_mul_ps, _mm_and_ps, _mm_andnot_ps, _mm_or_ps)

#undef ML_CPU_COMPUTE_PRELU_X86

#undef ML_CPU_COMPUTE_BIN_SSE2_F32
#undef ML_CPU_COMPUTE_BIN_SSE2_F64

#endif // ML_CPU_COMPUTE_BINARY_X86

#if ML_CPU_COMPUTE_BINARY_ARM64

// ---------------------------------------------------------------------------
// NEON (Advanced SIMD) implementations. Baseline on AArch64, so this needs no
// dedicated translation unit or extra compile flags.
// ---------------------------------------------------------------------------
#define ML_CPU_COMPUTE_BIN_NEON_F32(STEM, INTRIN, OPCH)                                            \
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

#define ML_CPU_COMPUTE_BIN_NEON_F64(STEM, INTRIN, OPCH)                                            \
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

ML_CPU_COMPUTE_BIN_NEON_F32(BinaryAddFloat32, vaddq_f32, +)
ML_CPU_COMPUTE_BIN_NEON_F32(BinarySubFloat32, vsubq_f32, -)
ML_CPU_COMPUTE_BIN_NEON_F32(BinaryMulFloat32, vmulq_f32, *)
ML_CPU_COMPUTE_BIN_NEON_F32(BinaryDivFloat32, vdivq_f32, /)
ML_CPU_COMPUTE_BIN_NEON_F64(BinaryAddFloat64, vaddq_f64, +)
ML_CPU_COMPUTE_BIN_NEON_F64(BinarySubFloat64, vsubq_f64, -)
ML_CPU_COMPUTE_BIN_NEON_F64(BinaryMulFloat64, vmulq_f64, *)
ML_CPU_COMPUTE_BIN_NEON_F64(BinaryDivFloat64, vdivq_f64, /)

#undef ML_CPU_COMPUTE_BIN_NEON_F32
#undef ML_CPU_COMPUTE_BIN_NEON_F64

#endif // ML_CPU_COMPUTE_BINARY_ARM64

// ---------------------------------------------------------------------------
// ISA pointer helpers. Each resolves to the real implementation's address
// when the translation unit was compiled for that ISA family, or to
// ``nullptr`` otherwise, so the platform-neutral selection code below never
// names a symbol that does not exist in this translation unit.
// ---------------------------------------------------------------------------
#if ML_CPU_COMPUTE_BINARY_X86
#define ML_CPU_COMPUTE_BIN_SSE2_PTR(STEM) (&STEM##_SSE2)
#define ML_CPU_COMPUTE_BIN_SSE2_LEFT_PTR(STEM) (&STEM##Left_SSE2)
#define ML_CPU_COMPUTE_BIN_SSE2_RIGHT_PTR(STEM) (&STEM##Right_SSE2)
#else
#define ML_CPU_COMPUTE_BIN_SSE2_PTR(STEM) (nullptr)
#define ML_CPU_COMPUTE_BIN_SSE2_LEFT_PTR(STEM) (nullptr)
#define ML_CPU_COMPUTE_BIN_SSE2_RIGHT_PTR(STEM) (nullptr)
#endif

#ifdef ML_CPU_COMPUTE_HAVE_AVX
#define ML_CPU_COMPUTE_BIN_AVX_PTR(STEM) (&STEM##_AVX)
#define ML_CPU_COMPUTE_BIN_AVX_LEFT_PTR(STEM) (&STEM##Left_AVX)
#define ML_CPU_COMPUTE_BIN_AVX_RIGHT_PTR(STEM) (&STEM##Right_AVX)
#else
#define ML_CPU_COMPUTE_BIN_AVX_PTR(STEM) (nullptr)
#define ML_CPU_COMPUTE_BIN_AVX_LEFT_PTR(STEM) (nullptr)
#define ML_CPU_COMPUTE_BIN_AVX_RIGHT_PTR(STEM) (nullptr)
#endif

#if ML_CPU_COMPUTE_BINARY_ARM64
#define ML_CPU_COMPUTE_BIN_NEON_PTR(STEM) (&STEM##_NEON)
#define ML_CPU_COMPUTE_BIN_NEON_LEFT_PTR(STEM) (&STEM##Left_NEON)
#define ML_CPU_COMPUTE_BIN_NEON_RIGHT_PTR(STEM) (&STEM##Right_NEON)
#else
#define ML_CPU_COMPUTE_BIN_NEON_PTR(STEM) (nullptr)
#define ML_CPU_COMPUTE_BIN_NEON_LEFT_PTR(STEM) (nullptr)
#define ML_CPU_COMPUTE_BIN_NEON_RIGHT_PTR(STEM) (nullptr)
#endif

#ifdef ML_CPU_COMPUTE_HAVE_AVX512
#define ML_CPU_COMPUTE_BIN_AVX512_PTR(STEM) (&STEM##_AVX512)
#define ML_CPU_COMPUTE_BIN_AVX512_LEFT_PTR(STEM) (&STEM##Left_AVX512)
#define ML_CPU_COMPUTE_BIN_AVX512_RIGHT_PTR(STEM) (&STEM##Right_AVX512)
#else
#define ML_CPU_COMPUTE_BIN_AVX512_PTR(STEM) (nullptr)
#define ML_CPU_COMPUTE_BIN_AVX512_LEFT_PTR(STEM) (nullptr)
#define ML_CPU_COMPUTE_BIN_AVX512_RIGHT_PTR(STEM) (nullptr)
#endif

#ifdef ML_CPU_COMPUTE_HAVE_SVE
#define ML_CPU_COMPUTE_BIN_SVE_PTR(STEM) (&STEM##_SVE)
#define ML_CPU_COMPUTE_BIN_SVE_LEFT_PTR(STEM) (&STEM##Left_SVE)
#define ML_CPU_COMPUTE_BIN_SVE_RIGHT_PTR(STEM) (&STEM##Right_SVE)
#else
#define ML_CPU_COMPUTE_BIN_SVE_PTR(STEM) (nullptr)
#define ML_CPU_COMPUTE_BIN_SVE_LEFT_PTR(STEM) (nullptr)
#define ML_CPU_COMPUTE_BIN_SVE_RIGHT_PTR(STEM) (nullptr)
#endif

// ---------------------------------------------------------------------------
// Runtime dispatch. Picks the widest available ISA once (cached in a
// function-local static) and forwards every subsequent call directly to it.
// Never touches a symbol from an unsupported ISA (each family's pointer is
// ``nullptr`` outside its own ``#if``/``#ifdef`` block above), so it never
// executes an unsupported instruction.
// ---------------------------------------------------------------------------
template <typename Fn>
inline Fn PickImpl(Fn scalar_fn, Fn sse2_fn, Fn avx_fn, Fn avx512_fn, Fn neon_fn, Fn sve_fn) {
#if ML_CPU_COMPUTE_BINARY_X86
  (void)neon_fn;
  (void)sve_fn;
  static const SimdLevel level = DetectSimdLevel();
  if (avx512_fn != nullptr && level >= SimdLevel::kAVX512) {
    return avx512_fn;
  }
  if (avx_fn != nullptr && level >= SimdLevel::kAVX) {
    return avx_fn;
  }
  if (level >= SimdLevel::kSSE2) {
    return sse2_fn;
  }
  return scalar_fn;
#elif ML_CPU_COMPUTE_BINARY_ARM64
  (void)sse2_fn;
  (void)avx_fn;
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
  (void)avx_fn;
  (void)avx512_fn;
  (void)neon_fn;
  (void)sve_fn;
  return scalar_fn;
#endif
}

template <typename T, void (*Fn)(const T *, const T *, T *, std::size_t)>
void ErasedContiguous(const void *left, const void *right, void *out, std::size_t count) {
  Fn(static_cast<const T *>(left), static_cast<const T *>(right), static_cast<T *>(out), count);
}

template <typename T, void (*Fn)(T, const T *, T *, std::size_t)>
void ErasedLeftScalar(const void *left, const void *right, void *out, std::size_t count) {
  Fn(*static_cast<const T *>(left), static_cast<const T *>(right), static_cast<T *>(out), count);
}

template <typename T, void (*Fn)(const T *, T, T *, std::size_t)>
void ErasedRightScalar(const void *left, const void *right, void *out, std::size_t count) {
  Fn(static_cast<const T *>(left), *static_cast<const T *>(right), static_cast<T *>(out), count);
}

template <typename T, void (*Contiguous)(const T *, const T *, T *, std::size_t),
          void (*LeftScalar)(T, const T *, T *, std::size_t),
          void (*RightScalar)(const T *, T, T *, std::size_t)>
const BinaryArithmeticBulkFunctions &MakeBulkFunctions() {
  static const BinaryArithmeticBulkFunctions functions{
      &ErasedContiguous<T, Contiguous>,
      &ErasedLeftScalar<T, LeftScalar>,
      &ErasedRightScalar<T, RightScalar>,
  };
  return functions;
}

#ifdef ML_CPU_COMPUTE_HAVE_AVX512
#define ML_CPU_COMPUTE_TRY_AVX512_BULK(STEM, T)                                                    \
  if (level >= SimdLevel::kAVX512) {                                                               \
    return MakeBulkFunctions<T, &STEM##_AVX512, &STEM##Left_AVX512, &STEM##Right_AVX512>();        \
  }
#else
#define ML_CPU_COMPUTE_TRY_AVX512_BULK(STEM, T)
#endif

#ifdef ML_CPU_COMPUTE_HAVE_AVX
#define ML_CPU_COMPUTE_TRY_AVX_BULK(STEM, T)                                                       \
  if (level >= SimdLevel::kAVX) {                                                                  \
    return MakeBulkFunctions<T, &STEM##_AVX, &STEM##Left_AVX, &STEM##Right_AVX>();                 \
  }
#else
#define ML_CPU_COMPUTE_TRY_AVX_BULK(STEM, T)
#endif

#if ML_CPU_COMPUTE_BINARY_X86
#define ML_CPU_COMPUTE_RESOLVE_X86_BULK(STEM, T)                                                   \
  static const SimdLevel level = DetectSimdLevel();                                                \
  ML_CPU_COMPUTE_TRY_AVX512_BULK(STEM, T)                                                          \
  ML_CPU_COMPUTE_TRY_AVX_BULK(STEM, T)                                                             \
  if (level >= SimdLevel::kSSE2) {                                                                 \
    return MakeBulkFunctions<T, &STEM##_SSE2, &STEM##Left_SSE2, &STEM##Right_SSE2>();              \
  }
#else
#define ML_CPU_COMPUTE_RESOLVE_X86_BULK(STEM, T)
#endif

#ifdef ML_CPU_COMPUTE_HAVE_SVE
#define ML_CPU_COMPUTE_TRY_SVE_BULK(STEM, T)                                                       \
  if (level == ArmSimdLevel::kSve || level == ArmSimdLevel::kSve2) {                               \
    return MakeBulkFunctions<T, &STEM##_SVE, &STEM##Left_SVE, &STEM##Right_SVE>();                 \
  }
#else
#define ML_CPU_COMPUTE_TRY_SVE_BULK(STEM, T)
#endif

#if ML_CPU_COMPUTE_BINARY_ARM64
#define ML_CPU_COMPUTE_RESOLVE_ARM_BULK(STEM, T)                                                   \
  static const ArmSimdLevel level = DetectArmSimdLevel();                                          \
  ML_CPU_COMPUTE_TRY_SVE_BULK(STEM, T)                                                             \
  if (level != ArmSimdLevel::kNone) {                                                              \
    return MakeBulkFunctions<T, &STEM##_NEON, &STEM##Left_NEON, &STEM##Right_NEON>();              \
  }
#else
#define ML_CPU_COMPUTE_RESOLVE_ARM_BULK(STEM, T)
#endif

#define ML_CPU_COMPUTE_DEFINE_BULK_RESOLVER(PUBLIC, STEM, T)                                       \
  const BinaryArithmeticBulkFunctions &PUBLIC##BulkFunctions() {                                   \
    ML_CPU_COMPUTE_RESOLVE_X86_BULK(STEM, T)                                                       \
    ML_CPU_COMPUTE_RESOLVE_ARM_BULK(STEM, T)                                                       \
    return MakeBulkFunctions<T, &STEM##_Scalar, &STEM##Left_Scalar, &STEM##Right_Scalar>();        \
  }

#define ML_CPU_COMPUTE_DEFINE_X86_BULK_RESOLVER(PUBLIC, STEM, T)                                   \
  const BinaryArithmeticBulkFunctions &PUBLIC##BulkFunctions() {                                   \
    ML_CPU_COMPUTE_RESOLVE_X86_BULK(STEM, T)                                                       \
    return MakeBulkFunctions<T, &STEM##_Scalar, &STEM##Left_Scalar, &STEM##Right_Scalar>();        \
  }

#ifdef ML_CPU_COMPUTE_HAVE_AVX512FP16
#define ML_CPU_COMPUTE_TRY_AVX512FP16_BULK(STEM)                                                   \
  static const bool use_avx512fp16 = CpuSupportsAvx512Fp16();                                      \
  if (use_avx512fp16) {                                                                            \
    return MakeBulkFunctions<std::uint16_t, &STEM##_AVX512FP16, &STEM##Left_AVX512FP16,            \
                             &STEM##Right_AVX512FP16>();                                           \
  }
#else
#define ML_CPU_COMPUTE_TRY_AVX512FP16_BULK(STEM)
#endif

#ifdef ML_CPU_COMPUTE_HAVE_F16C
#define ML_CPU_COMPUTE_TRY_F16C_BULK(STEM)                                                         \
  static const bool use_f16c = CpuSupportsF16C();                                                  \
  if (use_f16c) {                                                                                  \
    return MakeBulkFunctions<std::uint16_t, &STEM##_F16C, &STEM##Left_F16C, &STEM##Right_F16C>();  \
  }
#else
#define ML_CPU_COMPUTE_TRY_F16C_BULK(STEM)
#endif

#define ML_CPU_COMPUTE_DEFINE_FLOAT16_BULK_RESOLVER(PUBLIC, STEM)                                  \
  const BinaryArithmeticBulkFunctions &PUBLIC##BulkFunctions() {                                   \
    ML_CPU_COMPUTE_TRY_AVX512FP16_BULK(STEM)                                                       \
    ML_CPU_COMPUTE_TRY_F16C_BULK(STEM)                                                             \
    return MakeBulkFunctions<std::uint16_t, &STEM##_Scalar, &STEM##Left_Scalar,                    \
                             &STEM##Right_Scalar>();                                               \
  }

#define ML_CPU_COMPUTE_BIN_DISPATCH_CONTIG(PUBLIC, STEM, T)                                        \
  void PUBLIC##Contiguous(const T *left, const T *right, T *out, std::size_t count) {              \
    if (count == 0) {                                                                              \
      return;                                                                                      \
    }                                                                                              \
    using Fn = void (*)(const T *, const T *, T *, std::size_t);                                   \
    static const Fn fn =                                                                           \
        PickImpl<Fn>(&STEM##_Scalar, ML_CPU_COMPUTE_BIN_SSE2_PTR(STEM),                            \
                     ML_CPU_COMPUTE_BIN_AVX_PTR(STEM), ML_CPU_COMPUTE_BIN_AVX512_PTR(STEM),        \
                     ML_CPU_COMPUTE_BIN_NEON_PTR(STEM), ML_CPU_COMPUTE_BIN_SVE_PTR(STEM));         \
    fn(left, right, out, count);                                                                   \
  }

#define ML_CPU_COMPUTE_BIN_DISPATCH_LEFT(PUBLIC, STEM, T)                                          \
  void PUBLIC##LeftScalar(T left, const T *right, T *out, std::size_t count) {                     \
    if (count == 0) {                                                                              \
      return;                                                                                      \
    }                                                                                              \
    using Fn = void (*)(T, const T *, T *, std::size_t);                                           \
    static const Fn fn = PickImpl<Fn>(                                                             \
        &STEM##Left_Scalar, ML_CPU_COMPUTE_BIN_SSE2_LEFT_PTR(STEM),                                \
        ML_CPU_COMPUTE_BIN_AVX_LEFT_PTR(STEM), ML_CPU_COMPUTE_BIN_AVX512_LEFT_PTR(STEM),           \
        ML_CPU_COMPUTE_BIN_NEON_LEFT_PTR(STEM), ML_CPU_COMPUTE_BIN_SVE_LEFT_PTR(STEM));            \
    fn(left, right, out, count);                                                                   \
  }

#define ML_CPU_COMPUTE_BIN_DISPATCH_RIGHT(PUBLIC, STEM, T)                                         \
  void PUBLIC##RightScalar(const T *left, T right, T *out, std::size_t count) {                    \
    if (count == 0) {                                                                              \
      return;                                                                                      \
    }                                                                                              \
    using Fn = void (*)(const T *, T, T *, std::size_t);                                           \
    static const Fn fn = PickImpl<Fn>(                                                             \
        &STEM##Right_Scalar, ML_CPU_COMPUTE_BIN_SSE2_RIGHT_PTR(STEM),                              \
        ML_CPU_COMPUTE_BIN_AVX_RIGHT_PTR(STEM), ML_CPU_COMPUTE_BIN_AVX512_RIGHT_PTR(STEM),         \
        ML_CPU_COMPUTE_BIN_NEON_RIGHT_PTR(STEM), ML_CPU_COMPUTE_BIN_SVE_RIGHT_PTR(STEM));          \
    fn(left, right, out, count);                                                                   \
  }

#define ML_CPU_COMPUTE_BIN_DISPATCH(PUBLIC, STEM, T)                                               \
  ML_CPU_COMPUTE_BIN_DISPATCH_CONTIG(PUBLIC, STEM, T)                                              \
  ML_CPU_COMPUTE_BIN_DISPATCH_LEFT(PUBLIC, STEM, T)                                                \
  ML_CPU_COMPUTE_BIN_DISPATCH_RIGHT(PUBLIC, STEM, T)

} // namespace

ML_CPU_COMPUTE_DEFINE_BULK_RESOLVER(BinaryAddFloat32, BinaryAddFloat32, float)
ML_CPU_COMPUTE_DEFINE_BULK_RESOLVER(BinarySubFloat32, BinarySubFloat32, float)
ML_CPU_COMPUTE_DEFINE_BULK_RESOLVER(BinaryMulFloat32, BinaryMulFloat32, float)
ML_CPU_COMPUTE_DEFINE_BULK_RESOLVER(BinaryDivFloat32, BinaryDivFloat32, float)
ML_CPU_COMPUTE_DEFINE_X86_BULK_RESOLVER(BinaryPReluFloat32, BinaryPReluFloat32, float)

namespace {

bool UseBinaryFloat32Streaming(std::size_t count) {
#ifdef ML_CPU_COMPUTE_HAVE_AVX
  static const SimdLevel level = DetectSimdLevel();
  static const std::size_t cache_bytes = CpuCacheSizeBytesOrFallback(GetCpuCacheTopology(), 3, 0);
  // Avoid write allocation only when the three-array footprint exceeds LLC.
  // Unknown topology and AVX-512 retain their existing cached-store paths.
  if (level >= SimdLevel::kAVX && level < SimdLevel::kAVX512 && cache_bytes != 0 &&
      count > std::max<std::size_t>(32 * 1024 * 1024, cache_bytes) / (3 * sizeof(float))) {
    return true;
  }
#else
  (void)count;
#endif
  return false;
}

} // namespace

BinaryArithmeticBulkFunctions::Fn BinaryAddFloat32StreamingFunction(std::size_t count) {
  if (!UseBinaryFloat32Streaming(count)) {
    return nullptr;
  }
#ifdef ML_CPU_COMPUTE_HAVE_AVX
  return &ErasedContiguous<float, &BinaryAddFloat32Streaming_AVX>;
#else
  return nullptr;
#endif
}

BinaryArithmeticBulkFunctions::Fn BinaryDivFloat32StreamingFunction(std::size_t count) {
  if (!UseBinaryFloat32Streaming(count)) {
    return nullptr;
  }
#ifdef ML_CPU_COMPUTE_HAVE_AVX
  return &ErasedContiguous<float, &BinaryDivFloat32Streaming_AVX>;
#else
  return nullptr;
#endif
}
ML_CPU_COMPUTE_DEFINE_FLOAT16_BULK_RESOLVER(BinaryAddFloat16, BinaryAddFloat16)
ML_CPU_COMPUTE_DEFINE_FLOAT16_BULK_RESOLVER(BinarySubFloat16, BinarySubFloat16)
ML_CPU_COMPUTE_DEFINE_FLOAT16_BULK_RESOLVER(BinaryMulFloat16, BinaryMulFloat16)
ML_CPU_COMPUTE_DEFINE_FLOAT16_BULK_RESOLVER(BinaryDivFloat16, BinaryDivFloat16)

#undef ML_CPU_COMPUTE_DEFINE_FLOAT16_BULK_RESOLVER
#undef ML_CPU_COMPUTE_TRY_F16C_BULK
#undef ML_CPU_COMPUTE_TRY_AVX512FP16_BULK
#undef ML_CPU_COMPUTE_DEFINE_X86_BULK_RESOLVER
#undef ML_CPU_COMPUTE_DEFINE_BULK_RESOLVER
#undef ML_CPU_COMPUTE_RESOLVE_ARM_BULK
#undef ML_CPU_COMPUTE_TRY_SVE_BULK
#undef ML_CPU_COMPUTE_RESOLVE_X86_BULK
#undef ML_CPU_COMPUTE_TRY_AVX_BULK
#undef ML_CPU_COMPUTE_TRY_AVX512_BULK

ML_CPU_COMPUTE_BIN_DISPATCH(BinaryAddFloat32, BinaryAddFloat32, float)
ML_CPU_COMPUTE_BIN_DISPATCH(BinarySubFloat32, BinarySubFloat32, float)
ML_CPU_COMPUTE_BIN_DISPATCH(BinaryMulFloat32, BinaryMulFloat32, float)
ML_CPU_COMPUTE_BIN_DISPATCH(BinaryDivFloat32, BinaryDivFloat32, float)
ML_CPU_COMPUTE_BIN_DISPATCH(BinaryAddFloat64, BinaryAddFloat64, double)
ML_CPU_COMPUTE_BIN_DISPATCH(BinarySubFloat64, BinarySubFloat64, double)
ML_CPU_COMPUTE_BIN_DISPATCH(BinaryMulFloat64, BinaryMulFloat64, double)
ML_CPU_COMPUTE_BIN_DISPATCH(BinaryDivFloat64, BinaryDivFloat64, double)

void BinaryPReluFloat32Contiguous(const float *left, const float *right, float *out,
                                  std::size_t count) {
  using Fn = void (*)(const float *, const float *, float *, std::size_t);
  static const Fn fn =
      PickImpl<Fn>(&BinaryPReluFloat32_Scalar, ML_CPU_COMPUTE_BIN_SSE2_PTR(BinaryPReluFloat32),
                   ML_CPU_COMPUTE_BIN_AVX_PTR(BinaryPReluFloat32),
                   ML_CPU_COMPUTE_BIN_AVX512_PTR(BinaryPReluFloat32), &BinaryPReluFloat32_Scalar,
                   &BinaryPReluFloat32_Scalar);
  fn(left, right, out, count);
}

void BinaryPReluFloat32LeftScalar(float left, const float *right, float *out, std::size_t count) {
  using Fn = void (*)(float, const float *, float *, std::size_t);
  static const Fn fn = PickImpl<Fn>(&BinaryPReluFloat32Left_Scalar,
                                    ML_CPU_COMPUTE_BIN_SSE2_LEFT_PTR(BinaryPReluFloat32),
                                    ML_CPU_COMPUTE_BIN_AVX_LEFT_PTR(BinaryPReluFloat32),
                                    ML_CPU_COMPUTE_BIN_AVX512_LEFT_PTR(BinaryPReluFloat32),
                                    &BinaryPReluFloat32Left_Scalar, &BinaryPReluFloat32Left_Scalar);
  fn(left, right, out, count);
}

void BinaryPReluFloat32RightScalar(const float *left, float right, float *out, std::size_t count) {
  using Fn = void (*)(const float *, float, float *, std::size_t);
  static const Fn fn = PickImpl<Fn>(
      &BinaryPReluFloat32Right_Scalar, ML_CPU_COMPUTE_BIN_SSE2_RIGHT_PTR(BinaryPReluFloat32),
      ML_CPU_COMPUTE_BIN_AVX_RIGHT_PTR(BinaryPReluFloat32),
      ML_CPU_COMPUTE_BIN_AVX512_RIGHT_PTR(BinaryPReluFloat32), &BinaryPReluFloat32Right_Scalar,
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
#ifdef ML_CPU_COMPUTE_HAVE_AVX512
  static const bool use_avx512 = DetectSimdLevel() >= SimdLevel::kAVX512;
  if (use_avx512) {
    return BinarySquareInt32_AVX512(input, output, count);
  }
#endif
  return BinarySquare_Scalar<std::int32_t, 46340>(input, output, count);
}

bool BinarySquareInt64(const std::int64_t *input, std::int64_t *output, std::size_t count) {
#ifdef ML_CPU_COMPUTE_HAVE_AVX512
  static const bool use_avx512 = DetectSimdLevel() >= SimdLevel::kAVX512;
  if (use_avx512) {
    return BinarySquareInt64_AVX512(input, output, count);
  }
#endif
  return BinarySquare_Scalar<std::int64_t, 3037000499>(input, output, count);
}

template <typename T, bool Left>
void BinaryBitShiftContiguous(const T *left, const T *right, T *out, std::size_t count) {
#ifdef ML_CPU_COMPUTE_HAVE_AVX2
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
#ifdef ML_CPU_COMPUTE_HAVE_AVX2
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
#ifdef ML_CPU_COMPUTE_HAVE_AVX2
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
#ifdef ML_CPU_COMPUTE_HAVE_AVX2
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
#ifdef ML_CPU_COMPUTE_HAVE_AVX2
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

#define ML_CPU_COMPUTE_INSTANTIATE_BITSHIFT(T)                                                     \
  template void BinaryBitShiftContiguous<T, true>(const T *, const T *, T *, std::size_t);         \
  template void BinaryBitShiftContiguous<T, false>(const T *, const T *, T *, std::size_t);        \
  template void BinaryBitShiftLeftScalar<T, true>(T, const T *, T *, std::size_t);                 \
  template void BinaryBitShiftLeftScalar<T, false>(T, const T *, T *, std::size_t);                \
  template void BinaryBitShiftRightScalar<T, true>(const T *, T, T *, std::size_t);                \
  template void BinaryBitShiftRightScalar<T, false>(const T *, T, T *, std::size_t);               \
  template bool BinaryBitShiftHasInvalidAmount<T>(const T *, std::size_t);

ML_CPU_COMPUTE_INSTANTIATE_BITSHIFT(std::uint8_t)
ML_CPU_COMPUTE_INSTANTIATE_BITSHIFT(std::uint16_t)
ML_CPU_COMPUTE_INSTANTIATE_BITSHIFT(std::uint32_t)
ML_CPU_COMPUTE_INSTANTIATE_BITSHIFT(std::uint64_t)

#undef ML_CPU_COMPUTE_INSTANTIATE_BITSHIFT

} // namespace ml_cpu_compute
