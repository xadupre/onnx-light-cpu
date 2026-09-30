// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <immintrin.h>
#include <limits>

namespace onnx_light_cpu::detail {

inline constexpr float kExpLog2efAvx512 = 1.44269504088896341f;
inline constexpr float kExpC1Avx512 = -6.93145752e-1f;
inline constexpr float kExpC2Avx512 = -1.42860677e-6f;
inline constexpr float kExpP0Avx512 = 0x1.694000p-10f;
inline constexpr float kExpP1Avx512 = 0x1.125edcp-7f;
inline constexpr float kExpP2Avx512 = 0x1.555b5ap-5f;
inline constexpr float kExpP3Avx512 = 0x1.555450p-3f;
inline constexpr float kExpP4Avx512 = 0x1.fffff6p-2f;

#if defined(_MSC_VER)
#define ONNX_LIGHT_CPU_EXP_FORCE_INLINE __forceinline
#else
#define ONNX_LIGHT_CPU_EXP_FORCE_INLINE inline __attribute__((always_inline))
#endif

ONNX_LIGHT_CPU_EXP_FORCE_INLINE __m512 ExpFloat32Vector_AVX512(__m512 x) {
  constexpr float kExpHi = 88.7762626647950f;
  // Half of the smallest float32 subnormal, -150 * ln(2). Clamping here keeps
  // the reduced exponent representable while preserving correctly rounded
  // subnormal results.
  constexpr float kExpLo = -103.97208f;
  const __m512 one = _mm512_set1_ps(1.0f);
  const __m512 hi = _mm512_set1_ps(kExpHi);
  const __m512 lo = _mm512_set1_ps(kExpLo);

  const __mmask16 is_nan = _mm512_cmp_ps_mask(x, x, _CMP_UNORD_Q);
  const __mmask16 over = _mm512_cmp_ps_mask(x, hi, _CMP_GT_OQ);
  const __mmask16 under = _mm512_cmp_ps_mask(x, lo, _CMP_LT_OQ);

  __m512 reduced = _mm512_min_ps(_mm512_max_ps(x, lo), hi);
  const __m512 magic = _mm512_set1_ps(12582912.0f);
  const __m512 scaled = _mm512_fmadd_ps(reduced, _mm512_set1_ps(kExpLog2efAvx512), magic);
  const __m512i exponent_int =
      _mm512_sub_epi32(_mm512_castps_si512(scaled), _mm512_set1_epi32(0x4b400000));
  const __m512 exponent = _mm512_cvtepi32_ps(exponent_int);

  reduced = _mm512_fmadd_ps(exponent, _mm512_set1_ps(kExpC1Avx512), reduced);
  reduced = _mm512_fmadd_ps(exponent, _mm512_set1_ps(kExpC2Avx512), reduced);

  __m512 polynomial = _mm512_set1_ps(kExpP0Avx512);
  polynomial = _mm512_fmadd_ps(polynomial, reduced, _mm512_set1_ps(kExpP1Avx512));
  polynomial = _mm512_fmadd_ps(polynomial, reduced, _mm512_set1_ps(kExpP2Avx512));
  polynomial = _mm512_fmadd_ps(polynomial, reduced, _mm512_set1_ps(kExpP3Avx512));
  polynomial = _mm512_fmadd_ps(polynomial, reduced, _mm512_set1_ps(kExpP4Avx512));
  polynomial = _mm512_fmadd_ps(polynomial, reduced, one);
  polynomial = _mm512_fmadd_ps(polynomial, reduced, one);
  __m512 result = _mm512_scalef_ps(polynomial, exponent);

  result = _mm512_mask_mov_ps(result, under, _mm512_setzero_ps());
  result = _mm512_mask_mov_ps(result, over, _mm512_set1_ps(std::numeric_limits<float>::infinity()));
  return _mm512_mask_mov_ps(result, is_nan,
                            _mm512_set1_ps(std::numeric_limits<float>::quiet_NaN()));
}

#undef ONNX_LIGHT_CPU_EXP_FORCE_INLINE

} // namespace onnx_light_cpu::detail
