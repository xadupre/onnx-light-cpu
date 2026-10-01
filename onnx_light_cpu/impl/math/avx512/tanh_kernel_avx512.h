// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu/impl/math/avx512/exp_kernel_avx512.h"

#include <immintrin.h>

namespace onnx_light_cpu::detail {

inline __m512 TanhFloat32Vector_AVX512(__m512 value) {
  const __m512i bits = _mm512_castps_si512(value);
  const __m512i sign = _mm512_and_epi32(bits, _mm512_set1_epi32(static_cast<int>(0x80000000u)));
  const __m512 magnitude =
      _mm512_castsi512_ps(_mm512_and_epi32(bits, _mm512_set1_epi32(0x7fffffff)));
  // Tanh saturates beyond 10, so exp(-2*abs(x)) only needs the normal range.
  const __m512 bounded = _mm512_min_ps(magnitude, _mm512_set1_ps(10.0f));
  const __m512 x = _mm512_mul_ps(bounded, _mm512_set1_ps(-2.0f));
  const __m512 magic = _mm512_set1_ps(12582912.0f);
  const __m512 biased = _mm512_fmadd_ps(x, _mm512_set1_ps(kExpLog2efAvx512), magic);
  const __m512 exponent = _mm512_sub_ps(biased, magic);
  __m512 reduced = _mm512_fmadd_ps(exponent, _mm512_set1_ps(kExpC1Avx512), x);
  reduced = _mm512_fmadd_ps(exponent, _mm512_set1_ps(kExpC2Avx512), reduced);
  const __m512 scale = _mm512_castsi512_ps(_mm512_add_epi32(
      _mm512_slli_epi32(_mm512_castps_si512(biased), 23), _mm512_set1_epi32(0x3f800000)));
  const __m512 one = _mm512_set1_ps(1.0f);
  __m512 polynomial = _mm512_set1_ps(kExpP0Avx512);
  polynomial = _mm512_fmadd_ps(polynomial, reduced, _mm512_set1_ps(kExpP1Avx512));
  polynomial = _mm512_fmadd_ps(polynomial, reduced, _mm512_set1_ps(kExpP2Avx512));
  polynomial = _mm512_fmadd_ps(polynomial, reduced, _mm512_set1_ps(kExpP3Avx512));
  polynomial = _mm512_fmadd_ps(polynomial, reduced, _mm512_set1_ps(kExpP4Avx512));
  polynomial = _mm512_fmadd_ps(polynomial, reduced, one);
  const __m512 expm1 = _mm512_mul_ps(polynomial, reduced);
  // Reconstruct 1 +/- exp(x) without subtracting nearly equal numbers when scale=1.
  const __m512 numerator = _mm512_fnmadd_ps(scale, expm1, _mm512_sub_ps(one, scale));
  const __m512 denominator = _mm512_fmadd_ps(scale, expm1, _mm512_add_ps(one, scale));
  __m512 reciprocal = _mm512_rcp14_ps(denominator);
  reciprocal =
      _mm512_fmadd_ps(reciprocal, _mm512_fnmadd_ps(denominator, reciprocal, one), reciprocal);
  __m512 result = _mm512_mul_ps(numerator, reciprocal);
  result = _mm512_castsi512_ps(_mm512_or_epi32(_mm512_castps_si512(result), sign));
  // Preserve signed zero and subnormals, and quiet NaNs without losing their sign or payload.
  const __mmask16 tiny = _mm512_cmp_ps_mask(magnitude, _mm512_set1_ps(0x1p-13f), _CMP_LT_OQ);
  result = _mm512_mask_mov_ps(result, tiny, value);
  const __m512 quiet_nan =
      _mm512_castsi512_ps(_mm512_or_epi32(bits, _mm512_set1_epi32(0x00400000)));
  const __mmask16 is_nan = _mm512_cmp_ps_mask(value, value, _CMP_UNORD_Q);
  return _mm512_mask_mov_ps(result, is_nan, quiet_nan);
}

} // namespace onnx_light_cpu::detail
