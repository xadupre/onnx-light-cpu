// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu/impl/math/avx512/exp_kernel_avx512.h"

#include <immintrin.h>

namespace onnx_light_cpu::detail {

inline __m512 SigmoidFloat32Vector_AVX512(__m512 value) {
  const __m512 negative_abs = _mm512_castsi512_ps(_mm512_or_epi32(
      _mm512_castps_si512(value), _mm512_set1_epi32(static_cast<int>(0x80000000u))));
  const __m512 exponent = ExpFloat32Vector_AVX512(negative_abs);
  const __m512 one = _mm512_set1_ps(1.0f);
  const __mmask16 negative = _mm512_cmp_ps_mask(value, _mm512_setzero_ps(), _CMP_LT_OQ);
  const __m512 denominator = _mm512_add_ps(one, exponent);
  __m512 reciprocal = _mm512_rcp14_ps(denominator);
  reciprocal =
      _mm512_fmadd_ps(reciprocal, _mm512_fnmadd_ps(denominator, reciprocal, one), reciprocal);
  return _mm512_mul_ps(_mm512_mask_mov_ps(one, negative, exponent), reciprocal);
}

} // namespace onnx_light_cpu::detail
