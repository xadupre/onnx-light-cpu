// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/binary/binary_arithmetic_kernel.h"

#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace onnx_light_cpu {
namespace {

template <char Operation> __m512h Apply(__m512h left, __m512h right) {
  if constexpr (Operation == '+') {
    return _mm512_add_ph(left, right);
  } else if constexpr (Operation == '-') {
    return _mm512_sub_ph(left, right);
  } else if constexpr (Operation == '*') {
    return _mm512_mul_ph(left, right);
  } else {
    return _mm512_div_ph(left, right);
  }
}

template <char Operation> float ApplyScalar(float left, float right) {
  if constexpr (Operation == '+') {
    return left + right;
  } else if constexpr (Operation == '-') {
    return left - right;
  } else if constexpr (Operation == '*') {
    return left * right;
  } else {
    return left / right;
  }
}

template <char Operation, bool LeftScalar, bool RightScalar>
void StoreResult(__m512h result, const std::uint16_t *left, const std::uint16_t *right,
                 std::uint16_t *output) {
  _mm512_storeu_ph(output, result);
  const __mmask32 nan_mask = _mm512_cmp_ph_mask(result, result, _CMP_UNORD_Q);
  if (nan_mask == 0) {
    return;
  }
  for (std::size_t lane = 0; lane < 32; ++lane) {
    if ((nan_mask & (__mmask32{1} << lane)) != 0) {
      output[lane] = detail::FloatToFloat16Bits(
          ApplyScalar<Operation>(detail::Float16BitsToFloat(left[LeftScalar ? 0 : lane]),
                                 detail::Float16BitsToFloat(right[RightScalar ? 0 : lane])));
    }
  }
}

template <char Operation, bool LeftScalar, bool RightScalar>
void BinaryFloat16AVX512FP16(const std::uint16_t *left, const std::uint16_t *right,
                             std::uint16_t *output, std::size_t count) {
  const __m512h scalar_left =
      LeftScalar ? _mm512_set1_ph(std::bit_cast<_Float16>(*left)) : _mm512_setzero_ph();
  const __m512h scalar_right =
      RightScalar ? _mm512_set1_ph(std::bit_cast<_Float16>(*right)) : _mm512_setzero_ph();
  std::size_t i = 0;
  for (; i + 32 <= count; i += 32) {
    const __m512h a = LeftScalar ? scalar_left : _mm512_loadu_ph(left + i);
    const __m512h b = RightScalar ? scalar_right : _mm512_loadu_ph(right + i);
    StoreResult<Operation, LeftScalar, RightScalar>(Apply<Operation>(a, b),
                                                    LeftScalar ? left : left + i,
                                                    RightScalar ? right : right + i, output + i);
  }

  const float scalar_a = LeftScalar ? detail::Float16BitsToFloat(*left) : 0.0f;
  const float scalar_b = RightScalar ? detail::Float16BitsToFloat(*right) : 0.0f;
  for (; i < count; ++i) {
    const float a = LeftScalar ? scalar_a : detail::Float16BitsToFloat(left[i]);
    const float b = RightScalar ? scalar_b : detail::Float16BitsToFloat(right[i]);
    output[i] = detail::FloatToFloat16Bits(ApplyScalar<Operation>(a, b));
  }
}

} // namespace

#define ONNX_LIGHT_CPU_DEFINE_FLOAT16_AVX512FP16(STEM, OPERATION)                                  \
  void STEM##_AVX512FP16(const std::uint16_t *left, const std::uint16_t *right,                    \
                         std::uint16_t *output, std::size_t count) {                               \
    BinaryFloat16AVX512FP16<OPERATION, false, false>(left, right, output, count);                  \
  }                                                                                                \
  void STEM##Left_AVX512FP16(std::uint16_t left, const std::uint16_t *right,                       \
                             std::uint16_t *output, std::size_t count) {                           \
    BinaryFloat16AVX512FP16<OPERATION, true, false>(&left, right, output, count);                  \
  }                                                                                                \
  void STEM##Right_AVX512FP16(const std::uint16_t *left, std::uint16_t right,                      \
                              std::uint16_t *output, std::size_t count) {                          \
    BinaryFloat16AVX512FP16<OPERATION, false, true>(left, &right, output, count);                  \
  }

ONNX_LIGHT_CPU_DEFINE_FLOAT16_AVX512FP16(BinaryAddFloat16, '+')
ONNX_LIGHT_CPU_DEFINE_FLOAT16_AVX512FP16(BinarySubFloat16, '-')
ONNX_LIGHT_CPU_DEFINE_FLOAT16_AVX512FP16(BinaryMulFloat16, '*')
ONNX_LIGHT_CPU_DEFINE_FLOAT16_AVX512FP16(BinaryDivFloat16, '/')

#undef ONNX_LIGHT_CPU_DEFINE_FLOAT16_AVX512FP16

} // namespace onnx_light_cpu
