// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/binary/binary_arithmetic_kernel.h"

#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <immintrin.h>

namespace onnx_light_cpu {
namespace {

void StoreFloat16(__m256 values, std::uint16_t *output) {
  const __m128i halves = _mm256_cvtps_ph(values, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  _mm_storeu_si128(reinterpret_cast<__m128i *>(output), halves);
  const int nan_mask = _mm256_movemask_ps(_mm256_cmp_ps(values, values, _CMP_UNORD_Q));
  if (nan_mask == 0) {
    return;
  }
  alignas(32) float lanes[8];
  _mm256_store_ps(lanes, values);
  for (std::size_t lane = 0; lane < 8; ++lane) {
    if ((nan_mask & (1 << lane)) != 0) {
      output[lane] = detail::FloatToFloat16Bits(lanes[lane]);
    }
  }
}

template <char Operation> __m256 Apply(__m256 left, __m256 right) {
  if constexpr (Operation == '+') {
    return _mm256_add_ps(left, right);
  } else if constexpr (Operation == '-') {
    return _mm256_sub_ps(left, right);
  } else if constexpr (Operation == '*') {
    return _mm256_mul_ps(left, right);
  } else {
    return _mm256_div_ps(left, right);
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
void BinaryFloat16F16C(const std::uint16_t *left, const std::uint16_t *right, std::uint16_t *out,
                       std::size_t count) {
  const __m256 scalar_left =
      LeftScalar ? _mm256_set1_ps(detail::Float16BitsToFloat(*left)) : _mm256_setzero_ps();
  const __m256 scalar_right =
      RightScalar ? _mm256_set1_ps(detail::Float16BitsToFloat(*right)) : _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    const __m256 a =
        LeftScalar ? scalar_left
                   : _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(left + i)));
    const __m256 b =
        RightScalar
            ? scalar_right
            : _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(right + i)));
    StoreFloat16(Apply<Operation>(a, b), out + i);
  }

  const float scalar_a = LeftScalar ? detail::Float16BitsToFloat(*left) : 0.0f;
  const float scalar_b = RightScalar ? detail::Float16BitsToFloat(*right) : 0.0f;
  for (; i < count; ++i) {
    const float a = LeftScalar ? scalar_a : detail::Float16BitsToFloat(left[i]);
    const float b = RightScalar ? scalar_b : detail::Float16BitsToFloat(right[i]);
    out[i] = detail::FloatToFloat16Bits(ApplyScalar<Operation>(a, b));
  }
}

} // namespace

#define ONNX_LIGHT_CPU_DEFINE_FLOAT16_F16C(STEM, OPERATION)                                        \
  void STEM##_F16C(const std::uint16_t *left, const std::uint16_t *right, std::uint16_t *out,      \
                   std::size_t count) {                                                            \
    BinaryFloat16F16C<OPERATION, false, false>(left, right, out, count);                           \
  }                                                                                                \
  void STEM##Left_F16C(std::uint16_t left, const std::uint16_t *right, std::uint16_t *out,         \
                       std::size_t count) {                                                        \
    BinaryFloat16F16C<OPERATION, true, false>(&left, right, out, count);                           \
  }                                                                                                \
  void STEM##Right_F16C(const std::uint16_t *left, std::uint16_t right, std::uint16_t *out,        \
                        std::size_t count) {                                                       \
    BinaryFloat16F16C<OPERATION, false, true>(left, &right, out, count);                           \
  }

ONNX_LIGHT_CPU_DEFINE_FLOAT16_F16C(BinaryAddFloat16, '+')
ONNX_LIGHT_CPU_DEFINE_FLOAT16_F16C(BinarySubFloat16, '-')
ONNX_LIGHT_CPU_DEFINE_FLOAT16_F16C(BinaryMulFloat16, '*')
ONNX_LIGHT_CPU_DEFINE_FLOAT16_F16C(BinaryDivFloat16, '/')

#undef ONNX_LIGHT_CPU_DEFINE_FLOAT16_F16C

} // namespace onnx_light_cpu
