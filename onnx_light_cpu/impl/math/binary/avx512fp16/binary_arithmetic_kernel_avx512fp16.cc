// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/binary/binary_arithmetic_kernel.h"

#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace onnx_light_cpu {
#ifdef ONNX_LIGHT_CPU_HAVE_AVX_F16C
void BulkFloat16PReluF16C(const void *, const void *, void *, std::size_t);
void BulkFloat16PReluLeftF16C(const void *, const void *, void *, std::size_t);
void BulkFloat16PReluRightF16C(const void *, const void *, void *, std::size_t);
#endif

namespace {

constexpr std::size_t kNativePReluMaximumCount = 256 * 1024;

bool PreferF16CPRelu(std::size_t count) {
#ifdef ONNX_LIGHT_CPU_HAVE_AVX_F16C
  static const bool use_f16c = CpuSupportsF16C();
  return count > kNativePReluMaximumCount && use_f16c;
#else
  (void)count;
  return false;
#endif
}

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

void StorePRelu(__m512h result, std::uint16_t *output) {
  _mm512_storeu_ph(output, result);
  const __mmask32 nan_mask = _mm512_cmp_ph_mask(result, result, _CMP_UNORD_Q);
  if (nan_mask == 0) {
    return;
  }
  for (std::size_t lane = 0; lane < 32; ++lane) {
    if ((nan_mask & (__mmask32{1} << lane)) != 0) {
      output[lane] = detail::FloatToFloat16Bits(detail::Float16BitsToFloat(output[lane]));
    }
  }
}

__m512h PRelu(__m512h value, __m512h slope) {
  const __mmask32 negative = _mm512_cmp_ph_mask(value, _mm512_setzero_ph(), _CMP_LT_OQ);
  return _mm512_mask_mul_ph(value, negative, value, slope);
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

void BulkFloat16PReluAVX512FP16(const void *left, const void *right, void *out, std::size_t count) {
#ifdef ONNX_LIGHT_CPU_HAVE_AVX_F16C
  if (PreferF16CPRelu(count)) {
    BulkFloat16PReluF16C(left, right, out, count);
    return;
  }
#endif
  const auto *a = static_cast<const std::uint16_t *>(left);
  const auto *b = static_cast<const std::uint16_t *>(right);
  auto *y = static_cast<std::uint16_t *>(out);
  std::size_t i = 0;
  for (; i + 32 <= count; i += 32) {
    StorePRelu(PRelu(_mm512_loadu_ph(a + i), _mm512_loadu_ph(b + i)), y + i);
  }
  for (; i < count; ++i) {
    const float value = detail::Float16BitsToFloat(a[i]);
    const float slope = detail::Float16BitsToFloat(b[i]);
    y[i] = detail::FloatToFloat16Bits(value < 0.0f ? value * slope : value);
  }
}

void BulkFloat16PReluLeftAVX512FP16(const void *left, const void *right, void *out,
                                    std::size_t count) {
#ifdef ONNX_LIGHT_CPU_HAVE_AVX_F16C
  if (PreferF16CPRelu(count)) {
    BulkFloat16PReluLeftF16C(left, right, out, count);
    return;
  }
#endif
  const auto left_bits = *static_cast<const std::uint16_t *>(left);
  const float left_value = detail::Float16BitsToFloat(left_bits);
  auto *y = static_cast<std::uint16_t *>(out);
  if (!(left_value < 0.0f)) {
    std::fill_n(y, count, left_bits);
    return;
  }
  const auto *b = static_cast<const std::uint16_t *>(right);
  const __m512h value = _mm512_set1_ph(std::bit_cast<_Float16>(left_bits));
  std::size_t i = 0;
  for (; i + 32 <= count; i += 32) {
    StorePRelu(_mm512_mul_ph(value, _mm512_loadu_ph(b + i)), y + i);
  }
  for (; i < count; ++i) {
    y[i] = detail::FloatToFloat16Bits(left_value * detail::Float16BitsToFloat(b[i]));
  }
}

void BulkFloat16PReluRightAVX512FP16(const void *left, const void *right, void *out,
                                     std::size_t count) {
#ifdef ONNX_LIGHT_CPU_HAVE_AVX_F16C
  if (PreferF16CPRelu(count)) {
    BulkFloat16PReluRightF16C(left, right, out, count);
    return;
  }
#endif
  const auto *a = static_cast<const std::uint16_t *>(left);
  const auto right_bits = *static_cast<const std::uint16_t *>(right);
  const float right_value = detail::Float16BitsToFloat(right_bits);
  auto *y = static_cast<std::uint16_t *>(out);
  const __m512h slope = _mm512_set1_ph(std::bit_cast<_Float16>(right_bits));
  std::size_t i = 0;
  for (; i + 32 <= count; i += 32) {
    StorePRelu(PRelu(_mm512_loadu_ph(a + i), slope), y + i);
  }
  for (; i < count; ++i) {
    const float value = detail::Float16BitsToFloat(a[i]);
    y[i] = detail::FloatToFloat16Bits(value < 0.0f ? value * right_value : value);
  }
}

} // namespace onnx_light_cpu
