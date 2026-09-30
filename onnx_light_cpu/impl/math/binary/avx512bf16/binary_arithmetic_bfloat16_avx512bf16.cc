// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/binary/binary_arithmetic_kernel.h"

#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <cstring>
#include <immintrin.h>

namespace onnx_light_cpu {
namespace {

__m512 LoadBFloat16(const std::uint16_t *source) {
  const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(source));
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(packed), 16));
}

void StoreBFloat16(__m512 value, std::uint16_t *destination) {
  const __m512i bits = _mm512_castps_si512(value);
  const __mmask16 is_nan =
      _mm512_cmp_epu32_mask(_mm512_and_si512(bits, _mm512_set1_epi32(0x7fffffff)),
                            _mm512_set1_epi32(0x7f800000), _MM_CMPINT_GT);
  const __m512 quiet_value =
      _mm512_castsi512_ps(_mm512_mask_or_epi32(bits, is_nan, bits, _mm512_set1_epi32(0x00400000)));
  const __m256bh packed = _mm512_cvtneps_pbh(quiet_value);
  std::memcpy(destination, &packed, sizeof(packed));
}

struct Add {
  static __m512 Vector(__m512 left, __m512 right) { return _mm512_add_ps(left, right); }
  static float Scalar(float left, float right) { return left + right; }
};

struct Sub {
  static __m512 Vector(__m512 left, __m512 right) { return _mm512_sub_ps(left, right); }
  static float Scalar(float left, float right) { return left - right; }
};

struct Mul {
  static __m512 Vector(__m512 left, __m512 right) { return _mm512_mul_ps(left, right); }
  static float Scalar(float left, float right) { return left * right; }
};

template <typename Op, bool LeftScalar, bool RightScalar>
void BinaryBFloat16(const std::uint16_t *left, const std::uint16_t *right, std::uint16_t *out,
                    std::size_t count) {
  const float scalar_left = LeftScalar ? detail::Bfloat16BitsToFloat(*left) : 0.0f;
  const float scalar_right = RightScalar ? detail::Bfloat16BitsToFloat(*right) : 0.0f;
  const __m512 vector_left = _mm512_set1_ps(scalar_left);
  const __m512 vector_right = _mm512_set1_ps(scalar_right);
  std::size_t index = 0;
  for (; index + 16 <= count; index += 16) {
    const __m512 a = LeftScalar ? vector_left : LoadBFloat16(left + index);
    const __m512 b = RightScalar ? vector_right : LoadBFloat16(right + index);
    StoreBFloat16(Op::Vector(a, b), out + index);
  }
  for (; index < count; ++index) {
    const float a = LeftScalar ? scalar_left : detail::Bfloat16BitsToFloat(left[index]);
    const float b = RightScalar ? scalar_right : detail::Bfloat16BitsToFloat(right[index]);
    out[index] = detail::FloatToBFloat16Bits(Op::Scalar(a, b));
  }
}

} // namespace

#define ONNX_LIGHT_CPU_DEFINE_BINARY_BFLOAT16_AVX512BF16(NAME, OP)                                 \
  void NAME##_AVX512BF16(const std::uint16_t *left, const std::uint16_t *right,                    \
                         std::uint16_t *out, std::size_t count) {                                  \
    if (count < 16) {                                                                              \
      NAME##_AVX2(left, right, out, count);                                                        \
      return;                                                                                      \
    }                                                                                              \
    BinaryBFloat16<OP, false, false>(left, right, out, count);                                     \
  }                                                                                                \
  void NAME##Left_AVX512BF16(std::uint16_t left, const std::uint16_t *right, std::uint16_t *out,   \
                             std::size_t count) {                                                  \
    if (count < 16) {                                                                              \
      NAME##Left_AVX2(left, right, out, count);                                                    \
      return;                                                                                      \
    }                                                                                              \
    BinaryBFloat16<OP, true, false>(&left, right, out, count);                                     \
  }                                                                                                \
  void NAME##Right_AVX512BF16(const std::uint16_t *left, std::uint16_t right, std::uint16_t *out,  \
                              std::size_t count) {                                                 \
    if (count < 16) {                                                                              \
      NAME##Right_AVX2(left, right, out, count);                                                   \
      return;                                                                                      \
    }                                                                                              \
    BinaryBFloat16<OP, false, true>(left, &right, out, count);                                     \
  }

ONNX_LIGHT_CPU_DEFINE_BINARY_BFLOAT16_AVX512BF16(BinaryAddBFloat16, Add)
ONNX_LIGHT_CPU_DEFINE_BINARY_BFLOAT16_AVX512BF16(BinarySubBFloat16, Sub)
ONNX_LIGHT_CPU_DEFINE_BINARY_BFLOAT16_AVX512BF16(BinaryMulBFloat16, Mul)

#undef ONNX_LIGHT_CPU_DEFINE_BINARY_BFLOAT16_AVX512BF16

} // namespace onnx_light_cpu
