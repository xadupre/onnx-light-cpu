// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/binary/binary_arithmetic_kernel.h"

#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <immintrin.h>

namespace onnx_light_cpu {
namespace {

__m256 LoadBFloat16(const std::uint16_t *source) {
  const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i *>(source));
  return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(packed), 16));
}

void StoreBFloat16(__m256 value, std::uint16_t *destination) {
  const __m256i bits = _mm256_castps_si256(value);
  const __m256i upper = _mm256_srli_epi32(bits, 16);
  const __m256i rounding =
      _mm256_add_epi32(_mm256_set1_epi32(0x7fff), _mm256_and_si256(upper, _mm256_set1_epi32(1)));
  const __m256i rounded = _mm256_srli_epi32(_mm256_add_epi32(bits, rounding), 16);
  const __m256i is_nan = _mm256_cmpgt_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff)),
                                            _mm256_set1_epi32(0x7f800000));
  const __m256i narrowed =
      _mm256_blendv_epi8(rounded, _mm256_or_si256(upper, _mm256_set1_epi32(0x40)), is_nan);
  const __m128i packed =
      _mm_packus_epi32(_mm256_castsi256_si128(narrowed), _mm256_extracti128_si256(narrowed, 1));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(destination), packed);
}

struct Add {
  static __m256 Vector(__m256 left, __m256 right) { return _mm256_add_ps(left, right); }
  static float Scalar(float left, float right) { return left + right; }
};

struct Sub {
  static __m256 Vector(__m256 left, __m256 right) { return _mm256_sub_ps(left, right); }
  static float Scalar(float left, float right) { return left - right; }
};

struct Mul {
  static __m256 Vector(__m256 left, __m256 right) { return _mm256_mul_ps(left, right); }
  static float Scalar(float left, float right) { return left * right; }
};

template <typename Op, bool LeftScalar, bool RightScalar>
void BinaryBFloat16(const std::uint16_t *left, const std::uint16_t *right, std::uint16_t *out,
                    std::size_t count) {
  const float scalar_left = LeftScalar ? detail::Bfloat16BitsToFloat(*left) : 0.0f;
  const float scalar_right = RightScalar ? detail::Bfloat16BitsToFloat(*right) : 0.0f;
  const __m256 vector_left = _mm256_set1_ps(scalar_left);
  const __m256 vector_right = _mm256_set1_ps(scalar_right);
  std::size_t index = 0;
  for (; index + 8 <= count; index += 8) {
    const __m256 a = LeftScalar ? vector_left : LoadBFloat16(left + index);
    const __m256 b = RightScalar ? vector_right : LoadBFloat16(right + index);
    StoreBFloat16(Op::Vector(a, b), out + index);
  }
  for (; index < count; ++index) {
    const float a = LeftScalar ? scalar_left : detail::Bfloat16BitsToFloat(left[index]);
    const float b = RightScalar ? scalar_right : detail::Bfloat16BitsToFloat(right[index]);
    out[index] = detail::FloatToBFloat16Bits(Op::Scalar(a, b));
  }
}

} // namespace

#define ONNX_LIGHT_CPU_DEFINE_BINARY_BFLOAT16_AVX2(NAME, OP)                                       \
  void NAME##_AVX2(const std::uint16_t *left, const std::uint16_t *right, std::uint16_t *out,      \
                   std::size_t count) {                                                            \
    BinaryBFloat16<OP, false, false>(left, right, out, count);                                     \
  }                                                                                                \
  void NAME##Left_AVX2(std::uint16_t left, const std::uint16_t *right, std::uint16_t *out,         \
                       std::size_t count) {                                                        \
    BinaryBFloat16<OP, true, false>(&left, right, out, count);                                     \
  }                                                                                                \
  void NAME##Right_AVX2(const std::uint16_t *left, std::uint16_t right, std::uint16_t *out,        \
                        std::size_t count) {                                                       \
    BinaryBFloat16<OP, false, true>(left, &right, out, count);                                     \
  }

ONNX_LIGHT_CPU_DEFINE_BINARY_BFLOAT16_AVX2(BinaryAddBFloat16, Add)
ONNX_LIGHT_CPU_DEFINE_BINARY_BFLOAT16_AVX2(BinarySubBFloat16, Sub)
ONNX_LIGHT_CPU_DEFINE_BINARY_BFLOAT16_AVX2(BinaryMulBFloat16, Mul)

#undef ONNX_LIGHT_CPU_DEFINE_BINARY_BFLOAT16_AVX2

} // namespace onnx_light_cpu
