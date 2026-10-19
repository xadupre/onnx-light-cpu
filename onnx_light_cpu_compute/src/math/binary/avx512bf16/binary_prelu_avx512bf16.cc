// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace onnx_light_cpu {
namespace {

__m512 BFloat16ToFloat32(const std::uint16_t *values) {
  const __m256i halves = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(values));
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(halves), 16));
}

float BFloat16ToFloat32(std::uint16_t value) {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

void StoreBFloat16(__m512 values, std::uint16_t *output) {
  const __m512i bits = _mm512_castps_si512(values);
  const __mmask16 is_nan =
      _mm512_cmp_epu32_mask(_mm512_and_si512(bits, _mm512_set1_epi32(0x7fffffff)),
                            _mm512_set1_epi32(0x7f800000), _MM_CMPINT_GT);
  const __m512 quiet_values =
      _mm512_castsi512_ps(_mm512_mask_or_epi32(bits, is_nan, bits, _mm512_set1_epi32(0x00400000)));
  const __m256bh converted = _mm512_cvtneps_pbh(quiet_values);
  std::memcpy(output, &converted, sizeof(converted));
}

__m512 PRelu(__m512 value, __m512 slope) {
  const __mmask16 negative = _mm512_cmp_ps_mask(value, _mm512_setzero_ps(), _CMP_LT_OQ);
  return _mm512_mask_mul_ps(value, negative, value, slope);
}

template <bool LeftScalar, bool RightScalar>
void BulkPRelu(const void *left, const void *right, void *out, std::size_t count) {
  if (count == 0) {
    return;
  }
  const auto *values = static_cast<const std::uint16_t *>(left);
  const auto *slopes = static_cast<const std::uint16_t *>(right);
  auto *output = static_cast<std::uint16_t *>(out);
  const __m512 scalar_value =
      LeftScalar ? _mm512_set1_ps(BFloat16ToFloat32(*values)) : _mm512_setzero_ps();
  const __m512 scalar_slope =
      RightScalar ? _mm512_set1_ps(BFloat16ToFloat32(*slopes)) : _mm512_setzero_ps();
  std::size_t index = 0;
  for (; index + 16 <= count; index += 16) {
    const __m512 value = LeftScalar ? scalar_value : BFloat16ToFloat32(values + index);
    const __m512 slope = RightScalar ? scalar_slope : BFloat16ToFloat32(slopes + index);
    StoreBFloat16(PRelu(value, slope), output + index);
  }
  if (index < count) {
    alignas(32) std::uint16_t value_tail[16]{};
    alignas(32) std::uint16_t slope_tail[16]{};
    alignas(32) std::uint16_t output_tail[16];
    const std::size_t tail_bytes = (count - index) * sizeof(std::uint16_t);
    if constexpr (!LeftScalar) {
      std::memcpy(value_tail, values + index, tail_bytes);
    }
    if constexpr (!RightScalar) {
      std::memcpy(slope_tail, slopes + index, tail_bytes);
    }
    const __m512 value = LeftScalar ? scalar_value : BFloat16ToFloat32(value_tail);
    const __m512 slope = RightScalar ? scalar_slope : BFloat16ToFloat32(slope_tail);
    StoreBFloat16(PRelu(value, slope), output_tail);
    std::memcpy(output + index, output_tail, tail_bytes);
  }
}

} // namespace

void BulkBfloat16PRelu_AVX512BF16(const void *left, const void *right, void *out,
                                  std::size_t count) {
  BulkPRelu<false, false>(left, right, out, count);
}

void BulkBfloat16PReluLeft_AVX512BF16(const void *left, const void *right, void *out,
                                      std::size_t count) {
  BulkPRelu<true, false>(left, right, out, count);
}

void BulkBfloat16PReluRight_AVX512BF16(const void *left, const void *right, void *out,
                                       std::size_t count) {
  BulkPRelu<false, true>(left, right, out, count);
}

} // namespace onnx_light_cpu
