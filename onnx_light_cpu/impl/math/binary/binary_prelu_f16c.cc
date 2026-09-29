// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <immintrin.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace onnx_light_cpu {
namespace {

void StorePRelu(__m256 value, std::uint16_t *output) {
  _mm_storeu_si128(reinterpret_cast<__m128i *>(output),
                   _mm256_cvtps_ph(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  const int nan_mask = _mm256_movemask_ps(_mm256_cmp_ps(value, value, _CMP_UNORD_Q));
  if (nan_mask != 0) {
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, value);
    for (int lane = 0; lane < 8; ++lane) {
      if ((nan_mask & (1 << lane)) != 0) {
        output[static_cast<std::size_t>(lane)] = detail::FloatToFloat16Bits(lanes[lane]);
      }
    }
  }
}

__m256 PRelu(__m256 value, __m256 slope) {
  const __m256 scaled = _mm256_mul_ps(value, slope);
  return _mm256_blendv_ps(value, scaled, _mm256_cmp_ps(value, _mm256_setzero_ps(), _CMP_LT_OQ));
}

} // namespace

void BulkFloat16PReluF16C(const void *left, const void *right, void *out, std::size_t count) {
  const auto *a = static_cast<const std::uint16_t *>(left);
  const auto *b = static_cast<const std::uint16_t *>(right);
  auto *y = static_cast<std::uint16_t *>(out);
  std::size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    const __m256 value = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i)));
    const __m256 slope = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i)));
    StorePRelu(PRelu(value, slope), y + i);
  }
  for (; i < count; ++i) {
    const float value = detail::Float16BitsToFloat(a[i]);
    const float slope = detail::Float16BitsToFloat(b[i]);
    y[i] = detail::FloatToFloat16Bits(value < 0.0F ? value * slope : value);
  }
}

void BulkFloat16PReluLeftF16C(const void *left, const void *right, void *out, std::size_t count) {
  const auto left_bits = *static_cast<const std::uint16_t *>(left);
  const float left_value = detail::Float16BitsToFloat(left_bits);
  auto *y = static_cast<std::uint16_t *>(out);
  if (!(left_value < 0.0F)) {
    std::fill_n(y, count, left_bits);
    return;
  }
  const auto *b = static_cast<const std::uint16_t *>(right);
  const __m256 value = _mm256_set1_ps(left_value);
  std::size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    const __m256 slope = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i)));
    StorePRelu(_mm256_mul_ps(value, slope), y + i);
  }
  for (; i < count; ++i) {
    y[i] = detail::FloatToFloat16Bits(left_value * detail::Float16BitsToFloat(b[i]));
  }
}

void BulkFloat16PReluRightF16C(const void *left, const void *right, void *out, std::size_t count) {
  const auto *a = static_cast<const std::uint16_t *>(left);
  const float right_value = detail::Float16BitsToFloat(*static_cast<const std::uint16_t *>(right));
  auto *y = static_cast<std::uint16_t *>(out);
  const __m256 slope = _mm256_set1_ps(right_value);
  std::size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    const __m256 value = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i)));
    StorePRelu(PRelu(value, slope), y + i);
  }
  for (; i < count; ++i) {
    const float value = detail::Float16BitsToFloat(a[i]);
    y[i] = detail::FloatToFloat16Bits(value < 0.0F ? value * right_value : value);
  }
}

} // namespace onnx_light_cpu
