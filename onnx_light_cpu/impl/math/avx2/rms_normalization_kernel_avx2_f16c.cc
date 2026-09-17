// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/rms_normalization.h"

#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/impl/math/normalization_kernel.h"

#include <immintrin.h>

#include <cmath>

namespace onnx_light_cpu {
namespace {

float HorizontalSum(__m256 value) {
  __m128 sum = _mm_add_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1));
  sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
  sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
  return _mm_cvtss_f32(sum);
}

float MeanSquareFloat16(const std::uint16_t *input, std::size_t width) {
  // Four independent accumulators shorten the reduction's dependency chain
  // (matching the float32 and bfloat16 AVX2 reductions), letting the
  // out-of-order engine overlap the otherwise serialized multiply-adds.
  // This translation unit is compiled with only ``-mavx -mf16c`` (no FMA),
  // so the square is accumulated with a separate multiply and add.
  __m256 sums[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                    _mm256_setzero_ps()};
  std::size_t column = 0;
  for (; column + 32 <= width; column += 32) {
    for (std::size_t lane = 0; lane < 4; ++lane) {
      const __m128i packed =
          _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + column + lane * 8));
      const __m256 value = _mm256_cvtph_ps(packed);
      sums[lane] = _mm256_add_ps(sums[lane], _mm256_mul_ps(value, value));
    }
  }
  for (; column + 8 <= width; column += 8) {
    const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + column));
    const __m256 value = _mm256_cvtph_ps(packed);
    sums[0] = _mm256_add_ps(sums[0], _mm256_mul_ps(value, value));
  }
  float sum_squares = HorizontalSum(
      _mm256_add_ps(_mm256_add_ps(sums[0], sums[1]), _mm256_add_ps(sums[2], sums[3])));
  for (; column < width; ++column) {
    const float value = detail::Float16BitsToFloat(input[column]);
    sum_squares += value * value;
  }

  return sum_squares / static_cast<float>(width);
}

template <bool RoundNormalized>
void AffineFloat16(const std::uint16_t *input, const std::uint16_t *scale, std::uint16_t *output,
                   std::size_t width, float multiplier) {
  // VCVTPS2PH ignores _MM_FROUND_NO_EXC; the integer scalar codec never traps.
  const unsigned int mxcsr = _mm_getcsr();
  _mm_setcsr(mxcsr | _MM_MASK_MASK);
  const __m256 inverse_rms = _mm256_set1_ps(multiplier);
  std::size_t column = 0;
  for (; column + 8 <= width; column += 8) {
    const __m128i packed_input = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + column));
    const __m128i packed_scale = _mm_loadu_si128(reinterpret_cast<const __m128i *>(scale + column));
    const __m256 value = _mm256_cvtph_ps(packed_input);
    const __m256 weight = _mm256_cvtph_ps(packed_scale);
    __m256 normalized = _mm256_mul_ps(value, inverse_rms);
    if constexpr (RoundNormalized) {
      const __m128i packed_normalized = _mm256_cvtps_ph(normalized, _MM_FROUND_TO_NEAREST_INT);
      normalized = _mm256_cvtph_ps(packed_normalized);
    }
    const __m256 scaled = _mm256_mul_ps(normalized, weight);
    const __m128i packed_output = _mm256_cvtps_ph(scaled, _MM_FROUND_TO_NEAREST_INT);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output + column), packed_output);
    const int nan_mask = _mm256_movemask_ps(_mm256_cmp_ps(scaled, scaled, _CMP_UNORD_Q));
    if (nan_mask != 0) {
      alignas(32) float lanes[8];
      _mm256_store_ps(lanes, scaled);
      for (int lane = 0; lane < 8; ++lane) {
        if ((nan_mask & (1 << lane)) != 0) {
          output[column + static_cast<std::size_t>(lane)] = detail::FloatToFloat16Bits(lanes[lane]);
        }
      }
    }
  }
  _mm_setcsr(mxcsr);
  for (; column < width; ++column) {
    const float value = detail::Float16BitsToFloat(input[column]);
    const float weight = detail::Float16BitsToFloat(scale[column]);
    float normalized = value * multiplier;
    if constexpr (RoundNormalized) {
      normalized = detail::Float16BitsToFloat(detail::FloatToFloat16Bits(normalized));
    }
    output[column] = detail::FloatToFloat16Bits(normalized * weight);
  }
}

} // namespace

float ComputeNormalizationMeanSquareFloat16_F16C(const std::uint16_t *input, std::size_t count) {
  return MeanSquareFloat16(input, count);
}

void ApplyNormalizationAffineFloat16_F16C(const std::uint16_t *input, const std::uint16_t *scale,
                                          std::uint16_t *output, std::size_t count,
                                          float multiplier) {
  AffineFloat16<false>(input, scale, output, count, multiplier);
}

void RmsNormalizationFloat16_F16C(const std::uint16_t *input, const std::uint16_t *scale,
                                  std::uint16_t *output, std::size_t row_begin, std::size_t row_end,
                                  std::size_t width, float epsilon) {
  for (std::size_t row = row_begin; row < row_end; ++row) {
    const std::size_t offset = row * width;
    const float inverse_rms = 1.0F / std::sqrt(MeanSquareFloat16(input + offset, width) + epsilon);
    AffineFloat16<true>(input + offset, scale, output + offset, width, inverse_rms);
  }
}

} // namespace onnx_light_cpu
