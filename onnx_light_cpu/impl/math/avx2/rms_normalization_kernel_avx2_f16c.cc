// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/rms_normalization.h"

#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/impl/math/normalization_kernel.h"

#include <immintrin.h>

#include <cmath>
#include <limits>

namespace onnx_light_cpu {
namespace {

float HorizontalSum(__m256 value) {
  __m128 sum = _mm_add_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1));
  sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
  sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
  return _mm_cvtss_f32(sum);
}

double HorizontalSum(__m256d value) {
  const __m128d halves = _mm_add_pd(_mm256_castpd256_pd128(value), _mm256_extractf128_pd(value, 1));
  return _mm_cvtsd_f64(_mm_add_sd(halves, _mm_unpackhi_pd(halves, halves)));
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

void StoreHalf8(__m256 value, std::uint16_t *output) {
  const __m128i packed = _mm256_cvtps_ph(value, _MM_FROUND_TO_NEAREST_INT);
  _mm_storeu_si128(reinterpret_cast<__m128i *>(output), packed);
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

void ApplyScaleBiasFloat16Masked(const std::uint16_t *input, std::uint16_t *output,
                                 std::size_t count, float multiplier, float offset) {
  const __m256 scale = _mm256_set1_ps(multiplier);
  const __m256 bias = _mm256_set1_ps(offset);
  std::size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    const __m256 value =
        _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(input + i)));
    StoreHalf8(_mm256_add_ps(_mm256_mul_ps(value, scale), bias), output + i);
  }
  for (; i < count; ++i) {
    output[i] =
        detail::FloatToFloat16Bits(detail::Float16BitsToFloat(input[i]) * multiplier + offset);
  }
}

void ApplyLayerNormalizationFloat16Masked(const std::uint16_t *input,
                                          const std::uint16_t *scale_data,
                                          const std::uint16_t *bias_data, std::uint16_t *output,
                                          std::size_t count, float center_value, float multiplier) {
  const __m256 center = _mm256_set1_ps(center_value);
  const __m256 inverse = _mm256_set1_ps(multiplier);
  std::size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    const __m256 value =
        _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(input + i)));
    __m256 normalized = _mm256_mul_ps(_mm256_sub_ps(value, center), inverse);
    normalized = _mm256_cvtph_ps(_mm256_cvtps_ph(normalized, _MM_FROUND_TO_NEAREST_INT));
    const __m256 scale =
        _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(scale_data + i)));
    __m256 result = _mm256_mul_ps(normalized, scale);
    result = _mm256_cvtph_ps(_mm256_cvtps_ph(result, _MM_FROUND_TO_NEAREST_INT));
    if (bias_data != nullptr) {
      const __m256 bias =
          _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(bias_data + i)));
      result = _mm256_add_ps(result, bias);
    }
    StoreHalf8(result, output + i);
  }
  for (; i < count; ++i) {
    const float normalized = detail::Float16BitsToFloat(detail::FloatToFloat16Bits(
        (detail::Float16BitsToFloat(input[i]) - center_value) * multiplier));
    float result = detail::Float16BitsToFloat(
        detail::FloatToFloat16Bits(normalized * detail::Float16BitsToFloat(scale_data[i])));
    if (bias_data != nullptr) {
      result += detail::Float16BitsToFloat(bias_data[i]);
    }
    output[i] = detail::FloatToFloat16Bits(result);
  }
}

void ApplyGroupNormalizationFloat16Masked(const std::uint16_t *input, std::uint16_t *output,
                                          std::size_t count, float center_value, float multiplier,
                                          float scale_value, float bias_value) {
  const __m256 center = _mm256_set1_ps(center_value);
  const __m256 inverse = _mm256_set1_ps(multiplier);
  const __m256 scale = _mm256_set1_ps(scale_value);
  const __m256 bias = _mm256_set1_ps(bias_value);
  std::size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    const __m256 value =
        _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(input + i)));
    __m256 normalized = _mm256_mul_ps(_mm256_sub_ps(value, center), inverse);
    normalized = _mm256_cvtph_ps(_mm256_cvtps_ph(normalized, _MM_FROUND_TO_NEAREST_INT));
    __m256 result = _mm256_mul_ps(normalized, scale);
    result = _mm256_cvtph_ps(_mm256_cvtps_ph(result, _MM_FROUND_TO_NEAREST_INT));
    StoreHalf8(_mm256_add_ps(result, bias), output + i);
  }
  for (; i < count; ++i) {
    const float normalized = detail::Float16BitsToFloat(detail::FloatToFloat16Bits(
        (detail::Float16BitsToFloat(input[i]) - center_value) * multiplier));
    const float scaled =
        detail::Float16BitsToFloat(detail::FloatToFloat16Bits(normalized * scale_value));
    output[i] = detail::FloatToFloat16Bits(scaled + bias_value);
  }
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
    StoreHalf8(scaled, output + column);
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

Float32NormalizationMoments ComputeNormalizationMomentsFloat16_F16C(const std::uint16_t *input,
                                                                    std::size_t count) {
  __m256 sums[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                    _mm256_setzero_ps()};
  __m256 square_sums[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                           _mm256_setzero_ps()};
  std::size_t i = 0;
  for (; i + 32 <= count; i += 32) {
    for (std::size_t lane = 0; lane < 4; ++lane) {
      const __m256 value =
          _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(input + i + lane * 8)));
      sums[lane] = _mm256_add_ps(sums[lane], value);
      square_sums[lane] = _mm256_add_ps(square_sums[lane], _mm256_mul_ps(value, value));
    }
  }
  float sum = HorizontalSum(
      _mm256_add_ps(_mm256_add_ps(sums[0], sums[1]), _mm256_add_ps(sums[2], sums[3])));
  float square_sum = HorizontalSum(_mm256_add_ps(_mm256_add_ps(square_sums[0], square_sums[1]),
                                                 _mm256_add_ps(square_sums[2], square_sums[3])));
  for (; i < count; ++i) {
    const float value = detail::Float16BitsToFloat(input[i]);
    sum += value;
    square_sum += value * value;
  }
  const float mean = sum / static_cast<float>(count);
  const float second_moment = square_sum / static_cast<float>(count);
  float variance = second_moment - mean * mean;
  const float floor = second_moment * (std::numeric_limits<float>::epsilon() *
                                           static_cast<float>(count / 4 + (count % 4 != 0)) * 8.0F +
                                       std::sqrt(std::numeric_limits<float>::epsilon()) * 4.0F);
  if (!(variance > floor)) {
    __m256 centered[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                          _mm256_setzero_ps()};
    const __m256 center = _mm256_set1_ps(mean);
    i = 0;
    for (; i + 32 <= count; i += 32) {
      for (std::size_t lane = 0; lane < 4; ++lane) {
        const __m256 value = _mm256_cvtph_ps(
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + i + lane * 8)));
        const __m256 delta = _mm256_sub_ps(value, center);
        centered[lane] = _mm256_add_ps(centered[lane], _mm256_mul_ps(delta, delta));
      }
    }
    float centered_sum = HorizontalSum(_mm256_add_ps(_mm256_add_ps(centered[0], centered[1]),
                                                     _mm256_add_ps(centered[2], centered[3])));
    for (; i < count; ++i) {
      const float delta = detail::Float16BitsToFloat(input[i]) - mean;
      centered_sum += delta * delta;
    }
    variance = centered_sum / static_cast<float>(count);
  }
  return {mean, variance};
}

void ApplyNormalizationScaleBiasFloat16_F16C(const std::uint16_t *input, std::uint16_t *output,
                                             std::size_t count, float multiplier, float offset) {
  const unsigned int mxcsr = _mm_getcsr();
  _mm_setcsr(mxcsr | _MM_MASK_MASK);
  ApplyScaleBiasFloat16Masked(input, output, count, multiplier, offset);
  _mm_setcsr(mxcsr);
}

void InstanceNormalizationFloat16_F16C(const std::uint16_t *input, const std::uint16_t *scale,
                                       const std::uint16_t *bias, std::uint16_t *output,
                                       std::size_t slice_begin, std::size_t slice_end,
                                       std::size_t channels, std::size_t spatial, float epsilon) {
  const unsigned int mxcsr = _mm_getcsr();
  _mm_setcsr(mxcsr | _MM_MASK_MASK);
  for (std::size_t slice = slice_begin; slice < slice_end; ++slice) {
    const std::size_t channel = slice % channels;
    const std::size_t base = slice * spatial;
    const Float32NormalizationMoments moments =
        ComputeNormalizationMomentsFloat16_F16C(input + base, spatial);
    const float multiplier =
        detail::Float16BitsToFloat(scale[channel]) / std::sqrt(moments.variance + epsilon);
    const float offset = detail::Float16BitsToFloat(bias[channel]) - moments.mean * multiplier;
    ApplyScaleBiasFloat16Masked(input + base, output + base, spatial, multiplier, offset);
  }
  _mm_setcsr(mxcsr);
}

void ApplyLayerNormalizationFloat16_F16C(const std::uint16_t *input,
                                         const std::uint16_t *scale_data,
                                         const std::uint16_t *bias_data, std::uint16_t *output,
                                         std::size_t count, float center_value, float multiplier) {
  const unsigned int mxcsr = _mm_getcsr();
  _mm_setcsr(mxcsr | _MM_MASK_MASK);
  ApplyLayerNormalizationFloat16Masked(input, scale_data, bias_data, output, count, center_value,
                                       multiplier);
  _mm_setcsr(mxcsr);
}

void LayerNormalizationFloat16Rows_F16C(const std::uint16_t *input, const std::uint16_t *scale,
                                        const std::uint16_t *bias, std::uint16_t *output,
                                        float *mean_output, float *inv_output,
                                        std::size_t row_begin, std::size_t row_end,
                                        std::size_t width, float epsilon) {
  const unsigned int mxcsr = _mm_getcsr();
  _mm_setcsr(mxcsr | _MM_MASK_MASK);
  for (std::size_t row = row_begin; row < row_end; ++row) {
    const std::size_t base = row * width;
    const Float32NormalizationMoments moments =
        ComputeNormalizationMomentsFloat16_F16C(input + base, width);
    const float inverse = 1.0F / std::sqrt(moments.variance + epsilon);
    if (mean_output != nullptr) {
      mean_output[row] = moments.mean;
    }
    if (inv_output != nullptr) {
      inv_output[row] = inverse;
    }
    ApplyLayerNormalizationFloat16Masked(input + base, scale, bias, output + base, width,
                                         moments.mean, inverse);
  }
  _mm_setcsr(mxcsr);
}

void ApplyGroupNormalizationFloat16_F16C(const std::uint16_t *input, std::uint16_t *output,
                                         std::size_t count, float center_value, float multiplier,
                                         float scale_value, float bias_value) {
  const unsigned int mxcsr = _mm_getcsr();
  _mm_setcsr(mxcsr | _MM_MASK_MASK);
  ApplyGroupNormalizationFloat16Masked(input, output, count, center_value, multiplier, scale_value,
                                       bias_value);
  _mm_setcsr(mxcsr);
}

void ApplyGroupNormalizationFloat16Channels_F16C(const std::uint16_t *input,
                                                 const std::uint16_t *scale,
                                                 const std::uint16_t *bias, std::uint16_t *output,
                                                 std::size_t channels, std::size_t spatial,
                                                 float center, float multiplier) {
  const unsigned int mxcsr = _mm_getcsr();
  _mm_setcsr(mxcsr | _MM_MASK_MASK);
  for (std::size_t channel = 0; channel < channels; ++channel) {
    const std::size_t base = channel * spatial;
    ApplyGroupNormalizationFloat16Masked(input + base, output + base, spatial, center, multiplier,
                                         detail::Float16BitsToFloat(scale[channel]),
                                         detail::Float16BitsToFloat(bias[channel]));
  }
  _mm_setcsr(mxcsr);
}

void LpNormalizationFloat16_F16C(const std::uint16_t *input, std::uint16_t *output,
                                 std::size_t vectors, std::size_t width) {
  const unsigned int mxcsr = _mm_getcsr();
  _mm_setcsr(mxcsr | _MM_MASK_MASK);
  for (std::size_t vector = 0; vector < vectors; ++vector) {
    const std::size_t base = vector * width;
    __m256d sums[2] = {_mm256_setzero_pd(), _mm256_setzero_pd()};
    std::size_t i = 0;
    for (; i + 8 <= width; i += 8) {
      const __m256 value =
          _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(input + base + i)));
      const __m256d low = _mm256_cvtps_pd(_mm256_castps256_ps128(value));
      const __m256d high = _mm256_cvtps_pd(_mm256_extractf128_ps(value, 1));
      sums[0] = _mm256_add_pd(sums[0], _mm256_mul_pd(low, low));
      sums[1] = _mm256_add_pd(sums[1], _mm256_mul_pd(high, high));
    }
    double square_sum = HorizontalSum(_mm256_add_pd(sums[0], sums[1]));
    for (; i < width; ++i) {
      const double value = detail::Float16BitsToFloat(input[base + i]);
      square_sum += value * value;
    }
    const double norm = std::sqrt(square_sum);
    const float inverse = norm == 0.0 ? 0.0F : static_cast<float>(1.0 / norm);
    ApplyScaleBiasFloat16Masked(input + base, output + base, width, inverse, 0.0F);
  }
  _mm_setcsr(mxcsr);
}

} // namespace onnx_light_cpu
