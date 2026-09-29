// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/com_microsoft/skip_simplified_layer_normalization.h"

#include "onnx_light_cpu/impl/math/half_conversion.h"

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

void StoreHalf8(__m256 value, std::uint16_t *output) {
  _mm_storeu_si128(reinterpret_cast<__m128i *>(output),
                   _mm256_cvtps_ph(value, _MM_FROUND_TO_NEAREST_INT));
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

__m256 LoadHalf8(const std::uint16_t *input) {
  return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(input)));
}

} // namespace

void SkipSimplifiedLayerNormalizationFloat16_F16C(
    const std::uint16_t *input, const std::uint16_t *skip, const std::uint16_t *gamma,
    const std::uint16_t *bias, std::uint16_t *output, std::uint16_t *input_skip_bias_sum,
    float *mean, float *inv_std_var, std::size_t row_begin, std::size_t row_end, std::size_t width,
    std::size_t skip_rows, float epsilon) {
  const unsigned int mxcsr = _mm_getcsr();
  _mm_setcsr(mxcsr | _MM_MASK_MASK);
  for (std::size_t row = row_begin; row < row_end; ++row) {
    const std::size_t base = row * width;
    const std::size_t skip_base = (row % skip_rows) * width;
    __m256 squares[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                         _mm256_setzero_ps()};
    std::size_t i = 0;
    for (; i + 32 <= width; i += 32) {
      for (std::size_t lane = 0; lane < 4; ++lane) {
        const std::size_t offset = i + lane * 8;
        __m256 residual =
            _mm256_add_ps(LoadHalf8(input + base + offset), LoadHalf8(skip + skip_base + offset));
        if (bias != nullptr) {
          residual = _mm256_add_ps(residual, LoadHalf8(bias + offset));
        }
        squares[lane] = _mm256_add_ps(squares[lane], _mm256_mul_ps(residual, residual));
        if (input_skip_bias_sum != nullptr) {
          StoreHalf8(residual, input_skip_bias_sum + base + offset);
        }
      }
    }
    float square_sum = HorizontalSum(_mm256_add_ps(_mm256_add_ps(squares[0], squares[1]),
                                                   _mm256_add_ps(squares[2], squares[3])));
    for (; i < width; ++i) {
      float residual = detail::Float16BitsToFloat(input[base + i]) +
                       detail::Float16BitsToFloat(skip[skip_base + i]);
      if (bias != nullptr) {
        residual += detail::Float16BitsToFloat(bias[i]);
      }
      square_sum += residual * residual;
      if (input_skip_bias_sum != nullptr) {
        input_skip_bias_sum[base + i] = detail::FloatToFloat16Bits(residual);
      }
    }
    const float inverse = 1.0F / std::sqrt(square_sum / static_cast<float>(width) + epsilon);
    if (mean != nullptr) {
      mean[row] = 0.0F;
    }
    if (inv_std_var != nullptr) {
      inv_std_var[row] = inverse;
    }
    const __m256 inverse_vector = _mm256_set1_ps(inverse);
    i = 0;
    for (; i + 8 <= width; i += 8) {
      __m256 residual = _mm256_add_ps(LoadHalf8(input + base + i), LoadHalf8(skip + skip_base + i));
      if (bias != nullptr) {
        residual = _mm256_add_ps(residual, LoadHalf8(bias + i));
      }
      const __m256 result =
          _mm256_mul_ps(_mm256_mul_ps(residual, inverse_vector), LoadHalf8(gamma + i));
      StoreHalf8(result, output + base + i);
    }
    for (; i < width; ++i) {
      float residual = detail::Float16BitsToFloat(input[base + i]) +
                       detail::Float16BitsToFloat(skip[skip_base + i]);
      if (bias != nullptr) {
        residual += detail::Float16BitsToFloat(bias[i]);
      }
      output[base + i] =
          detail::FloatToFloat16Bits(residual * inverse * detail::Float16BitsToFloat(gamma[i]));
    }
  }
  _mm_setcsr(mxcsr);
}

} // namespace onnx_light_cpu
