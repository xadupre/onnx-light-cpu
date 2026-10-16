// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/com_microsoft/matmul_nbits_int8_tile.h"

#include <immintrin.h>

#include <cstdint>
#include <cstring>
#include <limits>

namespace onnx_light_cpu {

bool QuantizeMatMulNBitsInt8RowAvx2(const float *input, std::uint8_t *quantized, double *scales,
                                    std::size_t k) {
  const __m256 sign = _mm256_set1_ps(-0.0f);
  const __m256 finite_limit = _mm256_set1_ps(std::numeric_limits<float>::max());
  const __m256i offset = _mm256_set1_epi32(128);
  for (std::size_t block = 0; block < k / 32; ++block) {
    __m256 values[4];
    __m256 maximum = _mm256_setzero_ps();
    for (std::size_t part = 0; part < 4; ++part) {
      values[part] = _mm256_loadu_ps(input + block * 32 + part * 8);
      const __m256 absolute = _mm256_andnot_ps(sign, values[part]);
      if (_mm256_movemask_ps(_mm256_cmp_ps(absolute, finite_limit, _CMP_LE_OQ)) != 255) {
        return false;
      }
      maximum = _mm256_max_ps(maximum, absolute);
    }
    __m128 reduced = _mm_max_ps(_mm256_castps256_ps128(maximum), _mm256_extractf128_ps(maximum, 1));
    reduced = _mm_max_ps(reduced, _mm_movehl_ps(reduced, reduced));
    reduced = _mm_max_ss(reduced, _mm_shuffle_ps(reduced, reduced, 1));
    const float max_abs = _mm_cvtss_f32(reduced);
    if (max_abs != 0.0f && max_abs < 1e-36f) {
      return false;
    }
    scales[block] = static_cast<double>(max_abs) / 127.0;
    const __m256 multiplier = _mm256_set1_ps(max_abs == 0.0f ? 0.0f : 127.0f / max_abs);
    for (std::size_t part = 0; part < 4; ++part) {
      const __m256i integers =
          _mm256_add_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(values[part], multiplier)), offset);
      const __m128i shorts =
          _mm_packs_epi32(_mm256_castsi256_si128(integers), _mm256_extracti128_si256(integers, 1));
      _mm_storel_epi64(reinterpret_cast<__m128i *>(quantized + block * 32 + part * 8),
                       _mm_packus_epi16(shorts, shorts));
    }
  }
  return true;
}

static inline void MatMulNBitsInt8DotAvx2(const std::uint8_t *a, const std::uint8_t *b,
                                          std::int32_t *dots) {
  const __m256i ones = _mm256_set1_epi16(1);
  const __m256i offset = _mm256_set1_epi8(static_cast<char>(128));
  const __m256i low_mask = _mm256_set1_epi8(127);
  const __m256i high_mask = _mm256_set1_epi8(1);
  __m256i sum = _mm256_setzero_si256();
  for (int quad = 0; quad < 8; ++quad) {
    std::uint32_t value;
    std::memcpy(&value, a + quad * 4, sizeof(value));
    const __m256i activation = _mm256_set1_epi32(static_cast<int>(value));
    const __m256i weight = _mm256_xor_si256(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + quad * 32)), offset);
    // Split unsigned activations before maddubs so full-range INT8 products
    // cannot saturate its signed 16-bit pairwise sums.
    const __m256i low = _mm256_and_si256(activation, low_mask);
    const __m256i high = _mm256_and_si256(_mm256_srli_epi16(activation, 7), high_mask);
    const __m256i low_dot = _mm256_madd_epi16(_mm256_maddubs_epi16(low, weight), ones);
    const __m256i high_dot = _mm256_madd_epi16(_mm256_maddubs_epi16(high, weight), ones);
    sum = _mm256_add_epi32(sum, _mm256_add_epi32(low_dot, _mm256_slli_epi32(high_dot, 7)));
  }
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(dots), sum);
}

void MatMulNBitsInt8TileAvx2(const MatMulNBitsInt8Tile &tile) {
  ComputeMatMulNBitsInt8Tile<MatMulNBitsInt8DotAvx2>(tile);
}

} // namespace onnx_light_cpu
