// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/execution.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <immintrin.h>

namespace onnx_light_cpu {
namespace {

void QuantizeActivationRow(const float *input, std::int8_t *quantized, float *scales,
                           std::size_t k) {
  for (std::size_t block = 0; block < k / 32; ++block) {
    const float *source = input + block * 32;
    __m512 values0 = _mm512_loadu_ps(source);
    __m512 values1 = _mm512_loadu_ps(source + 16);
    const __m512 zero = _mm512_setzero_ps();
    const __m512 maximum = _mm512_max_ps(_mm512_max_ps(values0, _mm512_sub_ps(zero, values0)),
                                         _mm512_max_ps(values1, _mm512_sub_ps(zero, values1)));
    const float max_abs = _mm512_reduce_max_ps(maximum);
    scales[block] = max_abs / 127.0f;
    const __m512 multiplier = _mm512_set1_ps(max_abs == 0.0f ? 0.0f : 127.0f / max_abs);
    values0 = _mm512_mul_ps(values0, multiplier);
    values1 = _mm512_mul_ps(values1, multiplier);
    const __m512i integers0 =
        _mm512_cvtps_epi32(_mm512_roundscale_ps(values0, _MM_FROUND_TO_NEAREST_INT));
    const __m512i integers1 =
        _mm512_cvtps_epi32(_mm512_roundscale_ps(values1, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(quantized + block * 32),
                     _mm512_cvtepi32_epi8(integers0));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(quantized + block * 32 + 16),
                     _mm512_cvtepi32_epi8(integers1));
  }
}

void ComputeRows(const float *a, const std::int8_t *packed_weights, const float *block_scales,
                 const float *bias, float *y, std::size_t row_begin, std::size_t row_end,
                 std::size_t k, std::size_t n) {
  constexpr std::size_t kRowTile = 8;
  const std::size_t blocks = k / 32;
  const std::size_t column_groups = n / 16;
  std::vector<std::int8_t> quantized(kRowTile * k);
  std::vector<float> activation_scales(kRowTile * blocks);
  for (std::size_t row_base = row_begin; row_base < row_end; row_base += kRowTile) {
    const std::size_t row_count = std::min(kRowTile, row_end - row_base);
    for (std::size_t row = 0; row < row_count; ++row) {
      QuantizeActivationRow(a + (row_base + row) * k, quantized.data() + row * k,
                            activation_scales.data() + row * blocks, k);
    }
    for (std::size_t group = 0; group < column_groups; ++group) {
      __m512 sums[kRowTile];
      for (std::size_t row = 0; row < row_count; ++row) {
        sums[row] = bias == nullptr ? _mm512_setzero_ps() : _mm512_loadu_ps(bias + group * 16);
      }
      for (std::size_t block = 0; block < blocks; ++block) {
        __m512i dots[kRowTile];
        for (std::size_t row = 0; row < row_count; ++row) {
          dots[row] = _mm512_setzero_si512();
        }
        const std::int8_t *weights = packed_weights + (group * blocks + block) * 16 * 32;
        for (std::size_t pair = 0; pair < 16; ++pair) {
          const __m256i packed =
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(weights + pair * 32));
          const __m512i widened_weights = _mm512_cvtepi8_epi16(packed);
          for (std::size_t row = 0; row < row_count; ++row) {
            const std::int8_t *activation = quantized.data() + row * k + block * 32;
            const auto low =
                static_cast<std::uint16_t>(static_cast<std::int16_t>(activation[pair * 2]));
            const auto high =
                static_cast<std::uint16_t>(static_cast<std::int16_t>(activation[pair * 2 + 1]));
            const __m512i a_pair =
                _mm512_set1_epi32(static_cast<std::int32_t>(low | (std::uint32_t{high} << 16)));
            dots[row] = _mm512_dpwssd_epi32(dots[row], a_pair, widened_weights);
          }
        }
        const __m512 weights_scale = _mm512_loadu_ps(block_scales + (group * blocks + block) * 16);
        for (std::size_t row = 0; row < row_count; ++row) {
          const __m512 scale =
              _mm512_mul_ps(_mm512_set1_ps(activation_scales[row * blocks + block]), weights_scale);
          sums[row] = _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(dots[row]), scale), sums[row]);
        }
      }
      for (std::size_t row = 0; row < row_count; ++row) {
        _mm512_storeu_ps(y + (row_base + row) * n + group * 16, sums[row]);
      }
    }
  }
}

} // namespace

void MatMulNBitsAccuracy4Float32Avx512Vnni(const float *a, const std::int8_t *packed_weights,
                                           const float *block_scales, const float *bias, float *y,
                                           std::size_t rows, std::size_t k, std::size_t n,
                                           std::int64_t max_participants) {
  constexpr std::size_t kRowTile = 8;
  const std::size_t row_tiles = (rows + kRowTile - 1) / kRowTile;
  const ExecutionSchedule schedule{1, 1, max_participants};
  ExecuteRanges(static_cast<std::int64_t>(row_tiles), schedule,
                [&](std::int64_t begin, std::int64_t end) {
                  ComputeRows(a, packed_weights, block_scales, bias, y,
                              static_cast<std::size_t>(begin) * kRowTile,
                              std::min(rows, static_cast<std::size_t>(end) * kRowTile), k, n);
                });
}

} // namespace onnx_light_cpu
