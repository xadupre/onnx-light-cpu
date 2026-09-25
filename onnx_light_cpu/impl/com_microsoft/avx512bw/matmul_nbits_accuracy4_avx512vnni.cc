// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/execution.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <immintrin.h>

namespace onnx_light_cpu {
namespace {

void QuantizeActivationRow(const float *input, std::uint8_t *quantized, float *scales,
                           std::size_t k) {
  const __m512i offset = _mm512_set1_epi32(128);
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
    const __m512i integers0 = _mm512_add_epi32(
        _mm512_cvtps_epi32(_mm512_roundscale_ps(values0, _MM_FROUND_TO_NEAREST_INT)), offset);
    const __m512i integers1 = _mm512_add_epi32(
        _mm512_cvtps_epi32(_mm512_roundscale_ps(values1, _MM_FROUND_TO_NEAREST_INT)), offset);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(quantized + block * 32),
                     _mm512_cvtusepi32_epi8(integers0));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(quantized + block * 32 + 16),
                     _mm512_cvtusepi32_epi8(integers1));
  }
}

template <std::size_t Rows>
void ComputeRowTile(const float *a, const std::uint8_t *packed_weights,
                    const std::int32_t *weight_sums, const float *block_scales, const float *bias,
                    float *y, std::size_t row_base, std::size_t k, std::size_t n) {
  struct Scratch {
    std::vector<std::uint8_t> quantized;
    std::vector<float> activation_scales;
  };
  thread_local Scratch scratch;
  const std::size_t blocks = k / 32;
  const std::size_t column_groups = n / 16;
  scratch.quantized.resize(Rows * k);
  scratch.activation_scales.resize(Rows * blocks);
  for (std::size_t row = 0; row < Rows; ++row) {
    QuantizeActivationRow(a + (row_base + row) * k, scratch.quantized.data() + row * k,
                          scratch.activation_scales.data() + row * blocks, k);
  }
  const __m512i nibble_mask = _mm512_set1_epi16(0x0f);
  const __m512i eight = _mm512_set1_epi8(8);
  const __m512i correction_factor = _mm512_set1_epi32(128);
  for (std::size_t group = 0; group < column_groups; ++group) {
    __m512 sums[Rows];
    for (std::size_t row = 0; row < Rows; ++row) {
      sums[row] = bias == nullptr ? _mm512_setzero_ps() : _mm512_loadu_ps(bias + group * 16);
    }
    for (std::size_t block = 0; block < blocks; ++block) {
      __m512i dots[Rows];
      for (std::size_t row = 0; row < Rows; ++row) {
        dots[row] = _mm512_setzero_si512();
      }
      const std::uint8_t *weights = packed_weights + (group * blocks + block) * 8 * 32;
      for (std::size_t quad = 0; quad < 8; ++quad) {
        const __m256i packed =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(weights + quad * 32));
        const __m512i packed_words = _mm512_cvtepu8_epi16(packed);
        const __m512i unpacked = _mm512_sub_epi8(
            _mm512_or_si512(_mm512_and_si512(packed_words, nibble_mask),
                            _mm512_slli_epi16(_mm512_srli_epi16(packed_words, 4), 8)),
            eight);
        for (std::size_t row = 0; row < Rows; ++row) {
          std::uint32_t activation_quad;
          std::memcpy(&activation_quad, scratch.quantized.data() + row * k + block * 32 + quad * 4,
                      sizeof(activation_quad));
          dots[row] = _mm512_dpbusd_epi32(dots[row], _mm512_set1_epi32(activation_quad), unpacked);
        }
      }
      const __m512i correction = _mm512_mullo_epi32(
          _mm512_loadu_si512(weight_sums + (group * blocks + block) * 16), correction_factor);
      const __m512 weights_scale = _mm512_loadu_ps(block_scales + (group * blocks + block) * 16);
      for (std::size_t row = 0; row < Rows; ++row) {
        const __m512 scale = _mm512_mul_ps(
            _mm512_set1_ps(scratch.activation_scales[row * blocks + block]), weights_scale);
        const __m512 product = _mm512_cvtepi32_ps(_mm512_sub_epi32(dots[row], correction));
        sums[row] = _mm512_add_ps(_mm512_mul_ps(product, scale), sums[row]);
      }
    }
    for (std::size_t row = 0; row < Rows; ++row) {
      _mm512_storeu_ps(y + (row_base + row) * n + group * 16, sums[row]);
    }
  }
}

void ComputeRows(const float *a, const std::uint8_t *packed_weights,
                 const std::int32_t *weight_sums, const float *block_scales, const float *bias,
                 float *y, std::size_t row_begin, std::size_t row_end, std::size_t k,
                 std::size_t n) {
  constexpr std::size_t kRowTile = 8;
  std::size_t row = row_begin;
  for (; row + kRowTile <= row_end; row += kRowTile) {
    ComputeRowTile<kRowTile>(a, packed_weights, weight_sums, block_scales, bias, y, row, k, n);
  }
  for (; row < row_end; ++row) {
    ComputeRowTile<1>(a, packed_weights, weight_sums, block_scales, bias, y, row, k, n);
  }
}

} // namespace

void MatMulNBitsAccuracy4Float32Avx512Vnni(const float *a, const std::uint8_t *packed_weights,
                                           const std::int32_t *weight_sums,
                                           const float *block_scales, const float *bias, float *y,
                                           std::size_t rows, std::size_t k, std::size_t n,
                                           std::int64_t max_participants) {
  constexpr std::size_t kRowTile = 8;
  const std::size_t row_tiles = (rows + kRowTile - 1) / kRowTile;
  const ExecutionSchedule schedule{1, 1, max_participants};
  ExecuteRanges(static_cast<std::int64_t>(row_tiles), schedule,
                [&](std::int64_t begin, std::int64_t end) {
                  ComputeRows(a, packed_weights, weight_sums, block_scales, bias, y,
                              static_cast<std::size_t>(begin) * kRowTile,
                              std::min(rows, static_cast<std::size_t>(end) * kRowTile), k, n);
                });
}

} // namespace onnx_light_cpu
