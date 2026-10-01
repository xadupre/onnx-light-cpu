// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/execution.h"

#include <immintrin.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace onnx_light_cpu {
namespace {

float HorizontalMax(__m256 value) {
  __m128 reduced = _mm_max_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1));
  reduced = _mm_max_ps(reduced, _mm_movehl_ps(reduced, reduced));
  reduced = _mm_max_ss(reduced, _mm_shuffle_ps(reduced, reduced, 0x55));
  return _mm_cvtss_f32(reduced);
}

void QuantizeActivationRow(const float *input, std::uint8_t *quantized, float *scales,
                           std::size_t k) {
  const __m256 zero = _mm256_setzero_ps();
  const __m256 offset = _mm256_set1_ps(128.0f);
  const __m256i order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
  for (std::size_t block = 0; block < k / 32; ++block) {
    const float *source = input + block * 32;
    __m256 values[4];
    __m256 maximum = zero;
    for (std::size_t chunk = 0; chunk < 4; ++chunk) {
      values[chunk] = _mm256_loadu_ps(source + chunk * 8);
      maximum =
          _mm256_max_ps(maximum, _mm256_max_ps(values[chunk], _mm256_sub_ps(zero, values[chunk])));
    }
    const float max_abs = HorizontalMax(maximum);
    scales[block] = max_abs / 127.0f;
    const __m256 multiplier = _mm256_set1_ps(max_abs == 0.0f ? 0.0f : 127.0f / max_abs);
    __m256i integers[4];
    for (std::size_t chunk = 0; chunk < 4; ++chunk) {
      const __m256 rounded = _mm256_round_ps(_mm256_mul_ps(values[chunk], multiplier),
                                             _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
      integers[chunk] = _mm256_cvtps_epi32(_mm256_add_ps(rounded, offset));
    }
    const __m256i words0 = _mm256_packus_epi32(integers[0], integers[1]);
    const __m256i words1 = _mm256_packus_epi32(integers[2], integers[3]);
    const __m256i bytes = _mm256_permutevar8x32_epi32(_mm256_packus_epi16(words0, words1), order);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(quantized + block * 32), bytes);
  }
}

__m256i UnpackWeights(const std::uint8_t *packed) {
  const __m256i packed_words =
      _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i *>(packed)));
  const __m256i low = _mm256_and_si256(packed_words, _mm256_set1_epi16(0x0f));
  const __m256i high = _mm256_slli_epi16(_mm256_srli_epi16(packed_words, 4), 8);
  return _mm256_sub_epi8(_mm256_or_si256(low, high), _mm256_set1_epi8(8));
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

  const __m256i ones = _mm256_set1_epi16(1);
  const __m256i correction_factor = _mm256_set1_epi32(128);
  for (std::size_t group = 0; group < column_groups; ++group) {
    __m256 sums[Rows][2];
    for (std::size_t row = 0; row < Rows; ++row) {
      sums[row][0] = bias == nullptr ? _mm256_setzero_ps() : _mm256_loadu_ps(bias + group * 16);
      sums[row][1] = bias == nullptr ? _mm256_setzero_ps() : _mm256_loadu_ps(bias + group * 16 + 8);
    }
    for (std::size_t block = 0; block < blocks; ++block) {
      __m256i dots[Rows][2];
      for (std::size_t row = 0; row < Rows; ++row) {
        dots[row][0] = _mm256_setzero_si256();
        dots[row][1] = _mm256_setzero_si256();
      }
      const std::uint8_t *weights = packed_weights + (group * blocks + block) * 8 * 32;
      for (std::size_t quad = 0; quad < 8; ++quad) {
        const __m256i weights0 = UnpackWeights(weights + quad * 32);
        const __m256i weights1 = UnpackWeights(weights + quad * 32 + 16);
        for (std::size_t row = 0; row < Rows; ++row) {
          std::uint32_t activation_quad;
          std::memcpy(&activation_quad, scratch.quantized.data() + row * k + block * 32 + quad * 4,
                      sizeof(activation_quad));
          const __m256i activations = _mm256_set1_epi32(static_cast<int>(activation_quad));
          dots[row][0] = _mm256_add_epi32(
              dots[row][0], _mm256_madd_epi16(_mm256_maddubs_epi16(activations, weights0), ones));
          dots[row][1] = _mm256_add_epi32(
              dots[row][1], _mm256_madd_epi16(_mm256_maddubs_epi16(activations, weights1), ones));
        }
      }
      const std::int32_t *weight_sum = weight_sums + (group * blocks + block) * 16;
      const __m256i correction0 = _mm256_mullo_epi32(
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(weight_sum)), correction_factor);
      const __m256i correction1 = _mm256_mullo_epi32(
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(weight_sum + 8)), correction_factor);
      const float *weight_scale = block_scales + (group * blocks + block) * 16;
      const __m256 weight_scale0 = _mm256_loadu_ps(weight_scale);
      const __m256 weight_scale1 = _mm256_loadu_ps(weight_scale + 8);
      for (std::size_t row = 0; row < Rows; ++row) {
        const __m256 activation_scale =
            _mm256_set1_ps(scratch.activation_scales[row * blocks + block]);
        const __m256 scale0 = _mm256_mul_ps(activation_scale, weight_scale0);
        const __m256 scale1 = _mm256_mul_ps(activation_scale, weight_scale1);
        sums[row][0] = _mm256_add_ps(
            sums[row][0],
            _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(dots[row][0], correction0)), scale0));
        sums[row][1] = _mm256_add_ps(
            sums[row][1],
            _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(dots[row][1], correction1)), scale1));
      }
    }
    for (std::size_t row = 0; row < Rows; ++row) {
      _mm256_storeu_ps(y + (row_base + row) * n + group * 16, sums[row][0]);
      _mm256_storeu_ps(y + (row_base + row) * n + group * 16 + 8, sums[row][1]);
    }
  }
}

void ComputeRows(const float *a, const std::uint8_t *packed_weights,
                 const std::int32_t *weight_sums, const float *block_scales, const float *bias,
                 float *y, std::size_t row_begin, std::size_t row_end, std::size_t k,
                 std::size_t n) {
  constexpr std::size_t kRowTile = 2;
  std::size_t row = row_begin;
  for (; row + kRowTile <= row_end; row += kRowTile) {
    ComputeRowTile<kRowTile>(a, packed_weights, weight_sums, block_scales, bias, y, row, k, n);
  }
  for (; row < row_end; ++row) {
    ComputeRowTile<1>(a, packed_weights, weight_sums, block_scales, bias, y, row, k, n);
  }
}

} // namespace

void MatMulNBitsAccuracy4Float32Avx2(const float *a, const std::uint8_t *packed_weights,
                                     const std::int32_t *weight_sums, const float *block_scales,
                                     const float *bias, float *y, std::size_t rows, std::size_t k,
                                     std::size_t n, std::int64_t max_participants) {
  constexpr std::size_t kRowTile = 2;
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
