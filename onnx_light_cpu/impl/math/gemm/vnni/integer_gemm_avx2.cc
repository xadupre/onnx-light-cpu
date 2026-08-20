// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Exact AVX2 UINT8 x INT8 dot-product fallback for CPUs without VNNI.

#include "onnx_light_cpu/impl/math/gemm/vnni/integer_gemm_vnni.h"

#include <bit>
#include <cstdint>

#include <immintrin.h>

namespace onnx_light_cpu {

namespace detail {

namespace {

__m256i Accumulate(__m256i accumulator, __m256i a_low, __m256i a_high, __m256i b, __m256i ones) {
  const __m256i products_low = _mm256_maddubs_epi16(a_low, b);
  const __m256i products_high = _mm256_maddubs_epi16(a_high, b);
  accumulator = _mm256_add_epi32(accumulator, _mm256_madd_epi16(products_low, ones));
  return _mm256_add_epi32(accumulator, _mm256_madd_epi16(products_high, ones));
}

std::uint32_t Reduce(__m256i accumulator) {
  __m128i total128 =
      _mm_add_epi32(_mm256_castsi256_si128(accumulator), _mm256_extracti128_si256(accumulator, 1));
  total128 = _mm_add_epi32(total128, _mm_shuffle_epi32(total128, _MM_SHUFFLE(1, 0, 3, 2)));
  total128 = _mm_add_epi32(total128, _mm_shuffle_epi32(total128, _MM_SHUFFLE(2, 3, 0, 1)));
  return static_cast<std::uint32_t>(_mm_cvtsi128_si32(total128));
}

} // namespace

std::int32_t IntegerDotU8S8Avx2(const std::uint8_t *ua, const std::int8_t *sb, std::int64_t depth) {
  __m256i accumulator = _mm256_setzero_si256();
  const __m256i low_mask = _mm256_set1_epi8(0x7f);
  const __m256i ones = _mm256_set1_epi16(1);
  std::int64_t index = 0;
  for (; index + 32 <= depth; index += 32) {
    const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ua + index));
    const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(sb + index));
    const __m256i a_low = _mm256_and_si256(a, low_mask);
    const __m256i a_high = _mm256_andnot_si256(low_mask, a);
    accumulator = Accumulate(accumulator, a_low, a_high, b, ones);
  }

  std::uint32_t total = Reduce(accumulator);
  for (; index < depth; ++index) {
    total += static_cast<std::uint32_t>(static_cast<std::int32_t>(ua[index]) *
                                        static_cast<std::int32_t>(sb[index]));
  }
  return std::bit_cast<std::int32_t>(total);
}

void IntegerMatMulU8S8Avx2(const std::uint8_t *a, const std::int8_t *b, std::int32_t *c,
                           std::int64_t rows, std::int64_t cols, std::int64_t depth) {
  constexpr std::int64_t row_block = 2;
  constexpr std::int64_t column_block = 2;
  const __m256i low_mask = _mm256_set1_epi8(0x7f);
  const __m256i ones = _mm256_set1_epi16(1);

  std::int64_t row = 0;
  for (; row + row_block <= rows; row += row_block) {
    std::int64_t column = 0;
    for (; column + column_block <= cols; column += column_block) {
      __m256i accumulators[row_block][column_block] = {};
      std::int64_t inner = 0;
      for (; inner + 32 <= depth; inner += 32) {
        __m256i a_low[row_block];
        __m256i a_high[row_block];
        for (std::int64_t r = 0; r < row_block; ++r) {
          const __m256i values =
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + (row + r) * depth + inner));
          a_low[r] = _mm256_and_si256(values, low_mask);
          a_high[r] = _mm256_andnot_si256(low_mask, values);
        }
        for (std::int64_t col = 0; col < column_block; ++col) {
          const __m256i values = _mm256_loadu_si256(
              reinterpret_cast<const __m256i *>(b + (column + col) * depth + inner));
          for (std::int64_t r = 0; r < row_block; ++r) {
            accumulators[r][col] =
                Accumulate(accumulators[r][col], a_low[r], a_high[r], values, ones);
          }
        }
      }
      for (std::int64_t r = 0; r < row_block; ++r) {
        for (std::int64_t col = 0; col < column_block; ++col) {
          std::uint32_t total = Reduce(accumulators[r][col]);
          for (std::int64_t tail = inner; tail < depth; ++tail) {
            total += static_cast<std::uint32_t>(
                static_cast<std::int32_t>(a[(row + r) * depth + tail]) *
                static_cast<std::int32_t>(b[(column + col) * depth + tail]));
          }
          c[(row + r) * cols + column + col] = std::bit_cast<std::int32_t>(total);
        }
      }
    }
    for (; column < cols; ++column) {
      for (std::int64_t r = 0; r < row_block; ++r) {
        c[(row + r) * cols + column] =
            IntegerDotU8S8Avx2(a + (row + r) * depth, b + column * depth, depth);
      }
    }
  }
  for (; row < rows; ++row) {
    for (std::int64_t column = 0; column < cols; ++column) {
      c[row * cols + column] = IntegerDotU8S8Avx2(a + row * depth, b + column * depth, depth);
    }
  }
}

} // namespace detail

} // namespace onnx_light_cpu
