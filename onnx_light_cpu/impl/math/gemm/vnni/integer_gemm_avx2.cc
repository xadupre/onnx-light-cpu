// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Exact AVX2 UINT8 x INT8 dot-product fallback for CPUs without VNNI.

#include "onnx_light_cpu/impl/math/gemm/vnni/integer_gemm_vnni.h"

#include "onnx_light_cpu/impl/execution.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <type_traits>

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

std::uint32_t Reduce128(__m128i accumulator) {
  accumulator = _mm_add_epi32(accumulator, _mm_shuffle_epi32(accumulator, _MM_SHUFFLE(1, 0, 3, 2)));
  accumulator = _mm_add_epi32(accumulator, _mm_shuffle_epi32(accumulator, _MM_SHUFFLE(2, 3, 0, 1)));
  return static_cast<std::uint32_t>(_mm_cvtsi128_si32(accumulator));
}

template <typename T>
void RequantizeInt32ImplAvx2(const std::int32_t *src, T *dst, std::int64_t count, float scale,
                             std::int32_t zero_point, double min_value, double max_value) {
  const __m256d scale_pd = _mm256_set1_pd(static_cast<double>(scale));
  const __m256d zp_pd = _mm256_set1_pd(static_cast<double>(zero_point));
  const __m256d min_pd = _mm256_set1_pd(min_value);
  const __m256d max_pd = _mm256_set1_pd(max_value);
  std::int64_t index = 0;
  for (; index + 8 <= count; index += 8) {
    const __m256i input = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + index));
    __m256d lo = _mm256_cvtepi32_pd(_mm256_castsi256_si128(input));
    __m256d hi = _mm256_cvtepi32_pd(_mm256_extracti128_si256(input, 1));
    lo = _mm256_add_pd(
        _mm256_round_pd(_mm256_mul_pd(lo, scale_pd), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
        zp_pd);
    hi = _mm256_add_pd(
        _mm256_round_pd(_mm256_mul_pd(hi, scale_pd), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
        zp_pd);
    lo = _mm256_max_pd(min_pd, _mm256_min_pd(max_pd, lo));
    hi = _mm256_max_pd(min_pd, _mm256_min_pd(max_pd, hi));
    const __m128i lo_i = _mm256_cvtpd_epi32(lo);
    const __m128i hi_i = _mm256_cvtpd_epi32(hi);
    if constexpr (std::is_same_v<T, std::int8_t>) {
      const __m128i packed16 = _mm_packs_epi32(lo_i, hi_i);
      const __m128i packed8 = _mm_packs_epi16(packed16, packed16);
      _mm_storel_epi64(reinterpret_cast<__m128i *>(dst + index), packed8);
    } else {
      static_assert(std::is_same_v<T, std::uint8_t>);
      const __m128i packed16 = _mm_packs_epi32(lo_i, hi_i);
      const __m128i packed8 = _mm_packus_epi16(packed16, _mm_setzero_si128());
      _mm_storel_epi64(reinterpret_cast<__m128i *>(dst + index), packed8);
    }
  }
  for (; index < count; ++index) {
    const double rounded =
        std::nearbyint(static_cast<double>(src[index]) * static_cast<double>(scale));
    const double shifted = rounded + static_cast<double>(zero_point);
    const double clamped = std::clamp(shifted, min_value, max_value);
    dst[index] = static_cast<T>(clamped);
  }
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

std::int32_t IntegerDotU8S8Avx2ShortTail(const std::uint8_t *ua, const std::int8_t *sb,
                                         std::int64_t depth) {
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

  // Compact GEMM shapes with a short reduction depth (e.g. the transformer
  // ``direct`` case at depth 16) never reach the 32-byte main loop above and
  // used to fall through to a fully scalar tail, which measured far below
  // the vectorized cases. Cover the common 16..31 remainder with one 128-bit
  // SSSE3/SSE4.1 step so at least half of a short depth is still vectorized,
  // then finish any remaining 1..15 elements with the scalar loop below.
  // This variant is only selected by the dispatcher when ``depth % 32 != 0``
  // (see ``IntegerMatMul2D``/``IntegerMatMul4Bit2D``), so the depth-multiple
  // -of-32 shapes that dominate the square/large-K/skinny cases keep calling
  // the plain ``IntegerDotU8S8Avx2`` above unchanged, instead of paying for
  // this extra dead branch on every one of their (row, column) pairs.
  if (index + 16 <= depth) {
    const __m128i low_mask128 = _mm_set1_epi8(0x7f);
    const __m128i ones128 = _mm_set1_epi16(1);
    const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(ua + index));
    const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(sb + index));
    const __m128i a_low = _mm_and_si128(a, low_mask128);
    const __m128i a_high = _mm_andnot_si128(low_mask128, a);
    const __m128i products_low = _mm_maddubs_epi16(a_low, b);
    const __m128i products_high = _mm_maddubs_epi16(a_high, b);
    __m128i partial = _mm_madd_epi16(products_low, ones128);
    partial = _mm_add_epi32(partial, _mm_madd_epi16(products_high, ones128));
    total += Reduce128(partial);
    index += 16;
  }

  for (; index < depth; ++index) {
    total += static_cast<std::uint32_t>(static_cast<std::int32_t>(ua[index]) *
                                        static_cast<std::int32_t>(sb[index]));
  }
  return std::bit_cast<std::int32_t>(total);
}

namespace {

template <bool AIsSigned, bool BSigned>
void IntegerMatMulSkinnyMAvx2Impl(const std::uint8_t *a, const std::uint8_t *b, std::int32_t *c,
                                  std::int64_t cols, std::int64_t depth, std::int32_t a_zero_point,
                                  const std::int32_t *b_zero_point,
                                  std::int64_t b_zero_point_count) {
  ExecuteRanges(
      cols, static_cast<double>(depth) / 16.0, 8, [&](std::int64_t begin, std::int64_t end) {
        std::int64_t column = begin;
        for (; column + 8 <= end; column += 8) {
          __m256i accumulator = _mm256_setzero_si256();
          const __m256i bz =
              b_zero_point_count == 1
                  ? _mm256_set1_epi32(b_zero_point[0])
                  : _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b_zero_point + column));
          for (std::int64_t inner = 0; inner < depth; ++inner) {
            const std::int32_t av = AIsSigned ? static_cast<std::int8_t>(a[inner]) : a[inner];
            const __m128i packed_b =
                _mm_loadl_epi64(reinterpret_cast<const __m128i *>(b + inner * cols + column));
            const __m256i bv =
                BSigned ? _mm256_cvtepi8_epi32(packed_b) : _mm256_cvtepu8_epi32(packed_b);
            const __m256i adjusted_b = _mm256_sub_epi32(bv, bz);
            const __m256i adjusted_a = _mm256_set1_epi32(av - a_zero_point);
            accumulator = _mm256_add_epi32(accumulator, _mm256_mullo_epi32(adjusted_a, adjusted_b));
          }
          _mm256_storeu_si256(reinterpret_cast<__m256i *>(c + column), accumulator);
        }
        for (; column < end; ++column) {
          std::uint32_t accumulator = 0;
          const std::int32_t bz = b_zero_point_count == 1 ? b_zero_point[0] : b_zero_point[column];
          for (std::int64_t inner = 0; inner < depth; ++inner) {
            const std::int32_t av = AIsSigned ? static_cast<std::int8_t>(a[inner]) : a[inner];
            const std::uint8_t raw_b = b[inner * cols + column];
            const std::int32_t bv = BSigned ? static_cast<std::int8_t>(raw_b) : raw_b;
            accumulator += static_cast<std::uint32_t>((av - a_zero_point) * (bv - bz));
          }
          c[column] = std::bit_cast<std::int32_t>(accumulator);
        }
      });
}

} // namespace

void IntegerMatMulSkinnyMAvx2(const std::uint8_t *a, bool a_signed, const std::uint8_t *b,
                              bool b_signed, std::int32_t *c, std::int64_t cols, std::int64_t depth,
                              std::int32_t a_zero_point, const std::int32_t *b_zero_point,
                              std::int64_t b_zero_point_count) {
  if (a_signed) {
    if (b_signed) {
      IntegerMatMulSkinnyMAvx2Impl<true, true>(a, b, c, cols, depth, a_zero_point, b_zero_point,
                                               b_zero_point_count);
    } else {
      IntegerMatMulSkinnyMAvx2Impl<true, false>(a, b, c, cols, depth, a_zero_point, b_zero_point,
                                                b_zero_point_count);
    }
  } else if (b_signed) {
    IntegerMatMulSkinnyMAvx2Impl<false, true>(a, b, c, cols, depth, a_zero_point, b_zero_point,
                                              b_zero_point_count);
  } else {
    IntegerMatMulSkinnyMAvx2Impl<false, false>(a, b, c, cols, depth, a_zero_point, b_zero_point,
                                               b_zero_point_count);
  }
}

namespace {

constexpr std::int64_t kU8S8RowBlock = 2;
constexpr std::int64_t kU8S8ColumnBlock = 2;

// Processes the disjoint row range ``[row_begin, row_end)`` of the bulk
// depth>=32 microkernel. This is a plain (non-lambda) function so the
// single-participant fast path in ``IntegerMatMulU8S8Avx2`` below can call it
// directly with the exact same codegen as the original unparallelized loop,
// instead of indirecting through ``ExecuteRanges``'s generic closure
// plumbing, which measurably regressed this SIMD-heavy microkernel even for
// a single resulting block.
void ProcessU8S8RowRange(const std::uint8_t *a, const std::int8_t *b, std::int32_t *c,
                         std::int64_t row_begin, std::int64_t row_end, std::int64_t cols,
                         std::int64_t depth) {
  const __m256i low_mask = _mm256_set1_epi8(0x7f);
  const __m256i ones = _mm256_set1_epi16(1);
  std::int64_t row = row_begin;
  for (; row + kU8S8RowBlock <= row_end; row += kU8S8RowBlock) {
    std::int64_t column = 0;
    for (; column + kU8S8ColumnBlock <= cols; column += kU8S8ColumnBlock) {
      __m256i accumulators[kU8S8RowBlock][kU8S8ColumnBlock] = {};
      std::int64_t inner = 0;
      for (; inner + 32 <= depth; inner += 32) {
        __m256i a_low[kU8S8RowBlock];
        __m256i a_high[kU8S8RowBlock];
        for (std::int64_t r = 0; r < kU8S8RowBlock; ++r) {
          const __m256i values =
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + (row + r) * depth + inner));
          a_low[r] = _mm256_and_si256(values, low_mask);
          a_high[r] = _mm256_andnot_si256(low_mask, values);
        }
        for (std::int64_t col = 0; col < kU8S8ColumnBlock; ++col) {
          const __m256i values = _mm256_loadu_si256(
              reinterpret_cast<const __m256i *>(b + (column + col) * depth + inner));
          for (std::int64_t r = 0; r < kU8S8RowBlock; ++r) {
            accumulators[r][col] =
                Accumulate(accumulators[r][col], a_low[r], a_high[r], values, ones);
          }
        }
      }
      for (std::int64_t r = 0; r < kU8S8RowBlock; ++r) {
        for (std::int64_t col = 0; col < kU8S8ColumnBlock; ++col) {
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
      for (std::int64_t r = 0; r < kU8S8RowBlock; ++r) {
        c[(row + r) * cols + column] =
            IntegerDotU8S8Avx2(a + (row + r) * depth, b + column * depth, depth);
      }
    }
  }
  for (; row < row_end; ++row) {
    for (std::int64_t column = 0; column < cols; ++column) {
      c[row * cols + column] = IntegerDotU8S8Avx2(a + row * depth, b + column * depth, depth);
    }
  }
}

} // namespace

void IntegerMatMulU8S8Avx2(const std::uint8_t *a, const std::int8_t *b, std::int32_t *c,
                           std::int64_t rows, std::int64_t cols, std::int64_t depth) {
  // Bypass the executor/lambda plumbing entirely when no parallel executor
  // is installed: ``ProcessU8S8RowRange`` is then called exactly once for
  // the whole row range, identical in every respect to the original
  // unparallelized kernel, so the common single-thread case never regresses.
  // Only route through ``ExecuteRanges`` when there is a real executor to
  // hand row ranges to (e.g. the physical-core policy), which is where the
  // scaling benefit actually applies. Each worker owns a disjoint,
  // ``kU8S8RowBlock``-aligned row range (the only range that may fall short
  // of a full pair is the very last one handed out, which the per-range
  // leftover-row loop already covers).
  if (CurrentExecutionExecutor() == nullptr) {
    ProcessU8S8RowRange(a, b, c, 0, rows, cols, depth);
    return;
  }
  ExecuteRanges(rows, static_cast<double>(cols) * static_cast<double>(depth), kU8S8RowBlock,
                [a, b, c, cols, depth](std::int64_t row_begin, std::int64_t row_end) {
                  ProcessU8S8RowRange(a, b, c, row_begin, row_end, cols, depth);
                });
}

void RequantizeInt32ToInt8Avx2(const std::int32_t *src, std::int8_t *dst, std::int64_t count,
                               float scale, std::int32_t zero_point) {
  RequantizeInt32ImplAvx2(src, dst, count, scale, zero_point, -128.0, 127.0);
}

void RequantizeInt32ToUint8Avx2(const std::int32_t *src, std::uint8_t *dst, std::int64_t count,
                                float scale, std::int32_t zero_point) {
  RequantizeInt32ImplAvx2(src, dst, count, scale, zero_point, 0.0, 255.0);
}

} // namespace detail

} // namespace onnx_light_cpu
