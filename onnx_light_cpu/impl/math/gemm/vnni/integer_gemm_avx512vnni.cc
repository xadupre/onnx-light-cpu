// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Native AVX-512 VNNI INT8 dot-product (Roadmap PR09.2). This translation unit
// is compiled with an extra -mavx512f -mavx512vnni (see the per-file
// COMPILE_OPTIONS override in CMakeLists.txt) even though the rest of
// onnx_light_cpu keeps the project's baseline SIMD flags, so a single binary
// can carry this native kernel and still run on CPUs that lack the ISA: the
// integer matrix-multiplication driver in integer_gemm_vnni.cc only dispatches
// here when ``CpuSupportsAvx512Vnni()`` reports the instruction set at runtime.
//
// ``vpdpbusd`` reduces four consecutive UINT8 x INT8 products into each of the
// sixteen INT32 lanes, accumulating in INT32 (which naturally wraps modulo
// 2^32, matching the scalar sibling). Depths that are not a multiple of the
// 64-byte vector finish through the same scalar tail.

#include "onnx_light_cpu/impl/math/gemm/vnni/integer_gemm_vnni.h"

#include "onnx_light_cpu/impl/execution.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>

#include <immintrin.h>

namespace onnx_light_cpu {

namespace detail {

std::int32_t IntegerDotU8S8Avx512Vnni(const std::uint8_t *ua, const std::int8_t *sb,
                                      std::int64_t depth) {
  __m512i accumulator = _mm512_setzero_si512();
  std::int64_t i = 0;
  for (; i + 64 <= depth; i += 64) {
    const __m512i va = _mm512_loadu_si512(reinterpret_cast<const void *>(ua + i));
    const __m512i vb = _mm512_loadu_si512(reinterpret_cast<const void *>(sb + i));
    accumulator = _mm512_dpbusd_epi32(accumulator, va, vb);
  }
  std::uint32_t total = std::bit_cast<std::uint32_t>(_mm512_reduce_add_epi32(accumulator));
  for (; i < depth; ++i) {
    total += static_cast<std::uint32_t>(static_cast<std::int32_t>(ua[i]) *
                                        static_cast<std::int32_t>(sb[i]));
  }
  return std::bit_cast<std::int32_t>(total);
}

#ifdef ONNX_LIGHT_CPU_HAVE_AVX512BW
namespace {

void StoreSkinnyTile(__m512i x0, __m512i x1, __m512i x2, __m512i x3, std::int32_t *c,
                     const std::int32_t *b_zero_point, std::int64_t b_zero_point_count,
                     __m512i a_sum, __m512i b_offset, bool accumulate) {
  // Undo the lane-local interleave: each accumulator holds four columns per 128-bit lane.
  const __m512i lo01 = _mm512_shuffle_i32x4(x0, x1, 0x44);
  const __m512i lo23 = _mm512_shuffle_i32x4(x2, x3, 0x44);
  const __m512i hi01 = _mm512_shuffle_i32x4(x0, x1, 0xee);
  const __m512i hi23 = _mm512_shuffle_i32x4(x2, x3, 0xee);
  const __m512i values[] = {
      _mm512_shuffle_i32x4(lo01, lo23, 0x88), _mm512_shuffle_i32x4(lo01, lo23, 0xdd),
      _mm512_shuffle_i32x4(hi01, hi23, 0x88), _mm512_shuffle_i32x4(hi01, hi23, 0xdd)};
  for (int i = 0; i < 4; ++i) {
    const __m512i bz = b_zero_point_count == 1 ? _mm512_set1_epi32(b_zero_point[0])
                                               : _mm512_loadu_si512(b_zero_point + 16 * i);
    __m512i value =
        _mm512_add_epi32(values[i], _mm512_mullo_epi32(a_sum, _mm512_sub_epi32(b_offset, bz)));
    if (accumulate) {
      value = _mm512_add_epi32(value, _mm512_loadu_si512(c + 16 * i));
    }
    _mm512_storeu_si512(c + 16 * i, value);
  }
}

template <bool AIsSigned, bool BSigned>
void IntegerMatMulSkinnyMAvx512Impl(const std::uint8_t *a, const std::uint8_t *b, std::int32_t *c,
                                    std::int64_t cols, std::int64_t depth,
                                    std::int32_t a_zero_point, const std::int32_t *b_zero_point,
                                    std::int64_t b_zero_point_count) {
  const std::int64_t vector_depth = depth - depth % 4;
  const __m512i a_offset = _mm512_set1_epi32((AIsSigned ? 128 : 0) + a_zero_point);
  const __m512i b_offset = _mm512_set1_epi32(BSigned ? 0 : 128);
  const __m512i ones = _mm512_set1_epi8(1);
  const __m512i sign_bit = _mm512_set1_epi8(static_cast<char>(0x80));
  ExecuteRanges(
      cols, static_cast<double>(depth) / 16.0, 64, [&](std::int64_t begin, std::int64_t end) {
        std::int64_t column = begin;
        const std::int64_t tile_end = begin + (end - begin) / 64 * 64;
        // Keep a small band of B rows hot while streaming across columns, rather than revisiting
        // every page of B for each output tile. Only the partial outputs are stored between bands.
        for (std::int64_t depth_begin = 0;
             tile_end > begin && (depth_begin < vector_depth || depth_begin == 0);
             depth_begin += 64) {
          const std::int64_t depth_end = std::min(vector_depth, depth_begin + 64);
          std::uint32_t a_sum = 0;
          for (std::int64_t inner = depth_begin; inner < depth_end; ++inner) {
            const std::int32_t av = AIsSigned ? static_cast<std::int8_t>(a[inner]) : a[inner];
            a_sum += static_cast<std::uint32_t>(av - a_zero_point);
          }
          const __m512i va_sum = _mm512_set1_epi32(std::bit_cast<std::int32_t>(a_sum));
          for (column = begin; column < tile_end; column += 64) {
            __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512();
            __m512i acc2 = _mm512_setzero_si512(), acc3 = _mm512_setzero_si512();
            __m512i sum0 = _mm512_setzero_si512(), sum1 = _mm512_setzero_si512();
            __m512i sum2 = _mm512_setzero_si512(), sum3 = _mm512_setzero_si512();
            for (std::int64_t inner = depth_begin; inner < depth_end; inner += 4) {
              std::uint32_t raw_a;
              std::memcpy(&raw_a, a + inner, sizeof(raw_a));
              if constexpr (AIsSigned) {
                raw_a ^= 0x80808080u;
              }
              const __m512i av = _mm512_set1_epi32(std::bit_cast<std::int32_t>(raw_a));
              const std::uint8_t *bp = b + inner * cols + column;
              __m512i b0 = _mm512_loadu_si512(bp);
              __m512i b1 = _mm512_loadu_si512(bp + cols);
              __m512i b2 = _mm512_loadu_si512(bp + 2 * cols);
              __m512i b3 = _mm512_loadu_si512(bp + 3 * cols);
              if constexpr (!BSigned) {
                b0 = _mm512_xor_si512(b0, sign_bit);
                b1 = _mm512_xor_si512(b1, sign_bit);
                b2 = _mm512_xor_si512(b2, sign_bit);
                b3 = _mm512_xor_si512(b3, sign_bit);
              }
              const __m512i lo01 = _mm512_unpacklo_epi8(b0, b1);
              const __m512i hi01 = _mm512_unpackhi_epi8(b0, b1);
              const __m512i lo23 = _mm512_unpacklo_epi8(b2, b3);
              const __m512i hi23 = _mm512_unpackhi_epi8(b2, b3);
              b0 = _mm512_unpacklo_epi16(lo01, lo23);
              b1 = _mm512_unpackhi_epi16(lo01, lo23);
              b2 = _mm512_unpacklo_epi16(hi01, hi23);
              b3 = _mm512_unpackhi_epi16(hi01, hi23);
              acc0 = _mm512_dpbusd_epi32(acc0, av, b0);
              acc1 = _mm512_dpbusd_epi32(acc1, av, b1);
              acc2 = _mm512_dpbusd_epi32(acc2, av, b2);
              acc3 = _mm512_dpbusd_epi32(acc3, av, b3);
              sum0 = _mm512_dpbusd_epi32(sum0, ones, b0);
              sum1 = _mm512_dpbusd_epi32(sum1, ones, b1);
              sum2 = _mm512_dpbusd_epi32(sum2, ones, b2);
              sum3 = _mm512_dpbusd_epi32(sum3, ones, b3);
            }
            // uA = A + oa, sB = B - ob:
            // (A-az)(B-bz) = uA*sB - (oa+az)*sB + (ob-bz)*(A-az).
            StoreSkinnyTile(_mm512_sub_epi32(acc0, _mm512_mullo_epi32(a_offset, sum0)),
                            _mm512_sub_epi32(acc1, _mm512_mullo_epi32(a_offset, sum1)),
                            _mm512_sub_epi32(acc2, _mm512_mullo_epi32(a_offset, sum2)),
                            _mm512_sub_epi32(acc3, _mm512_mullo_epi32(a_offset, sum3)), c + column,
                            b_zero_point + (b_zero_point_count == 1 ? 0 : column),
                            b_zero_point_count, va_sum, b_offset, depth_begin != 0);
            for (std::int64_t inner = vector_depth; depth_end == vector_depth && inner < depth;
                 ++inner) {
              const std::int32_t av = AIsSigned ? static_cast<std::int8_t>(a[inner]) : a[inner];
              for (int offset = 0; offset < 64; offset += 16) {
                const __m512i bz = b_zero_point_count == 1
                                       ? _mm512_set1_epi32(b_zero_point[0])
                                       : _mm512_loadu_si512(b_zero_point + column + offset);
                const __m128i packed_b = _mm_loadu_si128(
                    reinterpret_cast<const __m128i *>(b + inner * cols + column + offset));
                const __m512i bv =
                    BSigned ? _mm512_cvtepi8_epi32(packed_b) : _mm512_cvtepu8_epi32(packed_b);
                const __m512i accumulator = _mm512_loadu_si512(c + column + offset);
                _mm512_storeu_si512(
                    c + column + offset,
                    _mm512_add_epi32(accumulator,
                                     _mm512_mullo_epi32(_mm512_set1_epi32(av - a_zero_point),
                                                        _mm512_sub_epi32(bv, bz))));
              }
            }
          }
        }
        for (; column + 16 <= end; column += 16) {
          __m512i accumulator = _mm512_setzero_si512();
          const __m512i bz =
              b_zero_point_count == 1
                  ? _mm512_set1_epi32(b_zero_point[0])
                  : _mm512_loadu_si512(reinterpret_cast<const void *>(b_zero_point + column));
          for (std::int64_t inner = 0; inner < depth; ++inner) {
            const std::int32_t av = AIsSigned ? static_cast<std::int8_t>(a[inner]) : a[inner];
            const __m128i packed_b =
                _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + inner * cols + column));
            const __m512i bv =
                BSigned ? _mm512_cvtepi8_epi32(packed_b) : _mm512_cvtepu8_epi32(packed_b);
            accumulator = _mm512_add_epi32(
                accumulator,
                _mm512_mullo_epi32(_mm512_set1_epi32(av - a_zero_point), _mm512_sub_epi32(bv, bz)));
          }
          _mm512_storeu_si512(reinterpret_cast<void *>(c + column), accumulator);
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

void IntegerMatMulSkinnyMAvx512(const std::uint8_t *a, bool a_signed, const std::uint8_t *b,
                                bool b_signed, std::int32_t *c, std::int64_t cols,
                                std::int64_t depth, std::int32_t a_zero_point,
                                const std::int32_t *b_zero_point, std::int64_t b_zero_point_count) {
  if (a_signed) {
    if (b_signed) {
      IntegerMatMulSkinnyMAvx512Impl<true, true>(a, b, c, cols, depth, a_zero_point, b_zero_point,
                                                 b_zero_point_count);
    } else {
      IntegerMatMulSkinnyMAvx512Impl<true, false>(a, b, c, cols, depth, a_zero_point, b_zero_point,
                                                  b_zero_point_count);
    }
  } else if (b_signed) {
    IntegerMatMulSkinnyMAvx512Impl<false, true>(a, b, c, cols, depth, a_zero_point, b_zero_point,
                                                b_zero_point_count);
  } else {
    IntegerMatMulSkinnyMAvx512Impl<false, false>(a, b, c, cols, depth, a_zero_point, b_zero_point,
                                                 b_zero_point_count);
  }
}
#endif

} // namespace detail

} // namespace onnx_light_cpu
