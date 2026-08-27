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

#include <bit>
#include <cstdint>

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

template <bool AIsSigned, bool BSigned>
void IntegerMatMulSkinnyMAvx512Impl(const std::uint8_t *a, const std::uint8_t *b, std::int32_t *c,
                                    std::int64_t cols, std::int64_t depth,
                                    std::int32_t a_zero_point, const std::int32_t *b_zero_point,
                                    std::int64_t b_zero_point_count) {
  ExecuteRanges(
      cols, static_cast<double>(depth) / 16.0, 16, [&](std::int64_t begin, std::int64_t end) {
        std::int64_t column = begin;
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
