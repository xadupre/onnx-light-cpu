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

} // namespace detail

} // namespace onnx_light_cpu
