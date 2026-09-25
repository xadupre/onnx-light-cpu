// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/tensor/cast_simd.h"

#include <cstring>
#include <immintrin.h>

namespace onnx_light_cpu::detail {

std::size_t CastInt64ToFloat32_AVX512DQ(const std::uint8_t *src, std::uint8_t *dst,
                                        std::size_t count) {
  std::size_t i = 0;
  for (; count - i >= 8; i += 8) {
    const __m512i integers =
        _mm512_loadu_si512(reinterpret_cast<const void *>(src + i * sizeof(std::int64_t)));
    const __m256 converted = _mm512_cvtepi64_ps(integers);
    std::memcpy(dst + i * sizeof(float), &converted, sizeof(converted));
  }
  return i;
}

} // namespace onnx_light_cpu::detail
