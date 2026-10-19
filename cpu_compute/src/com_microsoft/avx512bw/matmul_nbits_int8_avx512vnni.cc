// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "cpu_compute/com_microsoft/matmul_nbits_int8_tile.h"

#include <immintrin.h>

#include <cstdint>
#include <cstring>

namespace cpu_compute {

static inline void MatMulNBitsInt8DotAvx512Vnni(const std::uint8_t *a, const std::uint8_t *b,
                                                std::int32_t *dots) {
  const __m512i offset = _mm512_set1_epi8(static_cast<char>(128));
  __m512i sum = _mm512_setzero_si512();
  for (int pair = 0; pair < 4; ++pair) {
    std::uint32_t first, second;
    std::memcpy(&first, a + pair * 8, sizeof(first));
    std::memcpy(&second, a + pair * 8 + 4, sizeof(second));
    const __m512i activation =
        _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi32(static_cast<int>(first))),
                           _mm256_set1_epi32(static_cast<int>(second)), 1);
    const __m512i weight = _mm512_xor_si512(_mm512_loadu_si512(b + pair * 64), offset);
    sum = _mm512_dpbusd_epi32(sum, activation, weight);
  }
  _mm256_storeu_si256(
      reinterpret_cast<__m256i *>(dots),
      _mm256_add_epi32(_mm512_castsi512_si256(sum), _mm512_extracti64x4_epi64(sum, 1)));
}

void MatMulNBitsInt8TileAvx512Vnni(const MatMulNBitsInt8Tile &tile) {
  ComputeMatMulNBitsInt8Tile<MatMulNBitsInt8DotAvx512Vnni>(tile);
}

} // namespace cpu_compute
