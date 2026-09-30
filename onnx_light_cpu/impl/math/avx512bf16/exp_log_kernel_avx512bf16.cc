// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/avx512/exp_kernel_avx512.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace onnx_light_cpu {
namespace {

__m512 LoadBFloat16(const std::uint16_t *input) {
  const __m256i values = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input));
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(values), 16));
}

void StoreBFloat16(__m512 values, std::uint16_t *output) {
  const __m256bh converted = _mm512_cvtneps_pbh(values);
  std::memcpy(output, &converted, sizeof(converted));
}

} // namespace

void ExpBFloat16_AVX512BF16(const std::uint16_t *input, std::uint16_t *output, std::size_t count) {
  std::size_t index = 0;
  for (; index + 16 <= count; index += 16) {
    StoreBFloat16(detail::ExpFloat32Vector_AVX512(LoadBFloat16(input + index)), output + index);
  }
  if (index < count) {
    alignas(32) std::uint16_t input_tail[16]{};
    alignas(32) std::uint16_t output_tail[16];
    const std::size_t tail_bytes = (count - index) * sizeof(std::uint16_t);
    std::memcpy(input_tail, input + index, tail_bytes);
    StoreBFloat16(detail::ExpFloat32Vector_AVX512(LoadBFloat16(input_tail)), output_tail);
    std::memcpy(output + index, output_tail, tail_bytes);
  }
}

} // namespace onnx_light_cpu
