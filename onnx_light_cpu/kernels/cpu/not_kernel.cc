// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/cpu_kernels.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define ONNX_LIGHT_CPU_X86 1
#include <immintrin.h>
#else
#define ONNX_LIGHT_CPU_X86 0
#endif

namespace onnx_light_cpu {

// ---------------------------------------------------------------------------
// NotBool implementations
// ---------------------------------------------------------------------------
//
// The elementwise logical negation of an ONNX ``bool`` tensor is a pure byte
// operation: every zero byte becomes ``1`` and every non-zero byte becomes
// ``0`` (matching ``numpy.logical_not``). Comparing each byte against zero and
// keeping the low bit produces a canonical ``0``/``1`` result regardless of the
// input byte value.

namespace {

void NotBool_Scalar(const uint8_t *input, uint8_t *output, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    output[i] = static_cast<uint8_t>(input[i] == 0 ? 1 : 0);
  }
}

#if ONNX_LIGHT_CPU_X86

void NotBool_SSE2(const uint8_t *input, uint8_t *output, std::size_t count) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i one = _mm_set1_epi8(1);
  std::size_t i = 0;
  const std::size_t stride = 16;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + i));
    // 0xFF where the byte equals zero, 0x00 otherwise; keep the low bit -> 1/0.
    __m128i eq = _mm_cmpeq_epi8(v, zero);
    v = _mm_and_si128(eq, one);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output + i), v);
  }
  for (; i < count; ++i) {
    output[i] = static_cast<uint8_t>(input[i] == 0 ? 1 : 0);
  }
}

void NotBool_AVX2(const uint8_t *input, uint8_t *output, std::size_t count) {
  const __m256i zero = _mm256_setzero_si256();
  const __m256i one = _mm256_set1_epi8(1);
  std::size_t i = 0;
  const std::size_t stride = 32;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + i));
    __m256i eq = _mm256_cmpeq_epi8(v, zero);
    v = _mm256_and_si256(eq, one);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + i), v);
  }
  for (; i < count; ++i) {
    output[i] = static_cast<uint8_t>(input[i] == 0 ? 1 : 0);
  }
}

#ifdef __AVX512BW__
void NotBool_AVX512(const uint8_t *input, uint8_t *output, std::size_t count) {
  const __m512i zero = _mm512_setzero_si512();
  std::size_t i = 0;
  const std::size_t stride = 64;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    __m512i v = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(input + i));
    // Bitmask with a set bit for every byte equal to zero; expand to 1/0 bytes.
    __mmask64 m = _mm512_cmpeq_epi8_mask(v, zero);
    v = _mm512_maskz_set1_epi8(m, 1);
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(output + i), v);
  }
  for (; i < count; ++i) {
    output[i] = static_cast<uint8_t>(input[i] == 0 ? 1 : 0);
  }
}
#endif // __AVX512BW__

#endif // ONNX_LIGHT_CPU_X86

} // namespace

void NotBool(const uint8_t *input, uint8_t *output, std::size_t count) {
  if (count == 0)
    return;
#if ONNX_LIGHT_CPU_X86
  static const SimdLevel level = DetectSimdLevel();
#ifdef __AVX512BW__
  if (level >= SimdLevel::kAVX512) {
    NotBool_AVX512(input, output, count);
    return;
  }
#endif
  if (level >= SimdLevel::kAVX2) {
    NotBool_AVX2(input, output, count);
    return;
  }
  if (level >= SimdLevel::kSSE2) {
    NotBool_SSE2(input, output, count);
    return;
  }
#endif
  NotBool_Scalar(input, output, count);
}

} // namespace onnx_light_cpu
