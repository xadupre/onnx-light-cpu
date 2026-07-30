// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/cpu_kernels.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define ONNX_LIGHT_CPU_X86 1
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#else
#define ONNX_LIGHT_CPU_X86 0
#endif

namespace onnx_light_cpu {

// ---------------------------------------------------------------------------
// CPUID-based runtime SIMD detection (x86 only)
// ---------------------------------------------------------------------------

#if ONNX_LIGHT_CPU_X86

namespace {

struct CpuidResult {
  unsigned int eax, ebx, ecx, edx;
};

CpuidResult Cpuid(unsigned int leaf, unsigned int subleaf = 0) {
  CpuidResult r{};
#if defined(_MSC_VER)
  int regs[4];
  __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
  r.eax = static_cast<unsigned int>(regs[0]);
  r.ebx = static_cast<unsigned int>(regs[1]);
  r.ecx = static_cast<unsigned int>(regs[2]);
  r.edx = static_cast<unsigned int>(regs[3]);
#else
  __cpuid_count(leaf, subleaf, r.eax, r.ebx, r.ecx, r.edx);
#endif
  return r;
}

// Check if the OS has enabled AVX state saving (XGETBV).
bool OsSupportsAvx() {
  // OSXSAVE bit in CPUID.01H:ECX[27]
  auto info1 = Cpuid(1);
  if (!(info1.ecx & (1u << 27)))
    return false;
  // XCR0[2:1] = 0b11 means OS saves YMM state.
  unsigned long long xcr0;
#if defined(_MSC_VER)
  xcr0 = _xgetbv(0);
#else
  unsigned int lo, hi;
  __asm__ __volatile__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
  xcr0 = (static_cast<unsigned long long>(hi) << 32) | lo;
#endif
  return (xcr0 & 0x6) == 0x6;
}

bool OsSupportsAvx512() {
  if (!OsSupportsAvx())
    return false;
  unsigned long long xcr0;
#if defined(_MSC_VER)
  xcr0 = _xgetbv(0);
#else
  unsigned int lo, hi;
  __asm__ __volatile__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
  xcr0 = (static_cast<unsigned long long>(hi) << 32) | lo;
#endif
  // opmask (bit 5), ZMM_Hi256 (bit 6), Hi16_ZMM (bit 7)
  return (xcr0 & 0xE6) == 0xE6;
}

} // namespace

SimdLevel DetectSimdLevel() {
  auto info1 = Cpuid(1);
  bool has_sse2 = (info1.edx & (1u << 26)) != 0;
  bool has_avx = (info1.ecx & (1u << 28)) != 0;

  auto info7 = Cpuid(7);
  bool has_avx2 = (info7.ebx & (1u << 5)) != 0;
  bool has_avx512f = (info7.ebx & (1u << 16)) != 0;

  if (has_avx512f && OsSupportsAvx512())
    return SimdLevel::kAVX512;
  if (has_avx2 && OsSupportsAvx())
    return SimdLevel::kAVX2;
  if (has_avx && OsSupportsAvx())
    return SimdLevel::kAVX;
  if (has_sse2)
    return SimdLevel::kSSE2;
  return SimdLevel::kNone;
}

#else // Non-x86

SimdLevel DetectSimdLevel() { return SimdLevel::kNone; }

#endif // ONNX_LIGHT_CPU_X86

// ---------------------------------------------------------------------------
// AbsFloat32 implementations
// ---------------------------------------------------------------------------

namespace {

void AbsFloat32_Scalar(const float *input, float *output, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    output[i] = std::fabs(input[i]);
  }
}

#if ONNX_LIGHT_CPU_X86

void AbsFloat32_SSE2(const float *input, float *output, std::size_t count) {
  // Clear sign bit: AND with 0x7FFFFFFF
  const __m128 sign_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
  std::size_t i = 0;
  const std::size_t stride = 4;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    __m128 v = _mm_loadu_ps(input + i);
    v = _mm_and_ps(v, sign_mask);
    _mm_storeu_ps(output + i, v);
  }
  for (; i < count; ++i) {
    output[i] = std::fabs(input[i]);
  }
}

void AbsFloat32_AVX(const float *input, float *output, std::size_t count) {
  // Clear sign bit: AND with 0x7FFFFFFF using 256-bit registers
  const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
  std::size_t i = 0;
  const std::size_t stride = 8;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    __m256 v = _mm256_loadu_ps(input + i);
    v = _mm256_and_ps(v, sign_mask);
    _mm256_storeu_ps(output + i, v);
  }
  // Handle remainder with SSE2
  for (; i < count; ++i) {
    output[i] = std::fabs(input[i]);
  }
}

#ifdef __AVX512F__
void AbsFloat32_AVX512(const float *input, float *output, std::size_t count) {
  const __m512 sign_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));
  std::size_t i = 0;
  const std::size_t stride = 16;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    __m512 v = _mm512_loadu_ps(input + i);
    v = _mm512_and_ps(v, sign_mask);
    _mm512_storeu_ps(output + i, v);
  }
  for (; i < count; ++i) {
    output[i] = std::fabs(input[i]);
  }
}
#endif // __AVX512F__

#endif // ONNX_LIGHT_CPU_X86

} // namespace

void AbsFloat32(const float *input, float *output, std::size_t count) {
  if (count == 0)
    return;
#if ONNX_LIGHT_CPU_X86
  static const SimdLevel level = DetectSimdLevel();
#ifdef __AVX512F__
  if (level >= SimdLevel::kAVX512) {
    AbsFloat32_AVX512(input, output, count);
    return;
  }
#endif
  if (level >= SimdLevel::kAVX) {
    AbsFloat32_AVX(input, output, count);
    return;
  }
  if (level >= SimdLevel::kSSE2) {
    AbsFloat32_SSE2(input, output, count);
    return;
  }
#endif
  AbsFloat32_Scalar(input, output, count);
}

// ---------------------------------------------------------------------------
// AbsFloat64 implementations
// ---------------------------------------------------------------------------

namespace {

void AbsFloat64_Scalar(const double *input, double *output, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    output[i] = std::fabs(input[i]);
  }
}

#if ONNX_LIGHT_CPU_X86

void AbsFloat64_SSE2(const double *input, double *output, std::size_t count) {
  const __m128d sign_mask =
      _mm_castsi128_pd(_mm_set_epi64x(0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF));
  std::size_t i = 0;
  const std::size_t stride = 2;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    __m128d v = _mm_loadu_pd(input + i);
    v = _mm_and_pd(v, sign_mask);
    _mm_storeu_pd(output + i, v);
  }
  for (; i < count; ++i) {
    output[i] = std::fabs(input[i]);
  }
}

void AbsFloat64_AVX(const double *input, double *output, std::size_t count) {
  const __m256d sign_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
  std::size_t i = 0;
  const std::size_t stride = 4;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    __m256d v = _mm256_loadu_pd(input + i);
    v = _mm256_and_pd(v, sign_mask);
    _mm256_storeu_pd(output + i, v);
  }
  for (; i < count; ++i) {
    output[i] = std::fabs(input[i]);
  }
}

#ifdef __AVX512F__
void AbsFloat64_AVX512(const double *input, double *output, std::size_t count) {
  const __m512d sign_mask = _mm512_castsi512_pd(_mm512_set1_epi64(0x7FFFFFFFFFFFFFFF));
  std::size_t i = 0;
  const std::size_t stride = 8;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    __m512d v = _mm512_loadu_pd(input + i);
    v = _mm512_and_pd(v, sign_mask);
    _mm512_storeu_pd(output + i, v);
  }
  for (; i < count; ++i) {
    output[i] = std::fabs(input[i]);
  }
}
#endif // __AVX512F__

#endif // ONNX_LIGHT_CPU_X86

} // namespace

void AbsFloat64(const double *input, double *output, std::size_t count) {
  if (count == 0)
    return;
#if ONNX_LIGHT_CPU_X86
  static const SimdLevel level = DetectSimdLevel();
#ifdef __AVX512F__
  if (level >= SimdLevel::kAVX512) {
    AbsFloat64_AVX512(input, output, count);
    return;
  }
#endif
  if (level >= SimdLevel::kAVX) {
    AbsFloat64_AVX(input, output, count);
    return;
  }
  if (level >= SimdLevel::kSSE2) {
    AbsFloat64_SSE2(input, output, count);
    return;
  }
#endif
  AbsFloat64_Scalar(input, output, count);
}

// ---------------------------------------------------------------------------
// AbsInt32 implementations
// ---------------------------------------------------------------------------

namespace {

void AbsInt32_Scalar(const int32_t *input, int32_t *output, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    output[i] = input[i] < 0 ? -input[i] : input[i];
  }
}

#if ONNX_LIGHT_CPU_X86

void AbsInt32_SSE2(const int32_t *input, int32_t *output, std::size_t count) {
  // SSE2 pabsd is not available; use (x XOR mask) - mask where mask = x >> 31.
  std::size_t i = 0;
  const std::size_t stride = 4;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + i));
    __m128i mask = _mm_srai_epi32(v, 31);
    v = _mm_sub_epi32(_mm_xor_si128(v, mask), mask);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output + i), v);
  }
  for (; i < count; ++i) {
    output[i] = input[i] < 0 ? -input[i] : input[i];
  }
}

void AbsInt32_AVX2(const int32_t *input, int32_t *output, std::size_t count) {
  std::size_t i = 0;
  const std::size_t stride = 8;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + i));
    v = _mm256_abs_epi32(v);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + i), v);
  }
  for (; i < count; ++i) {
    output[i] = input[i] < 0 ? -input[i] : input[i];
  }
}

#ifdef __AVX512F__
void AbsInt32_AVX512(const int32_t *input, int32_t *output, std::size_t count) {
  std::size_t i = 0;
  const std::size_t stride = 16;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    __m512i v = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(input + i));
    v = _mm512_abs_epi32(v);
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(output + i), v);
  }
  for (; i < count; ++i) {
    output[i] = input[i] < 0 ? -input[i] : input[i];
  }
}
#endif // __AVX512F__

#endif // ONNX_LIGHT_CPU_X86

} // namespace

void AbsInt32(const int32_t *input, int32_t *output, std::size_t count) {
  if (count == 0)
    return;
#if ONNX_LIGHT_CPU_X86
  static const SimdLevel level = DetectSimdLevel();
#ifdef __AVX512F__
  if (level >= SimdLevel::kAVX512) {
    AbsInt32_AVX512(input, output, count);
    return;
  }
#endif
  if (level >= SimdLevel::kAVX2) {
    AbsInt32_AVX2(input, output, count);
    return;
  }
  if (level >= SimdLevel::kSSE2) {
    AbsInt32_SSE2(input, output, count);
    return;
  }
#endif
  AbsInt32_Scalar(input, output, count);
}

// ---------------------------------------------------------------------------
// AbsInt64 implementations
// ---------------------------------------------------------------------------

namespace {

void AbsInt64_Scalar(const int64_t *input, int64_t *output, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    output[i] = input[i] < 0 ? -input[i] : input[i];
  }
}

#if ONNX_LIGHT_CPU_X86

void AbsInt64_SSE2(const int64_t *input, int64_t *output, std::size_t count) {
  // No pabsq before AVX512; use conditional negate with arithmetic shift
  // emulation for 64-bit.
  std::size_t i = 0;
  const std::size_t stride = 2;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + i));
    // Arithmetic right shift by 63: shift hi 32 bits, then shuffle
    __m128i sign = _mm_srai_epi32(v, 31);
    sign = _mm_shuffle_epi32(sign, 0xF5); // replicate bits 63 across both halves
    __m128i abs_v = _mm_sub_epi64(_mm_xor_si128(v, sign), sign);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output + i), abs_v);
  }
  for (; i < count; ++i) {
    output[i] = input[i] < 0 ? -input[i] : input[i];
  }
}

void AbsInt64_AVX2(const int64_t *input, int64_t *output, std::size_t count) {
  // AVX2 has no _mm256_abs_epi64; emulate with arithmetic shift + xor + sub.
  std::size_t i = 0;
  const std::size_t stride = 4;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + i));
    // Emulate arithmetic right shift by 63 for epi64
    __m256i sign = _mm256_srai_epi32(v, 31);
    sign = _mm256_shuffle_epi32(sign, 0xF5);
    __m256i abs_v = _mm256_sub_epi64(_mm256_xor_si256(v, sign), sign);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + i), abs_v);
  }
  for (; i < count; ++i) {
    output[i] = input[i] < 0 ? -input[i] : input[i];
  }
}

#ifdef __AVX512F__
void AbsInt64_AVX512(const int64_t *input, int64_t *output, std::size_t count) {
  std::size_t i = 0;
  const std::size_t stride = 8;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    __m512i v = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(input + i));
    v = _mm512_abs_epi64(v);
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(output + i), v);
  }
  for (; i < count; ++i) {
    output[i] = input[i] < 0 ? -input[i] : input[i];
  }
}
#endif // __AVX512F__

#endif // ONNX_LIGHT_CPU_X86

} // namespace

void AbsInt64(const int64_t *input, int64_t *output, std::size_t count) {
  if (count == 0)
    return;
#if ONNX_LIGHT_CPU_X86
  static const SimdLevel level = DetectSimdLevel();
#ifdef __AVX512F__
  if (level >= SimdLevel::kAVX512) {
    AbsInt64_AVX512(input, output, count);
    return;
  }
#endif
  if (level >= SimdLevel::kAVX2) {
    AbsInt64_AVX2(input, output, count);
    return;
  }
  if (level >= SimdLevel::kSSE2) {
    AbsInt64_SSE2(input, output, count);
    return;
  }
#endif
  AbsInt64_Scalar(input, output, count);
}

} // namespace onnx_light_cpu
