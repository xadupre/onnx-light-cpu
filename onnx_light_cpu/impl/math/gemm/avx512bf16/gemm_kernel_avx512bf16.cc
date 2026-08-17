// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Native AVX-512BF16 Gemm micro-kernel (Roadmap PR07.4). This translation unit
// is compiled with an extra -mavx512bf16 (see the per-file COMPILE_OPTIONS
// override in CMakeLists.txt) even though the rest of onnx_light_cpu keeps the
// project's baseline ONNX_LIGHT_CPU_SIMD_FLAGS, so a single binary can carry
// this native BFLOAT16 kernel and still run correctly on CPUs that lack the
// ISA: the BFLOAT16 GEMM path in gemm_kernel.cc only dispatches here when
// ``CpuSupportsAvx512Bf16()`` reports the instruction set at runtime, and
// otherwise falls back to the convert-while-packing float32 path.
//
// Unlike the FP32 micro-kernels, both operands stay in BFLOAT16 up to the
// register file: ``Apack`` is a packed BFLOAT16 row panel and ``Bmat`` is the
// BFLOAT16 ``B`` matrix. Two consecutive ``k`` iterations are reduced at once
// with the AVX-512BF16 ``vdpbf16ps`` dot-product (``_mm512_dpbf16_ps``), which
// converts each BFLOAT16 pair to float32 and accumulates in float32, so the
// result matches the widen-then-float32 reference within float32 tolerance
// while halving the ``B`` memory traffic.

#include "onnx_light_cpu/impl/math/gemm/avx512bf16/gemm_kernel_avx512bf16.h"

#include <cassert>

#include <immintrin.h>

namespace onnx_light_cpu {

namespace {

// Number of ``k`` iterations ahead to issue a software prefetch for the next
// ``B`` row; see the identical rationale in gemm_kernel.cc.
constexpr int kGemmPrefetchDistanceK = 4;

inline void PrefetchT0(const std::uint16_t *ptr) {
  _mm_prefetch(reinterpret_cast<const char *>(ptr), _MM_HINT_T0);
}

// Interleaves two 16-lane BFLOAT16 rows (``b0`` = ``B[k]``, ``b1`` = ``B[k +
// 1]``) into the ``vdpbf16ps`` pair layout: the returned vector holds, for each
// of the 16 columns ``j``, the pair ``{B[k, j], B[k + 1, j]}`` as the low and
// high BFLOAT16 of lane ``j``. Zero-extending each 16-bit column to 32 bits
// preserves the column order across the whole ZMM register, unlike the
// lane-local ``unpack`` shuffles.
inline __m512bh MakeBPair(__m256i b0, __m256i b1) {
  const __m512i lo = _mm512_cvtepu16_epi32(b0);
  const __m512i hi = _mm512_cvtepu16_epi32(b1);
  return reinterpret_cast<__m512bh>(_mm512_or_si512(lo, _mm512_slli_epi32(hi, 16)));
}

// Broadcasts one BFLOAT16 pair ``{Apack[k], Apack[k + 1]}`` to all 16 lanes so
// ``vdpbf16ps`` multiplies the same ``A`` pair against every column's ``B``
// pair.
inline __m512bh BroadcastAPair(std::uint16_t a0, std::uint16_t a1) {
  const std::int32_t packed = static_cast<std::int32_t>(static_cast<std::uint32_t>(a0) |
                                                        (static_cast<std::uint32_t>(a1) << 16));
  return reinterpret_cast<__m512bh>(_mm512_set1_epi32(packed));
}

} // namespace

void GemmMicroKernel_AVX512BF16(std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                                float beta, const std::uint16_t *Bmat, std::size_t N,
                                const float *Crow_base, std::size_t Cstride, float *Yrow_base,
                                std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                                const std::uint16_t *Apack) {
  // ``acc`` is register-blocked over at most ``kGemmAVX512MR`` rows; callers
  // must cap ``mr`` to that bound (the dispatch path uses std::min).
  assert(mr <= kGemmAVX512MR && "mr exceeds kGemmAVX512MR register-row budget");
  const __m512 valpha = _mm512_set1_ps(alpha);
  const __m512 vbeta = _mm512_set1_ps(beta);
  const bool alpha_is_one = alpha == 1.0f;
  const bool beta_is_one = beta == 1.0f;
  std::size_t n = 0;
  // One 16-lane float32 accumulator (16 BFLOAT16 columns) per step,
  // register-blocked over ``mr`` rows across the whole K reduction.
  for (; n + 16 <= nb; n += 16) {
    __m512 acc[kGemmAVX512MR];
    for (std::size_t r = 0; r < mr; ++r) {
      acc[r] = _mm512_setzero_ps();
    }
    // Reduce two ``k`` iterations at a time: ``vdpbf16ps`` computes, per lane,
    // ``A[r,k]*B[k,n+j] + A[r,k+1]*B[k+1,n+j]``.
    std::size_t k = 0;
    for (; k + 2 <= K; k += 2) {
      const std::uint16_t *Brow0 = Bmat + k * N + n0 + n;
      const std::uint16_t *Brow1 = Brow0 + N;
      const __m256i b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(Brow0));
      const __m256i b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(Brow1));
      const __m512bh bpair = MakeBPair(b0, b1);
      if (k + kGemmPrefetchDistanceK < K) {
        PrefetchT0(Brow0 + kGemmPrefetchDistanceK * N);
      }
      for (std::size_t r = 0; r < mr; ++r) {
        const std::uint16_t *Apack_r = Apack + r * K + k;
        const __m512bh apair = BroadcastAPair(Apack_r[0], Apack_r[1]);
        acc[r] = _mm512_dpbf16_ps(acc[r], apair, bpair);
      }
    }
    // Odd ``K`` leftover: one ``k`` step through the same dot-product with the
    // second BFLOAT16 of each pair zeroed, so only ``A[r,k]*B[k,n+j]`` adds.
    if (k < K) {
      const std::uint16_t *Brow0 = Bmat + k * N + n0 + n;
      const __m256i b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(Brow0));
      const __m512bh bpair = MakeBPair(b0, _mm256_setzero_si256());
      for (std::size_t r = 0; r < mr; ++r) {
        const __m512bh apair = BroadcastAPair(Apack[r * K + k], 0);
        acc[r] = _mm512_dpbf16_ps(acc[r], apair, bpair);
      }
    }
    for (std::size_t r = 0; r < mr; ++r) {
      float *Yrow = Yrow_base + r * Ystride + n0 + n;
      __m512 res = alpha_is_one ? acc[r] : _mm512_mul_ps(valpha, acc[r]);
      if (mode == GemmAccumMode::kInitBias) {
        const __m512 vc = _mm512_loadu_ps(Crow_base + r * Cstride + n0 + n);
        res = _mm512_add_ps(res, beta_is_one ? vc : _mm512_mul_ps(vbeta, vc));
      } else if (mode == GemmAccumMode::kAccumulate) {
        res = _mm512_add_ps(res, _mm512_loadu_ps(Yrow));
      }
      _mm512_storeu_ps(Yrow, res);
    }
  }
  // Scalar tail (< 16 columns): reuse the shared scalar BFLOAT16 micro-kernel
  // instead of duplicating the mode-driven combine logic.
  if (n < nb) {
    GemmMicroKernel_ScalarBf16(mr, nb - n, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                               Ystride, n0 + n, mode, Apack);
  }
}

} // namespace onnx_light_cpu
