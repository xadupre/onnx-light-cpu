// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Native AVX-512FP16 Gemm micro-kernel (Roadmap PR07.3). This translation unit
// is compiled with an extra -mavx512fp16 (see the per-file COMPILE_OPTIONS
// override in CMakeLists.txt) even though the rest of onnx_light_cpu keeps the
// compiler's baseline ISA, so a single binary can carry
// this native FLOAT16 kernel and still run correctly on CPUs that lack the ISA:
// the FLOAT16 GEMM path in gemm_kernel.cc only dispatches here when
// ``CpuSupportsAvx512Fp16()`` reports the instruction set at runtime, and
// otherwise falls back to the convert-while-packing float32 path.
//
// Unlike the FP32 micro-kernels, both operands stay in FLOAT16 up to the
// register file: ``Apack`` is a packed FLOAT16 row panel and ``Bmat`` is the
// FLOAT16 ``B`` matrix. Each 16-lane column vector of ``B`` is widened with the
// AVX-512FP16 ``vcvtph2psx`` conversion (``_mm512_cvtxph_ps``) and the dot
// products accumulate in float32, so the result matches the widen-then-float32
// reference within float32 tolerance while halving the ``B`` memory traffic.

#include "onnx_light_cpu/impl/math/gemm/avx512fp16/gemm_kernel_avx512fp16.h"

#include "onnx_light_cpu/impl/execution.h"

#include <array>
#include <cassert>
#include <cstring>
#include <vector>

#include <immintrin.h>

namespace onnx_light_cpu {

namespace {

// Number of ``k`` iterations ahead to issue a software prefetch for the next
// ``B`` row; see the identical rationale in gemm_kernel.cc.
constexpr int kGemmPrefetchDistanceK = 4;
constexpr std::size_t kMaximumPackedK = 512;

inline void PrefetchT0(const std::uint16_t *ptr) {
  _mm_prefetch(reinterpret_cast<const char *>(ptr), _MM_HINT_T0);
}

// Widens one packed FLOAT16 element to float32. The raw 16-bit pattern is an
// IEEE binary16 value, so it is copied into a ``_Float16`` and widened with
// the compiler's standard conversion.
inline float WidenHalfScalar(const std::uint16_t *bits) {
  _Float16 value;
  std::memcpy(&value, bits, sizeof(value));
  return static_cast<float>(value);
}

} // namespace

void GemmMicroKernel_AVX512FP16(std::size_t mr, std::size_t nb, std::size_t K, float alpha,
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

  // ``Apack`` is reused unchanged across every 16-column block of ``nb`` below,
  // so widen each of its ``mr * K`` FLOAT16 entries to float32 exactly once
  // instead of re-widening the same scalar with every column block (the
  // previous version paid this conversion ``nb / 16`` times over).
  assert(K <= kMaximumPackedK && "K exceeds the packed micro-kernel depth");
  alignas(64) std::array<std::array<float, kMaximumPackedK>, kGemmAVX512MR> a_widened;
  for (std::size_t r = 0; r < mr; ++r) {
    const std::uint16_t *arow = Apack + r * K;
    std::size_t k = 0;
    for (; k + 16 <= K; k += 16) {
      const __m256h values = _mm256_loadu_ph(reinterpret_cast<const void *>(arow + k));
      _mm512_store_ps(a_widened[r].data() + k, _mm512_cvtxph_ps(values));
    }
    for (; k < K; ++k) {
      a_widened[r][k] = WidenHalfScalar(arow + k);
    }
  }

  std::size_t n = 0;
  // One 16-lane float32 vector (16 FLOAT16 columns) per step, register-blocked
  // over ``mr`` rows across the whole K reduction.
  for (; n + 16 <= nb; n += 16) {
    __m512 acc[kGemmAVX512MR];
    for (std::size_t r = 0; r < mr; ++r) {
      acc[r] = _mm512_setzero_ps();
    }
    const auto accumulate_k = [&](std::size_t k) {
      const std::uint16_t *Brow = Bmat + k * N + n0 + n;
      const __m256h bh = _mm256_loadu_ph(reinterpret_cast<const void *>(Brow));
      const __m512 vb = _mm512_cvtxph_ps(bh);
      if (k + kGemmPrefetchDistanceK < K) {
        PrefetchT0(Brow + kGemmPrefetchDistanceK * N);
      }
      for (std::size_t r = 0; r < mr; ++r) {
        const __m512 va = _mm512_set1_ps(a_widened[r][k]);
        acc[r] = _mm512_fmadd_ps(va, vb, acc[r]);
      }
    };
    std::size_t k = 0;
    for (; k + 4 <= K; k += 4) {
      accumulate_k(k);
      accumulate_k(k + 1);
      accumulate_k(k + 2);
      accumulate_k(k + 3);
    }
    for (; k < K; ++k) {
      accumulate_k(k);
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
  // Scalar tail (< 16 columns): reuse the shared scalar FLOAT16 micro-kernel
  // instead of duplicating the mode-driven combine logic.
  if (n < nb) {
    GemmMicroKernel_ScalarFp16(mr, nb - n, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                               Ystride, n0 + n, mode, Apack);
  }
}

void GemmFloat16SkinnyN_AVX512FP16(std::size_t M, std::size_t K, float alpha,
                                   const std::uint16_t *A, const std::uint16_t *B, float *Y) {
  std::vector<float> widened_b(K);
  std::size_t depth = 0;
  for (; depth + 16 <= K; depth += 16) {
    const __m256h values = _mm256_loadu_ph(reinterpret_cast<const void *>(B + depth));
    _mm512_storeu_ps(widened_b.data() + depth, _mm512_cvtxph_ps(values));
  }
  for (; depth < K; ++depth) {
    widened_b[depth] = WidenHalfScalar(B + depth);
  }

  const double cost = static_cast<double>(K) / 32768.0;
  ExecuteRanges(static_cast<std::int64_t>(M), cost, [&](std::int64_t begin, std::int64_t end) {
    std::int64_t row = begin;
    for (; row + 4 <= end; row += 4) {
      __m512 accumulators[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                                _mm512_setzero_ps()};
      std::size_t k = 0;
      for (; k + 16 <= K; k += 16) {
        const __m512 b = _mm512_loadu_ps(widened_b.data() + k);
        for (std::size_t r = 0; r < 4; ++r) {
          const std::uint16_t *a = A + (static_cast<std::size_t>(row) + r) * K + k;
          const __m256h values = _mm256_loadu_ph(reinterpret_cast<const void *>(a));
          accumulators[r] = _mm512_fmadd_ps(_mm512_cvtxph_ps(values), b, accumulators[r]);
        }
      }
      for (std::size_t r = 0; r < 4; ++r) {
        const std::uint16_t *a = A + (static_cast<std::size_t>(row) + r) * K;
        float sum = _mm512_reduce_add_ps(accumulators[r]);
        for (std::size_t tail = k; tail < K; ++tail) {
          sum += WidenHalfScalar(a + tail) * widened_b[tail];
        }
        Y[static_cast<std::size_t>(row) + r] = alpha * sum;
      }
    }
    for (; row < end; ++row) {
      const std::uint16_t *a = A + static_cast<std::size_t>(row) * K;
      __m512 accumulator = _mm512_setzero_ps();
      std::size_t k = 0;
      for (; k + 16 <= K; k += 16) {
        const __m256h values = _mm256_loadu_ph(reinterpret_cast<const void *>(a + k));
        accumulator = _mm512_fmadd_ps(_mm512_cvtxph_ps(values),
                                      _mm512_loadu_ps(widened_b.data() + k), accumulator);
      }
      float sum = _mm512_reduce_add_ps(accumulator);
      for (; k < K; ++k) {
        sum += WidenHalfScalar(a + k) * widened_b[k];
      }
      Y[row] = alpha * sum;
    }
  });
}

} // namespace onnx_light_cpu
