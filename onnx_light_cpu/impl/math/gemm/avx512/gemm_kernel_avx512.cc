// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// AVX-512 Gemm micro-kernels. This translation unit is compiled with an extra
// -mavx512f (see the per-file COMPILE_OPTIONS override in CMakeLists.txt)
// even though the rest of onnx_light_cpu keeps the project's baseline
// ONNX_LIGHT_CPU_SIMD_FLAGS (AVX2 by default). This lets a single binary
// carry both an AVX2 and this wider AVX-512 micro-kernel: GemmTileF32/F64 in
// gemm_kernel.cc pick whichever one matches DetectSimdLevel() at runtime, so
// a binary built on an AVX-512-capable machine still runs correctly (falling
// back to AVX2/SSE2/scalar) on a CPU that lacks it.
//
// Mirrors the AVX (256-bit) micro-kernels in gemm_kernel.cc one register
// width up: a 512-bit vector holds 16 floats / 8 doubles, and every
// micro-kernel call still processes ``kGemmMR`` (4) rows at once, register
// blocked over the whole (chunked) K reduction. Like the AVX/SSE2 kernels, the
// main loop processes two vectors of columns per step (``NR = 2``) so each
// broadcast A element is reused across twice the column width, amortizing the
// broadcast/loop overhead over more FMAs.

#include "onnx_light_cpu/impl/math/gemm/avx512/gemm_kernel_avx512.h"

#include <immintrin.h>

namespace onnx_light_cpu {

namespace {

inline __m512 MulAdd(__m512 a, __m512 b, __m512 acc) { return _mm512_fmadd_ps(a, b, acc); }
inline __m512d MulAdd(__m512d a, __m512d b, __m512d acc) { return _mm512_fmadd_pd(a, b, acc); }

// Number of ``k`` iterations ahead to issue a software prefetch for the next
// ``Bpack`` row; see the identical rationale in gemm_kernel.cc.
constexpr int kGemmPrefetchDistanceK = 4;

template <typename T> inline void PrefetchT0(const T *ptr) {
  _mm_prefetch(reinterpret_cast<const char *>(ptr), _MM_HINT_T0);
}

} // namespace

void GemmMicroKernel_AVX512_F32(std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                                float beta, const float *Bmat, std::size_t N,
                                const float *Crow_base, std::size_t Cstride, float *Yrow_base,
                                std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                                const float *Apack) {
  const __m512 valpha = _mm512_set1_ps(alpha);
  const __m512 vbeta = _mm512_set1_ps(beta);
  std::size_t n = 0;
  // NR == 2: 32 lanes (two 512-bit vectors) per step.
  for (; n + 32 <= nb; n += 32) {
    __m512 acc0[kGemmMR];
    __m512 acc1[kGemmMR];
    for (std::size_t r = 0; r < mr; ++r) {
      acc0[r] = _mm512_setzero_ps();
      acc1[r] = _mm512_setzero_ps();
    }
    for (std::size_t k = 0; k < K; ++k) {
      const float *Brow = Bmat + k * N + n0 + n;
      const __m512 vb0 = _mm512_loadu_ps(Brow);
      const __m512 vb1 = _mm512_loadu_ps(Brow + 16);
      if (k + kGemmPrefetchDistanceK < K) {
        PrefetchT0(Brow + kGemmPrefetchDistanceK * N);
      }
      for (std::size_t r = 0; r < mr; ++r) {
        const __m512 va = _mm512_set1_ps(Apack[r * K + k]);
        acc0[r] = MulAdd(va, vb0, acc0[r]);
        acc1[r] = MulAdd(va, vb1, acc1[r]);
      }
    }
    for (std::size_t r = 0; r < mr; ++r) {
      float *Yrow = Yrow_base + r * Ystride + n0 + n;
      __m512 res0 = _mm512_mul_ps(valpha, acc0[r]);
      __m512 res1 = _mm512_mul_ps(valpha, acc1[r]);
      if (mode == GemmAccumMode::kInitBias) {
        const float *Crow = Crow_base + r * Cstride + n0 + n;
        res0 = _mm512_add_ps(res0, _mm512_mul_ps(vbeta, _mm512_loadu_ps(Crow)));
        res1 = _mm512_add_ps(res1, _mm512_mul_ps(vbeta, _mm512_loadu_ps(Crow + 16)));
      } else if (mode == GemmAccumMode::kAccumulate) {
        res0 = _mm512_add_ps(res0, _mm512_loadu_ps(Yrow));
        res1 = _mm512_add_ps(res1, _mm512_loadu_ps(Yrow + 16));
      }
      _mm512_storeu_ps(Yrow, res0);
      _mm512_storeu_ps(Yrow + 16, res1);
    }
  }
  // NR == 1 remainder: a single 16-lane vector.
  for (; n + 16 <= nb; n += 16) {
    __m512 acc[kGemmMR];
    for (std::size_t r = 0; r < mr; ++r) {
      acc[r] = _mm512_setzero_ps();
    }
    for (std::size_t k = 0; k < K; ++k) {
      const __m512 vb = _mm512_loadu_ps(Bmat + k * N + n0 + n);
      for (std::size_t r = 0; r < mr; ++r) {
        const __m512 va = _mm512_set1_ps(Apack[r * K + k]);
        acc[r] = MulAdd(va, vb, acc[r]);
      }
    }
    for (std::size_t r = 0; r < mr; ++r) {
      float *Yrow = Yrow_base + r * Ystride + n0 + n;
      __m512 res = _mm512_mul_ps(valpha, acc[r]);
      if (mode == GemmAccumMode::kInitBias) {
        const __m512 vc = _mm512_loadu_ps(Crow_base + r * Cstride + n0 + n);
        res = _mm512_add_ps(res, _mm512_mul_ps(vbeta, vc));
      } else if (mode == GemmAccumMode::kAccumulate) {
        res = _mm512_add_ps(res, _mm512_loadu_ps(Yrow));
      }
      _mm512_storeu_ps(Yrow, res);
    }
  }
  // Scalar tail (< 16 lanes): reuse the shared scalar micro-kernel from
  // gemm_kernel.cc instead of duplicating the mode-driven combine logic.
  if (n < nb) {
    GemmMicroKernel_Scalar_F32(mr, nb - n, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                               Ystride, n0 + n, mode, Apack);
  }
}

void GemmMicroKernel_AVX512_F64(std::size_t mr, std::size_t nb, std::size_t K, double alpha,
                                double beta, const double *Bmat, std::size_t N,
                                const double *Crow_base, std::size_t Cstride, double *Yrow_base,
                                std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                                const double *Apack) {
  const __m512d valpha = _mm512_set1_pd(alpha);
  const __m512d vbeta = _mm512_set1_pd(beta);
  std::size_t n = 0;
  // NR == 2: 16 lanes (two 512-bit vectors) per step.
  for (; n + 16 <= nb; n += 16) {
    __m512d acc0[kGemmMR];
    __m512d acc1[kGemmMR];
    for (std::size_t r = 0; r < mr; ++r) {
      acc0[r] = _mm512_setzero_pd();
      acc1[r] = _mm512_setzero_pd();
    }
    for (std::size_t k = 0; k < K; ++k) {
      const double *Brow = Bmat + k * N + n0 + n;
      const __m512d vb0 = _mm512_loadu_pd(Brow);
      const __m512d vb1 = _mm512_loadu_pd(Brow + 8);
      if (k + kGemmPrefetchDistanceK < K) {
        PrefetchT0(Brow + kGemmPrefetchDistanceK * N);
      }
      for (std::size_t r = 0; r < mr; ++r) {
        const __m512d va = _mm512_set1_pd(Apack[r * K + k]);
        acc0[r] = MulAdd(va, vb0, acc0[r]);
        acc1[r] = MulAdd(va, vb1, acc1[r]);
      }
    }
    for (std::size_t r = 0; r < mr; ++r) {
      double *Yrow = Yrow_base + r * Ystride + n0 + n;
      __m512d res0 = _mm512_mul_pd(valpha, acc0[r]);
      __m512d res1 = _mm512_mul_pd(valpha, acc1[r]);
      if (mode == GemmAccumMode::kInitBias) {
        const double *Crow = Crow_base + r * Cstride + n0 + n;
        res0 = _mm512_add_pd(res0, _mm512_mul_pd(vbeta, _mm512_loadu_pd(Crow)));
        res1 = _mm512_add_pd(res1, _mm512_mul_pd(vbeta, _mm512_loadu_pd(Crow + 8)));
      } else if (mode == GemmAccumMode::kAccumulate) {
        res0 = _mm512_add_pd(res0, _mm512_loadu_pd(Yrow));
        res1 = _mm512_add_pd(res1, _mm512_loadu_pd(Yrow + 8));
      }
      _mm512_storeu_pd(Yrow, res0);
      _mm512_storeu_pd(Yrow + 8, res1);
    }
  }
  // NR == 1 remainder: a single 8-lane vector.
  for (; n + 8 <= nb; n += 8) {
    __m512d acc[kGemmMR];
    for (std::size_t r = 0; r < mr; ++r) {
      acc[r] = _mm512_setzero_pd();
    }
    for (std::size_t k = 0; k < K; ++k) {
      const __m512d vb = _mm512_loadu_pd(Bmat + k * N + n0 + n);
      for (std::size_t r = 0; r < mr; ++r) {
        const __m512d va = _mm512_set1_pd(Apack[r * K + k]);
        acc[r] = MulAdd(va, vb, acc[r]);
      }
    }
    for (std::size_t r = 0; r < mr; ++r) {
      double *Yrow = Yrow_base + r * Ystride + n0 + n;
      __m512d res = _mm512_mul_pd(valpha, acc[r]);
      if (mode == GemmAccumMode::kInitBias) {
        const __m512d vc = _mm512_loadu_pd(Crow_base + r * Cstride + n0 + n);
        res = _mm512_add_pd(res, _mm512_mul_pd(vbeta, vc));
      } else if (mode == GemmAccumMode::kAccumulate) {
        res = _mm512_add_pd(res, _mm512_loadu_pd(Yrow));
      }
      _mm512_storeu_pd(Yrow, res);
    }
  }
  // Scalar tail (< 8 lanes): reuse the shared scalar micro-kernel from
  // gemm_kernel.cc instead of duplicating the mode-driven combine logic.
  if (n < nb) {
    GemmMicroKernel_Scalar_F64(mr, nb - n, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                               Ystride, n0 + n, mode, Apack);
  }
}

} // namespace onnx_light_cpu
