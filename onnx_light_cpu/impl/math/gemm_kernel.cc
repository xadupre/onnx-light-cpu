// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Fast general matrix multiplication (ONNX ``Gemm``) kernels for float32 and
// float64.
//
// The kernel computes ``Y = alpha * op(A) @ op(B) + beta * C`` with a
// register-blocked micro-kernel. Instead of the memory-bound rank-1 update
// (AXPY) formulation -- which reloads and rewrites every ``Y`` element once per
// ``k`` and is therefore limited by memory bandwidth -- the micro-kernel keeps a
// tile of ``kGemmMR`` output rows by one or two SIMD vectors of columns
// resident in CPU registers while it accumulates a chunk of the ``k``
// reduction. For each ``k`` a ``B`` row vector is loaded once and reused
// across all ``kGemmMR`` rows: each row broadcasts its scalar ``A(m, k)`` and
// fuses it into its accumulator. This turns the hot loop into
// register-resident multiply-adds with an arithmetic intensity high enough to
// be compute bound, closing most of the gap with optimized BLAS-style GEMM.
//
// To keep a single fast inner kernel, a transposed ``B`` (``trans_b``) is packed
// into a contiguous ``K x N`` buffer first; this pre-pass is ``O(K*N)`` and
// negligible next to the ``O(M*N*K)`` multiplication. ``A`` is packed into a
// small contiguous ``kGemmMR x kc`` row-panel per (row block, k-chunk) by
// ``PackARowBlock``: this resolves ``trans_a`` once via a copy (or a strided
// gather, only when ``trans_a`` is set) instead of paying a strided load and a
// ``trans_a`` branch for every element on every one of the (up to
// ``kGemmTileN / vector_width``) column-vector iterations that reuse it.
//
// For cache efficiency the output ``Y`` is walked as a grid of ``kGemmTileM x
// kGemmTileN`` tiles instead of one full row at a time, and ``K`` is further
// split into ``kGemmTileK``-sized chunks (see the comment above
// ``kGemmTileK``). The (row block, column panel) grid is flattened into a
// single task list so :cpp:func:`ParallelFor` can spread work across threads
// on both the ``M`` and ``N`` axes: a pure M-only split leaves no parallelism
// for the "skinny" shapes common in inference (e.g. ``M == 1`` for a single
// example / matvec), no matter how wide ``N`` is or how many cores are
// available.
//
// On top of the AVX/SSE2/scalar micro-kernels below, an AVX-512 micro-kernel
// (gemm_kernel_avx512.cc, compiled with an extra -mavx512f) is picked instead
// when both the compiler supports it (``ONNX_LIGHT_CPU_HAVE_AVX512``) and
// ``DetectSimdLevel()`` reports it at runtime.

#include "onnx_light_cpu/impl/math/math_kernels.h"

#include "onnx_light_cpu/impl/math/gemm_common.h"
#include "onnx_light_cpu/impl/parallel_for.h"

#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
#include "onnx_light_cpu/impl/math/avx512/gemm_kernel_avx512.h"
#endif

#include <algorithm>
#include <cstddef>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define ONNX_LIGHT_CPU_X86 1
#include <immintrin.h>
#else
#define ONNX_LIGHT_CPU_X86 0
#endif

namespace onnx_light_cpu {

namespace {

// ---------------------------------------------------------------------------
// Register-blocked GEMM micro-kernels.
//
// Each micro-kernel computes a tile of ``mr`` (<= kGemmMR) output rows by
// ``nb`` columns:
//
//     acc[r][:] = sum_k Apack[r, k] * Bmat[k, n0 : n0 + nb]
//
// accumulating the k-chunk reduction in registers before it is scaled by
// ``alpha``, combined with the optional ``beta * C`` bias or existing ``Y``
// contents (see ``GemmAccumMode``) and written to ``Y``. ``Apack`` is a packed
// contiguous ``mr x K`` row-major panel (``PackARowBlock`` below): row ``r``,
// column ``k`` is ``A(m + r, k0 + k)``, already resolved for ``trans_a``. The
// ``B`` row ``Bmat + k * N + n0`` is loaded once per ``k`` and reused across
// all ``mr`` rows.
//
// The vectorized kernels process ``NR == 2`` vectors of columns per step so
// each broadcast ``A`` element is reused across twice the column width,
// amortizing the broadcast and loop overhead over more FMAs than a
// single-vector step would; a single-vector (``NR == 1``) loop handles the
// remainder before falling back to the scalar kernel for the final tail.

// Scalar micro-kernel: also the tail handler for every vectorized flavor and
// the fallback for non-x86 builds.
template <typename T>
void GemmMicroKernel_ScalarImpl(std::size_t mr, std::size_t nb, std::size_t K, T alpha, T beta,
                                const T *Bmat, std::size_t N, const T *Crow_base,
                                std::size_t Cstride, T *Yrow_base, std::size_t Ystride,
                                std::size_t n0, GemmAccumMode mode, const T *Apack) {
  for (std::size_t r = 0; r < mr; ++r) {
    T *Yrow = Yrow_base + r * Ystride + n0;
    // Initialize with the scaled bias (or zero) for the first chunk; a later
    // chunk (kAccumulate) leaves ``Y`` as-is and only adds to it below. This
    // path only runs for the scalar fallback and the (small) SIMD column tail,
    // so it avoids any temporary allocation.
    if (mode == GemmAccumMode::kInitBias) {
      const T *Crow = Crow_base + r * Cstride + n0;
      for (std::size_t n = 0; n < nb; ++n) {
        Yrow[n] = beta * Crow[n];
      }
    } else if (mode == GemmAccumMode::kInitZero) {
      for (std::size_t n = 0; n < nb; ++n) {
        Yrow[n] = T(0);
      }
    }
    const T *Apack_r = Apack + r * K;
    for (std::size_t k = 0; k < K; ++k) {
      const T a = alpha * Apack_r[k];
      const T *Brow = Bmat + k * N + n0;
      for (std::size_t n = 0; n < nb; ++n) {
        Yrow[n] += a * Brow[n];
      }
    }
  }
}

} // namespace

// Explicit non-template, external-linkage wrappers so gemm_kernel_avx512.cc (a
// separate translation unit compiled with -mavx512f) can reuse the scalar
// tail instead of duplicating the mode-driven combine logic.
void GemmMicroKernel_Scalar_F32(std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                                float beta, const float *Bmat, std::size_t N,
                                const float *Crow_base, std::size_t Cstride, float *Yrow_base,
                                std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                                const float *Apack) {
  GemmMicroKernel_ScalarImpl<float>(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                                    Ystride, n0, mode, Apack);
}

void GemmMicroKernel_Scalar_F64(std::size_t mr, std::size_t nb, std::size_t K, double alpha,
                                double beta, const double *Bmat, std::size_t N,
                                const double *Crow_base, std::size_t Cstride, double *Yrow_base,
                                std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                                const double *Apack) {
  GemmMicroKernel_ScalarImpl<double>(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                                     Ystride, n0, mode, Apack);
}

namespace {

#if ONNX_LIGHT_CPU_X86

// Fused multiply-add helpers: return ``acc + a * b``. When the translation unit
// is compiled with FMA support (``__FMA__``, e.g. ``-mfma`` or MSVC
// ``/arch:AVX2``) a single fused instruction is emitted; otherwise a separate
// multiply and add are used so plain AVX targets still compile.
inline __m256 MulAdd(__m256 a, __m256 b, __m256 acc) {
#ifdef __FMA__
  return _mm256_fmadd_ps(a, b, acc);
#else
  return _mm256_add_ps(acc, _mm256_mul_ps(a, b));
#endif
}
inline __m256d MulAdd(__m256d a, __m256d b, __m256d acc) {
#ifdef __FMA__
  return _mm256_fmadd_pd(a, b, acc);
#else
  return _mm256_add_pd(acc, _mm256_mul_pd(a, b));
#endif
}
inline __m128 MulAdd(__m128 a, __m128 b, __m128 acc) {
#ifdef __FMA__
  return _mm_fmadd_ps(a, b, acc);
#else
  return _mm_add_ps(acc, _mm_mul_ps(a, b));
#endif
}
inline __m128d MulAdd(__m128d a, __m128d b, __m128d acc) {
#ifdef __FMA__
  return _mm_fmadd_pd(a, b, acc);
#else
  return _mm_add_pd(acc, _mm_mul_pd(a, b));
#endif
}

// Number of ``k`` iterations ahead of the current one to prefetch B rows for.
// The packed B panel (``Bpack``, see ``PackBPanel``) can be up to
// ``kGemmTileK * kGemmTileN`` elements -- a few hundred KiB, larger than L1 --
// so streaming through it row by row benefits from hinting the prefetcher a
// handful of iterations ahead to hide L2 latency behind the FMA chain below. A
// is not prefetched: ``Apack`` is only a few KiB (``kGemmMR * kGemmTileK``)
// and stays L1-resident for the whole k loop.
constexpr int kGemmPrefetchDistanceK = 4;

// Issues a T0 (all cache levels) software prefetch hint for ``ptr``. A no-op
// wrapper so call sites don't need to spell out the intrinsic/hint constant.
template <typename T> inline void PrefetchT0(const T *ptr) {
  _mm_prefetch(reinterpret_cast<const char *>(ptr), _MM_HINT_T0);
}

// AVX (256-bit) micro-kernel, NR == 2: processes kGemmMR rows by 16 float
// lanes (two 8-lane vectors) at a time, falling back to an 8-lane loop and
// finally the scalar tail.
void GemmMicroKernel_AVX_F32(std::size_t mr, std::size_t nb, std::size_t K, float alpha, float beta,
                             const float *Bmat, std::size_t N, const float *Crow_base,
                             std::size_t Cstride, float *Yrow_base, std::size_t Ystride,
                             std::size_t n0, GemmAccumMode mode, const float *Apack) {
  const __m256 valpha = _mm256_set1_ps(alpha);
  const __m256 vbeta = _mm256_set1_ps(beta);
  std::size_t n = 0;
  for (; n + 16 <= nb; n += 16) {
    __m256 acc0[kGemmMR];
    __m256 acc1[kGemmMR];
    for (std::size_t r = 0; r < mr; ++r) {
      acc0[r] = _mm256_setzero_ps();
      acc1[r] = _mm256_setzero_ps();
    }
    for (std::size_t k = 0; k < K; ++k) {
      const float *Brow = Bmat + k * N + n0 + n;
      const __m256 vb0 = _mm256_loadu_ps(Brow);
      const __m256 vb1 = _mm256_loadu_ps(Brow + 8);
      if (k + kGemmPrefetchDistanceK < K) {
        PrefetchT0(Brow + kGemmPrefetchDistanceK * N);
      }
      for (std::size_t r = 0; r < mr; ++r) {
        const __m256 va = _mm256_set1_ps(Apack[r * K + k]);
        acc0[r] = MulAdd(va, vb0, acc0[r]);
        acc1[r] = MulAdd(va, vb1, acc1[r]);
      }
    }
    for (std::size_t r = 0; r < mr; ++r) {
      float *Yrow = Yrow_base + r * Ystride + n0 + n;
      __m256 res0 = _mm256_mul_ps(valpha, acc0[r]);
      __m256 res1 = _mm256_mul_ps(valpha, acc1[r]);
      if (mode == GemmAccumMode::kInitBias) {
        const float *Crow = Crow_base + r * Cstride + n0 + n;
        res0 = _mm256_add_ps(res0, _mm256_mul_ps(vbeta, _mm256_loadu_ps(Crow)));
        res1 = _mm256_add_ps(res1, _mm256_mul_ps(vbeta, _mm256_loadu_ps(Crow + 8)));
      } else if (mode == GemmAccumMode::kAccumulate) {
        res0 = _mm256_add_ps(res0, _mm256_loadu_ps(Yrow));
        res1 = _mm256_add_ps(res1, _mm256_loadu_ps(Yrow + 8));
      }
      _mm256_storeu_ps(Yrow, res0);
      _mm256_storeu_ps(Yrow + 8, res1);
    }
  }
  for (; n + 8 <= nb; n += 8) {
    __m256 acc[kGemmMR];
    for (std::size_t r = 0; r < mr; ++r) {
      acc[r] = _mm256_setzero_ps();
    }
    for (std::size_t k = 0; k < K; ++k) {
      const __m256 vb = _mm256_loadu_ps(Bmat + k * N + n0 + n);
      for (std::size_t r = 0; r < mr; ++r) {
        const __m256 va = _mm256_set1_ps(Apack[r * K + k]);
        acc[r] = MulAdd(va, vb, acc[r]);
      }
    }
    for (std::size_t r = 0; r < mr; ++r) {
      float *Yrow = Yrow_base + r * Ystride + n0 + n;
      __m256 res = _mm256_mul_ps(valpha, acc[r]);
      if (mode == GemmAccumMode::kInitBias) {
        const __m256 vc = _mm256_loadu_ps(Crow_base + r * Cstride + n0 + n);
        res = _mm256_add_ps(res, _mm256_mul_ps(vbeta, vc));
      } else if (mode == GemmAccumMode::kAccumulate) {
        res = _mm256_add_ps(res, _mm256_loadu_ps(Yrow));
      }
      _mm256_storeu_ps(Yrow, res);
    }
  }
  if (n < nb) {
    GemmMicroKernel_ScalarImpl<float>(mr, nb - n, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                      Yrow_base, Ystride, n0 + n, mode, Apack);
  }
}

// SSE2 (128-bit) micro-kernel, NR == 2: processes kGemmMR rows by 8 float
// lanes (two 4-lane vectors) at a time.
void GemmMicroKernel_SSE2_F32(std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                              float beta, const float *Bmat, std::size_t N, const float *Crow_base,
                              std::size_t Cstride, float *Yrow_base, std::size_t Ystride,
                              std::size_t n0, GemmAccumMode mode, const float *Apack) {
  const __m128 valpha = _mm_set1_ps(alpha);
  const __m128 vbeta = _mm_set1_ps(beta);
  std::size_t n = 0;
  for (; n + 8 <= nb; n += 8) {
    __m128 acc0[kGemmMR];
    __m128 acc1[kGemmMR];
    for (std::size_t r = 0; r < mr; ++r) {
      acc0[r] = _mm_setzero_ps();
      acc1[r] = _mm_setzero_ps();
    }
    for (std::size_t k = 0; k < K; ++k) {
      const float *Brow = Bmat + k * N + n0 + n;
      const __m128 vb0 = _mm_loadu_ps(Brow);
      const __m128 vb1 = _mm_loadu_ps(Brow + 4);
      for (std::size_t r = 0; r < mr; ++r) {
        const __m128 va = _mm_set1_ps(Apack[r * K + k]);
        acc0[r] = MulAdd(va, vb0, acc0[r]);
        acc1[r] = MulAdd(va, vb1, acc1[r]);
      }
    }
    for (std::size_t r = 0; r < mr; ++r) {
      float *Yrow = Yrow_base + r * Ystride + n0 + n;
      __m128 res0 = _mm_mul_ps(valpha, acc0[r]);
      __m128 res1 = _mm_mul_ps(valpha, acc1[r]);
      if (mode == GemmAccumMode::kInitBias) {
        const float *Crow = Crow_base + r * Cstride + n0 + n;
        res0 = _mm_add_ps(res0, _mm_mul_ps(vbeta, _mm_loadu_ps(Crow)));
        res1 = _mm_add_ps(res1, _mm_mul_ps(vbeta, _mm_loadu_ps(Crow + 4)));
      } else if (mode == GemmAccumMode::kAccumulate) {
        res0 = _mm_add_ps(res0, _mm_loadu_ps(Yrow));
        res1 = _mm_add_ps(res1, _mm_loadu_ps(Yrow + 4));
      }
      _mm_storeu_ps(Yrow, res0);
      _mm_storeu_ps(Yrow + 4, res1);
    }
  }
  for (; n + 4 <= nb; n += 4) {
    __m128 acc[kGemmMR];
    for (std::size_t r = 0; r < mr; ++r) {
      acc[r] = _mm_setzero_ps();
    }
    for (std::size_t k = 0; k < K; ++k) {
      const __m128 vb = _mm_loadu_ps(Bmat + k * N + n0 + n);
      for (std::size_t r = 0; r < mr; ++r) {
        const __m128 va = _mm_set1_ps(Apack[r * K + k]);
        acc[r] = MulAdd(va, vb, acc[r]);
      }
    }
    for (std::size_t r = 0; r < mr; ++r) {
      float *Yrow = Yrow_base + r * Ystride + n0 + n;
      __m128 res = _mm_mul_ps(valpha, acc[r]);
      if (mode == GemmAccumMode::kInitBias) {
        const __m128 vc = _mm_loadu_ps(Crow_base + r * Cstride + n0 + n);
        res = _mm_add_ps(res, _mm_mul_ps(vbeta, vc));
      } else if (mode == GemmAccumMode::kAccumulate) {
        res = _mm_add_ps(res, _mm_loadu_ps(Yrow));
      }
      _mm_storeu_ps(Yrow, res);
    }
  }
  if (n < nb) {
    GemmMicroKernel_ScalarImpl<float>(mr, nb - n, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                      Yrow_base, Ystride, n0 + n, mode, Apack);
  }
}

// AVX (256-bit) micro-kernel, NR == 2: processes kGemmMR rows by 8 double
// lanes (two 4-lane vectors) at a time.
void GemmMicroKernel_AVX_F64(std::size_t mr, std::size_t nb, std::size_t K, double alpha,
                             double beta, const double *Bmat, std::size_t N,
                             const double *Crow_base, std::size_t Cstride, double *Yrow_base,
                             std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                             const double *Apack) {
  const __m256d valpha = _mm256_set1_pd(alpha);
  const __m256d vbeta = _mm256_set1_pd(beta);
  std::size_t n = 0;
  for (; n + 8 <= nb; n += 8) {
    __m256d acc0[kGemmMR];
    __m256d acc1[kGemmMR];
    for (std::size_t r = 0; r < mr; ++r) {
      acc0[r] = _mm256_setzero_pd();
      acc1[r] = _mm256_setzero_pd();
    }
    for (std::size_t k = 0; k < K; ++k) {
      const double *Brow = Bmat + k * N + n0 + n;
      const __m256d vb0 = _mm256_loadu_pd(Brow);
      const __m256d vb1 = _mm256_loadu_pd(Brow + 4);
      if (k + kGemmPrefetchDistanceK < K) {
        PrefetchT0(Brow + kGemmPrefetchDistanceK * N);
      }
      for (std::size_t r = 0; r < mr; ++r) {
        const __m256d va = _mm256_set1_pd(Apack[r * K + k]);
        acc0[r] = MulAdd(va, vb0, acc0[r]);
        acc1[r] = MulAdd(va, vb1, acc1[r]);
      }
    }
    for (std::size_t r = 0; r < mr; ++r) {
      double *Yrow = Yrow_base + r * Ystride + n0 + n;
      __m256d res0 = _mm256_mul_pd(valpha, acc0[r]);
      __m256d res1 = _mm256_mul_pd(valpha, acc1[r]);
      if (mode == GemmAccumMode::kInitBias) {
        const double *Crow = Crow_base + r * Cstride + n0 + n;
        res0 = _mm256_add_pd(res0, _mm256_mul_pd(vbeta, _mm256_loadu_pd(Crow)));
        res1 = _mm256_add_pd(res1, _mm256_mul_pd(vbeta, _mm256_loadu_pd(Crow + 4)));
      } else if (mode == GemmAccumMode::kAccumulate) {
        res0 = _mm256_add_pd(res0, _mm256_loadu_pd(Yrow));
        res1 = _mm256_add_pd(res1, _mm256_loadu_pd(Yrow + 4));
      }
      _mm256_storeu_pd(Yrow, res0);
      _mm256_storeu_pd(Yrow + 4, res1);
    }
  }
  for (; n + 4 <= nb; n += 4) {
    __m256d acc[kGemmMR];
    for (std::size_t r = 0; r < mr; ++r) {
      acc[r] = _mm256_setzero_pd();
    }
    for (std::size_t k = 0; k < K; ++k) {
      const __m256d vb = _mm256_loadu_pd(Bmat + k * N + n0 + n);
      for (std::size_t r = 0; r < mr; ++r) {
        const __m256d va = _mm256_set1_pd(Apack[r * K + k]);
        acc[r] = MulAdd(va, vb, acc[r]);
      }
    }
    for (std::size_t r = 0; r < mr; ++r) {
      double *Yrow = Yrow_base + r * Ystride + n0 + n;
      __m256d res = _mm256_mul_pd(valpha, acc[r]);
      if (mode == GemmAccumMode::kInitBias) {
        const __m256d vc = _mm256_loadu_pd(Crow_base + r * Cstride + n0 + n);
        res = _mm256_add_pd(res, _mm256_mul_pd(vbeta, vc));
      } else if (mode == GemmAccumMode::kAccumulate) {
        res = _mm256_add_pd(res, _mm256_loadu_pd(Yrow));
      }
      _mm256_storeu_pd(Yrow, res);
    }
  }
  if (n < nb) {
    GemmMicroKernel_ScalarImpl<double>(mr, nb - n, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                       Yrow_base, Ystride, n0 + n, mode, Apack);
  }
}

// SSE2 (128-bit) micro-kernel, NR == 2: processes kGemmMR rows by 4 double
// lanes (two 2-lane vectors) at a time.
void GemmMicroKernel_SSE2_F64(std::size_t mr, std::size_t nb, std::size_t K, double alpha,
                              double beta, const double *Bmat, std::size_t N,
                              const double *Crow_base, std::size_t Cstride, double *Yrow_base,
                              std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                              const double *Apack) {
  const __m128d valpha = _mm_set1_pd(alpha);
  const __m128d vbeta = _mm_set1_pd(beta);
  std::size_t n = 0;
  for (; n + 4 <= nb; n += 4) {
    __m128d acc0[kGemmMR];
    __m128d acc1[kGemmMR];
    for (std::size_t r = 0; r < mr; ++r) {
      acc0[r] = _mm_setzero_pd();
      acc1[r] = _mm_setzero_pd();
    }
    for (std::size_t k = 0; k < K; ++k) {
      const double *Brow = Bmat + k * N + n0 + n;
      const __m128d vb0 = _mm_loadu_pd(Brow);
      const __m128d vb1 = _mm_loadu_pd(Brow + 2);
      for (std::size_t r = 0; r < mr; ++r) {
        const __m128d va = _mm_set1_pd(Apack[r * K + k]);
        acc0[r] = MulAdd(va, vb0, acc0[r]);
        acc1[r] = MulAdd(va, vb1, acc1[r]);
      }
    }
    for (std::size_t r = 0; r < mr; ++r) {
      double *Yrow = Yrow_base + r * Ystride + n0 + n;
      __m128d res0 = _mm_mul_pd(valpha, acc0[r]);
      __m128d res1 = _mm_mul_pd(valpha, acc1[r]);
      if (mode == GemmAccumMode::kInitBias) {
        const double *Crow = Crow_base + r * Cstride + n0 + n;
        res0 = _mm_add_pd(res0, _mm_mul_pd(vbeta, _mm_loadu_pd(Crow)));
        res1 = _mm_add_pd(res1, _mm_mul_pd(vbeta, _mm_loadu_pd(Crow + 2)));
      } else if (mode == GemmAccumMode::kAccumulate) {
        res0 = _mm_add_pd(res0, _mm_loadu_pd(Yrow));
        res1 = _mm_add_pd(res1, _mm_loadu_pd(Yrow + 2));
      }
      _mm_storeu_pd(Yrow, res0);
      _mm_storeu_pd(Yrow + 2, res1);
    }
  }
  for (; n + 2 <= nb; n += 2) {
    __m128d acc[kGemmMR];
    for (std::size_t r = 0; r < mr; ++r) {
      acc[r] = _mm_setzero_pd();
    }
    for (std::size_t k = 0; k < K; ++k) {
      const __m128d vb = _mm_loadu_pd(Bmat + k * N + n0 + n);
      for (std::size_t r = 0; r < mr; ++r) {
        const __m128d va = _mm_set1_pd(Apack[r * K + k]);
        acc[r] = MulAdd(va, vb, acc[r]);
      }
    }
    for (std::size_t r = 0; r < mr; ++r) {
      double *Yrow = Yrow_base + r * Ystride + n0 + n;
      __m128d res = _mm_mul_pd(valpha, acc[r]);
      if (mode == GemmAccumMode::kInitBias) {
        const __m128d vc = _mm_loadu_pd(Crow_base + r * Cstride + n0 + n);
        res = _mm_add_pd(res, _mm_mul_pd(vbeta, vc));
      } else if (mode == GemmAccumMode::kAccumulate) {
        res = _mm_add_pd(res, _mm_loadu_pd(Yrow));
      }
      _mm_storeu_pd(Yrow, res);
    }
  }
  if (n < nb) {
    GemmMicroKernel_ScalarImpl<double>(mr, nb - n, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                       Yrow_base, Ystride, n0 + n, mode, Apack);
  }
}

#endif // ONNX_LIGHT_CPU_X86

// Cache-blocking tile sizes, in elements. The output ``Y`` is processed as a
// grid of ``kGemmTileM x kGemmTileN`` tiles so that, within a column panel, the
// packed ``B`` panel is reused across the rows of a tile while it is still hot
// in cache. The values are a pragmatic default: a column panel of a few hundred
// elements keeps the reused ``B`` rows small, and a modest row block bounds the
// live ``Y`` tile.
//
// ``K`` is additionally blocked into ``kGemmTileK``-sized chunks
// (``GemmAccumMode`` stitches the partial per-chunk results back together).
// Without this, a full ``K x kGemmTileN`` ``B`` panel for a moderately large
// ``K`` (e.g. 4096 for a typical transformer hidden size) would be several
// MiB -- larger than most L2 caches -- so the "keep the B panel hot across the
// row block" reuse the tiling above is meant to provide would be defeated by
// cache eviction. Chunking ``K`` bounds every panel to
// ``kGemmTileK x kGemmTileN`` elements, which comfortably fits in L2 for both
// float32 and float64.
// ParallelFor's grain is calibrated for full element-wise loop iterations.
// A GEMM multiply-add is only one instruction inside a heavily vectorized,
// register-blocked inner loop, so counting every scalar FMA as a full work unit
// wakes workers far too early (for example as soon as M crosses one 64-row
// tile). This divisor keeps sub-million-FMA tiles inline while preserving
// parallelism for genuinely compute-heavy panels.
constexpr double kGemmFmasPerParallelWorkUnit = 256.0;

// Runtime-selected micro-kernel flavor for a given element type.
enum class GemmKernelKind { kScalar, kSSE2, kAVX, kAVX512 };

template <typename T> GemmKernelKind SelectGemmKernelKind() {
#if ONNX_LIGHT_CPU_X86
  static const SimdLevel level = DetectSimdLevel();
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
  if (level >= SimdLevel::kAVX512) {
    return GemmKernelKind::kAVX512;
  }
#endif
  if (level >= SimdLevel::kAVX) {
    return GemmKernelKind::kAVX;
  }
  if (level >= SimdLevel::kSSE2) {
    return GemmKernelKind::kSSE2;
  }
#endif
  return GemmKernelKind::kScalar;
}

// Dispatches the register-blocked micro-kernel matching ``kind`` for float32.
void GemmTileF32(GemmKernelKind kind, std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                 float beta, const float *Bmat, std::size_t N, const float *Crow_base,
                 std::size_t Cstride, float *Yrow_base, std::size_t Ystride, std::size_t n0,
                 GemmAccumMode mode, const float *Apack) {
#if ONNX_LIGHT_CPU_X86
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
  if (kind == GemmKernelKind::kAVX512) {
    GemmMicroKernel_AVX512_F32(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                               Ystride, n0, mode, Apack);
    return;
  }
#endif
  if (kind == GemmKernelKind::kAVX) {
    GemmMicroKernel_AVX_F32(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base, Ystride,
                            n0, mode, Apack);
    return;
  }
  if (kind == GemmKernelKind::kSSE2) {
    GemmMicroKernel_SSE2_F32(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                             Ystride, n0, mode, Apack);
    return;
  }
#else
  (void)kind;
#endif
  GemmMicroKernel_Scalar_F32(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                             Ystride, n0, mode, Apack);
}

// Dispatches the register-blocked micro-kernel matching ``kind`` for float64.
void GemmTileF64(GemmKernelKind kind, std::size_t mr, std::size_t nb, std::size_t K, double alpha,
                 double beta, const double *Bmat, std::size_t N, const double *Crow_base,
                 std::size_t Cstride, double *Yrow_base, std::size_t Ystride, std::size_t n0,
                 GemmAccumMode mode, const double *Apack) {
#if ONNX_LIGHT_CPU_X86
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
  if (kind == GemmKernelKind::kAVX512) {
    GemmMicroKernel_AVX512_F64(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                               Ystride, n0, mode, Apack);
    return;
  }
#endif
  if (kind == GemmKernelKind::kAVX) {
    GemmMicroKernel_AVX_F64(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base, Ystride,
                            n0, mode, Apack);
    return;
  }
  if (kind == GemmKernelKind::kSSE2) {
    GemmMicroKernel_SSE2_F64(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                             Ystride, n0, mode, Apack);
    return;
  }
#else
  (void)kind;
#endif
  GemmMicroKernel_Scalar_F64(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                             Ystride, n0, mode, Apack);
}

// Packs a ``kc x nb`` row-major panel of ``Bmat`` (already resolved to a
// contiguous K x N row-major matrix, see ``GemmImpl``) into a small contiguous
// buffer reused across every row block of ``A`` within a (row-block panel,
// k-chunk): without this, each row block's inner loop would touch an
// ``nb``-wide slice embedded in a stride-``N`` matrix that can span many
// megabytes end to end, so once a task straddles more than one ``kGemmMR``
// row block (i.e. whenever a row block spans more than ``kGemmTileM`` rows)
// repeated re-reads of the same logical columns are not guaranteed to still
// be cache-resident. Packing the panel once per (task, k-chunk) bounds the
// footprint every following row block reuses to
// ``kGemmTileK * kGemmTileN`` elements, comfortably L2-resident, mirroring
// the ``A``-packing rationale above.
template <typename T>
void PackBPanel(const T *Bmat, std::size_t N, std::size_t k0, std::size_t kc, std::size_t n0,
                std::size_t nb, T *Bpack) {
  for (std::size_t k = 0; k < kc; ++k) {
    const T *src = Bmat + (k0 + k) * N + n0;
    std::copy(src, src + nb, Bpack + k * nb);
  }
}

// Packs a ``mr x kc`` row-major panel of ``A`` (or ``A^T`` when ``trans_a``)
// into a contiguous buffer: ``Apack[r * kc + k] == A(m + r, k0 + k)``. This
// replaces a per-(row, k, column-vector) indexed/strided lookup with a single
// linear read per micro-kernel call: without packing, every one of the (up
// to ``kGemmTileN / vector_width``) column-vector iterations that share the
// same ``(row, k)`` element would repeat the ``trans_a`` branch and, when
// ``trans_a`` is set, a strided (stride ``M``) load. When ``!trans_a``, ``A``
// is already contiguous along ``k`` so the pack is a plain memcpy-equivalent.
template <typename T>
void PackARowBlock(bool trans_a, const T *A, std::size_t M, std::size_t K, std::size_t m,
                   std::size_t mr, std::size_t k0, std::size_t kc, T *Apack) {
  for (std::size_t r = 0; r < mr; ++r) {
    const std::size_t row = m + r;
    T *dst = Apack + r * kc;
    if (trans_a) {
      // A is stored as K x M; A(row, k0 + k) == A_storage[(k0 + k) * M + row].
      for (std::size_t k = 0; k < kc; ++k) {
        dst[k] = A[(k0 + k) * M + row];
      }
    } else {
      const T *src = A + row * K + k0;
      std::copy(src, src + kc, dst);
    }
  }
}

// One (row block, column panel) output tile, part of the flattened work list
// below.
struct GemmTask {
  std::size_t m0;
  std::size_t m_end;
  std::size_t n0;
  std::size_t nb;
};

// Generic GEMM driver shared by the float32 and float64 entry points. ``tile``
// dispatches to the register-blocked micro-kernel of the selected SIMD level.
template <typename T, typename TileFn>
void GemmImpl(bool trans_a, bool trans_b, std::size_t M, std::size_t N, std::size_t K, T alpha,
              const T *A, const T *B, T beta, const T *C, T *Y, GemmKernelKind kind, TileFn tile) {
  if (M == 0 || N == 0) {
    return;
  }

  // Ensure B is available as a contiguous K x N (row-major) matrix so the inner
  // micro-kernel reads a contiguous row. When B is not transposed it already is.
  const T *Bmat = B;
  std::vector<T> packed;
  if (trans_b && K != 0) {
    // B is stored as N x K; transpose into K x N.
    packed.resize(K * N);
    for (std::size_t n = 0; n < N; ++n) {
      const T *src = B + n * K;
      for (std::size_t k = 0; k < K; ++k) {
        packed[k * N + n] = src[k];
      }
    }
    Bmat = packed.data();
  }

  const bool has_bias = C != nullptr && beta != T(0);

  // Flatten the output grid into a list of (row block, column panel) tasks
  // instead of only splitting the M dimension. A pure M-split leaves no
  // parallelism at all for the common "skinny" shapes seen in inference
  // (e.g. M == 1 for a single-example matvec, or a small batch): with only a
  // handful of rows, every one of those rows would run on a single thread no
  // matter how wide N is or how many cores are available. Flattening M and N
  // row/column blocks into one task list lets those column panels be
  // distributed across threads too.
  std::vector<GemmTask> tasks;
  const std::size_t num_row_blocks = (M + kGemmTileM - 1) / kGemmTileM;
  const std::size_t num_col_panels = (N + kGemmTileN - 1) / kGemmTileN;
  tasks.reserve(num_row_blocks * num_col_panels);
  for (std::size_t n0 = 0; n0 < N; n0 += kGemmTileN) {
    const std::size_t nb = std::min(kGemmTileN, N - n0);
    for (std::size_t m0 = 0; m0 < M; m0 += kGemmTileM) {
      const std::size_t m_end = std::min(m0 + kGemmTileM, M);
      tasks.push_back(GemmTask{m0, m_end, n0, nb});
    }
  }

  // Average per-task cost converted from scalar multiply-adds to ParallelFor's
  // coarser work units, used to decide how many threads are worth waking.
  const double total_work =
      static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K == 0 ? 1 : K);
  const double cost_per_task =
      total_work / (static_cast<double>(tasks.size()) * kGemmFmasPerParallelWorkUnit);

  ParallelFor(static_cast<std::int64_t>(tasks.size()), cost_per_task,
              [&](std::int64_t begin, std::int64_t end) {
                // Bpack holds one packed kGemmTileK x kGemmTileN panel of B, reused
                // across every A row block of a task for a given k-chunk (see
                // PackBPanel). Sized for the worst case and stack-allocated once per
                // ParallelFor block (not per task/k-chunk/row-block) since its
                // contents are fully overwritten before each use.
                alignas(64) T Bpack[kGemmTileK * kGemmTileN];
                for (std::int64_t t = begin; t < end; ++t) {
                  const GemmTask &task = tasks[static_cast<std::size_t>(t)];
                  // Block K into kGemmTileK-sized chunks so the B panel a chunk reads
                  // (kGemmTileK x nb elements) stays cache-resident; see the comment
                  // above kGemmTileK. Each chunk after the first adds its scaled partial
                  // result into Y instead of overwriting it (GemmAccumMode). The k-chunk
                  // loop is outermost (rather than nested inside the row-block loop) so
                  // the packed B panel below is computed once per (task, k-chunk) and
                  // reused across every row block of A in the task, instead of being
                  // implicitly re-read from Bmat by each row block.
                  std::size_t k0 = 0;
                  bool first_chunk = true;
                  do {
                    const std::size_t kc = std::min(kGemmTileK, K - k0);
                    const GemmAccumMode mode = first_chunk ? (has_bias ? GemmAccumMode::kInitBias
                                                                       : GemmAccumMode::kInitZero)
                                                           : GemmAccumMode::kAccumulate;
                    PackBPanel<T>(Bmat, N, k0, kc, task.n0, task.nb, Bpack);
                    for (std::size_t m = task.m0; m < task.m_end; m += kGemmMR) {
                      const std::size_t mr = std::min(kGemmMR, task.m_end - m);
                      // Bpack is already sliced to [task.n0, task.n0 + task.nb), so
                      // Yrow_base/Crow_base must be pre-offset by task.n0 too and the
                      // tile() call below passes n0 == 0.
                      T *Yrow_base = Y + m * N + task.n0;
                      const T *Crow_base = has_bias ? C + m * N + task.n0 : nullptr;
                      // Pack A(m : m + mr, k0 : k0 + kc) into a small contiguous buffer
                      // once, then reuse it for every column-vector step below instead of
                      // recomputing the trans_a-aware index (and, for trans_a, a strided
                      // load) on each one. kGemmMR * kGemmTileK elements is a few KiB,
                      // trivially stack-allocated and L1-resident.
                      alignas(64) T Apack[kGemmMR * kGemmTileK];
                      PackARowBlock<T>(trans_a, A, M, K, m, mr, k0, kc, Apack);
                      tile(kind, mr, task.nb, kc, alpha, beta, Bpack, task.nb, Crow_base, N,
                           Yrow_base, N, 0, mode, Apack);
                    }
                    k0 += kc;
                    first_chunk = false;
                  } while (k0 < K);
                }
              });
}

} // namespace

void GemmFloat32(bool trans_a, bool trans_b, std::size_t M, std::size_t N, std::size_t K,
                 float alpha, const float *A, const float *B, float beta, const float *C,
                 float *Y) {
  const auto tile = [](GemmKernelKind kind, std::size_t mr, std::size_t nb, std::size_t K,
                       float alpha, float beta, const float *Bmat, std::size_t N,
                       const float *Crow_base, std::size_t Cstride, float *Yrow_base,
                       std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                       const float *Apack) {
    GemmTileF32(kind, mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base, Ystride, n0,
                mode, Apack);
  };
  GemmImpl<float>(trans_a, trans_b, M, N, K, alpha, A, B, beta, C, Y, SelectGemmKernelKind<float>(),
                  tile);
}

void GemmFloat64(bool trans_a, bool trans_b, std::size_t M, std::size_t N, std::size_t K,
                 double alpha, const double *A, const double *B, double beta, const double *C,
                 double *Y) {
  const auto tile = [](GemmKernelKind kind, std::size_t mr, std::size_t nb, std::size_t K,
                       double alpha, double beta, const double *Bmat, std::size_t N,
                       const double *Crow_base, std::size_t Cstride, double *Yrow_base,
                       std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                       const double *Apack) {
    GemmTileF64(kind, mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base, Ystride, n0,
                mode, Apack);
  };
  GemmImpl<double>(trans_a, trans_b, M, N, K, alpha, A, B, beta, C, Y,
                   SelectGemmKernelKind<double>(), tile);
}

} // namespace onnx_light_cpu
