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
// tile of ``MR`` output rows by one SIMD vector of columns resident in CPU
// registers while it accumulates the whole ``k`` reduction. For each ``k`` a
// single ``B`` row vector is loaded once and reused across all ``MR`` rows: each
// row broadcasts its scalar ``A(m, k)`` and fuses it into its accumulator. This
// turns the hot loop into register-resident multiply-adds with an arithmetic
// intensity high enough to be compute bound, closing most of the gap with
// optimized BLAS-style GEMM.
//
// To keep a single fast inner kernel, a transposed ``B`` (``trans_b``) is packed
// into a contiguous ``K x N`` buffer first; this pre-pass is ``O(K*N)`` and
// negligible next to the ``O(M*N*K)`` multiplication. ``A`` is only read one
// scalar at a time, so ``trans_a`` is handled by index arithmetic without any
// copy.
//
// For cache efficiency the output ``Y`` is walked as a grid of ``MB x NB`` tiles
// instead of one full row at a time. For every column panel of width ``NB`` the
// packed ``B`` panel (``K x NB``) is reused across all rows of an ``MB`` row
// block, so it stays resident in cache rather than being streamed once per
// output row.

#include "onnx_light_cpu/impl/math/math_kernels.h"

#include "onnx_light_cpu/impl/parallel_for.h"

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
// Each micro-kernel computes a tile of ``MR`` output rows by ``nb`` columns:
//
//     acc[r][:] = sum_k A(m0 + r, k) * Bmat[k, n0 : n0 + nb]
//
// accumulating the whole ``k`` reduction in registers before it is scaled by
// ``alpha``, combined with the optional ``beta * C`` bias and written to ``Y``.
// ``a_at(r, k)`` returns ``A(m0 + r, k)`` accounting for ``trans_a``. The B row
// ``Bmat + k * N + n0`` is loaded once per ``k`` and reused across all ``MR``
// rows.
//
// ``MR`` is the number of rows processed together (register blocking on M). A
// larger ``MR`` amortizes each B load over more FMAs but needs more live
// accumulator registers; 4 is a good default for both AVX (16 YMM registers)
// and SSE2.
constexpr std::size_t kMR = 4;

// Scalar reference micro-kernel (also the tail handler and non-x86 fallback).
template <typename T, typename AAt>
void GemmMicroKernel_Scalar(std::size_t mr, std::size_t nb, std::size_t K, T alpha, T beta,
                            const T *Bmat, std::size_t N, const T *Crow_base, std::size_t Cstride,
                            T *Yrow_base, std::size_t Ystride, std::size_t n0, bool has_bias,
                            AAt a_at) {
  for (std::size_t r = 0; r < mr; ++r) {
    T *Yrow = Yrow_base + r * Ystride;
    // Initialize with the scaled bias (or zero) then accumulate alpha * A * B.
    // This path only runs for the scalar fallback and the (small) SIMD column
    // tail, so it avoids any temporary allocation.
    if (has_bias) {
      const T *Crow = Crow_base + r * Cstride + n0;
      for (std::size_t n = 0; n < nb; ++n) {
        Yrow[n] = beta * Crow[n];
      }
    } else {
      for (std::size_t n = 0; n < nb; ++n) {
        Yrow[n] = T(0);
      }
    }
    for (std::size_t k = 0; k < K; ++k) {
      const T a = alpha * a_at(r, k);
      const T *Brow = Bmat + k * N + n0;
      for (std::size_t n = 0; n < nb; ++n) {
        Yrow[n] += a * Brow[n];
      }
    }
  }
}

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

// AVX (256-bit) micro-kernel processing ``kMR`` rows by 8 float lanes at a time.
template <typename AAt>
void GemmMicroKernel_AVX_F32(std::size_t mr, std::size_t nb, std::size_t K, float alpha, float beta,
                             const float *Bmat, std::size_t N, const float *Crow_base,
                             std::size_t Cstride, float *Yrow_base, std::size_t Ystride,
                             std::size_t n0, bool has_bias, AAt a_at) {
  std::size_t n = 0;
  for (; n + 8 <= nb; n += 8) {
    __m256 acc[kMR];
    for (std::size_t r = 0; r < mr; ++r) {
      acc[r] = _mm256_setzero_ps();
    }
    for (std::size_t k = 0; k < K; ++k) {
      const __m256 vb = _mm256_loadu_ps(Bmat + k * N + n0 + n);
      for (std::size_t r = 0; r < mr; ++r) {
        const __m256 va = _mm256_set1_ps(a_at(r, k));
        acc[r] = MulAdd(va, vb, acc[r]);
      }
    }
    const __m256 valpha = _mm256_set1_ps(alpha);
    const __m256 vbeta = _mm256_set1_ps(beta);
    for (std::size_t r = 0; r < mr; ++r) {
      __m256 res = _mm256_mul_ps(valpha, acc[r]);
      float *Yrow = Yrow_base + r * Ystride + n;
      if (has_bias) {
        const __m256 vc = _mm256_loadu_ps(Crow_base + r * Cstride + n0 + n);
        res = _mm256_add_ps(res, _mm256_mul_ps(vbeta, vc));
      }
      _mm256_storeu_ps(Yrow, res);
    }
  }
  if (n < nb) {
    GemmMicroKernel_Scalar<float>(mr, nb - n, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                  Yrow_base + n, Ystride, n0 + n, has_bias, a_at);
  }
}

// SSE2 (128-bit) micro-kernel processing ``kMR`` rows by 4 float lanes at a time.
template <typename AAt>
void GemmMicroKernel_SSE2_F32(std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                              float beta, const float *Bmat, std::size_t N, const float *Crow_base,
                              std::size_t Cstride, float *Yrow_base, std::size_t Ystride,
                              std::size_t n0, bool has_bias, AAt a_at) {
  std::size_t n = 0;
  for (; n + 4 <= nb; n += 4) {
    __m128 acc[kMR];
    for (std::size_t r = 0; r < mr; ++r) {
      acc[r] = _mm_setzero_ps();
    }
    for (std::size_t k = 0; k < K; ++k) {
      const __m128 vb = _mm_loadu_ps(Bmat + k * N + n0 + n);
      for (std::size_t r = 0; r < mr; ++r) {
        const __m128 va = _mm_set1_ps(a_at(r, k));
        acc[r] = MulAdd(va, vb, acc[r]);
      }
    }
    const __m128 valpha = _mm_set1_ps(alpha);
    const __m128 vbeta = _mm_set1_ps(beta);
    for (std::size_t r = 0; r < mr; ++r) {
      __m128 res = _mm_mul_ps(valpha, acc[r]);
      float *Yrow = Yrow_base + r * Ystride + n;
      if (has_bias) {
        const __m128 vc = _mm_loadu_ps(Crow_base + r * Cstride + n0 + n);
        res = _mm_add_ps(res, _mm_mul_ps(vbeta, vc));
      }
      _mm_storeu_ps(Yrow, res);
    }
  }
  if (n < nb) {
    GemmMicroKernel_Scalar<float>(mr, nb - n, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                  Yrow_base + n, Ystride, n0 + n, has_bias, a_at);
  }
}

// AVX (256-bit) micro-kernel processing ``kMR`` rows by 4 double lanes at a time.
template <typename AAt>
void GemmMicroKernel_AVX_F64(std::size_t mr, std::size_t nb, std::size_t K, double alpha,
                             double beta, const double *Bmat, std::size_t N,
                             const double *Crow_base, std::size_t Cstride, double *Yrow_base,
                             std::size_t Ystride, std::size_t n0, bool has_bias, AAt a_at) {
  std::size_t n = 0;
  for (; n + 4 <= nb; n += 4) {
    __m256d acc[kMR];
    for (std::size_t r = 0; r < mr; ++r) {
      acc[r] = _mm256_setzero_pd();
    }
    for (std::size_t k = 0; k < K; ++k) {
      const __m256d vb = _mm256_loadu_pd(Bmat + k * N + n0 + n);
      for (std::size_t r = 0; r < mr; ++r) {
        const __m256d va = _mm256_set1_pd(a_at(r, k));
        acc[r] = MulAdd(va, vb, acc[r]);
      }
    }
    const __m256d valpha = _mm256_set1_pd(alpha);
    const __m256d vbeta = _mm256_set1_pd(beta);
    for (std::size_t r = 0; r < mr; ++r) {
      __m256d res = _mm256_mul_pd(valpha, acc[r]);
      double *Yrow = Yrow_base + r * Ystride + n;
      if (has_bias) {
        const __m256d vc = _mm256_loadu_pd(Crow_base + r * Cstride + n0 + n);
        res = _mm256_add_pd(res, _mm256_mul_pd(vbeta, vc));
      }
      _mm256_storeu_pd(Yrow, res);
    }
  }
  if (n < nb) {
    GemmMicroKernel_Scalar<double>(mr, nb - n, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                   Yrow_base + n, Ystride, n0 + n, has_bias, a_at);
  }
}

// SSE2 (128-bit) micro-kernel processing ``kMR`` rows by 2 double lanes at a time.
template <typename AAt>
void GemmMicroKernel_SSE2_F64(std::size_t mr, std::size_t nb, std::size_t K, double alpha,
                              double beta, const double *Bmat, std::size_t N,
                              const double *Crow_base, std::size_t Cstride, double *Yrow_base,
                              std::size_t Ystride, std::size_t n0, bool has_bias, AAt a_at) {
  std::size_t n = 0;
  for (; n + 2 <= nb; n += 2) {
    __m128d acc[kMR];
    for (std::size_t r = 0; r < mr; ++r) {
      acc[r] = _mm_setzero_pd();
    }
    for (std::size_t k = 0; k < K; ++k) {
      const __m128d vb = _mm_loadu_pd(Bmat + k * N + n0 + n);
      for (std::size_t r = 0; r < mr; ++r) {
        const __m128d va = _mm_set1_pd(a_at(r, k));
        acc[r] = MulAdd(va, vb, acc[r]);
      }
    }
    const __m128d valpha = _mm_set1_pd(alpha);
    const __m128d vbeta = _mm_set1_pd(beta);
    for (std::size_t r = 0; r < mr; ++r) {
      __m128d res = _mm_mul_pd(valpha, acc[r]);
      double *Yrow = Yrow_base + r * Ystride + n;
      if (has_bias) {
        const __m128d vc = _mm_loadu_pd(Crow_base + r * Cstride + n0 + n);
        res = _mm_add_pd(res, _mm_mul_pd(vbeta, vc));
      }
      _mm_storeu_pd(Yrow, res);
    }
  }
  if (n < nb) {
    GemmMicroKernel_Scalar<double>(mr, nb - n, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                   Yrow_base + n, Ystride, n0 + n, has_bias, a_at);
  }
}

#endif // ONNX_LIGHT_CPU_X86

// Cache-blocking tile sizes, in elements. The output ``Y`` is processed as a
// grid of ``kGemmTileM x kGemmTileN`` tiles so that, within a column panel, the
// packed ``B`` panel is reused across the rows of a tile while it is still hot
// in cache. The values are a pragmatic default: a column panel of a few hundred
// elements keeps the reused ``B`` rows small, and a modest row block bounds the
// live ``Y`` tile.
constexpr std::size_t kGemmTileN = 256;
constexpr std::size_t kGemmTileM = 64;

// Runtime-selected micro-kernel flavor for a given element type.
enum class GemmKernelKind { kScalar, kSSE2, kAVX };

template <typename T> GemmKernelKind SelectGemmKernelKind() {
#if ONNX_LIGHT_CPU_X86
  static const SimdLevel level = DetectSimdLevel();
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
template <typename AAt>
void GemmTileF32(GemmKernelKind kind, std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                 float beta, const float *Bmat, std::size_t N, const float *Crow_base,
                 std::size_t Cstride, float *Yrow_base, std::size_t Ystride, std::size_t n0,
                 bool has_bias, AAt a_at) {
#if ONNX_LIGHT_CPU_X86
  if (kind == GemmKernelKind::kAVX) {
    GemmMicroKernel_AVX_F32(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base, Ystride,
                            n0, has_bias, a_at);
    return;
  }
  if (kind == GemmKernelKind::kSSE2) {
    GemmMicroKernel_SSE2_F32(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                             Ystride, n0, has_bias, a_at);
    return;
  }
#else
  (void)kind;
#endif
  GemmMicroKernel_Scalar<float>(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                                Ystride, n0, has_bias, a_at);
}

// Dispatches the register-blocked micro-kernel matching ``kind`` for float64.
template <typename AAt>
void GemmTileF64(GemmKernelKind kind, std::size_t mr, std::size_t nb, std::size_t K, double alpha,
                 double beta, const double *Bmat, std::size_t N, const double *Crow_base,
                 std::size_t Cstride, double *Yrow_base, std::size_t Ystride, std::size_t n0,
                 bool has_bias, AAt a_at) {
#if ONNX_LIGHT_CPU_X86
  if (kind == GemmKernelKind::kAVX) {
    GemmMicroKernel_AVX_F64(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base, Ystride,
                            n0, has_bias, a_at);
    return;
  }
  if (kind == GemmKernelKind::kSSE2) {
    GemmMicroKernel_SSE2_F64(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                             Ystride, n0, has_bias, a_at);
    return;
  }
#else
  (void)kind;
#endif
  GemmMicroKernel_Scalar<double>(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                                 Ystride, n0, has_bias, a_at);
}

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

  // The per-row work is roughly N*K multiply-adds; pass that as the per-element
  // cost so even a handful of large rows is parallelized, while tiny problems
  // stay on the calling thread.
  const double cost = static_cast<double>(N) * static_cast<double>(K == 0 ? 1 : K);

  ParallelFor(static_cast<std::int64_t>(M), cost, [&](std::int64_t begin, std::int64_t end) {
    const std::size_t row_begin = static_cast<std::size_t>(begin);
    const std::size_t row_end = static_cast<std::size_t>(end);
    // Walk the [row_begin, row_end) x N output slice as a grid of tiles. For a
    // fixed column panel the packed B panel (K x nb) is reused across every row
    // block, keeping it in cache; within a row block the micro-kernel keeps the
    // MR x vector output tile in registers across the whole k reduction.
    for (std::size_t n0 = 0; n0 < N; n0 += kGemmTileN) {
      const std::size_t nb = std::min(kGemmTileN, N - n0);
      for (std::size_t m0 = row_begin; m0 < row_end; m0 += kGemmTileM) {
        const std::size_t m_end = std::min(m0 + kGemmTileM, row_end);
        for (std::size_t m = m0; m < m_end; m += kMR) {
          const std::size_t mr = std::min(kMR, m_end - m);
          // a_at(r, k) returns A(m + r, k), honoring trans_a without copying A.
          const auto a_at = [&](std::size_t r, std::size_t k) -> T {
            const std::size_t row = m + r;
            return trans_a ? A[k * M + row] : A[row * K + k];
          };
          T *Yrow_base = Y + m * N;
          const T *Crow_base = has_bias ? C + m * N : nullptr;
          tile(kind, mr, nb, K, alpha, beta, Bmat, N, Crow_base, N, Yrow_base, N, n0, has_bias,
               a_at);
        }
      }
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
                       std::size_t Ystride, std::size_t n0, bool has_bias, const auto &a_at) {
    GemmTileF32(kind, mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base, Ystride, n0,
                has_bias, a_at);
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
                       std::size_t Ystride, std::size_t n0, bool has_bias, const auto &a_at) {
    GemmTileF64(kind, mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base, Ystride, n0,
                has_bias, a_at);
  };
  GemmImpl<double>(trans_a, trans_b, M, N, K, alpha, A, B, beta, C, Y,
                   SelectGemmKernelKind<double>(), tile);
}

} // namespace onnx_light_cpu
