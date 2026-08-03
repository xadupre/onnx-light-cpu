// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Fast general matrix multiplication (ONNX ``Gemm``) kernels for float32 and
// float64.
//
// The kernel computes ``Y = alpha * op(A) @ op(B) + beta * C`` using the classic
// rank-1 update (AXPY) formulation: for every output row ``m`` the row is
// initialized with ``beta * C[m]`` (or zero) and then, for every ``k``, the
// scaled ``B`` row ``alpha * A(m, k) * B[k, :]`` is added to it. The inner
// ``Y_row += a * B_row`` update is memory friendly (both operands are contiguous
// over ``N``) and vectorizes cleanly with runtime AVX-512/AVX/SSE2 dispatch.
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
// output row. The tiling only changes the iteration order, not the order in
// which each output element accumulates over ``k``, so the result stays
// bit-exact.

#include "onnx_light_cpu/impl/math/math_kernels.h"

#include "onnx_light_cpu/impl/parallel_for.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
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
// AXPY inner kernels: y[i] += a * x[i] for i in [0, n).
// ---------------------------------------------------------------------------

void AxpyF32_Scalar(float a, const float *x, float *y, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    y[i] += a * x[i];
  }
}

void AxpyF64_Scalar(double a, const double *x, double *y, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    y[i] += a * x[i];
  }
}

#if ONNX_LIGHT_CPU_X86

void AxpyF32_SSE2(float a, const float *x, float *y, std::size_t n) {
  const __m128 va = _mm_set1_ps(a);
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    __m128 vy = _mm_loadu_ps(y + i);
    vy = _mm_add_ps(vy, _mm_mul_ps(va, _mm_loadu_ps(x + i)));
    _mm_storeu_ps(y + i, vy);
  }
  for (; i < n; ++i) {
    y[i] += a * x[i];
  }
}

void AxpyF64_SSE2(double a, const double *x, double *y, std::size_t n) {
  const __m128d va = _mm_set1_pd(a);
  std::size_t i = 0;
  for (; i + 2 <= n; i += 2) {
    __m128d vy = _mm_loadu_pd(y + i);
    vy = _mm_add_pd(vy, _mm_mul_pd(va, _mm_loadu_pd(x + i)));
    _mm_storeu_pd(y + i, vy);
  }
  for (; i < n; ++i) {
    y[i] += a * x[i];
  }
}

void AxpyF32_AVX(float a, const float *x, float *y, std::size_t n) {
  const __m256 va = _mm256_set1_ps(a);
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 vy = _mm256_loadu_ps(y + i);
    vy = _mm256_add_ps(vy, _mm256_mul_ps(va, _mm256_loadu_ps(x + i)));
    _mm256_storeu_ps(y + i, vy);
  }
  for (; i < n; ++i) {
    y[i] += a * x[i];
  }
}

void AxpyF64_AVX(double a, const double *x, double *y, std::size_t n) {
  const __m256d va = _mm256_set1_pd(a);
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    __m256d vy = _mm256_loadu_pd(y + i);
    vy = _mm256_add_pd(vy, _mm256_mul_pd(va, _mm256_loadu_pd(x + i)));
    _mm256_storeu_pd(y + i, vy);
  }
  for (; i < n; ++i) {
    y[i] += a * x[i];
  }
}

#ifdef __AVX512F__
void AxpyF32_AVX512(float a, const float *x, float *y, std::size_t n) {
  const __m512 va = _mm512_set1_ps(a);
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m512 vy = _mm512_loadu_ps(y + i);
    vy = _mm512_add_ps(vy, _mm512_mul_ps(va, _mm512_loadu_ps(x + i)));
    _mm512_storeu_ps(y + i, vy);
  }
  for (; i < n; ++i) {
    y[i] += a * x[i];
  }
}

void AxpyF64_AVX512(double a, const double *x, double *y, std::size_t n) {
  const __m512d va = _mm512_set1_pd(a);
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m512d vy = _mm512_loadu_pd(y + i);
    vy = _mm512_add_pd(vy, _mm512_mul_pd(va, _mm512_loadu_pd(x + i)));
    _mm512_storeu_pd(y + i, vy);
  }
  for (; i < n; ++i) {
    y[i] += a * x[i];
  }
}
#endif // __AVX512F__

#endif // ONNX_LIGHT_CPU_X86

using AxpyF32Fn = void (*)(float, const float *, float *, std::size_t);
using AxpyF64Fn = void (*)(double, const double *, double *, std::size_t);

AxpyF32Fn SelectAxpyF32() {
#if ONNX_LIGHT_CPU_X86
  static const SimdLevel level = DetectSimdLevel();
#ifdef __AVX512F__
  if (level >= SimdLevel::kAVX512) {
    return &AxpyF32_AVX512;
  }
#endif
  if (level >= SimdLevel::kAVX) {
    return &AxpyF32_AVX;
  }
  if (level >= SimdLevel::kSSE2) {
    return &AxpyF32_SSE2;
  }
#endif
  return &AxpyF32_Scalar;
}

AxpyF64Fn SelectAxpyF64() {
#if ONNX_LIGHT_CPU_X86
  static const SimdLevel level = DetectSimdLevel();
#ifdef __AVX512F__
  if (level >= SimdLevel::kAVX512) {
    return &AxpyF64_AVX512;
  }
#endif
  if (level >= SimdLevel::kAVX) {
    return &AxpyF64_AVX;
  }
  if (level >= SimdLevel::kSSE2) {
    return &AxpyF64_SSE2;
  }
#endif
  return &AxpyF64_Scalar;
}

// Cache-blocking tile sizes, in elements. The output ``Y`` is processed as a
// grid of ``kGemmTileM x kGemmTileN`` tiles so that, within a column panel, the
// packed ``B`` panel is reused across the rows of a tile while it is still hot
// in cache. The values are a pragmatic default: a column panel of a few hundred
// elements keeps the reused ``B`` rows small, and a modest row block bounds the
// live ``Y`` tile.
constexpr std::size_t kGemmTileN = 256;
constexpr std::size_t kGemmTileM = 64;

// Generic GEMM driver shared by the float32 and float64 entry points.
template <typename T, typename AxpyFn>
void GemmImpl(bool trans_a, bool trans_b, std::size_t M, std::size_t N, std::size_t K, T alpha,
              const T *A, const T *B, T beta, const T *C, T *Y, AxpyFn axpy) {
  if (M == 0 || N == 0) {
    return;
  }

  // Ensure B is available as a contiguous K x N (row-major) matrix so the inner
  // AXPY reads a contiguous row. When B is not transposed it already is.
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

  // The per-row work is roughly N*K rank-1 updates; pass that as the per-element
  // cost so even a handful of large rows is parallelized, while tiny problems
  // stay on the calling thread.
  const double cost = static_cast<double>(N) * static_cast<double>(K == 0 ? 1 : K);

  ParallelFor(static_cast<std::int64_t>(M), cost, [&](std::int64_t begin, std::int64_t end) {
    const std::size_t row_begin = static_cast<std::size_t>(begin);
    const std::size_t row_end = static_cast<std::size_t>(end);
    // Walk the [row_begin, row_end) x N output slice as a grid of tiles. For a
    // fixed column panel the packed B panel (K x nb) is reused across every row
    // of the row block, keeping it in cache.
    for (std::size_t n0 = 0; n0 < N; n0 += kGemmTileN) {
      const std::size_t nb = std::min(kGemmTileN, N - n0);
      for (std::size_t m0 = row_begin; m0 < row_end; m0 += kGemmTileM) {
        const std::size_t m_end = std::min(m0 + kGemmTileM, row_end);
        for (std::size_t m = m0; m < m_end; ++m) {
          T *Yrow = Y + m * N + n0;
          if (has_bias) {
            const T *Crow = C + m * N + n0;
            for (std::size_t n = 0; n < nb; ++n) {
              Yrow[n] = beta * Crow[n];
            }
          } else {
            std::memset(Yrow, 0, nb * sizeof(T));
          }
          for (std::size_t k = 0; k < K; ++k) {
            const T a = alpha * (trans_a ? A[k * M + m] : A[m * K + k]);
            axpy(a, Bmat + k * N + n0, Yrow, nb);
          }
        }
      }
    }
  });
}

} // namespace

void GemmFloat32(bool trans_a, bool trans_b, std::size_t M, std::size_t N, std::size_t K,
                 float alpha, const float *A, const float *B, float beta, const float *C,
                 float *Y) {
  GemmImpl<float>(trans_a, trans_b, M, N, K, alpha, A, B, beta, C, Y, SelectAxpyF32());
}

void GemmFloat64(bool trans_a, bool trans_b, std::size_t M, std::size_t N, std::size_t K,
                 double alpha, const double *A, const double *B, double beta, const double *C,
                 double *Y) {
  GemmImpl<double>(trans_a, trans_b, M, N, K, alpha, A, B, beta, C, Y, SelectAxpyF64());
}

} // namespace onnx_light_cpu
