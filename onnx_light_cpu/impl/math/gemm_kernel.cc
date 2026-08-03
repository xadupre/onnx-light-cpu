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

#include "onnx_light_cpu/impl/math/math_kernels.h"

#include "onnx_light_cpu/impl/parallel_for.h"

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
    for (std::int64_t m = begin; m < end; ++m) {
      T *Yrow = Y + static_cast<std::size_t>(m) * N;
      if (has_bias) {
        const T *Crow = C + static_cast<std::size_t>(m) * N;
        for (std::size_t n = 0; n < N; ++n) {
          Yrow[n] = beta * Crow[n];
        }
      } else {
        std::memset(Yrow, 0, N * sizeof(T));
      }
      for (std::size_t k = 0; k < K; ++k) {
        const T a = alpha * (trans_a ? A[k * M + static_cast<std::size_t>(m)]
                                     : A[static_cast<std::size_t>(m) * K + k]);
        axpy(a, Bmat + k * N, Yrow, N);
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
