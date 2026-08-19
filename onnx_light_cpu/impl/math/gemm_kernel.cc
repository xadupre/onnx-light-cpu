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
// tile of ``blocking.mr`` output rows by one or two SIMD vectors of columns
// resident in CPU registers while it accumulates a chunk of the ``k``
// reduction. For each ``k`` a ``B`` row vector is loaded once and reused
// across all register-blocked rows: each row broadcasts its scalar ``A(m, k)`` and
// fuses it into its accumulator. This turns the hot loop into
// register-resident multiply-adds with an arithmetic intensity high enough to
// be compute bound, closing most of the gap with optimized BLAS-style GEMM.
//
// To keep a single fast inner kernel, a transposed ``B`` (``trans_b``) is packed
// into a contiguous ``K x N`` buffer first; this pre-pass is ``O(K*N)`` and
// negligible next to the ``O(M*N*K)`` multiplication. ``A`` is packed into a
// small contiguous ``mr x kc`` row-panel per (row block, k-chunk) by
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

#include "onnx_light_cpu/impl/math/gemm/arm/gemm_kernel_arm.h"
#include "onnx_light_cpu/impl/math/gemm/float8/float8_conversion.h"
#include "onnx_light_cpu/impl/math/gemm/gemm_common.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/impl/parallel_for.h"

#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
#include "onnx_light_cpu/impl/math/gemm/avx2/gemm_kernel_avx2_fma.h"
#endif

#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
#include "onnx_light_cpu/impl/math/gemm/avx512/gemm_kernel_avx512.h"
#endif

#ifdef ONNX_LIGHT_CPU_HAVE_AVX512FP16
#include "onnx_light_cpu/impl/math/gemm/avx512fp16/gemm_kernel_avx512fp16.h"
#endif

#ifdef ONNX_LIGHT_CPU_HAVE_AVX512BF16
#include "onnx_light_cpu/impl/math/gemm/avx512bf16/gemm_kernel_avx512bf16.h"
#endif

#ifdef ONNX_LIGHT_CPU_HAVE_AMX_BF16
#include "onnx_light_cpu/impl/math/gemm/amx/gemm_amx_bf16.h"
#include "onnx_light_cpu/impl/math/gemm/amx/gemm_amx_tile.h"
#endif

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define ONNX_LIGHT_CPU_X86 1
#include <immintrin.h>
#else
#define ONNX_LIGHT_CPU_X86 0
#endif

namespace onnx_light_cpu {

namespace {

template <typename T, std::size_t Alignment> struct AlignedAllocator {
  using value_type = T;

  AlignedAllocator() = default;
  template <typename U> constexpr AlignedAllocator(const AlignedAllocator<U, Alignment> &) {}

  [[nodiscard]] T *allocate(std::size_t count) {
    return static_cast<T *>(::operator new(count * sizeof(T), std::align_val_t{Alignment}));
  }

  void deallocate(T *pointer, std::size_t) noexcept {
    ::operator delete(pointer, std::align_val_t{Alignment});
  }

  template <typename U> struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };
};

template <typename T, typename U, std::size_t Alignment>
bool operator==(const AlignedAllocator<T, Alignment> &, const AlignedAllocator<U, Alignment> &) {
  return true;
}

template <typename T, typename U, std::size_t Alignment>
bool operator!=(const AlignedAllocator<T, Alignment> &, const AlignedAllocator<U, Alignment> &) {
  return false;
}

template <typename T> using AlignedVector = std::vector<T, AlignedAllocator<T, 64>>;

std::size_t AlignUp(std::size_t value, std::size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

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
  const bool alpha_is_one = alpha == T(1);
  const bool beta_is_one = beta == T(1);
  for (std::size_t r = 0; r < mr; ++r) {
    T *Yrow = Yrow_base + r * Ystride + n0;
    // Initialize with the scaled bias (or zero) for the first chunk; a later
    // chunk (kAccumulate) leaves ``Y`` as-is and only adds to it below. This
    // path only runs for the scalar fallback and the (small) SIMD column tail,
    // so it avoids any temporary allocation.
    if (mode == GemmAccumMode::kInitBias) {
      const T *Crow = Crow_base + r * Cstride + n0;
      for (std::size_t n = 0; n < nb; ++n) {
        Yrow[n] = beta_is_one ? Crow[n] : beta * Crow[n];
      }
    } else if (mode == GemmAccumMode::kInitZero) {
      for (std::size_t n = 0; n < nb; ++n) {
        Yrow[n] = T(0);
      }
    }
    const T *Apack_r = Apack + r * K;
    for (std::size_t k = 0; k < K; ++k) {
      const T a = alpha_is_one ? Apack_r[k] : alpha * Apack_r[k];
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

// Portable scalar member of the native FLOAT16 micro-kernel family (Roadmap
// PR07.3): also the column-tail handler for the AVX-512FP16 kernel. Both
// operands are raw FLOAT16 patterns converted to float32 on access, so the
// float32 accumulation matches the widen-then-float32 reference. Unlike the
// float32 scalar kernel, ``Bmat`` and ``Apack`` are ``std::uint16_t`` FLOAT16
// panels while the ``C``/``Y`` epilogue stays in float32.
void GemmMicroKernel_ScalarFp16(std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                                float beta, const std::uint16_t *Bmat, std::size_t N,
                                const float *Crow_base, std::size_t Cstride, float *Yrow_base,
                                std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                                const std::uint16_t *Apack) {
  const bool alpha_is_one = alpha == 1.0f;
  const bool beta_is_one = beta == 1.0f;
  for (std::size_t r = 0; r < mr; ++r) {
    float *Yrow = Yrow_base + r * Ystride + n0;
    if (mode == GemmAccumMode::kInitBias) {
      const float *Crow = Crow_base + r * Cstride + n0;
      for (std::size_t n = 0; n < nb; ++n) {
        Yrow[n] = beta_is_one ? Crow[n] : beta * Crow[n];
      }
    } else if (mode == GemmAccumMode::kInitZero) {
      for (std::size_t n = 0; n < nb; ++n) {
        Yrow[n] = 0.0f;
      }
    }
    const std::uint16_t *Apack_r = Apack + r * K;
    for (std::size_t k = 0; k < K; ++k) {
      const float a_raw = detail::Float16BitsToFloat(Apack_r[k]);
      const float a = alpha_is_one ? a_raw : alpha * a_raw;
      const std::uint16_t *Brow = Bmat + k * N + n0;
      for (std::size_t n = 0; n < nb; ++n) {
        Yrow[n] += a * detail::Float16BitsToFloat(Brow[n]);
      }
    }
  }
}

// Portable scalar member of the native BFLOAT16 micro-kernel family (Roadmap
// PR07.4): also the column-tail handler for the AVX-512BF16 kernel. Both
// operands are raw BFLOAT16 patterns converted to float32 on access, so the
// float32 accumulation matches the widen-then-float32 reference. Unlike the
// float32 scalar kernel, ``Bmat`` and ``Apack`` are ``std::uint16_t`` BFLOAT16
// panels while the ``C``/``Y`` epilogue stays in float32.
void GemmMicroKernel_ScalarBf16(std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                                float beta, const std::uint16_t *Bmat, std::size_t N,
                                const float *Crow_base, std::size_t Cstride, float *Yrow_base,
                                std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                                const std::uint16_t *Apack) {
  const bool alpha_is_one = alpha == 1.0f;
  const bool beta_is_one = beta == 1.0f;
  for (std::size_t r = 0; r < mr; ++r) {
    float *Yrow = Yrow_base + r * Ystride + n0;
    if (mode == GemmAccumMode::kInitBias) {
      const float *Crow = Crow_base + r * Cstride + n0;
      for (std::size_t n = 0; n < nb; ++n) {
        Yrow[n] = beta_is_one ? Crow[n] : beta * Crow[n];
      }
    } else if (mode == GemmAccumMode::kInitZero) {
      for (std::size_t n = 0; n < nb; ++n) {
        Yrow[n] = 0.0f;
      }
    }
    const std::uint16_t *Apack_r = Apack + r * K;
    for (std::size_t k = 0; k < K; ++k) {
      const float a_raw = detail::Bfloat16BitsToFloat(Apack_r[k]);
      const float a = alpha_is_one ? a_raw : alpha * a_raw;
      const std::uint16_t *Brow = Bmat + k * N + n0;
      for (std::size_t n = 0; n < nb; ++n) {
        Yrow[n] += a * detail::Bfloat16BitsToFloat(Brow[n]);
      }
    }
  }
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
// The tile loops walk one contiguous ``kc x column_block`` micro-panel of the
// packed B panel at a time (see ``PackBPanel``), which is still tens of KiB --
// larger than L1 -- so streaming through it row by row benefits from hinting
// the prefetcher a handful of iterations ahead to hide L2 latency behind the
// FMA chain below. A is not prefetched: ``Apack`` is only a few KiB
// (``kGemmMR * kGemmTileK``) and stays L1-resident for the whole k loop.
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
  const bool alpha_is_one = alpha == 1.0f;
  const bool beta_is_one = beta == 1.0f;
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
      __m256 res0 = alpha_is_one ? acc0[r] : _mm256_mul_ps(valpha, acc0[r]);
      __m256 res1 = alpha_is_one ? acc1[r] : _mm256_mul_ps(valpha, acc1[r]);
      if (mode == GemmAccumMode::kInitBias) {
        const float *Crow = Crow_base + r * Cstride + n0 + n;
        const __m256 vc0 = _mm256_loadu_ps(Crow);
        const __m256 vc1 = _mm256_loadu_ps(Crow + 8);
        res0 = _mm256_add_ps(res0, beta_is_one ? vc0 : _mm256_mul_ps(vbeta, vc0));
        res1 = _mm256_add_ps(res1, beta_is_one ? vc1 : _mm256_mul_ps(vbeta, vc1));
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
      __m256 res = alpha_is_one ? acc[r] : _mm256_mul_ps(valpha, acc[r]);
      if (mode == GemmAccumMode::kInitBias) {
        const __m256 vc = _mm256_loadu_ps(Crow_base + r * Cstride + n0 + n);
        res = _mm256_add_ps(res, beta_is_one ? vc : _mm256_mul_ps(vbeta, vc));
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
  const bool alpha_is_one = alpha == 1.0f;
  const bool beta_is_one = beta == 1.0f;
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
      __m128 res0 = alpha_is_one ? acc0[r] : _mm_mul_ps(valpha, acc0[r]);
      __m128 res1 = alpha_is_one ? acc1[r] : _mm_mul_ps(valpha, acc1[r]);
      if (mode == GemmAccumMode::kInitBias) {
        const float *Crow = Crow_base + r * Cstride + n0 + n;
        const __m128 vc0 = _mm_loadu_ps(Crow);
        const __m128 vc1 = _mm_loadu_ps(Crow + 4);
        res0 = _mm_add_ps(res0, beta_is_one ? vc0 : _mm_mul_ps(vbeta, vc0));
        res1 = _mm_add_ps(res1, beta_is_one ? vc1 : _mm_mul_ps(vbeta, vc1));
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
      __m128 res = alpha_is_one ? acc[r] : _mm_mul_ps(valpha, acc[r]);
      if (mode == GemmAccumMode::kInitBias) {
        const __m128 vc = _mm_loadu_ps(Crow_base + r * Cstride + n0 + n);
        res = _mm_add_ps(res, beta_is_one ? vc : _mm_mul_ps(vbeta, vc));
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
  const bool alpha_is_one = alpha == 1.0;
  const bool beta_is_one = beta == 1.0;
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
      __m256d res0 = alpha_is_one ? acc0[r] : _mm256_mul_pd(valpha, acc0[r]);
      __m256d res1 = alpha_is_one ? acc1[r] : _mm256_mul_pd(valpha, acc1[r]);
      if (mode == GemmAccumMode::kInitBias) {
        const double *Crow = Crow_base + r * Cstride + n0 + n;
        const __m256d vc0 = _mm256_loadu_pd(Crow);
        const __m256d vc1 = _mm256_loadu_pd(Crow + 4);
        res0 = _mm256_add_pd(res0, beta_is_one ? vc0 : _mm256_mul_pd(vbeta, vc0));
        res1 = _mm256_add_pd(res1, beta_is_one ? vc1 : _mm256_mul_pd(vbeta, vc1));
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
      __m256d res = alpha_is_one ? acc[r] : _mm256_mul_pd(valpha, acc[r]);
      if (mode == GemmAccumMode::kInitBias) {
        const __m256d vc = _mm256_loadu_pd(Crow_base + r * Cstride + n0 + n);
        res = _mm256_add_pd(res, beta_is_one ? vc : _mm256_mul_pd(vbeta, vc));
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
  const bool alpha_is_one = alpha == 1.0;
  const bool beta_is_one = beta == 1.0;
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
      __m128d res0 = alpha_is_one ? acc0[r] : _mm_mul_pd(valpha, acc0[r]);
      __m128d res1 = alpha_is_one ? acc1[r] : _mm_mul_pd(valpha, acc1[r]);
      if (mode == GemmAccumMode::kInitBias) {
        const double *Crow = Crow_base + r * Cstride + n0 + n;
        const __m128d vc0 = _mm_loadu_pd(Crow);
        const __m128d vc1 = _mm_loadu_pd(Crow + 2);
        res0 = _mm_add_pd(res0, beta_is_one ? vc0 : _mm_mul_pd(vbeta, vc0));
        res1 = _mm_add_pd(res1, beta_is_one ? vc1 : _mm_mul_pd(vbeta, vc1));
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
      __m128d res = alpha_is_one ? acc[r] : _mm_mul_pd(valpha, acc[r]);
      if (mode == GemmAccumMode::kInitBias) {
        const __m128d vc = _mm_loadu_pd(Crow_base + r * Cstride + n0 + n);
        res = _mm_add_pd(res, beta_is_one ? vc : _mm_mul_pd(vbeta, vc));
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
// Runtime-selected micro-kernel flavor for a given element type.
enum class GemmKernelKind { kScalar, kSSE2, kAVX, kAVX2FMA, kAVX512, kNeon, kSve };

template <typename T> GemmKernelKind SelectGemmKernelKind() {
#if ONNX_LIGHT_CPU_X86
  static const SimdLevel level = DetectSimdLevel();
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
  if (level >= SimdLevel::kAVX512) {
    return GemmKernelKind::kAVX512;
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
  if (level >= SimdLevel::kAVX2 && CpuSupportsFma()) {
    return GemmKernelKind::kAVX2FMA;
  }
#endif
  if (level >= SimdLevel::kAVX) {
    return GemmKernelKind::kAVX;
  }
  if (level >= SimdLevel::kSSE2) {
    return GemmKernelKind::kSSE2;
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_NEON
  static const ArmGemmProfile arm_profile = DetectArmGemmProfile();
#ifdef ONNX_LIGHT_CPU_HAVE_SVE
  if (arm_profile.kind == ArmGemmKernelKind::kSve) {
    return GemmKernelKind::kSve;
  }
#endif
  if (arm_profile.kind == ArmGemmKernelKind::kNeon) {
    return GemmKernelKind::kNeon;
  }
#endif
  return GemmKernelKind::kScalar;
}

template <typename T> std::size_t GemmVectorLanes(GemmKernelKind kind) {
  switch (kind) {
  case GemmKernelKind::kAVX512:
    return 64 / sizeof(T);
  case GemmKernelKind::kAVX2FMA:
  case GemmKernelKind::kAVX:
    return 32 / sizeof(T);
  case GemmKernelKind::kSSE2:
  case GemmKernelKind::kNeon:
    return 16 / sizeof(T);
  case GemmKernelKind::kSve:
    return DetectArmGemmProfile().vector_bytes / sizeof(T);
  case GemmKernelKind::kScalar:
    return 1;
  }
  return 1;
}

std::size_t GemmRegisterRows(GemmKernelKind kind) {
  if (kind == GemmKernelKind::kAVX512) {
    return detail::SelectGemmRegisterRows(SimdLevel::kAVX512, true);
  }
  if (kind == GemmKernelKind::kAVX2FMA) {
    return detail::SelectGemmRegisterRows(SimdLevel::kAVX2, true);
  }
  if (kind == GemmKernelKind::kNeon || kind == GemmKernelKind::kSve) {
    return DetectArmGemmProfile().register_rows;
  }
  return kGemmMR;
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
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
  if (kind == GemmKernelKind::kAVX2FMA) {
    GemmMicroKernel_AVX2FMA_F32(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                                Ystride, n0, mode, Apack);
    return;
  }
#endif
  if (kind == GemmKernelKind::kAVX && mr <= kGemmMR) {
    GemmMicroKernel_AVX_F32(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base, Ystride,
                            n0, mode, Apack);
    return;
  }
  if (kind == GemmKernelKind::kSSE2 && mr <= kGemmMR) {
    GemmMicroKernel_SSE2_F32(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                             Ystride, n0, mode, Apack);
    return;
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_SVE
  if (kind == GemmKernelKind::kSve) {
    GemmMicroKernel_SVE_F32(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base, Ystride,
                            n0, mode, Apack);
    return;
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_NEON
  if (kind == GemmKernelKind::kNeon) {
    GemmMicroKernel_NEON_F32(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                             Ystride, n0, mode, Apack);
    return;
  }
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
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
  if (kind == GemmKernelKind::kAVX2FMA) {
    GemmMicroKernel_AVX2FMA_F64(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                                Ystride, n0, mode, Apack);
    return;
  }
#endif
  if (kind == GemmKernelKind::kAVX && mr <= kGemmMR) {
    GemmMicroKernel_AVX_F64(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base, Ystride,
                            n0, mode, Apack);
    return;
  }
  if (kind == GemmKernelKind::kSSE2 && mr <= kGemmMR) {
    GemmMicroKernel_SSE2_F64(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                             Ystride, n0, mode, Apack);
    return;
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_SVE
  if (kind == GemmKernelKind::kSve) {
    GemmMicroKernel_SVE_F64(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base, Ystride,
                            n0, mode, Apack);
    return;
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_NEON
  if (kind == GemmKernelKind::kNeon) {
    GemmMicroKernel_NEON_F64(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                             Ystride, n0, mode, Apack);
    return;
  }
#endif
  GemmMicroKernel_Scalar_F64(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                             Ystride, n0, mode, Apack);
}

// Packs one ``kc x nb`` block of ``B`` as a sequence of contiguous
// ``kc x column_block`` column micro-panels. Micro-panels are laid out at the
// nominal ``column_block`` pitch, so micro-panel starting at column ``j`` is at
// ``Bpack + j * kc`` even when the (last) micro-panel is narrower; the tile
// loops must locate it the same way. Each micro-panel stores ``k`` rows of
// ``jb`` contiguous elements, so the micro-kernel walks it sequentially and the whole slice stays
// cache-resident while the row tiles reuse it. Keeping one wide ``nb``-strided
// panel instead would make every ``k`` step jump by the panel width, which both
// defeats the hardware prefetcher and maps the rows of a micro-panel onto very
// few L1 sets once that width is a large power of two.
// Reads a FLOAT16 (``Bfloat == false``) or BFLOAT16 (``Bfloat == true``)
// element stored as a raw 16-bit pattern and converts it to ``float`` on
// access, so the packing loops materialize FP32 micro-panels directly from
// half-precision inputs without a separate full-tensor widening pass. Its size
// and alignment match ``std::uint16_t`` so a ``const std::uint16_t *`` input
// buffer can be viewed as ``const HalfSource *``.
template <bool Bfloat> struct HalfSource {
  std::uint16_t bits;
  operator float() const {
    return Bfloat ? detail::Bfloat16BitsToFloat(bits) : detail::Float16BitsToFloat(bits);
  }
};
using Float16Source = HalfSource<false>;
using BFloat16Source = HalfSource<true>;

// Reads a Float8 element stored as a raw one-byte pattern and decodes it to
// ``float`` on access (Roadmap PR09.5), so the packing loops materialize FP32
// micro-panels directly from Float8 inputs without a separate full-tensor
// conversion pass. ``Format`` selects the ONNX Float8 storage format. Its size
// and alignment match ``std::uint8_t`` so a ``const std::uint8_t *`` input
// buffer can be viewed as ``const Float8Source<Format> *``. The strided
// (transposed) gathers use this per-element decode while the contiguous packing
// copies use the exact 256-entry table gather in ``PackConvertContiguous``.
template <detail::Float8Format Format> struct Float8Source {
  std::uint8_t bits;
  operator float() const { return detail::Float8BitsToFloat(Format, bits); }
};

// The exact 256-entry decode table for ``Format``, built once on first use from
// the scalar decoder so the vectorized packing gather is bit-for-bit identical
// to the per-element decode.
template <detail::Float8Format Format> inline const float *Float8DecodeTable() {
  static const std::array<float, 256> table = detail::BuildFloat8DecodeTable(Format);
  return table.data();
}

// Widens a contiguous run of ``n`` source elements into the packed float panel.
// The generic template is a scalar per-element convert (a plain copy for the
// native FP32/FP64 paths). The FP16/BF16 overloads use the vectorized AVX2 F16C
// / shift conversion (or the NEON ``FCVTL`` / shift conversion on ARM) when the
// running CPU supports it -- which produces exactly the same float bits as the
// scalar decode -- and otherwise fall back to the scalar bit decode. Only the
// contiguous packing copies use this; the strided (transposed) gathers keep the
// per-element decode.
template <typename T, typename SrcT>
inline void PackConvertContiguous(const SrcT *src, T *dst, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    dst[i] = static_cast<T>(src[i]);
  }
}

inline void PackConvertContiguous(const Float16Source *src, float *dst, std::size_t n) {
#ifdef ONNX_LIGHT_CPU_HAVE_F16C
  // ``_mm256_cvtph_ps`` needs F16C and OS-enabled AVX state (both checked by
  // ``CpuSupportsF16C``); it does not require AVX2, which is an independent ISA
  // extension.
  static const bool use_f16c = CpuSupportsF16C();
  if (use_f16c) {
    GemmConvertFloat16ToFloat32_F16C(reinterpret_cast<const std::uint16_t *>(src), dst, n);
    return;
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_NEON_FP16
  // The NEON ``vcvt_f32_f16`` (``FCVTL``) conversion is baseline on the AArch64
  // targets that compile this translation unit, so no runtime gate is needed.
  GemmConvertFloat16ToFloat32_NEON(reinterpret_cast<const std::uint16_t *>(src), dst, n);
  return;
#endif
  for (std::size_t i = 0; i < n; ++i) {
    dst[i] = static_cast<float>(src[i]);
  }
}

inline void PackConvertContiguous(const BFloat16Source *src, float *dst, std::size_t n) {
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
  static const bool use_avx2 = DetectSimdLevel() >= SimdLevel::kAVX2;
  if (use_avx2) {
    GemmConvertBFloat16ToFloat32_AVX2(reinterpret_cast<const std::uint16_t *>(src), dst, n);
    return;
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_NEON
  // BF16 -> FP32 widening is baseline NEON (zero-extend then 16-bit shift).
  GemmConvertBFloat16ToFloat32_NEON(reinterpret_cast<const std::uint16_t *>(src), dst, n);
  return;
#endif
  for (std::size_t i = 0; i < n; ++i) {
    dst[i] = static_cast<float>(src[i]);
  }
}

// Decodes a contiguous run of ``n`` Float8 source elements into the packed
// float panel (Roadmap PR09.5). It gathers from the exact per-format decode
// table -- through the AVX2 ``vgatherdps`` helper when the running CPU supports
// AVX2, otherwise a scalar table lookup -- so the contiguous packing copies
// decode while packing with the same float bits as the per-element
// ``operator float`` used by the strided gathers.
template <detail::Float8Format Format>
inline void PackConvertContiguous(const Float8Source<Format> *src, float *dst, std::size_t n) {
  const float *table = Float8DecodeTable<Format>();
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
  static const bool use_avx2 = DetectSimdLevel() >= SimdLevel::kAVX2;
  if (use_avx2) {
    GemmDecodeFloat8ToFloat32_AVX2(table, reinterpret_cast<const std::uint8_t *>(src), dst, n);
    return;
  }
#endif
  for (std::size_t i = 0; i < n; ++i) {
    dst[i] = table[reinterpret_cast<const std::uint8_t *>(src)[i]];
  }
}

// ``SrcT`` is the element type of the input matrices; it equals the packed type
// ``T`` for the native FP32/FP64 paths (a plain copy) and is a ``HalfSource``
// for the FP16/BF16 path, which converts to ``T`` element by element while
// packing. The packed panels themselves are always ``T`` (float or double).
template <typename T, typename SrcT = T>
void PackBPanel(bool trans_b, const SrcT *B, std::size_t K, std::size_t N, std::size_t k0,
                std::size_t kc, std::size_t n0, std::size_t nb, std::size_t column_block,
                T *Bpack) {
  for (std::size_t j = 0; j < nb; j += column_block) {
    const std::size_t jb = std::min(column_block, nb - j);
    T *dst = Bpack + j * kc;
    if (!trans_b) {
      for (std::size_t k = 0; k < kc; ++k) {
        const SrcT *src = B + (k0 + k) * N + n0 + j;
        T *out = dst + k * jb;
        PackConvertContiguous(src, out, jb);
      }
      continue;
    }
    for (std::size_t k = 0; k < kc; ++k) {
      for (std::size_t n = 0; n < jb; ++n) {
        dst[k * jb + n] = static_cast<T>(B[(n0 + j + n) * K + k0 + k]);
      }
    }
  }
}

template <typename T, typename SrcT = T>
void PackAPanel(bool trans_a, const SrcT *A, std::size_t M, std::size_t K, std::size_t m0,
                std::size_t mc, std::size_t k0, std::size_t kc, T *Apack) {
  for (std::size_t m = 0; m < mc; ++m) {
    T *dst = Apack + m * kc;
    if (trans_a) {
      for (std::size_t k = 0; k < kc; ++k) {
        dst[k] = static_cast<T>(A[(k0 + k) * M + m0 + m]);
      }
    } else {
      const SrcT *src = A + (m0 + m) * K + k0;
      PackConvertContiguous(src, dst, kc);
    }
  }
}

template <typename T>
void InitializeOutput(std::size_t M, std::size_t N, T beta, const T *C, T *Y) {
  const bool has_bias = C != nullptr && beta != T(0);
  ParallelFor(static_cast<std::int64_t>(M), static_cast<double>(N),
              [=](std::int64_t begin, std::int64_t end) {
                for (std::int64_t m = begin; m < end; ++m) {
                  T *y = Y + static_cast<std::size_t>(m) * N;
                  if (has_bias) {
                    const T *c = C + static_cast<std::size_t>(m) * N;
                    for (std::size_t n = 0; n < N; ++n) {
                      y[n] = beta * c[n];
                    }
                  } else {
                    std::fill(y, y + N, T(0));
                  }
                }
              });
}

template <typename T, typename TileFn>
void GemmDirect(std::size_t M, std::size_t N, std::size_t K, T alpha, const T *A, const T *B,
                T beta, const T *C, T *Y, GemmKernelKind kind, TileFn tile,
                const GemmBlocking &blocking) {
  const bool has_bias = C != nullptr && beta != T(0);
  const std::size_t row_blocks = (M + blocking.mr - 1) / blocking.mr;
  const std::size_t column_panels = (N + blocking.nc - 1) / blocking.nc;
  const std::size_t task_count = row_blocks * column_panels;
  const double cost =
      static_cast<double>(K) * blocking.mr * blocking.nc / kGemmFmasPerParallelWorkUnit;
  ParallelFor(
      static_cast<std::int64_t>(task_count), cost, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t task = begin; task < end; ++task) {
          const std::size_t index = static_cast<std::size_t>(task);
          const std::size_t panel = index / row_blocks;
          const std::size_t block = index % row_blocks;
          const std::size_t m = block * blocking.mr;
          const std::size_t n0 = panel * blocking.nc;
          const std::size_t mr = std::min(blocking.mr, M - m);
          const std::size_t nb = std::min(blocking.nc, N - n0);
          tile(kind, mr, nb, K, alpha, beta, B, N, has_bias ? C + m * N : nullptr, N, Y + m * N, N,
               n0, has_bias ? GemmAccumMode::kInitBias : GemmAccumMode::kInitZero, A + m * K);
        }
      });
}

// Dot product of two ``K``-length sequences read with independent strides,
// accumulated through several partial sums so the compiler can vectorize the
// unit-stride case and extract instruction-level parallelism otherwise. The
// tail (``K`` not a multiple of the unroll factor) is summed exactly.
//
// ``SrcT`` is the element type of the input sequences; it equals the
// accumulator type ``T`` for the native FP32/FP64 paths and is a ``HalfSource``
// for the FP16/BF16 path, which converts each element to ``T`` on access so no
// full-tensor widening buffer is allocated. Accumulation always happens in the
// wider ``T`` (float or double).
template <typename T, typename SrcT = T>
T SkinnyDotProduct(const SrcT *a, std::size_t a_stride, const SrcT *b, std::size_t b_stride,
                   std::size_t K) {
  std::size_t k = 0;
  if (a_stride == 1 && b_stride == 1) {
    // Carry enough independent partial sums for the SLP vectorizer to fill
    // several full-width AVX accumulators (two 8-wide float or four 4-wide
    // double vectors) instead of a single 128-bit pack. On AVX2 this lifts the
    // unit-stride reduction -- the ``N == 1`` skinny-N and common inference
    // layouts -- from a four-lane SSE dot product to full-width vectors, which
    // is the dominant cost of the memory-streaming GEMV shapes.
    constexpr std::size_t kLanes = 16;
    T acc[kLanes] = {};
    for (; k + kLanes <= K; k += kLanes) {
      for (std::size_t lane = 0; lane < kLanes; ++lane) {
        acc[lane] += a[k + lane] * b[k + lane];
      }
    }
    T sum = T(0);
    for (std::size_t lane = 0; lane < kLanes; ++lane) {
      sum += acc[lane];
    }
    for (; k < K; ++k) {
      sum += a[k] * b[k];
    }
    return sum;
  }
  // Strided reads cannot vectorize; keep four accumulators for instruction-level
  // parallelism without spilling.
  constexpr std::size_t kUnroll = 4;
  T acc0 = T(0);
  T acc1 = T(0);
  T acc2 = T(0);
  T acc3 = T(0);
  for (; k + kUnroll <= K; k += kUnroll) {
    acc0 += a[(k + 0) * a_stride] * b[(k + 0) * b_stride];
    acc1 += a[(k + 1) * a_stride] * b[(k + 1) * b_stride];
    acc2 += a[(k + 2) * a_stride] * b[(k + 2) * b_stride];
    acc3 += a[(k + 3) * a_stride] * b[(k + 3) * b_stride];
  }
  T acc = (acc0 + acc1) + (acc2 + acc3);
  for (; k < K; ++k) {
    acc += a[k * a_stride] * b[k * b_stride];
  }
  return acc;
}

template <typename T, typename SrcT = T>
void GemmSkinnyN(bool trans_a, bool trans_b, std::size_t M, std::size_t N, std::size_t K, T alpha,
                 const SrcT *A, const SrcT *B, T beta, const T *C, T *Y) {
  const bool has_bias = C != nullptr && beta != T(0);
  // ``A(m, k)`` and ``B(k, n)`` reduce to unit-stride reads for the common
  // inference layouts (non-transposed A and transposed weights, or ``N == 1``),
  // which lets ``SkinnyDotProduct`` vectorize over K instead of walking each
  // reduction with a serial dependency chain.
  const std::size_t a_stride = trans_a ? M : 1;
  const std::size_t b_stride = trans_b ? 1 : N;
  const double cost = static_cast<double>(N) * K / kGemmFmasPerParallelWorkUnit;
  ParallelFor(static_cast<std::int64_t>(M), cost, [&](std::int64_t begin, std::int64_t end) {
    for (std::int64_t row = begin; row < end; ++row) {
      const std::size_t m = static_cast<std::size_t>(row);
      const SrcT *a_row = trans_a ? A + m : A + m * K;
      T *y_row = Y + m * N;
      const T *c_row = has_bias ? C + m * N : nullptr;
      for (std::size_t n = 0; n < N; ++n) {
        const SrcT *b_col = trans_b ? B + n * K : B + n;
        const T acc = SkinnyDotProduct<T>(a_row, a_stride, b_col, b_stride, K);
        y_row[n] = alpha * acc + (has_bias ? beta * c_row[n] : T(0));
      }
    }
  });
}

// Dedicated GEMV / skinny-M kernel. When ``M`` is small (a single example or a
// short batch), the multiplication is bound by streaming ``B`` rather than by
// register-blocked FMAs, so the register-tiled five-loop engine wastes work
// packing a nearly empty ``A`` panel. This kernel instead streams each ``B``
// row ``B(k, n0:n0+nb)`` once per ``k`` and reuses it across the few output
// rows: for every ``k`` it broadcasts the scalar ``A(m, k)`` into an axpy over
// the output columns. That axpy is unit-stride for the common
// non-transposed-``B`` layout, so the compiler vectorizes over ``N`` -- the
// useful dimension when ``M`` is tiny. Accumulators for one column panel stay
// in a small ``M x nb`` buffer, and the ``alpha``/``beta`` epilogue is applied
// once per output element. Work is parallelized over ``N`` column panels.
template <typename T, typename SrcT = T>
void GemmSkinnyM(bool trans_a, bool trans_b, std::size_t M, std::size_t N, std::size_t K, T alpha,
                 const SrcT *A, const SrcT *B, T beta, const T *C, T *Y,
                 const GemmBlocking &blocking) {
  const bool has_bias = C != nullptr && beta != T(0);
  // ``B(k, n)`` reduces to unit-stride column reads for the common inference
  // layout (non-transposed weights), which lets the axpy vectorize over N.
  const std::size_t b_col_stride = trans_b ? K : 1;
  const std::size_t b_row_stride = trans_b ? 1 : N;
  const std::size_t nc = blocking.nc;
  const std::size_t panel_count = (N + nc - 1) / nc;
  const double cost = static_cast<double>(M) * nc * K / kGemmFmasPerParallelWorkUnit;
  ParallelFor(static_cast<std::int64_t>(panel_count), cost,
              [&](std::int64_t begin, std::int64_t end) {
                AlignedVector<T> acc(M * nc);
                for (std::int64_t panel = begin; panel < end; ++panel) {
                  const std::size_t n0 = static_cast<std::size_t>(panel) * nc;
                  const std::size_t nb = std::min(nc, N - n0);
                  std::fill(acc.data(), acc.data() + M * nb, T(0));
                  for (std::size_t k = 0; k < K; ++k) {
                    const SrcT *b_row = B + k * b_row_stride + n0 * b_col_stride;
                    for (std::size_t m = 0; m < M; ++m) {
                      const T a_val = trans_a ? A[k * M + m] : A[m * K + k];
                      T *acc_row = acc.data() + m * nb;
                      for (std::size_t j = 0; j < nb; ++j) {
                        acc_row[j] += a_val * b_row[j * b_col_stride];
                      }
                    }
                  }
                  for (std::size_t m = 0; m < M; ++m) {
                    const T *acc_row = acc.data() + m * nb;
                    T *y_row = Y + m * N + n0;
                    const T *c_row = has_bias ? C + m * N + n0 : nullptr;
                    for (std::size_t j = 0; j < nb; ++j) {
                      y_row[j] = alpha * acc_row[j] + (has_bias ? beta * c_row[j] : T(0));
                    }
                  }
                }
              });
}

template <typename T, typename TileFn, typename SrcT = T>
void GemmFiveLoopRange(bool trans_a, bool trans_b, std::size_t M, std::size_t N, std::size_t K,
                       std::size_t k_begin, std::size_t k_end, T alpha, const SrcT *A,
                       const SrcT *B, T beta, const T *C, T *Y, GemmKernelKind kind, TileFn tile,
                       const GemmBlocking &blocking) {
  const bool has_bias = C != nullptr && beta != T(0);
  const std::size_t column_panels = (N + blocking.nc - 1) / blocking.nc;
  const std::size_t row_panels = (M + blocking.mc - 1) / blocking.mc;
  const std::size_t thread_count = static_cast<std::size_t>(ParallelForThreadCount());
  const std::size_t panels_per_wave = std::min(
      column_panels, std::max<std::size_t>(1, (thread_count + row_panels - 1) / row_panels));
  // Column micro-panels are the outer tile loop and row tiles the inner one, so
  // the contiguous ``kc x column_block`` micro-panel of B is reused from cache
  // by every row tile of the L2-resident packed A panel. The opposite order
  // streams the whole ``kc x nc`` B panel -- which is sized for L3 -- once per
  // row tile, and that traffic caps large-matrix throughput well below the
  // micro-kernel rate.
  const std::size_t column_block = detail::SelectGemmColumnBlock(blocking, sizeof(T));
  const std::size_t panel_capacity = blocking.kc * AlignUp(blocking.nc, column_block);
  AlignedVector<T> bpack(panels_per_wave * panel_capacity);

  for (std::size_t k0 = k_begin; k0 < k_end; k0 += blocking.kc) {
    const std::size_t kc = std::min(blocking.kc, k_end - k0);
    const GemmAccumMode mode =
        k0 == k_begin ? (has_bias ? GemmAccumMode::kInitBias : GemmAccumMode::kInitZero)
                      : GemmAccumMode::kAccumulate;
    for (std::size_t first_panel = 0; first_panel < column_panels; first_panel += panels_per_wave) {
      const std::size_t wave_panels = std::min(panels_per_wave, column_panels - first_panel);
      for (std::size_t panel = 0; panel < wave_panels; ++panel) {
        const std::size_t n0 = (first_panel + panel) * blocking.nc;
        const std::size_t nb = std::min(blocking.nc, N - n0);
        PackBPanel(trans_b, B, K, N, k0, kc, n0, nb, column_block,
                   bpack.data() + panel * panel_capacity);
      }

      const std::size_t task_count = row_panels * wave_panels;
      const std::size_t first_n = first_panel * blocking.nc;
      const std::size_t task_columns = std::min(blocking.nc, N - first_n);
      const double cost = static_cast<double>(std::min(blocking.mc, M)) * task_columns * kc /
                          kGemmFmasPerParallelWorkUnit;
      ParallelFor(static_cast<std::int64_t>(task_count), cost,
                  [&](std::int64_t begin, std::int64_t end) {
                    AlignedVector<T> apack(blocking.mc * kc);
                    std::size_t packed_row_panel = row_panels;
                    for (std::int64_t task = begin; task < end; ++task) {
                      const std::size_t row_panel = static_cast<std::size_t>(task) / wave_panels;
                      const std::size_t wave_panel = static_cast<std::size_t>(task) % wave_panels;
                      const std::size_t m0 = row_panel * blocking.mc;
                      const std::size_t n0 = (first_panel + wave_panel) * blocking.nc;
                      const std::size_t mc = std::min(blocking.mc, M - m0);
                      const std::size_t nb = std::min(blocking.nc, N - n0);
                      if (packed_row_panel != row_panel) {
                        PackAPanel(trans_a, A, M, K, m0, mc, k0, kc, apack.data());
                        packed_row_panel = row_panel;
                      }
                      const T *panel_b = bpack.data() + wave_panel * panel_capacity;
                      for (std::size_t jr = 0; jr < nb; jr += column_block) {
                        const std::size_t jb = std::min(column_block, nb - jr);
                        const T *micro_b = panel_b + jr * kc;
                        for (std::size_t ir = 0; ir < mc; ir += blocking.mr) {
                          const std::size_t mr = std::min(blocking.mr, mc - ir);
                          tile(kind, mr, jb, kc, alpha, beta, micro_b, jb,
                               has_bias ? C + (m0 + ir) * N + n0 + jr : nullptr, N,
                               Y + (m0 + ir) * N + n0 + jr, N, 0, mode, apack.data() + ir * kc);
                        }
                      }
                    }
                  });
    }
  }
}

template <typename T, typename TileFn, typename SrcT = T>
void GemmFiveLoop(bool trans_a, bool trans_b, std::size_t M, std::size_t N, std::size_t K, T alpha,
                  const SrcT *A, const SrcT *B, T beta, const T *C, T *Y, GemmKernelKind kind,
                  TileFn tile, const GemmBlocking &blocking) {
  GemmFiveLoopRange(trans_a, trans_b, M, N, K, 0, K, alpha, A, B, beta, C, Y, kind, tile, blocking);
}

template <typename T, typename TileFn, typename SrcT = T>
void GemmSplitK(bool trans_a, bool trans_b, std::size_t M, std::size_t N, std::size_t K, T alpha,
                const SrcT *A, const SrcT *B, T beta, const T *C, T *Y, GemmKernelKind kind,
                TileFn tile, const GemmBlocking &blocking) {
  if (ParallelForInParallelRegion()) {
    GemmFiveLoop(trans_a, trans_b, M, N, K, alpha, A, B, beta, C, Y, kind, tile, blocking);
    return;
  }

  const std::size_t part_count = std::min<std::size_t>(
      static_cast<std::size_t>(ParallelForThreadCount()), (K + blocking.kc - 1) / blocking.kc);
  if (part_count <= 1) {
    GemmFiveLoop(trans_a, trans_b, M, N, K, alpha, A, B, beta, C, Y, kind, tile, blocking);
    return;
  }

  AlignedVector<T> partials(part_count * M * N);
  const double cost = static_cast<double>(M) * N * K /
                      (static_cast<double>(part_count) * kGemmFmasPerParallelWorkUnit);
  ParallelFor(static_cast<std::int64_t>(part_count), cost,
              [&](std::int64_t begin, std::int64_t end) {
                for (std::int64_t part = begin; part < end; ++part) {
                  const std::size_t p = static_cast<std::size_t>(part);
                  const std::size_t k_begin = K * p / part_count;
                  const std::size_t k_end = K * (p + 1) / part_count;
                  GemmFiveLoopRange(trans_a, trans_b, M, N, K, k_begin, k_end, T(1), A, B, T(0),
                                    static_cast<const T *>(nullptr), partials.data() + p * M * N,
                                    kind, tile, blocking);
                }
              });

  const bool has_bias = C != nullptr && beta != T(0);
  ParallelFor(static_cast<std::int64_t>(M * N), static_cast<double>(part_count),
              [&](std::int64_t begin, std::int64_t end) {
                for (std::int64_t offset = begin; offset < end; ++offset) {
                  const std::size_t index = static_cast<std::size_t>(offset);
                  T sum = T(0);
                  for (std::size_t part = 0; part < part_count; ++part) {
                    sum += partials[part * M * N + index];
                  }
                  Y[index] = alpha * sum + (has_bias ? beta * C[index] : T(0));
                }
              });
}

template <GemmAlgorithm Algorithm, typename T, typename TileFn, typename SrcT = T>
void GemmImpl(bool trans_a, bool trans_b, std::size_t M, std::size_t N, std::size_t K, T alpha,
              const SrcT *A, const SrcT *B, T beta, const T *C, T *Y, GemmKernelKind kind,
              TileFn tile, const GemmBlocking &blocking) {
  if (M == 0 || N == 0) {
    return;
  }
  if (K == 0) {
    InitializeOutput(M, N, beta, C, Y);
    return;
  }

  if constexpr (Algorithm == GemmAlgorithm::kDirect) {
    // The direct small-K path hands raw A/B pointers to the SIMD micro-kernel,
    // which only consumes packed ``T`` panels. For FP16/BF16 inputs (``SrcT !=
    // T``) that conversion happens while packing, so route them through the
    // five-loop engine, which materializes ``T`` panels from the typed source
    // without a full-tensor widening buffer.
    if constexpr (!std::is_same_v<SrcT, T>) {
      GemmFiveLoop(trans_a, trans_b, M, N, K, alpha, A, B, beta, C, Y, kind, tile, blocking);
    } else if (trans_a || trans_b || K > 32) {
      GemmFiveLoop(trans_a, trans_b, M, N, K, alpha, A, B, beta, C, Y, kind, tile, blocking);
    } else {
      GemmDirect(M, N, K, alpha, A, B, beta, C, Y, kind, tile, blocking);
    }
  } else if constexpr (Algorithm == GemmAlgorithm::kSkinnyM) {
    if (M > blocking.mr) {
      GemmFiveLoop(trans_a, trans_b, M, N, K, alpha, A, B, beta, C, Y, kind, tile, blocking);
    } else {
      GemmSkinnyM<T, SrcT>(trans_a, trans_b, M, N, K, alpha, A, B, beta, C, Y, blocking);
    }
  } else if constexpr (Algorithm == GemmAlgorithm::kSkinnyN) {
    GemmSkinnyN<T, SrcT>(trans_a, trans_b, M, N, K, alpha, A, B, beta, C, Y);
  } else if constexpr (Algorithm == GemmAlgorithm::kSplitK) {
    GemmSplitK<T, TileFn, SrcT>(trans_a, trans_b, M, N, K, alpha, A, B, beta, C, Y, kind, tile,
                                blocking);
  } else {
    static_assert(Algorithm == GemmAlgorithm::kGeneral);
    GemmFiveLoop(trans_a, trans_b, M, N, K, alpha, A, B, beta, C, Y, kind, tile, blocking);
  }
}

} // namespace

namespace detail {

template <GemmAlgorithm Algorithm>
void GemmFloat32Planned(bool trans_a, bool trans_b, std::size_t M, std::size_t N, std::size_t K,
                        float alpha, const float *A, const float *B, float beta, const float *C,
                        float *Y, const GemmBlocking *blocking) {
  const auto tile = [](GemmKernelKind kind, std::size_t mr, std::size_t nb, std::size_t K,
                       float alpha, float beta, const float *Bmat, std::size_t N,
                       const float *Crow_base, std::size_t Cstride, float *Yrow_base,
                       std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                       const float *Apack) {
    GemmTileF32(kind, mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base, Ystride, n0,
                mode, Apack);
  };
  static const GemmKernelKind default_kind = SelectGemmKernelKind<float>();
  static const GemmBlocking default_blocking = SelectGemmBlocking(
      sizeof(float), GemmVectorLanes<float>(default_kind), GemmRegisterRows(default_kind));
  const GemmBlocking selected =
      ConstrainGemmBlockingForTasks(blocking == nullptr ? default_blocking : *blocking, M, N,
                                    static_cast<std::size_t>(ParallelForThreadCount()));
  GemmImpl<Algorithm, float>(trans_a, trans_b, M, N, K, alpha, A, B, beta, C, Y,
                             SelectGemmKernelKind<float>(), tile, selected);
}

template <GemmAlgorithm Algorithm>
void GemmFloat64Planned(bool trans_a, bool trans_b, std::size_t M, std::size_t N, std::size_t K,
                        double alpha, const double *A, const double *B, double beta,
                        const double *C, double *Y, const GemmBlocking *blocking) {
  const auto tile = [](GemmKernelKind kind, std::size_t mr, std::size_t nb, std::size_t K,
                       double alpha, double beta, const double *Bmat, std::size_t N,
                       const double *Crow_base, std::size_t Cstride, double *Yrow_base,
                       std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                       const double *Apack) {
    GemmTileF64(kind, mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base, Ystride, n0,
                mode, Apack);
  };
  static const GemmKernelKind default_kind = SelectGemmKernelKind<double>();
  static const GemmBlocking default_blocking = SelectGemmBlocking(
      sizeof(double), GemmVectorLanes<double>(default_kind), GemmRegisterRows(default_kind));
  const GemmBlocking selected =
      ConstrainGemmBlockingForTasks(blocking == nullptr ? default_blocking : *blocking, M, N,
                                    static_cast<std::size_t>(ParallelForThreadCount()));
  GemmImpl<Algorithm, double>(trans_a, trans_b, M, N, K, alpha, A, B, beta, C, Y,
                              SelectGemmKernelKind<double>(), tile, selected);
}

// Native FLOAT16 general GEMM driver (Roadmap PR07.3). See the declaration in
// gemm_common.h: it packs each ``mr``-row block of ``A`` into a FLOAT16 panel
// (resolving ``trans_a``) and streams the non-transposed FLOAT16 ``B`` matrix
// through ``kernel`` with float32 accumulation, writing ``alpha * (A @ B)`` to
// the float32 ``Y`` with no bias. The kernel is injected so the same driver,
// packing, and column-tail logic can be tested with the portable scalar member
// and dispatched to the AVX-512FP16 member in production.
void GemmFp16NativeGeneral(bool trans_a, std::size_t M, std::size_t N, std::size_t K, float alpha,
                           const std::uint16_t *A, const std::uint16_t *B, float *Y,
                           GemmFp16MicroKernel kernel, std::size_t mr) {
  if (M == 0 || N == 0) {
    return;
  }
  if (K == 0) {
    InitializeOutput(M, N, 0.0f, static_cast<const float *>(nullptr), Y);
    return;
  }
  const std::size_t mr_block = std::max<std::size_t>(1, mr);
  const std::size_t row_blocks = (M + mr_block - 1) / mr_block;
  const double cost = static_cast<double>(K) * mr_block * N / kGemmFmasPerParallelWorkUnit;
  ParallelFor(static_cast<std::int64_t>(row_blocks), cost,
              [&](std::int64_t begin, std::int64_t end) {
                AlignedVector<std::uint16_t> apack(mr_block * K);
                for (std::int64_t task = begin; task < end; ++task) {
                  const std::size_t m0 = static_cast<std::size_t>(task) * mr_block;
                  const std::size_t rows = std::min(mr_block, M - m0);
                  for (std::size_t r = 0; r < rows; ++r) {
                    std::uint16_t *dst = apack.data() + r * K;
                    if (trans_a) {
                      // ``A`` is ``K x M`` row-major: ``A(k, m)`` is ``A[k * M + m]``.
                      for (std::size_t k = 0; k < K; ++k) {
                        dst[k] = A[k * M + (m0 + r)];
                      }
                    } else {
                      std::memcpy(dst, A + (m0 + r) * K, K * sizeof(std::uint16_t));
                    }
                  }
                  kernel(rows, N, K, alpha, 0.0f, B, N, nullptr, 0, Y + m0 * N, N, 0,
                         GemmAccumMode::kInitZero, apack.data());
                }
              });
}

// Native BFLOAT16 general GEMM driver (Roadmap PR07.4). See the declaration in
// gemm_common.h: it mirrors :cpp:func:`GemmFp16NativeGeneral` for BFLOAT16,
// packing each ``mr``-row block of ``A`` into a BFLOAT16 panel (resolving
// ``trans_a``) and streaming the non-transposed BFLOAT16 ``B`` matrix through
// ``kernel`` with float32 accumulation, writing ``alpha * (A @ B)`` to the
// float32 ``Y`` with no bias. The kernel is injected so the same driver,
// packing, and column-tail logic can be tested with the portable scalar member
// and dispatched to the AVX-512BF16 member in production.
void GemmBf16NativeGeneral(bool trans_a, std::size_t M, std::size_t N, std::size_t K, float alpha,
                           const std::uint16_t *A, const std::uint16_t *B, float *Y,
                           GemmBf16MicroKernel kernel, std::size_t mr) {
  if (M == 0 || N == 0) {
    return;
  }
  if (K == 0) {
    InitializeOutput(M, N, 0.0f, static_cast<const float *>(nullptr), Y);
    return;
  }
  const std::size_t mr_block = std::max<std::size_t>(1, mr);
  const std::size_t row_blocks = (M + mr_block - 1) / mr_block;
  const double cost = static_cast<double>(K) * mr_block * N / kGemmFmasPerParallelWorkUnit;
  ParallelFor(static_cast<std::int64_t>(row_blocks), cost,
              [&](std::int64_t begin, std::int64_t end) {
                AlignedVector<std::uint16_t> apack(mr_block * K);
                for (std::int64_t task = begin; task < end; ++task) {
                  const std::size_t m0 = static_cast<std::size_t>(task) * mr_block;
                  const std::size_t rows = std::min(mr_block, M - m0);
                  for (std::size_t r = 0; r < rows; ++r) {
                    std::uint16_t *dst = apack.data() + r * K;
                    if (trans_a) {
                      // ``A`` is ``K x M`` row-major: ``A(k, m)`` is ``A[k * M + m]``.
                      for (std::size_t k = 0; k < K; ++k) {
                        dst[k] = A[k * M + (m0 + r)];
                      }
                    } else {
                      std::memcpy(dst, A + (m0 + r) * K, K * sizeof(std::uint16_t));
                    }
                  }
                  kernel(rows, N, K, alpha, 0.0f, B, N, nullptr, 0, Y + m0 * N, N, 0,
                         GemmAccumMode::kInitZero, apack.data());
                }
              });
}

// FP16/BF16 GEMM accumulated in float32, executed through the typed source
// path for one prepared algorithm. ``A`` and ``B`` are raw 16-bit patterns
// viewed as ``HalfSource`` and converted to float32 element by element while
// the operands are packed (general/direct/skinny-M) or reduced (skinny-N), so
// no full-tensor widening buffer is allocated for any shape. Split-K reuses the
// same converting five-loop range per partition. The reduction accumulates in
// float32 and the raw ``M x N`` product is written to ``Y`` for the caller's
// narrowing epilogue. ``blocking`` overrides the cached plan blocking when
// non-null; otherwise the default blocking is derived here.
template <GemmAlgorithm Algorithm>
void GemmHalfPlanned(bool is_bfloat16, bool trans_a, bool trans_b, std::size_t M, std::size_t N,
                     std::size_t K, float alpha, const std::uint16_t *A, const std::uint16_t *B,
                     float *Y, const GemmBlocking *blocking) {
  const auto tile = [](GemmKernelKind kind, std::size_t mr, std::size_t nb, std::size_t k,
                       float alpha, float beta, const float *Bmat, std::size_t N,
                       const float *Crow_base, std::size_t Cstride, float *Yrow_base,
                       std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                       const float *Apack) {
    GemmTileF32(kind, mr, nb, k, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base, Ystride, n0,
                mode, Apack);
  };
  static const GemmKernelKind default_kind = SelectGemmKernelKind<float>();
  static const GemmBlocking default_blocking = SelectGemmBlocking(
      sizeof(float), GemmVectorLanes<float>(default_kind), GemmRegisterRows(default_kind));
  const GemmBlocking selected =
      ConstrainGemmBlockingForTasks(blocking == nullptr ? default_blocking : *blocking, M, N,
                                    static_cast<std::size_t>(ParallelForThreadCount()));
  if (is_bfloat16) {
    const auto *a = reinterpret_cast<const BFloat16Source *>(A);
    const auto *b = reinterpret_cast<const BFloat16Source *>(B);
#ifdef ONNX_LIGHT_CPU_HAVE_AMX_BF16
    // Roadmap PR07.6: prefer the native AMX-BF16 tile kernel when the CPU
    // supports AMX-BF16 and the OS has enabled tile state. Like the AVX-512BF16
    // path it keeps both operands in BFLOAT16 and requires a non-transposed
    // ``B``; it falls back to the AVX-512BF16 kernel (below) or the converting
    // path for every other shape or ISA.
    if constexpr (Algorithm == GemmAlgorithm::kGeneral) {
      static const bool use_amx_bf16 = CpuSupportsAmxBf16() && AmxTileStateAvailable();
      if (use_amx_bf16 && !trans_b) {
        GemmBf16NativeGeneral(trans_a, M, N, K, alpha, A, B, Y, &GemmMicroKernel_AMXBF16,
                              kGemmAmxBf16MR);
        return;
      }
    }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512BF16
    // Roadmap PR07.4: when the CPU natively supports AVX-512BF16, run the
    // general BFLOAT16 algorithm through the native ``vdpbf16ps`` micro-kernel,
    // which keeps both operands in BFLOAT16 (halving the ``B`` traffic) instead
    // of widening while packing. It requires a non-transposed ``B`` so the
    // kernel can read it with a plain row stride; every other shape keeps the
    // converting path.
    if constexpr (Algorithm == GemmAlgorithm::kGeneral) {
      static const bool use_avx512bf16 = CpuSupportsAvx512Bf16();
      if (use_avx512bf16 && !trans_b) {
        const std::size_t mr = std::min<std::size_t>(selected.mr, kGemmAVX512MR);
        GemmBf16NativeGeneral(trans_a, M, N, K, alpha, A, B, Y, &GemmMicroKernel_AVX512BF16, mr);
        return;
      }
    }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_SVE
    // Roadmap PR08.3: on a machine whose runtime ARM profile selects SVE (a
    // vector length of at least 256 bits; shorter vectors keep the better
    // unrolled NEON kernel below), run the native SVE BFLOAT16 kernel. Like the
    // NEON path both operands stay BFLOAT16 and ``B`` is widened on the fly with
    // float32 accumulation; it only requires a non-transposed ``B``.
    if constexpr (Algorithm == GemmAlgorithm::kGeneral) {
      static const ArmGemmProfile arm_profile = DetectArmGemmProfile();
      if (!trans_b && arm_profile.kind == ArmGemmKernelKind::kSve) {
        const std::size_t mr = std::min<std::size_t>(selected.mr, kGemmSveMR);
        GemmBf16NativeGeneral(trans_a, M, N, K, alpha, A, B, Y, &GemmMicroKernel_SVE_BF16, mr);
        return;
      }
    }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_NEON
    // Roadmap PR08.2: native NEON BFLOAT16 arithmetic keeps both operands in
    // BFLOAT16 to the register file and widens ``B`` on the fly (baseline NEON
    // zero-extend / 16-bit shift) with float32 accumulation, instead of widening
    // while packing. NEON is baseline on this build, so it only requires a
    // non-transposed ``B`` for the plain row stride; every other shape keeps the
    // converting path.
    if constexpr (Algorithm == GemmAlgorithm::kGeneral) {
      if (!trans_b) {
        const std::size_t mr = std::min<std::size_t>(selected.mr, kGemmNeonMR);
        GemmBf16NativeGeneral(trans_a, M, N, K, alpha, A, B, Y, &GemmMicroKernel_NEON_BF16, mr);
        return;
      }
    }
#endif
    GemmImpl<Algorithm, float, decltype(tile), BFloat16Source>(
        trans_a, trans_b, M, N, K, alpha, a, b, 0.0f, static_cast<const float *>(nullptr), Y,
        default_kind, tile, selected);
  } else {
    const auto *a = reinterpret_cast<const Float16Source *>(A);
    const auto *b = reinterpret_cast<const Float16Source *>(B);
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512FP16
    // Roadmap PR07.3: when the CPU natively supports AVX-512FP16, run the
    // general FLOAT16 algorithm through the native micro-kernel, which keeps
    // both operands in FLOAT16 (halving the ``B`` traffic) instead of widening
    // while packing. It requires a non-transposed ``B`` so the kernel can read
    // it with a plain row stride; every other shape keeps the converting path.
    if constexpr (Algorithm == GemmAlgorithm::kGeneral) {
      static const bool use_avx512fp16 = CpuSupportsAvx512Fp16();
      if (use_avx512fp16 && !trans_b) {
        const std::size_t mr = std::min<std::size_t>(selected.mr, kGemmAVX512MR);
        GemmFp16NativeGeneral(trans_a, M, N, K, alpha, A, B, Y, &GemmMicroKernel_AVX512FP16, mr);
        return;
      }
    }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_SVE
    // Roadmap PR08.3: on a machine whose runtime ARM profile selects SVE (a
    // vector length of at least 256 bits; shorter vectors keep the NEON kernel
    // below), run the native SVE FLOAT16 kernel. Half precision is baseline SVE,
    // so unlike NEON it needs no separate FP16 feature gate; both operands stay
    // FLOAT16 and ``B`` is widened on the fly with float32 accumulation, and it
    // only requires a non-transposed ``B``.
    if constexpr (Algorithm == GemmAlgorithm::kGeneral) {
      static const ArmGemmProfile arm_profile = DetectArmGemmProfile();
      if (!trans_b && arm_profile.kind == ArmGemmKernelKind::kSve) {
        const std::size_t mr = std::min<std::size_t>(selected.mr, kGemmSveMR);
        GemmFp16NativeGeneral(trans_a, M, N, K, alpha, A, B, Y, &GemmMicroKernel_SVE_FP16, mr);
        return;
      }
    }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_NEON_FP16
    // Roadmap PR08.2: native NEON FLOAT16 arithmetic keeps both operands in
    // FLOAT16 to the register file and widens ``B`` on the fly with the NEON
    // ``vcvt_f32_f16`` (``FCVTL``) instruction with float32 accumulation,
    // instead of widening while packing. It requires the FP16 vector intrinsics
    // and a non-transposed ``B``; every other shape or toolchain keeps the
    // converting path.
    if constexpr (Algorithm == GemmAlgorithm::kGeneral) {
      if (!trans_b) {
        const std::size_t mr = std::min<std::size_t>(selected.mr, kGemmNeonMR);
        GemmFp16NativeGeneral(trans_a, M, N, K, alpha, A, B, Y, &GemmMicroKernel_NEON_FP16, mr);
        return;
      }
    }
#endif
    GemmImpl<Algorithm, float, decltype(tile), Float16Source>(
        trans_a, trans_b, M, N, K, alpha, a, b, 0.0f, static_cast<const float *>(nullptr), Y,
        default_kind, tile, selected);
  }
}

// Runtime-dispatched FP16/BF16 GEMM used by the non-plan entry point. It
// selects the algorithm the same way the FP32 path does and forwards to the
// typed :cpp:func:`GemmHalfPlanned` so every algorithm reads the half operands
// directly, without a full-tensor widening buffer.
void GemmHalfToFloat(bool is_bfloat16, bool trans_a, bool trans_b, std::size_t M, std::size_t N,
                     std::size_t K, float alpha, const std::uint16_t *A, const std::uint16_t *B,
                     float *Y) {
  const GemmKernelKind kind = SelectGemmKernelKind<float>();
  const GemmAlgorithm algorithm = SelectGemmAlgorithm(
      trans_a, trans_b, M, N, K, GemmVectorLanes<float>(kind), GemmRegisterRows(kind));
  switch (algorithm) {
  case GemmAlgorithm::kDirect:
    return GemmHalfPlanned<GemmAlgorithm::kDirect>(is_bfloat16, trans_a, trans_b, M, N, K, alpha, A,
                                                   B, Y, nullptr);
  case GemmAlgorithm::kSkinnyM:
    return GemmHalfPlanned<GemmAlgorithm::kSkinnyM>(is_bfloat16, trans_a, trans_b, M, N, K, alpha,
                                                    A, B, Y, nullptr);
  case GemmAlgorithm::kSkinnyN:
    return GemmHalfPlanned<GemmAlgorithm::kSkinnyN>(is_bfloat16, trans_a, trans_b, M, N, K, alpha,
                                                    A, B, Y, nullptr);
  case GemmAlgorithm::kSplitK:
    return GemmHalfPlanned<GemmAlgorithm::kSplitK>(is_bfloat16, trans_a, trans_b, M, N, K, alpha, A,
                                                   B, Y, nullptr);
  case GemmAlgorithm::kGeneral:
    return GemmHalfPlanned<GemmAlgorithm::kGeneral>(is_bfloat16, trans_a, trans_b, M, N, K, alpha,
                                                    A, B, Y, nullptr);
  }
}

// Float8 GEMM accumulated in float32 for one prepared algorithm (Roadmap
// PR09.5). ``A`` and ``B`` are raw one-byte Float8 patterns of ``format``
// (both operands share the format) viewed as ``Float8Source<Format>`` and
// decoded to float32 while the operands are packed (general/direct/skinny-M) or
// reduced (skinny-N), so no full-tensor conversion buffer is allocated for any
// shape. The float32 ``M x N`` product is written to ``Y`` for the caller's
// narrowing epilogue. Each format is handled as a separate packing format, not
// a branch in the FP32 inner loop.
template <GemmAlgorithm Algorithm, detail::Float8Format Format>
void GemmFloat8PlannedTyped(bool trans_a, bool trans_b, std::size_t M, std::size_t N, std::size_t K,
                            float alpha, const std::uint8_t *A, const std::uint8_t *B, float *Y,
                            const GemmBlocking *blocking) {
  const auto tile = [](GemmKernelKind kind, std::size_t mr, std::size_t nb, std::size_t k,
                       float alpha, float beta, const float *Bmat, std::size_t N,
                       const float *Crow_base, std::size_t Cstride, float *Yrow_base,
                       std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                       const float *Apack) {
    GemmTileF32(kind, mr, nb, k, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base, Ystride, n0,
                mode, Apack);
  };
  static const GemmKernelKind default_kind = SelectGemmKernelKind<float>();
  static const GemmBlocking default_blocking = SelectGemmBlocking(
      sizeof(float), GemmVectorLanes<float>(default_kind), GemmRegisterRows(default_kind));
  const GemmBlocking selected =
      ConstrainGemmBlockingForTasks(blocking == nullptr ? default_blocking : *blocking, M, N,
                                    static_cast<std::size_t>(ParallelForThreadCount()));
  const auto *a = reinterpret_cast<const Float8Source<Format> *>(A);
  const auto *b = reinterpret_cast<const Float8Source<Format> *>(B);
  GemmImpl<Algorithm, float, decltype(tile), Float8Source<Format>>(
      trans_a, trans_b, M, N, K, alpha, a, b, 0.0f, static_cast<const float *>(nullptr), Y,
      default_kind, tile, selected);
}

template <GemmAlgorithm Algorithm>
void GemmFloat8Planned(Float8Format format, bool trans_a, bool trans_b, std::size_t M,
                       std::size_t N, std::size_t K, float alpha, const std::uint8_t *A,
                       const std::uint8_t *B, float *Y, const GemmBlocking *blocking) {
  switch (format) {
  case Float8Format::kE4M3FN:
    return GemmFloat8PlannedTyped<Algorithm, Float8Format::kE4M3FN>(trans_a, trans_b, M, N, K,
                                                                    alpha, A, B, Y, blocking);
  case Float8Format::kE4M3FNUZ:
    return GemmFloat8PlannedTyped<Algorithm, Float8Format::kE4M3FNUZ>(trans_a, trans_b, M, N, K,
                                                                      alpha, A, B, Y, blocking);
  case Float8Format::kE5M2:
    return GemmFloat8PlannedTyped<Algorithm, Float8Format::kE5M2>(trans_a, trans_b, M, N, K, alpha,
                                                                  A, B, Y, blocking);
  case Float8Format::kE5M2FNUZ:
    return GemmFloat8PlannedTyped<Algorithm, Float8Format::kE5M2FNUZ>(trans_a, trans_b, M, N, K,
                                                                      alpha, A, B, Y, blocking);
  }
}

// Runtime-dispatched Float8 GEMM used by the non-plan entry point (Roadmap
// PR09.5). It selects the algorithm the same way the FP32 path does and
// forwards to the typed :cpp:func:`GemmFloat8Planned` so every algorithm
// decodes the Float8 operands directly while packing.
void GemmFloat8ToFloat(Float8Format format, bool trans_a, bool trans_b, std::size_t M,
                       std::size_t N, std::size_t K, float alpha, const std::uint8_t *A,
                       const std::uint8_t *B, float *Y) {
  const GemmKernelKind kind = SelectGemmKernelKind<float>();
  const GemmAlgorithm algorithm = SelectGemmAlgorithm(
      trans_a, trans_b, M, N, K, GemmVectorLanes<float>(kind), GemmRegisterRows(kind));
  switch (algorithm) {
  case GemmAlgorithm::kDirect:
    return GemmFloat8Planned<GemmAlgorithm::kDirect>(format, trans_a, trans_b, M, N, K, alpha, A, B,
                                                     Y, nullptr);
  case GemmAlgorithm::kSkinnyM:
    return GemmFloat8Planned<GemmAlgorithm::kSkinnyM>(format, trans_a, trans_b, M, N, K, alpha, A,
                                                      B, Y, nullptr);
  case GemmAlgorithm::kSkinnyN:
    return GemmFloat8Planned<GemmAlgorithm::kSkinnyN>(format, trans_a, trans_b, M, N, K, alpha, A,
                                                      B, Y, nullptr);
  case GemmAlgorithm::kSplitK:
    return GemmFloat8Planned<GemmAlgorithm::kSplitK>(format, trans_a, trans_b, M, N, K, alpha, A, B,
                                                     Y, nullptr);
  case GemmAlgorithm::kGeneral:
    return GemmFloat8Planned<GemmAlgorithm::kGeneral>(format, trans_a, trans_b, M, N, K, alpha, A,
                                                      B, Y, nullptr);
  }
}

#define INSTANTIATE_PLANNED_GEMM(Algorithm)                                                        \
  template void GemmFloat32Planned<Algorithm>(bool, bool, std::size_t, std::size_t, std::size_t,   \
                                              float, const float *, const float *, float,          \
                                              const float *, float *, const GemmBlocking *);       \
  template void GemmFloat64Planned<Algorithm>(bool, bool, std::size_t, std::size_t, std::size_t,   \
                                              double, const double *, const double *, double,      \
                                              const double *, double *, const GemmBlocking *);     \
  template void GemmHalfPlanned<Algorithm>(bool, bool, bool, std::size_t, std::size_t,             \
                                           std::size_t, float, const std::uint16_t *,              \
                                           const std::uint16_t *, float *, const GemmBlocking *)

INSTANTIATE_PLANNED_GEMM(GemmAlgorithm::kGeneral);
INSTANTIATE_PLANNED_GEMM(GemmAlgorithm::kDirect);
INSTANTIATE_PLANNED_GEMM(GemmAlgorithm::kSkinnyM);
INSTANTIATE_PLANNED_GEMM(GemmAlgorithm::kSkinnyN);
INSTANTIATE_PLANNED_GEMM(GemmAlgorithm::kSplitK);

#undef INSTANTIATE_PLANNED_GEMM

} // namespace detail

void GemmFloat32(bool trans_a, bool trans_b, std::size_t M, std::size_t N, std::size_t K,
                 float alpha, const float *A, const float *B, float beta, const float *C,
                 float *Y) {
  const GemmKernelKind kind = SelectGemmKernelKind<float>();
  const GemmAlgorithm algorithm = detail::SelectGemmAlgorithm(
      trans_a, trans_b, M, N, K, GemmVectorLanes<float>(kind), GemmRegisterRows(kind));
  switch (algorithm) {
  case GemmAlgorithm::kDirect:
    return detail::GemmFloat32Planned<GemmAlgorithm::kDirect>(trans_a, trans_b, M, N, K, alpha, A,
                                                              B, beta, C, Y);
  case GemmAlgorithm::kSkinnyM:
    return detail::GemmFloat32Planned<GemmAlgorithm::kSkinnyM>(trans_a, trans_b, M, N, K, alpha, A,
                                                               B, beta, C, Y);
  case GemmAlgorithm::kSkinnyN:
    return detail::GemmFloat32Planned<GemmAlgorithm::kSkinnyN>(trans_a, trans_b, M, N, K, alpha, A,
                                                               B, beta, C, Y);
  case GemmAlgorithm::kSplitK:
    return detail::GemmFloat32Planned<GemmAlgorithm::kSplitK>(trans_a, trans_b, M, N, K, alpha, A,
                                                              B, beta, C, Y);
  case GemmAlgorithm::kGeneral:
    return detail::GemmFloat32Planned<GemmAlgorithm::kGeneral>(trans_a, trans_b, M, N, K, alpha, A,
                                                               B, beta, C, Y);
  }
}

void GemmFloat64(bool trans_a, bool trans_b, std::size_t M, std::size_t N, std::size_t K,
                 double alpha, const double *A, const double *B, double beta, const double *C,
                 double *Y) {
  const GemmKernelKind kind = SelectGemmKernelKind<double>();
  const GemmAlgorithm algorithm = detail::SelectGemmAlgorithm(
      trans_a, trans_b, M, N, K, GemmVectorLanes<double>(kind), GemmRegisterRows(kind));
  switch (algorithm) {
  case GemmAlgorithm::kDirect:
    return detail::GemmFloat64Planned<GemmAlgorithm::kDirect>(trans_a, trans_b, M, N, K, alpha, A,
                                                              B, beta, C, Y);
  case GemmAlgorithm::kSkinnyM:
    return detail::GemmFloat64Planned<GemmAlgorithm::kSkinnyM>(trans_a, trans_b, M, N, K, alpha, A,
                                                               B, beta, C, Y);
  case GemmAlgorithm::kSkinnyN:
    return detail::GemmFloat64Planned<GemmAlgorithm::kSkinnyN>(trans_a, trans_b, M, N, K, alpha, A,
                                                               B, beta, C, Y);
  case GemmAlgorithm::kSplitK:
    return detail::GemmFloat64Planned<GemmAlgorithm::kSplitK>(trans_a, trans_b, M, N, K, alpha, A,
                                                              B, beta, C, Y);
  case GemmAlgorithm::kGeneral:
    return detail::GemmFloat64Planned<GemmAlgorithm::kGeneral>(trans_a, trans_b, M, N, K, alpha, A,
                                                               B, beta, C, Y);
  }
}

namespace {

template <typename T>
T BroadcastValue(const T *values, GemmBroadcast layout, std::size_t m, std::size_t n,
                 std::size_t N) {
  switch (layout) {
  case GemmBroadcast::kScalar:
    return values[0];
  case GemmBroadcast::kRow:
    return values[n];
  case GemmBroadcast::kColumn:
    return values[m];
  case GemmBroadcast::kMatrix:
    return values[m * N + n];
  case GemmBroadcast::kNone:
    return T(0);
  }
  return T(0);
}

template <typename T> std::uint16_t ConvertOutput(T value, GemmOutputConversion conversion) {
  const float narrowed = static_cast<float>(value);
  return conversion == GemmOutputConversion::kFloat16 ? detail::FloatToFloat16Bits(narrowed)
                                                      : detail::FloatToBFloat16Bits(narrowed);
}

template <typename T, typename GemmFn>
void GemmWithEpilogue(bool trans_a, bool trans_b, std::size_t M, std::size_t N, std::size_t K,
                      T alpha, const T *A, const T *B, const GemmEpilogue<T> &epilogue, T *Y,
                      GemmFn gemm) {
  ValidateGemmEpilogue(M, N, epilogue);
  const bool no_epilogue = (epilogue.bias == nullptr || epilogue.beta == T(0) ||
                            epilogue.bias_layout == GemmBroadcast::kNone) &&
                           (epilogue.residual == nullptr || epilogue.residual_scale == T(0) ||
                            epilogue.residual_layout == GemmBroadcast::kNone) &&
                           epilogue.activation == GemmActivation::kNone &&
                           epilogue.output_conversion == GemmOutputConversion::kNone;
  if (no_epilogue) {
    gemm(trans_a, trans_b, M, N, K, alpha, A, B, T(0), nullptr, Y);
    return;
  }
  const bool matrix_bias_only = epilogue.bias != nullptr && epilogue.beta != T(0) &&
                                epilogue.bias_layout == GemmBroadcast::kMatrix &&
                                epilogue.residual == nullptr &&
                                epilogue.activation == GemmActivation::kNone &&
                                epilogue.output_conversion == GemmOutputConversion::kNone;
  if (matrix_bias_only) {
    gemm(trans_a, trans_b, M, N, K, alpha, A, B, epilogue.beta, epilogue.bias, Y);
    return;
  }

  gemm(trans_a, trans_b, M, N, K, alpha, A, B, T(0), nullptr, Y);
  ApplyGemmEpilogue(M, N, epilogue, Y);
}

} // namespace

template <typename T>
void ValidateGemmEpilogue(std::size_t M, std::size_t N, const GemmEpilogue<T> &epilogue) {
  const auto validate_layout = [](GemmBroadcast layout) {
    switch (layout) {
    case GemmBroadcast::kNone:
    case GemmBroadcast::kScalar:
    case GemmBroadcast::kRow:
    case GemmBroadcast::kColumn:
    case GemmBroadcast::kMatrix:
      return;
    }
    throw std::invalid_argument("Unsupported GEMM broadcast layout.");
  };
  validate_layout(epilogue.bias_layout);
  validate_layout(epilogue.residual_layout);
  switch (epilogue.activation) {
  case GemmActivation::kNone:
  case GemmActivation::kRelu:
    break;
  default:
    throw std::invalid_argument("Unsupported GEMM activation.");
  }
  switch (epilogue.output_conversion) {
  case GemmOutputConversion::kNone:
  case GemmOutputConversion::kFloat16:
  case GemmOutputConversion::kBFloat16:
    break;
  default:
    throw std::invalid_argument("Unsupported GEMM output conversion.");
  }
  if (M == 0 || N == 0) {
    return;
  }
  if (epilogue.beta != T(0) && epilogue.bias_layout != GemmBroadcast::kNone &&
      epilogue.bias == nullptr) {
    throw std::invalid_argument("GEMM epilogue bias must not be null.");
  }
  if (epilogue.residual_scale != T(0) && epilogue.residual_layout != GemmBroadcast::kNone &&
      epilogue.residual == nullptr) {
    throw std::invalid_argument("GEMM epilogue residual must not be null.");
  }
  if (epilogue.output_conversion != GemmOutputConversion::kNone &&
      epilogue.converted_output == nullptr) {
    throw std::invalid_argument("GEMM epilogue converted output must not be null.");
  }
}

template <typename T>
void ApplyGemmEpilogue(std::size_t M, std::size_t N, const GemmEpilogue<T> &epilogue, T *Y) {
  const bool has_bias = epilogue.bias != nullptr && epilogue.beta != T(0) &&
                        epilogue.bias_layout != GemmBroadcast::kNone;
  const bool has_residual = epilogue.residual != nullptr && epilogue.residual_scale != T(0) &&
                            epilogue.residual_layout != GemmBroadcast::kNone;
  const bool has_activation = epilogue.activation != GemmActivation::kNone;
  const bool converts_output = epilogue.output_conversion != GemmOutputConversion::kNone;
  if (!has_bias && !has_residual && !has_activation && !converts_output) {
    return;
  }

  const double cost = static_cast<double>(N) *
                      (1.0 + static_cast<double>(has_bias) + static_cast<double>(has_residual) +
                       static_cast<double>(has_activation) + static_cast<double>(converts_output));
  ParallelFor(static_cast<std::int64_t>(M), cost, [&](std::int64_t begin, std::int64_t end) {
    for (std::int64_t row = begin; row < end; ++row) {
      const std::size_t m = static_cast<std::size_t>(row);
      for (std::size_t n = 0; n < N; ++n) {
        const std::size_t index = m * N + n;
        T value = Y[index];
        if (has_bias) {
          value += epilogue.beta * BroadcastValue(epilogue.bias, epilogue.bias_layout, m, n, N);
        }
        if (has_residual) {
          value += epilogue.residual_scale *
                   BroadcastValue(epilogue.residual, epilogue.residual_layout, m, n, N);
        }
        if (epilogue.activation == GemmActivation::kRelu && value < T(0)) {
          value = T(0);
        }
        if (converts_output) {
          epilogue.converted_output[index] = ConvertOutput(value, epilogue.output_conversion);
        } else {
          Y[index] = value;
        }
      }
    }
  });
}

template void ValidateGemmEpilogue<float>(std::size_t, std::size_t, const GemmEpilogue<float> &);
template void ValidateGemmEpilogue<double>(std::size_t, std::size_t, const GemmEpilogue<double> &);
template void ApplyGemmEpilogue<float>(std::size_t, std::size_t, const GemmEpilogue<float> &,
                                       float *);
template void ApplyGemmEpilogue<double>(std::size_t, std::size_t, const GemmEpilogue<double> &,
                                        double *);

void GemmFloat32WithEpilogue(bool trans_a, bool trans_b, std::size_t M, std::size_t N,
                             std::size_t K, float alpha, const float *A, const float *B,
                             const GemmEpilogue<float> &epilogue, float *Y) {
  GemmWithEpilogue(trans_a, trans_b, M, N, K, alpha, A, B, epilogue, Y, GemmFloat32);
}

void GemmFloat64WithEpilogue(bool trans_a, bool trans_b, std::size_t M, std::size_t N,
                             std::size_t K, double alpha, const double *A, const double *B,
                             const GemmEpilogue<double> &epilogue, double *Y) {
  GemmWithEpilogue(trans_a, trans_b, M, N, K, alpha, A, B, epilogue, Y, GemmFloat64);
}

void GemmHalfWithEpilogue(bool is_bfloat16, bool trans_a, bool trans_b, std::size_t M,
                          std::size_t N, std::size_t K, float alpha, const std::uint16_t *A,
                          const std::uint16_t *B, const GemmEpilogue<float> &epilogue, float *Y) {
  ValidateGemmEpilogue(M, N, epilogue);
  detail::GemmHalfToFloat(is_bfloat16, trans_a, trans_b, M, N, K, alpha, A, B, Y);
  ApplyGemmEpilogue(M, N, epilogue, Y);
}

void GemmFloat8WithEpilogue(GemmFloat8Format format, bool trans_a, bool trans_b, std::size_t M,
                            std::size_t N, std::size_t K, float alpha, const std::uint8_t *A,
                            const std::uint8_t *B, const GemmEpilogue<float> &epilogue, float *Y) {
  ValidateGemmEpilogue(M, N, epilogue);
  detail::Float8Format internal_format;
  switch (format) {
  case GemmFloat8Format::kE4M3FN:
    internal_format = detail::Float8Format::kE4M3FN;
    break;
  case GemmFloat8Format::kE4M3FNUZ:
    internal_format = detail::Float8Format::kE4M3FNUZ;
    break;
  case GemmFloat8Format::kE5M2:
    internal_format = detail::Float8Format::kE5M2;
    break;
  case GemmFloat8Format::kE5M2FNUZ:
    internal_format = detail::Float8Format::kE5M2FNUZ;
    break;
  default:
    throw std::invalid_argument("Unsupported Float8 GEMM format.");
  }
  detail::GemmFloat8ToFloat(internal_format, trans_a, trans_b, M, N, K, alpha, A, B, Y);
  ApplyGemmEpilogue(M, N, epilogue, Y);
}

} // namespace onnx_light_cpu
