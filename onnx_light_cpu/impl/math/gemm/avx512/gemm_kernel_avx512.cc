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
// micro-kernel call processes up to ``kGemmAVX512MR`` (12) rows at once,
// register blocked over the whole (chunked) K reduction. Like the AVX/SSE2
// kernels, the main loop processes two vectors of columns per step (``NR = 2``)
// so each broadcast A element is reused across twice the column width,
// amortizing the broadcast/loop overhead over more FMAs.

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

template <std::size_t MR>
void GemmMicroKernel_AVX512_F32Impl(std::size_t nb, std::size_t K, float alpha, float beta,
                                    const float *Bmat, std::size_t N, const float *Crow_base,
                                    std::size_t Cstride, float *Yrow_base, std::size_t Ystride,
                                    std::size_t n0, GemmAccumMode mode, const float *Apack,
                                    std::size_t AStride = 0) {
  static_assert(MR >= 1 && MR <= kGemmAVX512MR);
  AStride = AStride == 0 ? K : AStride;
  const __m512 valpha = _mm512_set1_ps(alpha);
  const __m512 vbeta = _mm512_set1_ps(beta);
  const bool alpha_is_one = alpha == 1.0f;
  const bool beta_is_one = beta == 1.0f;
  std::size_t n = 0;
  // NR == 2: 32 lanes (two 512-bit vectors) per step.
  for (; n + 32 <= nb; n += 32) {
    __m512 acc0[MR];
    __m512 acc1[MR];
    for (std::size_t r = 0; r < MR; ++r) {
      acc0[r] = _mm512_setzero_ps();
      acc1[r] = _mm512_setzero_ps();
    }
    const auto accumulate_k = [&](std::size_t k) {
      const float *Brow = Bmat + k * N + n0 + n;
      const __m512 vb0 = _mm512_loadu_ps(Brow);
      const __m512 vb1 = _mm512_loadu_ps(Brow + 16);
      if (k + kGemmPrefetchDistanceK < K) {
        PrefetchT0(Brow + kGemmPrefetchDistanceK * N);
      }
      for (std::size_t r = 0; r < MR; ++r) {
        const __m512 va = _mm512_set1_ps(Apack[r * AStride + k]);
        acc0[r] = MulAdd(va, vb0, acc0[r]);
        acc1[r] = MulAdd(va, vb1, acc1[r]);
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
    for (std::size_t r = 0; r < MR; ++r) {
      float *Yrow = Yrow_base + r * Ystride + n0 + n;
      __m512 res0 = alpha_is_one ? acc0[r] : _mm512_mul_ps(valpha, acc0[r]);
      __m512 res1 = alpha_is_one ? acc1[r] : _mm512_mul_ps(valpha, acc1[r]);
      if (mode == GemmAccumMode::kInitBias) {
        const float *Crow = Crow_base + r * Cstride + n0 + n;
        const __m512 vc0 = _mm512_loadu_ps(Crow);
        const __m512 vc1 = _mm512_loadu_ps(Crow + 16);
        res0 = _mm512_add_ps(res0, beta_is_one ? vc0 : _mm512_mul_ps(vbeta, vc0));
        res1 = _mm512_add_ps(res1, beta_is_one ? vc1 : _mm512_mul_ps(vbeta, vc1));
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
    __m512 acc[MR];
    for (std::size_t r = 0; r < MR; ++r) {
      acc[r] = _mm512_setzero_ps();
    }
    const auto accumulate_k = [&](std::size_t k) {
      const __m512 vb = _mm512_loadu_ps(Bmat + k * N + n0 + n);
      for (std::size_t r = 0; r < MR; ++r) {
        const __m512 va = _mm512_set1_ps(Apack[r * AStride + k]);
        acc[r] = MulAdd(va, vb, acc[r]);
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
    for (std::size_t r = 0; r < MR; ++r) {
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
  // Scalar tail (< 16 lanes): reuse the shared scalar micro-kernel from
  // gemm_kernel.cc instead of duplicating the mode-driven combine logic.
  if (n < nb) {
    GemmMicroKernel_Scalar_F32(MR, nb - n, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                               Ystride, n0 + n, mode, Apack);
  }
}

template <std::size_t MR>
void GemmMicroKernel_AVX512_F64Impl(std::size_t nb, std::size_t K, double alpha, double beta,
                                    const double *Bmat, std::size_t N, const double *Crow_base,
                                    std::size_t Cstride, double *Yrow_base, std::size_t Ystride,
                                    std::size_t n0, GemmAccumMode mode, const double *Apack) {
  static_assert(MR >= 1 && MR <= kGemmAVX512MR);
  const __m512d valpha = _mm512_set1_pd(alpha);
  const __m512d vbeta = _mm512_set1_pd(beta);
  const bool alpha_is_one = alpha == 1.0;
  const bool beta_is_one = beta == 1.0;
  std::size_t n = 0;
  // NR == 2: 16 lanes (two 512-bit vectors) per step.
  for (; n + 16 <= nb; n += 16) {
    __m512d acc0[MR];
    __m512d acc1[MR];
    for (std::size_t r = 0; r < MR; ++r) {
      acc0[r] = _mm512_setzero_pd();
      acc1[r] = _mm512_setzero_pd();
    }
    const auto accumulate_k = [&](std::size_t k) {
      const double *Brow = Bmat + k * N + n0 + n;
      const __m512d vb0 = _mm512_loadu_pd(Brow);
      const __m512d vb1 = _mm512_loadu_pd(Brow + 8);
      if (k + kGemmPrefetchDistanceK < K) {
        PrefetchT0(Brow + kGemmPrefetchDistanceK * N);
      }
      for (std::size_t r = 0; r < MR; ++r) {
        const __m512d va = _mm512_set1_pd(Apack[r * K + k]);
        acc0[r] = MulAdd(va, vb0, acc0[r]);
        acc1[r] = MulAdd(va, vb1, acc1[r]);
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
    for (std::size_t r = 0; r < MR; ++r) {
      double *Yrow = Yrow_base + r * Ystride + n0 + n;
      __m512d res0 = alpha_is_one ? acc0[r] : _mm512_mul_pd(valpha, acc0[r]);
      __m512d res1 = alpha_is_one ? acc1[r] : _mm512_mul_pd(valpha, acc1[r]);
      if (mode == GemmAccumMode::kInitBias) {
        const double *Crow = Crow_base + r * Cstride + n0 + n;
        const __m512d vc0 = _mm512_loadu_pd(Crow);
        const __m512d vc1 = _mm512_loadu_pd(Crow + 8);
        res0 = _mm512_add_pd(res0, beta_is_one ? vc0 : _mm512_mul_pd(vbeta, vc0));
        res1 = _mm512_add_pd(res1, beta_is_one ? vc1 : _mm512_mul_pd(vbeta, vc1));
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
    __m512d acc[MR];
    for (std::size_t r = 0; r < MR; ++r) {
      acc[r] = _mm512_setzero_pd();
    }
    const auto accumulate_k = [&](std::size_t k) {
      const __m512d vb = _mm512_loadu_pd(Bmat + k * N + n0 + n);
      for (std::size_t r = 0; r < MR; ++r) {
        const __m512d va = _mm512_set1_pd(Apack[r * K + k]);
        acc[r] = MulAdd(va, vb, acc[r]);
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
    for (std::size_t r = 0; r < MR; ++r) {
      double *Yrow = Yrow_base + r * Ystride + n0 + n;
      __m512d res = alpha_is_one ? acc[r] : _mm512_mul_pd(valpha, acc[r]);
      if (mode == GemmAccumMode::kInitBias) {
        const __m512d vc = _mm512_loadu_pd(Crow_base + r * Cstride + n0 + n);
        res = _mm512_add_pd(res, beta_is_one ? vc : _mm512_mul_pd(vbeta, vc));
      } else if (mode == GemmAccumMode::kAccumulate) {
        res = _mm512_add_pd(res, _mm512_loadu_pd(Yrow));
      }
      _mm512_storeu_pd(Yrow, res);
    }
  }
  // Scalar tail (< 8 lanes): reuse the shared scalar micro-kernel from
  // gemm_kernel.cc instead of duplicating the mode-driven combine logic.
  if (n < nb) {
    GemmMicroKernel_Scalar_F64(MR, nb - n, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                               Ystride, n0 + n, mode, Apack);
  }
}

void GemmMicroKernel_AVX512_F32_MR12(std::size_t nb, std::size_t K, float alpha, float beta,
                                     const float *Bmat, std::size_t N, const float *Crow_base,
                                     std::size_t Cstride, float *Yrow_base, std::size_t Ystride,
                                     std::size_t n0, GemmAccumMode mode, const float *Apack,
                                     std::size_t AStride) {
  if (nb != 32) {
    return GemmMicroKernel_AVX512_F32Impl<12>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                              Yrow_base, Ystride, n0, mode, Apack, AStride);
  }

#define DECLARE_ACCUMULATORS(R)                                                                    \
  __m512 acc0_##R = _mm512_setzero_ps();                                                           \
  __m512 acc1_##R = _mm512_setzero_ps()
  DECLARE_ACCUMULATORS(0);
  DECLARE_ACCUMULATORS(1);
  DECLARE_ACCUMULATORS(2);
  DECLARE_ACCUMULATORS(3);
  DECLARE_ACCUMULATORS(4);
  DECLARE_ACCUMULATORS(5);
  DECLARE_ACCUMULATORS(6);
  DECLARE_ACCUMULATORS(7);
  DECLARE_ACCUMULATORS(8);
  DECLARE_ACCUMULATORS(9);
  DECLARE_ACCUMULATORS(10);
  DECLARE_ACCUMULATORS(11);
#undef DECLARE_ACCUMULATORS

  for (std::size_t k = 0; k < K; ++k) {
    const float *b_row = Bmat + k * N + n0;
    const __m512 vb0 = _mm512_loadu_ps(b_row);
    const __m512 vb1 = _mm512_loadu_ps(b_row + 16);
    if (k + kGemmPrefetchDistanceK < K) {
      PrefetchT0(b_row + kGemmPrefetchDistanceK * N);
    }
#define ACCUMULATE_ROW(R)                                                                          \
  do {                                                                                             \
    const __m512 va = _mm512_set1_ps(Apack[(R) * AStride + k]);                                    \
    acc0_##R = MulAdd(va, vb0, acc0_##R);                                                          \
    acc1_##R = MulAdd(va, vb1, acc1_##R);                                                          \
  } while (false)
    ACCUMULATE_ROW(0);
    ACCUMULATE_ROW(1);
    ACCUMULATE_ROW(2);
    ACCUMULATE_ROW(3);
    ACCUMULATE_ROW(4);
    ACCUMULATE_ROW(5);
    ACCUMULATE_ROW(6);
    ACCUMULATE_ROW(7);
    ACCUMULATE_ROW(8);
    ACCUMULATE_ROW(9);
    ACCUMULATE_ROW(10);
    ACCUMULATE_ROW(11);
#undef ACCUMULATE_ROW
  }

  const __m512 valpha = _mm512_set1_ps(alpha);
  const __m512 vbeta = _mm512_set1_ps(beta);
  const bool alpha_is_one = alpha == 1.0f;
  const bool beta_is_one = beta == 1.0f;
#define STORE_ROW(R)                                                                               \
  do {                                                                                             \
    float *y_row = Yrow_base + (R) * Ystride + n0;                                                 \
    __m512 result0 = alpha_is_one ? acc0_##R : _mm512_mul_ps(valpha, acc0_##R);                    \
    __m512 result1 = alpha_is_one ? acc1_##R : _mm512_mul_ps(valpha, acc1_##R);                    \
    if (mode == GemmAccumMode::kInitBias) {                                                        \
      const float *c_row = Crow_base + (R) * Cstride + n0;                                         \
      const __m512 bias0 = _mm512_loadu_ps(c_row);                                                 \
      const __m512 bias1 = _mm512_loadu_ps(c_row + 16);                                            \
      result0 = _mm512_add_ps(result0, beta_is_one ? bias0 : _mm512_mul_ps(vbeta, bias0));         \
      result1 = _mm512_add_ps(result1, beta_is_one ? bias1 : _mm512_mul_ps(vbeta, bias1));         \
    } else if (mode == GemmAccumMode::kAccumulate) {                                               \
      result0 = _mm512_add_ps(result0, _mm512_loadu_ps(y_row));                                    \
      result1 = _mm512_add_ps(result1, _mm512_loadu_ps(y_row + 16));                               \
    }                                                                                              \
    _mm512_storeu_ps(y_row, result0);                                                              \
    _mm512_storeu_ps(y_row + 16, result1);                                                         \
  } while (false)
  STORE_ROW(0);
  STORE_ROW(1);
  STORE_ROW(2);
  STORE_ROW(3);
  STORE_ROW(4);
  STORE_ROW(5);
  STORE_ROW(6);
  STORE_ROW(7);
  STORE_ROW(8);
  STORE_ROW(9);
  STORE_ROW(10);
  STORE_ROW(11);
#undef STORE_ROW
}

void GemmMicroKernel_AVX512_F64_MR6_NR32(std::size_t nb, std::size_t K, double alpha, double beta,
                                         const double *Bmat, std::size_t N, const double *Crow_base,
                                         std::size_t Cstride, double *Yrow_base,
                                         std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                                         const double *Apack) {
  if (nb != 32) {
    return GemmMicroKernel_AVX512_F64Impl<6>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                             Yrow_base, Ystride, n0, mode, Apack);
  }

#define DECLARE_ACCUMULATORS(R)                                                                    \
  __m512d acc0_##R = _mm512_setzero_pd();                                                          \
  __m512d acc1_##R = _mm512_setzero_pd();                                                          \
  __m512d acc2_##R = _mm512_setzero_pd();                                                          \
  __m512d acc3_##R = _mm512_setzero_pd()
  DECLARE_ACCUMULATORS(0);
  DECLARE_ACCUMULATORS(1);
  DECLARE_ACCUMULATORS(2);
  DECLARE_ACCUMULATORS(3);
  DECLARE_ACCUMULATORS(4);
  DECLARE_ACCUMULATORS(5);
#undef DECLARE_ACCUMULATORS

  for (std::size_t k = 0; k < K; ++k) {
    const double *b_row = Bmat + k * N + n0;
    const __m512d vb0 = _mm512_loadu_pd(b_row);
    const __m512d vb1 = _mm512_loadu_pd(b_row + 8);
    const __m512d vb2 = _mm512_loadu_pd(b_row + 16);
    const __m512d vb3 = _mm512_loadu_pd(b_row + 24);
    if (k + kGemmPrefetchDistanceK < K) {
      PrefetchT0(b_row + kGemmPrefetchDistanceK * N);
    }
#define ACCUMULATE_ROW(R)                                                                          \
  do {                                                                                             \
    const __m512d va = _mm512_set1_pd(Apack[(R) * K + k]);                                         \
    acc0_##R = MulAdd(va, vb0, acc0_##R);                                                          \
    acc1_##R = MulAdd(va, vb1, acc1_##R);                                                          \
    acc2_##R = MulAdd(va, vb2, acc2_##R);                                                          \
    acc3_##R = MulAdd(va, vb3, acc3_##R);                                                          \
  } while (false)
    ACCUMULATE_ROW(0);
    ACCUMULATE_ROW(1);
    ACCUMULATE_ROW(2);
    ACCUMULATE_ROW(3);
    ACCUMULATE_ROW(4);
    ACCUMULATE_ROW(5);
#undef ACCUMULATE_ROW
  }

  const __m512d valpha = _mm512_set1_pd(alpha);
  const __m512d vbeta = _mm512_set1_pd(beta);
  const bool alpha_is_one = alpha == 1.0;
  const bool beta_is_one = beta == 1.0;
#define STORE_ROW(R)                                                                               \
  do {                                                                                             \
    double *y_row = Yrow_base + (R) * Ystride + n0;                                                \
    __m512d result0 = alpha_is_one ? acc0_##R : _mm512_mul_pd(valpha, acc0_##R);                   \
    __m512d result1 = alpha_is_one ? acc1_##R : _mm512_mul_pd(valpha, acc1_##R);                   \
    __m512d result2 = alpha_is_one ? acc2_##R : _mm512_mul_pd(valpha, acc2_##R);                   \
    __m512d result3 = alpha_is_one ? acc3_##R : _mm512_mul_pd(valpha, acc3_##R);                   \
    if (mode == GemmAccumMode::kInitBias) {                                                        \
      const double *c_row = Crow_base + (R) * Cstride + n0;                                        \
      const __m512d bias0 = _mm512_loadu_pd(c_row);                                                \
      const __m512d bias1 = _mm512_loadu_pd(c_row + 8);                                            \
      const __m512d bias2 = _mm512_loadu_pd(c_row + 16);                                           \
      const __m512d bias3 = _mm512_loadu_pd(c_row + 24);                                           \
      result0 = _mm512_add_pd(result0, beta_is_one ? bias0 : _mm512_mul_pd(vbeta, bias0));         \
      result1 = _mm512_add_pd(result1, beta_is_one ? bias1 : _mm512_mul_pd(vbeta, bias1));         \
      result2 = _mm512_add_pd(result2, beta_is_one ? bias2 : _mm512_mul_pd(vbeta, bias2));         \
      result3 = _mm512_add_pd(result3, beta_is_one ? bias3 : _mm512_mul_pd(vbeta, bias3));         \
    } else if (mode == GemmAccumMode::kAccumulate) {                                               \
      result0 = _mm512_add_pd(result0, _mm512_loadu_pd(y_row));                                    \
      result1 = _mm512_add_pd(result1, _mm512_loadu_pd(y_row + 8));                                \
      result2 = _mm512_add_pd(result2, _mm512_loadu_pd(y_row + 16));                               \
      result3 = _mm512_add_pd(result3, _mm512_loadu_pd(y_row + 24));                               \
    }                                                                                              \
    _mm512_storeu_pd(y_row, result0);                                                              \
    _mm512_storeu_pd(y_row + 8, result1);                                                          \
    _mm512_storeu_pd(y_row + 16, result2);                                                         \
    _mm512_storeu_pd(y_row + 24, result3);                                                         \
  } while (false)
  STORE_ROW(0);
  STORE_ROW(1);
  STORE_ROW(2);
  STORE_ROW(3);
  STORE_ROW(4);
  STORE_ROW(5);
#undef STORE_ROW
}

void GemmMicroKernel_AVX512_F32_MR12_NR16(std::size_t K, float alpha, float beta, const float *Bmat,
                                          std::size_t N, const float *Crow_base,
                                          std::size_t Cstride, float *Yrow_base,
                                          std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                                          const float *A, std::size_t AStride) {
#define DECLARE_ACCUMULATOR(R) __m512 acc_##R = _mm512_setzero_ps()
  DECLARE_ACCUMULATOR(0);
  DECLARE_ACCUMULATOR(1);
  DECLARE_ACCUMULATOR(2);
  DECLARE_ACCUMULATOR(3);
  DECLARE_ACCUMULATOR(4);
  DECLARE_ACCUMULATOR(5);
  DECLARE_ACCUMULATOR(6);
  DECLARE_ACCUMULATOR(7);
  DECLARE_ACCUMULATOR(8);
  DECLARE_ACCUMULATOR(9);
  DECLARE_ACCUMULATOR(10);
  DECLARE_ACCUMULATOR(11);
#undef DECLARE_ACCUMULATOR

  for (std::size_t k = 0; k < K; ++k) {
    const float *b_row = Bmat + k * N + n0;
    const __m512 vb = _mm512_loadu_ps(b_row);
    if (k + kGemmPrefetchDistanceK < K) {
      PrefetchT0(b_row + kGemmPrefetchDistanceK * N);
    }
#define ACCUMULATE_ROW(R) acc_##R = MulAdd(_mm512_set1_ps(A[(R) * AStride + k]), vb, acc_##R)
    ACCUMULATE_ROW(0);
    ACCUMULATE_ROW(1);
    ACCUMULATE_ROW(2);
    ACCUMULATE_ROW(3);
    ACCUMULATE_ROW(4);
    ACCUMULATE_ROW(5);
    ACCUMULATE_ROW(6);
    ACCUMULATE_ROW(7);
    ACCUMULATE_ROW(8);
    ACCUMULATE_ROW(9);
    ACCUMULATE_ROW(10);
    ACCUMULATE_ROW(11);
#undef ACCUMULATE_ROW
  }

  const __m512 valpha = _mm512_set1_ps(alpha);
  const __m512 vbeta = _mm512_set1_ps(beta);
  const bool alpha_is_one = alpha == 1.0f;
  const bool beta_is_one = beta == 1.0f;
#define STORE_ROW(R)                                                                               \
  do {                                                                                             \
    float *y_row = Yrow_base + (R) * Ystride + n0;                                                 \
    __m512 result = alpha_is_one ? acc_##R : _mm512_mul_ps(valpha, acc_##R);                       \
    if (mode == GemmAccumMode::kInitBias) {                                                        \
      const __m512 bias = _mm512_loadu_ps(Crow_base + (R) * Cstride + n0);                         \
      result = _mm512_add_ps(result, beta_is_one ? bias : _mm512_mul_ps(vbeta, bias));             \
    } else if (mode == GemmAccumMode::kAccumulate) {                                               \
      result = _mm512_add_ps(result, _mm512_loadu_ps(y_row));                                      \
    }                                                                                              \
    _mm512_storeu_ps(y_row, result);                                                               \
  } while (false)
  STORE_ROW(0);
  STORE_ROW(1);
  STORE_ROW(2);
  STORE_ROW(3);
  STORE_ROW(4);
  STORE_ROW(5);
  STORE_ROW(6);
  STORE_ROW(7);
  STORE_ROW(8);
  STORE_ROW(9);
  STORE_ROW(10);
  STORE_ROW(11);
#undef STORE_ROW
}

void GemmMicroKernel_AVX512_F32_MR24_NR16(std::size_t K, float alpha, float beta, const float *Bmat,
                                          std::size_t N, const float *Crow_base,
                                          std::size_t Cstride, float *Yrow_base,
                                          std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                                          const float *A, std::size_t AStride) {
#define DECLARE_ACCUMULATOR(R) __m512 acc_##R = _mm512_setzero_ps()
  DECLARE_ACCUMULATOR(0);
  DECLARE_ACCUMULATOR(1);
  DECLARE_ACCUMULATOR(2);
  DECLARE_ACCUMULATOR(3);
  DECLARE_ACCUMULATOR(4);
  DECLARE_ACCUMULATOR(5);
  DECLARE_ACCUMULATOR(6);
  DECLARE_ACCUMULATOR(7);
  DECLARE_ACCUMULATOR(8);
  DECLARE_ACCUMULATOR(9);
  DECLARE_ACCUMULATOR(10);
  DECLARE_ACCUMULATOR(11);
  DECLARE_ACCUMULATOR(12);
  DECLARE_ACCUMULATOR(13);
  DECLARE_ACCUMULATOR(14);
  DECLARE_ACCUMULATOR(15);
  DECLARE_ACCUMULATOR(16);
  DECLARE_ACCUMULATOR(17);
  DECLARE_ACCUMULATOR(18);
  DECLARE_ACCUMULATOR(19);
  DECLARE_ACCUMULATOR(20);
  DECLARE_ACCUMULATOR(21);
  DECLARE_ACCUMULATOR(22);
  DECLARE_ACCUMULATOR(23);
#undef DECLARE_ACCUMULATOR

  for (std::size_t k = 0; k < K; ++k) {
    const float *b_row = Bmat + k * N + n0;
    const __m512 vb = _mm512_loadu_ps(b_row);
#define ACCUMULATE_ROW(R) acc_##R = MulAdd(_mm512_set1_ps(A[(R) * AStride + k]), vb, acc_##R)
    ACCUMULATE_ROW(0);
    ACCUMULATE_ROW(1);
    ACCUMULATE_ROW(2);
    ACCUMULATE_ROW(3);
    ACCUMULATE_ROW(4);
    ACCUMULATE_ROW(5);
    ACCUMULATE_ROW(6);
    ACCUMULATE_ROW(7);
    ACCUMULATE_ROW(8);
    ACCUMULATE_ROW(9);
    ACCUMULATE_ROW(10);
    ACCUMULATE_ROW(11);
    ACCUMULATE_ROW(12);
    ACCUMULATE_ROW(13);
    ACCUMULATE_ROW(14);
    ACCUMULATE_ROW(15);
    ACCUMULATE_ROW(16);
    ACCUMULATE_ROW(17);
    ACCUMULATE_ROW(18);
    ACCUMULATE_ROW(19);
    ACCUMULATE_ROW(20);
    ACCUMULATE_ROW(21);
    ACCUMULATE_ROW(22);
    ACCUMULATE_ROW(23);
#undef ACCUMULATE_ROW
  }

  const __m512 valpha = _mm512_set1_ps(alpha);
  const __m512 vbeta = _mm512_set1_ps(beta);
  const bool alpha_is_one = alpha == 1.0f;
  const bool beta_is_one = beta == 1.0f;
#define STORE_ROW(R)                                                                               \
  do {                                                                                             \
    float *y_row = Yrow_base + (R) * Ystride + n0;                                                 \
    __m512 result = alpha_is_one ? acc_##R : _mm512_mul_ps(valpha, acc_##R);                       \
    if (mode == GemmAccumMode::kInitBias) {                                                        \
      const __m512 bias = _mm512_loadu_ps(Crow_base + (R) * Cstride + n0);                         \
      result = _mm512_add_ps(result, beta_is_one ? bias : _mm512_mul_ps(vbeta, bias));             \
    } else if (mode == GemmAccumMode::kAccumulate) {                                               \
      result = _mm512_add_ps(result, _mm512_loadu_ps(y_row));                                      \
    }                                                                                              \
    _mm512_storeu_ps(y_row, result);                                                               \
  } while (false)
  STORE_ROW(0);
  STORE_ROW(1);
  STORE_ROW(2);
  STORE_ROW(3);
  STORE_ROW(4);
  STORE_ROW(5);
  STORE_ROW(6);
  STORE_ROW(7);
  STORE_ROW(8);
  STORE_ROW(9);
  STORE_ROW(10);
  STORE_ROW(11);
  STORE_ROW(12);
  STORE_ROW(13);
  STORE_ROW(14);
  STORE_ROW(15);
  STORE_ROW(16);
  STORE_ROW(17);
  STORE_ROW(18);
  STORE_ROW(19);
  STORE_ROW(20);
  STORE_ROW(21);
  STORE_ROW(22);
  STORE_ROW(23);
#undef STORE_ROW
}

void GemmMicroKernel_AVX512_F32(std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                                float beta, const float *Bmat, std::size_t N,
                                const float *Crow_base, std::size_t Cstride, float *Yrow_base,
                                std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                                const float *Apack) {
  switch (mr) {
  case 1:
    return GemmMicroKernel_AVX512_F32Impl<1>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                             Yrow_base, Ystride, n0, mode, Apack);
  case 2:
    return GemmMicroKernel_AVX512_F32Impl<2>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                             Yrow_base, Ystride, n0, mode, Apack);
  case 3:
    return GemmMicroKernel_AVX512_F32Impl<3>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                             Yrow_base, Ystride, n0, mode, Apack);
  case 4:
    return GemmMicroKernel_AVX512_F32Impl<4>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                             Yrow_base, Ystride, n0, mode, Apack);
  case 5:
    return GemmMicroKernel_AVX512_F32Impl<5>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                             Yrow_base, Ystride, n0, mode, Apack);
  case 6:
    return GemmMicroKernel_AVX512_F32Impl<6>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                             Yrow_base, Ystride, n0, mode, Apack);
  case 7:
    return GemmMicroKernel_AVX512_F32Impl<7>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                             Yrow_base, Ystride, n0, mode, Apack);
  case 8:
    return GemmMicroKernel_AVX512_F32Impl<8>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                             Yrow_base, Ystride, n0, mode, Apack);
  case 12:
    if (nb == 16) {
      return GemmMicroKernel_AVX512_F32_MR12_NR16(K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                                  Yrow_base, Ystride, n0, mode, Apack, K);
    }
    return GemmMicroKernel_AVX512_F32_MR12(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                           Yrow_base, Ystride, n0, mode, Apack, K);
  default:
    return GemmMicroKernel_Scalar_F32(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                      Yrow_base, Ystride, n0, mode, Apack);
  }
}

void GemmMicroKernel_AVX512_F32_StridedA(std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                                         float beta, const float *Bmat, std::size_t N,
                                         const float *Crow_base, std::size_t Cstride,
                                         float *Yrow_base, std::size_t Ystride, std::size_t n0,
                                         GemmAccumMode mode, const float *A, std::size_t AStride) {
  switch (mr) {
  case 1:
    return GemmMicroKernel_AVX512_F32Impl<1>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                             Yrow_base, Ystride, n0, mode, A, AStride);
  case 2:
    return GemmMicroKernel_AVX512_F32Impl<2>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                             Yrow_base, Ystride, n0, mode, A, AStride);
  case 3:
    return GemmMicroKernel_AVX512_F32Impl<3>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                             Yrow_base, Ystride, n0, mode, A, AStride);
  case 4:
    return GemmMicroKernel_AVX512_F32Impl<4>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                             Yrow_base, Ystride, n0, mode, A, AStride);
  case 5:
    return GemmMicroKernel_AVX512_F32Impl<5>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                             Yrow_base, Ystride, n0, mode, A, AStride);
  case 6:
    return GemmMicroKernel_AVX512_F32Impl<6>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                             Yrow_base, Ystride, n0, mode, A, AStride);
  case 7:
    return GemmMicroKernel_AVX512_F32Impl<7>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                             Yrow_base, Ystride, n0, mode, A, AStride);
  case 8:
    return GemmMicroKernel_AVX512_F32Impl<8>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                             Yrow_base, Ystride, n0, mode, A, AStride);
  case 12:
    if (nb == 16) {
      return GemmMicroKernel_AVX512_F32_MR12_NR16(K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                                  Yrow_base, Ystride, n0, mode, A, AStride);
    }
    return GemmMicroKernel_AVX512_F32_MR12(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                           Yrow_base, Ystride, n0, mode, A, AStride);
  default:
    return GemmMicroKernel_Scalar_F32(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                      Yrow_base, Ystride, n0, mode, A);
  }
}

void GemmMicroKernel_AVX512_F64(std::size_t mr, std::size_t nb, std::size_t K, double alpha,
                                double beta, const double *Bmat, std::size_t N,
                                const double *Crow_base, std::size_t Cstride, double *Yrow_base,
                                std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                                const double *Apack) {
  switch (mr) {
  case 1:
    return GemmMicroKernel_AVX512_F64Impl<1>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                             Yrow_base, Ystride, n0, mode, Apack);
  case 2:
    return GemmMicroKernel_AVX512_F64Impl<2>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                             Yrow_base, Ystride, n0, mode, Apack);
  case 3:
    return GemmMicroKernel_AVX512_F64Impl<3>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                             Yrow_base, Ystride, n0, mode, Apack);
  case 4:
    return GemmMicroKernel_AVX512_F64Impl<4>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                             Yrow_base, Ystride, n0, mode, Apack);
  case 5:
    return GemmMicroKernel_AVX512_F64Impl<5>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                             Yrow_base, Ystride, n0, mode, Apack);
  case 6:
    if (nb == 32) {
      return GemmMicroKernel_AVX512_F64_MR6_NR32(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                                 Yrow_base, Ystride, n0, mode, Apack);
    }
    return GemmMicroKernel_AVX512_F64Impl<6>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                             Yrow_base, Ystride, n0, mode, Apack);
  case 7:
    return GemmMicroKernel_AVX512_F64Impl<7>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                             Yrow_base, Ystride, n0, mode, Apack);
  case 8:
    return GemmMicroKernel_AVX512_F64Impl<8>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                             Yrow_base, Ystride, n0, mode, Apack);
  default:
    return GemmMicroKernel_Scalar_F64(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                      Yrow_base, Ystride, n0, mode, Apack);
  }
}

} // namespace onnx_light_cpu
