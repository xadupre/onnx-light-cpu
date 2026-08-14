// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/gemm/avx2/gemm_kernel_avx2_fma.h"

#include <immintrin.h>

namespace onnx_light_cpu {

namespace {

constexpr int kGemmPrefetchDistanceK = 4;

template <typename T> inline void PrefetchT0(const T *ptr) {
  _mm_prefetch(reinterpret_cast<const char *>(ptr), _MM_HINT_T0);
}

} // namespace

template <std::size_t MR>
void GemmMicroKernel_AVX2FMA_F32Impl(std::size_t nb, std::size_t K, float alpha, float beta,
                                     const float *Bmat, std::size_t N, const float *Crow_base,
                                     std::size_t Cstride, float *Yrow_base, std::size_t Ystride,
                                     std::size_t n0, GemmAccumMode mode, const float *Apack) {
  static_assert(MR >= 1 && MR <= kGemmMR);
  const __m256 valpha = _mm256_set1_ps(alpha);
  const __m256 vbeta = _mm256_set1_ps(beta);
  const bool alpha_is_one = alpha == 1.0f;
  const bool beta_is_one = beta == 1.0f;
  std::size_t n = 0;
  for (; n + 16 <= nb; n += 16) {
    __m256 acc0[MR];
    __m256 acc1[MR];
    for (std::size_t r = 0; r < MR; ++r) {
      acc0[r] = _mm256_setzero_ps();
      acc1[r] = _mm256_setzero_ps();
    }
    const auto accumulate_k = [&](std::size_t k) {
      const float *Brow = Bmat + k * N + n0 + n;
      const __m256 vb0 = _mm256_loadu_ps(Brow);
      const __m256 vb1 = _mm256_loadu_ps(Brow + 8);
      if (k + kGemmPrefetchDistanceK < K) {
        PrefetchT0(Brow + kGemmPrefetchDistanceK * N);
      }
      for (std::size_t r = 0; r < MR; ++r) {
        const __m256 va = _mm256_set1_ps(Apack[r * K + k]);
        acc0[r] = _mm256_fmadd_ps(va, vb0, acc0[r]);
        acc1[r] = _mm256_fmadd_ps(va, vb1, acc1[r]);
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
    __m256 acc[MR];
    for (std::size_t r = 0; r < MR; ++r) {
      acc[r] = _mm256_setzero_ps();
    }
    const auto accumulate_k = [&](std::size_t k) {
      const __m256 vb = _mm256_loadu_ps(Bmat + k * N + n0 + n);
      for (std::size_t r = 0; r < MR; ++r) {
        const __m256 va = _mm256_set1_ps(Apack[r * K + k]);
        acc[r] = _mm256_fmadd_ps(va, vb, acc[r]);
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
    GemmMicroKernel_Scalar_F32(MR, nb - n, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                               Ystride, n0 + n, mode, Apack);
  }
}

template <std::size_t MR>
void GemmMicroKernel_AVX2FMA_F64Impl(std::size_t nb, std::size_t K, double alpha, double beta,
                                     const double *Bmat, std::size_t N, const double *Crow_base,
                                     std::size_t Cstride, double *Yrow_base, std::size_t Ystride,
                                     std::size_t n0, GemmAccumMode mode, const double *Apack) {
  static_assert(MR >= 1 && MR <= kGemmMR);
  const __m256d valpha = _mm256_set1_pd(alpha);
  const __m256d vbeta = _mm256_set1_pd(beta);
  const bool alpha_is_one = alpha == 1.0;
  const bool beta_is_one = beta == 1.0;
  std::size_t n = 0;
  for (; n + 8 <= nb; n += 8) {
    __m256d acc0[MR];
    __m256d acc1[MR];
    for (std::size_t r = 0; r < MR; ++r) {
      acc0[r] = _mm256_setzero_pd();
      acc1[r] = _mm256_setzero_pd();
    }
    const auto accumulate_k = [&](std::size_t k) {
      const double *Brow = Bmat + k * N + n0 + n;
      const __m256d vb0 = _mm256_loadu_pd(Brow);
      const __m256d vb1 = _mm256_loadu_pd(Brow + 4);
      if (k + kGemmPrefetchDistanceK < K) {
        PrefetchT0(Brow + kGemmPrefetchDistanceK * N);
      }
      for (std::size_t r = 0; r < MR; ++r) {
        const __m256d va = _mm256_set1_pd(Apack[r * K + k]);
        acc0[r] = _mm256_fmadd_pd(va, vb0, acc0[r]);
        acc1[r] = _mm256_fmadd_pd(va, vb1, acc1[r]);
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
    __m256d acc[MR];
    for (std::size_t r = 0; r < MR; ++r) {
      acc[r] = _mm256_setzero_pd();
    }
    const auto accumulate_k = [&](std::size_t k) {
      const __m256d vb = _mm256_loadu_pd(Bmat + k * N + n0 + n);
      for (std::size_t r = 0; r < MR; ++r) {
        const __m256d va = _mm256_set1_pd(Apack[r * K + k]);
        acc[r] = _mm256_fmadd_pd(va, vb, acc[r]);
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
    GemmMicroKernel_Scalar_F64(MR, nb - n, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                               Ystride, n0 + n, mode, Apack);
  }
}

void GemmMicroKernel_AVX2FMA_F32(std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                                 float beta, const float *Bmat, std::size_t N,
                                 const float *Crow_base, std::size_t Cstride, float *Yrow_base,
                                 std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                                 const float *Apack) {
  switch (mr) {
  case 1:
    return GemmMicroKernel_AVX2FMA_F32Impl<1>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                              Yrow_base, Ystride, n0, mode, Apack);
  case 2:
    return GemmMicroKernel_AVX2FMA_F32Impl<2>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                              Yrow_base, Ystride, n0, mode, Apack);
  case 3:
    return GemmMicroKernel_AVX2FMA_F32Impl<3>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                              Yrow_base, Ystride, n0, mode, Apack);
  case 4:
    return GemmMicroKernel_AVX2FMA_F32Impl<4>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                              Yrow_base, Ystride, n0, mode, Apack);
  default:
    return GemmMicroKernel_Scalar_F32(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                      Yrow_base, Ystride, n0, mode, Apack);
  }
}

void GemmMicroKernel_AVX2FMA_F64(std::size_t mr, std::size_t nb, std::size_t K, double alpha,
                                 double beta, const double *Bmat, std::size_t N,
                                 const double *Crow_base, std::size_t Cstride, double *Yrow_base,
                                 std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                                 const double *Apack) {
  switch (mr) {
  case 1:
    return GemmMicroKernel_AVX2FMA_F64Impl<1>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                              Yrow_base, Ystride, n0, mode, Apack);
  case 2:
    return GemmMicroKernel_AVX2FMA_F64Impl<2>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                              Yrow_base, Ystride, n0, mode, Apack);
  case 3:
    return GemmMicroKernel_AVX2FMA_F64Impl<3>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                              Yrow_base, Ystride, n0, mode, Apack);
  case 4:
    return GemmMicroKernel_AVX2FMA_F64Impl<4>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                              Yrow_base, Ystride, n0, mode, Apack);
  default:
    return GemmMicroKernel_Scalar_F64(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                      Yrow_base, Ystride, n0, mode, Apack);
  }
}

} // namespace onnx_light_cpu
