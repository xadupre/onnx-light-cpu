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

void GemmMicroKernel_AVX2FMA_F32(std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                                 float beta, const float *Bmat, std::size_t N,
                                 const float *Crow_base, std::size_t Cstride, float *Yrow_base,
                                 std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                                 const float *Apack) {
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
    const auto accumulate_k = [&](std::size_t k) {
      const float *Brow = Bmat + k * N + n0 + n;
      const __m256 vb0 = _mm256_loadu_ps(Brow);
      const __m256 vb1 = _mm256_loadu_ps(Brow + 8);
      if (k + kGemmPrefetchDistanceK < K) {
        PrefetchT0(Brow + kGemmPrefetchDistanceK * N);
      }
      for (std::size_t r = 0; r < mr; ++r) {
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
    const auto accumulate_k = [&](std::size_t k) {
      const __m256 vb = _mm256_loadu_ps(Bmat + k * N + n0 + n);
      for (std::size_t r = 0; r < mr; ++r) {
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
    GemmMicroKernel_Scalar_F32(mr, nb - n, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                               Ystride, n0 + n, mode, Apack);
  }
}

void GemmMicroKernel_AVX2FMA_F64(std::size_t mr, std::size_t nb, std::size_t K, double alpha,
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
    const auto accumulate_k = [&](std::size_t k) {
      const double *Brow = Bmat + k * N + n0 + n;
      const __m256d vb0 = _mm256_loadu_pd(Brow);
      const __m256d vb1 = _mm256_loadu_pd(Brow + 4);
      if (k + kGemmPrefetchDistanceK < K) {
        PrefetchT0(Brow + kGemmPrefetchDistanceK * N);
      }
      for (std::size_t r = 0; r < mr; ++r) {
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
    const auto accumulate_k = [&](std::size_t k) {
      const __m256d vb = _mm256_loadu_pd(Bmat + k * N + n0 + n);
      for (std::size_t r = 0; r < mr; ++r) {
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
    GemmMicroKernel_Scalar_F64(mr, nb - n, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                               Ystride, n0 + n, mode, Apack);
  }
}

} // namespace onnx_light_cpu
