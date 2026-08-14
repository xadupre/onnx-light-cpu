// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/gemm/arm/gemm_kernel_arm.h"

#if defined(_MSC_VER)
#include <arm64_neon.h>
#else
#include <arm_neon.h>
#endif

namespace onnx_light_cpu {

namespace {

constexpr std::size_t kPrefetchDistanceK = 4;

template <typename T> inline void Prefetch(const T *pointer) {
#if !defined(_MSC_VER)
  __builtin_prefetch(pointer);
#else
  (void)pointer;
#endif
}

template <std::size_t MR>
void GemmMicroKernel_NEON_F32Impl(std::size_t nb, std::size_t K, float alpha, float beta,
                                  const float *Bmat, std::size_t N, const float *Crow_base,
                                  std::size_t Cstride, float *Yrow_base, std::size_t Ystride,
                                  std::size_t n0, GemmAccumMode mode, const float *Apack) {
  static_assert(MR >= 1 && MR <= kGemmNeonMR);
  const float32x4_t valpha = vdupq_n_f32(alpha);
  const float32x4_t vbeta = vdupq_n_f32(beta);
  const bool alpha_is_one = alpha == 1.0f;
  const bool beta_is_one = beta == 1.0f;
  std::size_t n = 0;
  for (; n + 8 <= nb; n += 8) {
    float32x4_t acc0[MR];
    float32x4_t acc1[MR];
    for (std::size_t r = 0; r < MR; ++r) {
      acc0[r] = vdupq_n_f32(0.0f);
      acc1[r] = vdupq_n_f32(0.0f);
    }
    const auto accumulate_k = [&](std::size_t k) {
      const float *Brow = Bmat + k * N + n0 + n;
      const float32x4_t vb0 = vld1q_f32(Brow);
      const float32x4_t vb1 = vld1q_f32(Brow + 4);
      if (k + kPrefetchDistanceK < K) {
        Prefetch(Brow + kPrefetchDistanceK * N);
      }
      for (std::size_t r = 0; r < MR; ++r) {
        acc0[r] = vfmaq_n_f32(acc0[r], vb0, Apack[r * K + k]);
        acc1[r] = vfmaq_n_f32(acc1[r], vb1, Apack[r * K + k]);
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
      float32x4_t res0 = alpha_is_one ? acc0[r] : vmulq_f32(valpha, acc0[r]);
      float32x4_t res1 = alpha_is_one ? acc1[r] : vmulq_f32(valpha, acc1[r]);
      if (mode == GemmAccumMode::kInitBias) {
        const float *Crow = Crow_base + r * Cstride + n0 + n;
        const float32x4_t vc0 = vld1q_f32(Crow);
        const float32x4_t vc1 = vld1q_f32(Crow + 4);
        res0 = vaddq_f32(res0, beta_is_one ? vc0 : vmulq_f32(vbeta, vc0));
        res1 = vaddq_f32(res1, beta_is_one ? vc1 : vmulq_f32(vbeta, vc1));
      } else if (mode == GemmAccumMode::kAccumulate) {
        res0 = vaddq_f32(res0, vld1q_f32(Yrow));
        res1 = vaddq_f32(res1, vld1q_f32(Yrow + 4));
      }
      vst1q_f32(Yrow, res0);
      vst1q_f32(Yrow + 4, res1);
    }
  }
  for (; n + 4 <= nb; n += 4) {
    float32x4_t acc[MR];
    for (std::size_t r = 0; r < MR; ++r) {
      acc[r] = vdupq_n_f32(0.0f);
    }
    for (std::size_t k = 0; k < K; ++k) {
      const float32x4_t vb = vld1q_f32(Bmat + k * N + n0 + n);
      for (std::size_t r = 0; r < MR; ++r) {
        acc[r] = vfmaq_n_f32(acc[r], vb, Apack[r * K + k]);
      }
    }
    for (std::size_t r = 0; r < MR; ++r) {
      float *Yrow = Yrow_base + r * Ystride + n0 + n;
      float32x4_t res = alpha_is_one ? acc[r] : vmulq_f32(valpha, acc[r]);
      if (mode == GemmAccumMode::kInitBias) {
        const float32x4_t vc = vld1q_f32(Crow_base + r * Cstride + n0 + n);
        res = vaddq_f32(res, beta_is_one ? vc : vmulq_f32(vbeta, vc));
      } else if (mode == GemmAccumMode::kAccumulate) {
        res = vaddq_f32(res, vld1q_f32(Yrow));
      }
      vst1q_f32(Yrow, res);
    }
  }
  if (n < nb) {
    GemmMicroKernel_Scalar_F32(MR, nb - n, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                               Ystride, n0 + n, mode, Apack);
  }
}

template <std::size_t MR>
void GemmMicroKernel_NEON_F64Impl(std::size_t nb, std::size_t K, double alpha, double beta,
                                  const double *Bmat, std::size_t N, const double *Crow_base,
                                  std::size_t Cstride, double *Yrow_base, std::size_t Ystride,
                                  std::size_t n0, GemmAccumMode mode, const double *Apack) {
  static_assert(MR >= 1 && MR <= kGemmNeonMR);
  const float64x2_t valpha = vdupq_n_f64(alpha);
  const float64x2_t vbeta = vdupq_n_f64(beta);
  const bool alpha_is_one = alpha == 1.0;
  const bool beta_is_one = beta == 1.0;
  std::size_t n = 0;
  for (; n + 4 <= nb; n += 4) {
    float64x2_t acc0[MR];
    float64x2_t acc1[MR];
    for (std::size_t r = 0; r < MR; ++r) {
      acc0[r] = vdupq_n_f64(0.0);
      acc1[r] = vdupq_n_f64(0.0);
    }
    const auto accumulate_k = [&](std::size_t k) {
      const double *Brow = Bmat + k * N + n0 + n;
      const float64x2_t vb0 = vld1q_f64(Brow);
      const float64x2_t vb1 = vld1q_f64(Brow + 2);
      if (k + kPrefetchDistanceK < K) {
        Prefetch(Brow + kPrefetchDistanceK * N);
      }
      for (std::size_t r = 0; r < MR; ++r) {
        acc0[r] = vfmaq_n_f64(acc0[r], vb0, Apack[r * K + k]);
        acc1[r] = vfmaq_n_f64(acc1[r], vb1, Apack[r * K + k]);
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
      float64x2_t res0 = alpha_is_one ? acc0[r] : vmulq_f64(valpha, acc0[r]);
      float64x2_t res1 = alpha_is_one ? acc1[r] : vmulq_f64(valpha, acc1[r]);
      if (mode == GemmAccumMode::kInitBias) {
        const double *Crow = Crow_base + r * Cstride + n0 + n;
        const float64x2_t vc0 = vld1q_f64(Crow);
        const float64x2_t vc1 = vld1q_f64(Crow + 2);
        res0 = vaddq_f64(res0, beta_is_one ? vc0 : vmulq_f64(vbeta, vc0));
        res1 = vaddq_f64(res1, beta_is_one ? vc1 : vmulq_f64(vbeta, vc1));
      } else if (mode == GemmAccumMode::kAccumulate) {
        res0 = vaddq_f64(res0, vld1q_f64(Yrow));
        res1 = vaddq_f64(res1, vld1q_f64(Yrow + 2));
      }
      vst1q_f64(Yrow, res0);
      vst1q_f64(Yrow + 2, res1);
    }
  }
  for (; n + 2 <= nb; n += 2) {
    float64x2_t acc[MR];
    for (std::size_t r = 0; r < MR; ++r) {
      acc[r] = vdupq_n_f64(0.0);
    }
    for (std::size_t k = 0; k < K; ++k) {
      const float64x2_t vb = vld1q_f64(Bmat + k * N + n0 + n);
      for (std::size_t r = 0; r < MR; ++r) {
        acc[r] = vfmaq_n_f64(acc[r], vb, Apack[r * K + k]);
      }
    }
    for (std::size_t r = 0; r < MR; ++r) {
      double *Yrow = Yrow_base + r * Ystride + n0 + n;
      float64x2_t res = alpha_is_one ? acc[r] : vmulq_f64(valpha, acc[r]);
      if (mode == GemmAccumMode::kInitBias) {
        const float64x2_t vc = vld1q_f64(Crow_base + r * Cstride + n0 + n);
        res = vaddq_f64(res, beta_is_one ? vc : vmulq_f64(vbeta, vc));
      } else if (mode == GemmAccumMode::kAccumulate) {
        res = vaddq_f64(res, vld1q_f64(Yrow));
      }
      vst1q_f64(Yrow, res);
    }
  }
  if (n < nb) {
    GemmMicroKernel_Scalar_F64(MR, nb - n, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                               Ystride, n0 + n, mode, Apack);
  }
}

} // namespace

#define DISPATCH_NEON_MR(Function, mr, ...)                                                        \
  switch (mr) {                                                                                    \
  case 1:                                                                                          \
    return Function<1>(__VA_ARGS__);                                                               \
  case 2:                                                                                          \
    return Function<2>(__VA_ARGS__);                                                               \
  case 3:                                                                                          \
    return Function<3>(__VA_ARGS__);                                                               \
  case 4:                                                                                          \
    return Function<4>(__VA_ARGS__);                                                               \
  case 5:                                                                                          \
    return Function<5>(__VA_ARGS__);                                                               \
  case 6:                                                                                          \
    return Function<6>(__VA_ARGS__);                                                               \
  default:                                                                                         \
    break;                                                                                         \
  }

void GemmMicroKernel_NEON_F32(std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                              float beta, const float *Bmat, std::size_t N, const float *Crow_base,
                              std::size_t Cstride, float *Yrow_base, std::size_t Ystride,
                              std::size_t n0, GemmAccumMode mode, const float *Apack) {
  DISPATCH_NEON_MR(GemmMicroKernel_NEON_F32Impl, mr, nb, K, alpha, beta, Bmat, N, Crow_base,
                   Cstride, Yrow_base, Ystride, n0, mode, Apack)
  GemmMicroKernel_Scalar_F32(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                             Ystride, n0, mode, Apack);
}

void GemmMicroKernel_NEON_F64(std::size_t mr, std::size_t nb, std::size_t K, double alpha,
                              double beta, const double *Bmat, std::size_t N,
                              const double *Crow_base, std::size_t Cstride, double *Yrow_base,
                              std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                              const double *Apack) {
  DISPATCH_NEON_MR(GemmMicroKernel_NEON_F64Impl, mr, nb, K, alpha, beta, Bmat, N, Crow_base,
                   Cstride, Yrow_base, Ystride, n0, mode, Apack)
  GemmMicroKernel_Scalar_F64(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                             Ystride, n0, mode, Apack);
}

#undef DISPATCH_NEON_MR

} // namespace onnx_light_cpu
