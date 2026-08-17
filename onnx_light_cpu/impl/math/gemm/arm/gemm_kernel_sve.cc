// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/gemm/arm/gemm_kernel_arm.h"

#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <arm_sve.h>

#include <cstdint>

namespace onnx_light_cpu {

namespace {

inline svfloat32_t FinishF32(svbool_t pg, svfloat32_t acc, float alpha, float beta,
                             const float *Crow, float *Yrow, GemmAccumMode mode) {
  svfloat32_t result = alpha == 1.0f ? acc : svmul_n_f32_x(pg, acc, alpha);
  if (mode == GemmAccumMode::kInitBias) {
    svfloat32_t bias = svld1_f32(pg, Crow);
    if (beta != 1.0f) {
      bias = svmul_n_f32_x(pg, bias, beta);
    }
    result = svadd_f32_x(pg, result, bias);
  } else if (mode == GemmAccumMode::kAccumulate) {
    result = svadd_f32_x(pg, result, svld1_f32(pg, Yrow));
  }
  return result;
}

inline svfloat64_t FinishF64(svbool_t pg, svfloat64_t acc, double alpha, double beta,
                             const double *Crow, double *Yrow, GemmAccumMode mode) {
  svfloat64_t result = alpha == 1.0 ? acc : svmul_n_f64_x(pg, acc, alpha);
  if (mode == GemmAccumMode::kInitBias) {
    svfloat64_t bias = svld1_f64(pg, Crow);
    if (beta != 1.0) {
      bias = svmul_n_f64_x(pg, bias, beta);
    }
    result = svadd_f64_x(pg, result, bias);
  } else if (mode == GemmAccumMode::kAccumulate) {
    result = svadd_f64_x(pg, result, svld1_f64(pg, Yrow));
  }
  return result;
}

// Widens up to ``lanes`` contiguous BFLOAT16 patterns to a float32 vector with
// an SVE zero-extending halfword load and a 16-bit shift, matching the scalar
// bit decode. The predicate governs the 32-bit destination lanes so the
// predicated tail never reads inactive columns.
inline svfloat32_t WidenBf16_SVE(svbool_t pg, const std::uint16_t *src) {
  const svuint32_t raw = svld1uh_u32(pg, src);
  return svreinterpret_f32_u32(svlsl_n_u32_x(pg, raw, 16));
}

// Widens up to ``lanes`` contiguous FLOAT16 patterns to a float32 vector. The
// zero-extending halfword load places each pattern in the low 16 bits of its
// 32-bit lane, which is exactly where the SVE ``FCVT`` (``svcvt_f32_f16``)
// widening reads the half-precision source, so the result matches the scalar
// bit decode. Half precision is part of baseline SVE.
inline svfloat32_t WidenFp16_SVE(svbool_t pg, const std::uint16_t *src) {
  const svuint32_t raw = svld1uh_u32(pg, src);
  return svcvt_f32_f16_x(pg, svreinterpret_f16_u32(raw));
}

template <std::size_t MR>
void GemmMicroKernel_SVE_F32Impl(std::size_t nb, std::size_t K, float alpha, float beta,
                                 const float *Bmat, std::size_t N, const float *Crow_base,
                                 std::size_t Cstride, float *Yrow_base, std::size_t Ystride,
                                 std::size_t n0, GemmAccumMode mode, const float *Apack) {
  static_assert(MR >= 1 && MR <= kGemmSveMR);
  const std::size_t lanes = svcntw();
  const svbool_t all = svptrue_b32();
  std::size_t n = 0;
  for (; n + 2 * lanes <= nb; n += 2 * lanes) {
    svfloat32_t acc00 = svdup_f32(0.0f);
    svfloat32_t acc01 = svdup_f32(0.0f);
    svfloat32_t acc10 = svdup_f32(0.0f);
    svfloat32_t acc11 = svdup_f32(0.0f);
    svfloat32_t acc20 = svdup_f32(0.0f);
    svfloat32_t acc21 = svdup_f32(0.0f);
    svfloat32_t acc30 = svdup_f32(0.0f);
    svfloat32_t acc31 = svdup_f32(0.0f);
    for (std::size_t k = 0; k < K; ++k) {
      const float *Brow = Bmat + k * N + n0 + n;
      const svfloat32_t vb0 = svld1_f32(all, Brow);
      const svfloat32_t vb1 = svld1_f32(all, Brow + lanes);
      acc00 = svmla_n_f32_x(all, acc00, vb0, Apack[k]);
      acc01 = svmla_n_f32_x(all, acc01, vb1, Apack[k]);
      if constexpr (MR >= 2) {
        acc10 = svmla_n_f32_x(all, acc10, vb0, Apack[K + k]);
        acc11 = svmla_n_f32_x(all, acc11, vb1, Apack[K + k]);
      }
      if constexpr (MR >= 3) {
        acc20 = svmla_n_f32_x(all, acc20, vb0, Apack[2 * K + k]);
        acc21 = svmla_n_f32_x(all, acc21, vb1, Apack[2 * K + k]);
      }
      if constexpr (MR >= 4) {
        acc30 = svmla_n_f32_x(all, acc30, vb0, Apack[3 * K + k]);
        acc31 = svmla_n_f32_x(all, acc31, vb1, Apack[3 * K + k]);
      }
    }
    const auto store_row = [&](std::size_t r, svfloat32_t acc0, svfloat32_t acc1) {
      float *Yrow = Yrow_base + r * Ystride + n0 + n;
      const float *Crow =
          mode == GemmAccumMode::kInitBias ? Crow_base + r * Cstride + n0 + n : nullptr;
      svst1_f32(all, Yrow, FinishF32(all, acc0, alpha, beta, Crow, Yrow, mode));
      svst1_f32(all, Yrow + lanes,
                FinishF32(all, acc1, alpha, beta, Crow == nullptr ? nullptr : Crow + lanes,
                          Yrow + lanes, mode));
    };
    store_row(0, acc00, acc01);
    if constexpr (MR >= 2) {
      store_row(1, acc10, acc11);
    }
    if constexpr (MR >= 3) {
      store_row(2, acc20, acc21);
    }
    if constexpr (MR >= 4) {
      store_row(3, acc30, acc31);
    }
  }
  for (; n < nb; n += lanes) {
    const svbool_t pg =
        svwhilelt_b32(static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(nb));
    svfloat32_t acc0 = svdup_f32(0.0f);
    svfloat32_t acc1 = svdup_f32(0.0f);
    svfloat32_t acc2 = svdup_f32(0.0f);
    svfloat32_t acc3 = svdup_f32(0.0f);
    for (std::size_t k = 0; k < K; ++k) {
      const svfloat32_t vb = svld1_f32(pg, Bmat + k * N + n0 + n);
      acc0 = svmla_n_f32_x(pg, acc0, vb, Apack[k]);
      if constexpr (MR >= 2) {
        acc1 = svmla_n_f32_x(pg, acc1, vb, Apack[K + k]);
      }
      if constexpr (MR >= 3) {
        acc2 = svmla_n_f32_x(pg, acc2, vb, Apack[2 * K + k]);
      }
      if constexpr (MR >= 4) {
        acc3 = svmla_n_f32_x(pg, acc3, vb, Apack[3 * K + k]);
      }
    }
    const auto store_row = [&](std::size_t r, svfloat32_t acc) {
      float *Yrow = Yrow_base + r * Ystride + n0 + n;
      const float *Crow =
          mode == GemmAccumMode::kInitBias ? Crow_base + r * Cstride + n0 + n : nullptr;
      svst1_f32(pg, Yrow, FinishF32(pg, acc, alpha, beta, Crow, Yrow, mode));
    };
    store_row(0, acc0);
    if constexpr (MR >= 2) {
      store_row(1, acc1);
    }
    if constexpr (MR >= 3) {
      store_row(2, acc2);
    }
    if constexpr (MR >= 4) {
      store_row(3, acc3);
    }
  }
}

template <std::size_t MR>
void GemmMicroKernel_SVE_F64Impl(std::size_t nb, std::size_t K, double alpha, double beta,
                                 const double *Bmat, std::size_t N, const double *Crow_base,
                                 std::size_t Cstride, double *Yrow_base, std::size_t Ystride,
                                 std::size_t n0, GemmAccumMode mode, const double *Apack) {
  static_assert(MR >= 1 && MR <= kGemmSveMR);
  const std::size_t lanes = svcntd();
  const svbool_t all = svptrue_b64();
  std::size_t n = 0;
  for (; n + 2 * lanes <= nb; n += 2 * lanes) {
    svfloat64_t acc00 = svdup_f64(0.0);
    svfloat64_t acc01 = svdup_f64(0.0);
    svfloat64_t acc10 = svdup_f64(0.0);
    svfloat64_t acc11 = svdup_f64(0.0);
    svfloat64_t acc20 = svdup_f64(0.0);
    svfloat64_t acc21 = svdup_f64(0.0);
    svfloat64_t acc30 = svdup_f64(0.0);
    svfloat64_t acc31 = svdup_f64(0.0);
    for (std::size_t k = 0; k < K; ++k) {
      const double *Brow = Bmat + k * N + n0 + n;
      const svfloat64_t vb0 = svld1_f64(all, Brow);
      const svfloat64_t vb1 = svld1_f64(all, Brow + lanes);
      acc00 = svmla_n_f64_x(all, acc00, vb0, Apack[k]);
      acc01 = svmla_n_f64_x(all, acc01, vb1, Apack[k]);
      if constexpr (MR >= 2) {
        acc10 = svmla_n_f64_x(all, acc10, vb0, Apack[K + k]);
        acc11 = svmla_n_f64_x(all, acc11, vb1, Apack[K + k]);
      }
      if constexpr (MR >= 3) {
        acc20 = svmla_n_f64_x(all, acc20, vb0, Apack[2 * K + k]);
        acc21 = svmla_n_f64_x(all, acc21, vb1, Apack[2 * K + k]);
      }
      if constexpr (MR >= 4) {
        acc30 = svmla_n_f64_x(all, acc30, vb0, Apack[3 * K + k]);
        acc31 = svmla_n_f64_x(all, acc31, vb1, Apack[3 * K + k]);
      }
    }
    const auto store_row = [&](std::size_t r, svfloat64_t acc0, svfloat64_t acc1) {
      double *Yrow = Yrow_base + r * Ystride + n0 + n;
      const double *Crow =
          mode == GemmAccumMode::kInitBias ? Crow_base + r * Cstride + n0 + n : nullptr;
      svst1_f64(all, Yrow, FinishF64(all, acc0, alpha, beta, Crow, Yrow, mode));
      svst1_f64(all, Yrow + lanes,
                FinishF64(all, acc1, alpha, beta, Crow == nullptr ? nullptr : Crow + lanes,
                          Yrow + lanes, mode));
    };
    store_row(0, acc00, acc01);
    if constexpr (MR >= 2) {
      store_row(1, acc10, acc11);
    }
    if constexpr (MR >= 3) {
      store_row(2, acc20, acc21);
    }
    if constexpr (MR >= 4) {
      store_row(3, acc30, acc31);
    }
  }
  for (; n < nb; n += lanes) {
    const svbool_t pg =
        svwhilelt_b64(static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(nb));
    svfloat64_t acc0 = svdup_f64(0.0);
    svfloat64_t acc1 = svdup_f64(0.0);
    svfloat64_t acc2 = svdup_f64(0.0);
    svfloat64_t acc3 = svdup_f64(0.0);
    for (std::size_t k = 0; k < K; ++k) {
      const svfloat64_t vb = svld1_f64(pg, Bmat + k * N + n0 + n);
      acc0 = svmla_n_f64_x(pg, acc0, vb, Apack[k]);
      if constexpr (MR >= 2) {
        acc1 = svmla_n_f64_x(pg, acc1, vb, Apack[K + k]);
      }
      if constexpr (MR >= 3) {
        acc2 = svmla_n_f64_x(pg, acc2, vb, Apack[2 * K + k]);
      }
      if constexpr (MR >= 4) {
        acc3 = svmla_n_f64_x(pg, acc3, vb, Apack[3 * K + k]);
      }
    }
    const auto store_row = [&](std::size_t r, svfloat64_t acc) {
      double *Yrow = Yrow_base + r * Ystride + n0 + n;
      const double *Crow =
          mode == GemmAccumMode::kInitBias ? Crow_base + r * Cstride + n0 + n : nullptr;
      svst1_f64(pg, Yrow, FinishF64(pg, acc, alpha, beta, Crow, Yrow, mode));
    };
    store_row(0, acc0);
    if constexpr (MR >= 2) {
      store_row(1, acc1);
    }
    if constexpr (MR >= 3) {
      store_row(2, acc2);
    }
    if constexpr (MR >= 4) {
      store_row(3, acc3);
    }
  }
}

// Native BFLOAT16 micro-kernel body (Roadmap PR08.3), register-blocked over
// ``MR`` rows. ``Bmat`` and ``Apack`` stay BFLOAT16 to the register file; each
// ``B`` row is widened on the fly with the SVE zero-extend / 16-bit shift while
// the dot products accumulate in float32. The runtime vector length drives the
// lane count, the leading loop consumes two vectors per column step and the
// predicated tail covers the column remainder, so the result is identical to
// the widen-then-float32 reference. SVE vectors are sizeless and cannot live in
// arrays, so the accumulators are named per row like the FP32 kernel.
template <std::size_t MR>
void GemmMicroKernel_SVE_BF16Impl(std::size_t nb, std::size_t K, float alpha, float beta,
                                  const std::uint16_t *Bmat, std::size_t N, const float *Crow_base,
                                  std::size_t Cstride, float *Yrow_base, std::size_t Ystride,
                                  std::size_t n0, GemmAccumMode mode, const std::uint16_t *Apack) {
  static_assert(MR >= 1 && MR <= kGemmSveMR);
  const std::size_t lanes = svcntw();
  const svbool_t all = svptrue_b32();
  std::size_t n = 0;
  for (; n + 2 * lanes <= nb; n += 2 * lanes) {
    svfloat32_t acc00 = svdup_f32(0.0f);
    svfloat32_t acc01 = svdup_f32(0.0f);
    svfloat32_t acc10 = svdup_f32(0.0f);
    svfloat32_t acc11 = svdup_f32(0.0f);
    svfloat32_t acc20 = svdup_f32(0.0f);
    svfloat32_t acc21 = svdup_f32(0.0f);
    svfloat32_t acc30 = svdup_f32(0.0f);
    svfloat32_t acc31 = svdup_f32(0.0f);
    for (std::size_t k = 0; k < K; ++k) {
      const std::uint16_t *Brow = Bmat + k * N + n0 + n;
      const svfloat32_t vb0 = WidenBf16_SVE(all, Brow);
      const svfloat32_t vb1 = WidenBf16_SVE(all, Brow + lanes);
      const float a0 = detail::Bfloat16BitsToFloat(Apack[k]);
      acc00 = svmla_n_f32_x(all, acc00, vb0, a0);
      acc01 = svmla_n_f32_x(all, acc01, vb1, a0);
      if constexpr (MR >= 2) {
        const float a1 = detail::Bfloat16BitsToFloat(Apack[K + k]);
        acc10 = svmla_n_f32_x(all, acc10, vb0, a1);
        acc11 = svmla_n_f32_x(all, acc11, vb1, a1);
      }
      if constexpr (MR >= 3) {
        const float a2 = detail::Bfloat16BitsToFloat(Apack[2 * K + k]);
        acc20 = svmla_n_f32_x(all, acc20, vb0, a2);
        acc21 = svmla_n_f32_x(all, acc21, vb1, a2);
      }
      if constexpr (MR >= 4) {
        const float a3 = detail::Bfloat16BitsToFloat(Apack[3 * K + k]);
        acc30 = svmla_n_f32_x(all, acc30, vb0, a3);
        acc31 = svmla_n_f32_x(all, acc31, vb1, a3);
      }
    }
    const auto store_row = [&](std::size_t r, svfloat32_t acc0, svfloat32_t acc1) {
      float *Yrow = Yrow_base + r * Ystride + n0 + n;
      const float *Crow =
          mode == GemmAccumMode::kInitBias ? Crow_base + r * Cstride + n0 + n : nullptr;
      svst1_f32(all, Yrow, FinishF32(all, acc0, alpha, beta, Crow, Yrow, mode));
      svst1_f32(all, Yrow + lanes,
                FinishF32(all, acc1, alpha, beta, Crow == nullptr ? nullptr : Crow + lanes,
                          Yrow + lanes, mode));
    };
    store_row(0, acc00, acc01);
    if constexpr (MR >= 2) {
      store_row(1, acc10, acc11);
    }
    if constexpr (MR >= 3) {
      store_row(2, acc20, acc21);
    }
    if constexpr (MR >= 4) {
      store_row(3, acc30, acc31);
    }
  }
  for (; n < nb; n += lanes) {
    const svbool_t pg =
        svwhilelt_b32(static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(nb));
    svfloat32_t acc0 = svdup_f32(0.0f);
    svfloat32_t acc1 = svdup_f32(0.0f);
    svfloat32_t acc2 = svdup_f32(0.0f);
    svfloat32_t acc3 = svdup_f32(0.0f);
    for (std::size_t k = 0; k < K; ++k) {
      const svfloat32_t vb = WidenBf16_SVE(pg, Bmat + k * N + n0 + n);
      acc0 = svmla_n_f32_x(pg, acc0, vb, detail::Bfloat16BitsToFloat(Apack[k]));
      if constexpr (MR >= 2) {
        acc1 = svmla_n_f32_x(pg, acc1, vb, detail::Bfloat16BitsToFloat(Apack[K + k]));
      }
      if constexpr (MR >= 3) {
        acc2 = svmla_n_f32_x(pg, acc2, vb, detail::Bfloat16BitsToFloat(Apack[2 * K + k]));
      }
      if constexpr (MR >= 4) {
        acc3 = svmla_n_f32_x(pg, acc3, vb, detail::Bfloat16BitsToFloat(Apack[3 * K + k]));
      }
    }
    const auto store_row = [&](std::size_t r, svfloat32_t acc) {
      float *Yrow = Yrow_base + r * Ystride + n0 + n;
      const float *Crow =
          mode == GemmAccumMode::kInitBias ? Crow_base + r * Cstride + n0 + n : nullptr;
      svst1_f32(pg, Yrow, FinishF32(pg, acc, alpha, beta, Crow, Yrow, mode));
    };
    store_row(0, acc0);
    if constexpr (MR >= 2) {
      store_row(1, acc1);
    }
    if constexpr (MR >= 3) {
      store_row(2, acc2);
    }
    if constexpr (MR >= 4) {
      store_row(3, acc3);
    }
  }
}

// Native FLOAT16 micro-kernel body (Roadmap PR08.3). It mirrors
// ``GemmMicroKernel_SVE_BF16Impl`` but widens the ``B`` rows with the SVE
// ``FCVT`` (``svcvt_f32_f16``) instruction; ``Bmat`` and ``Apack`` stay FLOAT16
// to the register file and the dot products accumulate in float32.
template <std::size_t MR>
void GemmMicroKernel_SVE_FP16Impl(std::size_t nb, std::size_t K, float alpha, float beta,
                                  const std::uint16_t *Bmat, std::size_t N, const float *Crow_base,
                                  std::size_t Cstride, float *Yrow_base, std::size_t Ystride,
                                  std::size_t n0, GemmAccumMode mode, const std::uint16_t *Apack) {
  static_assert(MR >= 1 && MR <= kGemmSveMR);
  const std::size_t lanes = svcntw();
  const svbool_t all = svptrue_b32();
  std::size_t n = 0;
  for (; n + 2 * lanes <= nb; n += 2 * lanes) {
    svfloat32_t acc00 = svdup_f32(0.0f);
    svfloat32_t acc01 = svdup_f32(0.0f);
    svfloat32_t acc10 = svdup_f32(0.0f);
    svfloat32_t acc11 = svdup_f32(0.0f);
    svfloat32_t acc20 = svdup_f32(0.0f);
    svfloat32_t acc21 = svdup_f32(0.0f);
    svfloat32_t acc30 = svdup_f32(0.0f);
    svfloat32_t acc31 = svdup_f32(0.0f);
    for (std::size_t k = 0; k < K; ++k) {
      const std::uint16_t *Brow = Bmat + k * N + n0 + n;
      const svfloat32_t vb0 = WidenFp16_SVE(all, Brow);
      const svfloat32_t vb1 = WidenFp16_SVE(all, Brow + lanes);
      const float a0 = detail::Float16BitsToFloat(Apack[k]);
      acc00 = svmla_n_f32_x(all, acc00, vb0, a0);
      acc01 = svmla_n_f32_x(all, acc01, vb1, a0);
      if constexpr (MR >= 2) {
        const float a1 = detail::Float16BitsToFloat(Apack[K + k]);
        acc10 = svmla_n_f32_x(all, acc10, vb0, a1);
        acc11 = svmla_n_f32_x(all, acc11, vb1, a1);
      }
      if constexpr (MR >= 3) {
        const float a2 = detail::Float16BitsToFloat(Apack[2 * K + k]);
        acc20 = svmla_n_f32_x(all, acc20, vb0, a2);
        acc21 = svmla_n_f32_x(all, acc21, vb1, a2);
      }
      if constexpr (MR >= 4) {
        const float a3 = detail::Float16BitsToFloat(Apack[3 * K + k]);
        acc30 = svmla_n_f32_x(all, acc30, vb0, a3);
        acc31 = svmla_n_f32_x(all, acc31, vb1, a3);
      }
    }
    const auto store_row = [&](std::size_t r, svfloat32_t acc0, svfloat32_t acc1) {
      float *Yrow = Yrow_base + r * Ystride + n0 + n;
      const float *Crow =
          mode == GemmAccumMode::kInitBias ? Crow_base + r * Cstride + n0 + n : nullptr;
      svst1_f32(all, Yrow, FinishF32(all, acc0, alpha, beta, Crow, Yrow, mode));
      svst1_f32(all, Yrow + lanes,
                FinishF32(all, acc1, alpha, beta, Crow == nullptr ? nullptr : Crow + lanes,
                          Yrow + lanes, mode));
    };
    store_row(0, acc00, acc01);
    if constexpr (MR >= 2) {
      store_row(1, acc10, acc11);
    }
    if constexpr (MR >= 3) {
      store_row(2, acc20, acc21);
    }
    if constexpr (MR >= 4) {
      store_row(3, acc30, acc31);
    }
  }
  for (; n < nb; n += lanes) {
    const svbool_t pg =
        svwhilelt_b32(static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(nb));
    svfloat32_t acc0 = svdup_f32(0.0f);
    svfloat32_t acc1 = svdup_f32(0.0f);
    svfloat32_t acc2 = svdup_f32(0.0f);
    svfloat32_t acc3 = svdup_f32(0.0f);
    for (std::size_t k = 0; k < K; ++k) {
      const svfloat32_t vb = WidenFp16_SVE(pg, Bmat + k * N + n0 + n);
      acc0 = svmla_n_f32_x(pg, acc0, vb, detail::Float16BitsToFloat(Apack[k]));
      if constexpr (MR >= 2) {
        acc1 = svmla_n_f32_x(pg, acc1, vb, detail::Float16BitsToFloat(Apack[K + k]));
      }
      if constexpr (MR >= 3) {
        acc2 = svmla_n_f32_x(pg, acc2, vb, detail::Float16BitsToFloat(Apack[2 * K + k]));
      }
      if constexpr (MR >= 4) {
        acc3 = svmla_n_f32_x(pg, acc3, vb, detail::Float16BitsToFloat(Apack[3 * K + k]));
      }
    }
    const auto store_row = [&](std::size_t r, svfloat32_t acc) {
      float *Yrow = Yrow_base + r * Ystride + n0 + n;
      const float *Crow =
          mode == GemmAccumMode::kInitBias ? Crow_base + r * Cstride + n0 + n : nullptr;
      svst1_f32(pg, Yrow, FinishF32(pg, acc, alpha, beta, Crow, Yrow, mode));
    };
    store_row(0, acc0);
    if constexpr (MR >= 2) {
      store_row(1, acc1);
    }
    if constexpr (MR >= 3) {
      store_row(2, acc2);
    }
    if constexpr (MR >= 4) {
      store_row(3, acc3);
    }
  }
}

} // namespace

std::size_t GemmSveVectorBytes() { return svcntb(); }

#define DISPATCH_SVE_MR(Function, mr, ...)                                                         \
  switch (mr) {                                                                                    \
  case 1:                                                                                          \
    return Function<1>(__VA_ARGS__);                                                               \
  case 2:                                                                                          \
    return Function<2>(__VA_ARGS__);                                                               \
  case 3:                                                                                          \
    return Function<3>(__VA_ARGS__);                                                               \
  case 4:                                                                                          \
    return Function<4>(__VA_ARGS__);                                                               \
  default:                                                                                         \
    break;                                                                                         \
  }

void GemmMicroKernel_SVE_F32(std::size_t mr, std::size_t nb, std::size_t K, float alpha, float beta,
                             const float *Bmat, std::size_t N, const float *Crow_base,
                             std::size_t Cstride, float *Yrow_base, std::size_t Ystride,
                             std::size_t n0, GemmAccumMode mode, const float *Apack) {
  DISPATCH_SVE_MR(GemmMicroKernel_SVE_F32Impl, mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                  Yrow_base, Ystride, n0, mode, Apack)
  GemmMicroKernel_Scalar_F32(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                             Ystride, n0, mode, Apack);
}

void GemmMicroKernel_SVE_F64(std::size_t mr, std::size_t nb, std::size_t K, double alpha,
                             double beta, const double *Bmat, std::size_t N,
                             const double *Crow_base, std::size_t Cstride, double *Yrow_base,
                             std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                             const double *Apack) {
  DISPATCH_SVE_MR(GemmMicroKernel_SVE_F64Impl, mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                  Yrow_base, Ystride, n0, mode, Apack)
  GemmMicroKernel_Scalar_F64(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                             Ystride, n0, mode, Apack);
}

void GemmMicroKernel_SVE_BF16(std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                              float beta, const std::uint16_t *Bmat, std::size_t N,
                              const float *Crow_base, std::size_t Cstride, float *Yrow_base,
                              std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                              const std::uint16_t *Apack) {
  DISPATCH_SVE_MR(GemmMicroKernel_SVE_BF16Impl, mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                  Yrow_base, Ystride, n0, mode, Apack)
  GemmMicroKernel_ScalarBf16(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                             Ystride, n0, mode, Apack);
}

void GemmMicroKernel_SVE_FP16(std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                              float beta, const std::uint16_t *Bmat, std::size_t N,
                              const float *Crow_base, std::size_t Cstride, float *Yrow_base,
                              std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                              const std::uint16_t *Apack) {
  DISPATCH_SVE_MR(GemmMicroKernel_SVE_FP16Impl, mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                  Yrow_base, Ystride, n0, mode, Apack)
  GemmMicroKernel_ScalarFp16(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                             Ystride, n0, mode, Apack);
}

#undef DISPATCH_SVE_MR

} // namespace onnx_light_cpu
