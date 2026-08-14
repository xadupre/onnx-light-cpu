// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu/impl/arm_simd_level.h"
#include "onnx_light_cpu/impl/math/gemm/gemm_common.h"

#include <cstddef>

namespace onnx_light_cpu {

enum class ArmGemmKernelKind {
  kScalar,
  kNeon,
  kSve,
};

struct ArmGemmProfile {
  ArmGemmKernelKind kind = ArmGemmKernelKind::kScalar;
  std::size_t vector_bytes = 0;
  std::size_t register_rows = kGemmMR;
};

ArmGemmProfile SelectArmGemmProfile(ArmSimdLevel level, std::size_t sve_vector_bytes,
                                    bool sve_kernel_available);
ArmGemmProfile DetectArmGemmProfile();

#ifdef ONNX_LIGHT_CPU_HAVE_NEON
void GemmMicroKernel_NEON_F32(std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                              float beta, const float *Bmat, std::size_t N, const float *Crow_base,
                              std::size_t Cstride, float *Yrow_base, std::size_t Ystride,
                              std::size_t n0, GemmAccumMode mode, const float *Apack);

void GemmMicroKernel_NEON_F64(std::size_t mr, std::size_t nb, std::size_t K, double alpha,
                              double beta, const double *Bmat, std::size_t N,
                              const double *Crow_base, std::size_t Cstride, double *Yrow_base,
                              std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                              const double *Apack);
#endif

#ifdef ONNX_LIGHT_CPU_HAVE_SVE
std::size_t GemmSveVectorBytes();

void GemmMicroKernel_SVE_F32(std::size_t mr, std::size_t nb, std::size_t K, float alpha, float beta,
                             const float *Bmat, std::size_t N, const float *Crow_base,
                             std::size_t Cstride, float *Yrow_base, std::size_t Ystride,
                             std::size_t n0, GemmAccumMode mode, const float *Apack);

void GemmMicroKernel_SVE_F64(std::size_t mr, std::size_t nb, std::size_t K, double alpha,
                             double beta, const double *Bmat, std::size_t N,
                             const double *Crow_base, std::size_t Cstride, double *Yrow_base,
                             std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                             const double *Apack);
#endif

} // namespace onnx_light_cpu
