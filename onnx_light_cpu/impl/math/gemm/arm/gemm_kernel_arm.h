// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu/impl/arm_simd_level.h"
#include "onnx_light_cpu/impl/math/gemm/gemm_common.h"

#include <cstddef>
#include <cstdint>

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

// Converts ``n`` contiguous BFLOAT16 patterns to float32, eight at a time
// through a NEON zero-extend and 16-bit left shift, with an exact scalar tail.
// Used by the GEMM packing loops to widen while packing without a full-tensor
// conversion pass. Only requires baseline NEON, so it is always available in
// this translation unit.
void GemmConvertBFloat16ToFloat32_NEON(const std::uint16_t *src, float *dst, std::size_t n);

// Native BFLOAT16 member of the GEMM micro-kernel family (Roadmap PR08.2). It
// matches ``GemmBf16MicroKernel``: ``Bmat`` and ``Apack`` stay BFLOAT16 to the
// register file and each eight-column ``B`` row is widened on the fly with the
// baseline NEON zero-extend / 16-bit shift while the dot products accumulate in
// float32, so no full-tensor widening buffer is needed. The scalar column tail
// reuses ``GemmMicroKernel_ScalarBf16``. Only requires baseline NEON, so it is
// always available in this translation unit.
void GemmMicroKernel_NEON_BF16(std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                               float beta, const std::uint16_t *Bmat, std::size_t N,
                               const float *Crow_base, std::size_t Cstride, float *Yrow_base,
                               std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                               const std::uint16_t *Apack);

#ifdef ONNX_LIGHT_CPU_HAVE_NEON_FP16
// Converts ``n`` contiguous FLOAT16 patterns to float32 with the NEON
// ``vcvt_f32_f16`` (``FCVTL``) instruction, eight at a time, with an exact
// scalar tail. Widens while packing in the GEMM packing loops. Requires the
// FP16 vector load/convert intrinsics, so it is only declared/defined when the
// translation unit is compiled with support for them.
void GemmConvertFloat16ToFloat32_NEON(const std::uint16_t *src, float *dst, std::size_t n);

// Native FLOAT16 member of the GEMM micro-kernel family (Roadmap PR08.2). It
// matches ``GemmFp16MicroKernel``: ``Bmat`` and ``Apack`` stay FLOAT16 to the
// register file and each eight-column ``B`` row is widened on the fly with the
// NEON ``vcvt_f32_f16`` (``FCVTL``) instruction while the dot products
// accumulate in float32, so no full-tensor widening buffer is needed. The
// scalar column tail reuses ``GemmMicroKernel_ScalarFp16``. Requires the FP16
// vector load/convert intrinsics, so it is only declared/defined when the
// translation unit is compiled with support for them.
void GemmMicroKernel_NEON_FP16(std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                               float beta, const std::uint16_t *Bmat, std::size_t N,
                               const float *Crow_base, std::size_t Cstride, float *Yrow_base,
                               std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                               const std::uint16_t *Apack);
#endif
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
