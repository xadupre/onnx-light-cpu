// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu/impl/math/gemm/gemm_common.h"

#include <cstddef>
#include <cstdint>

namespace onnx_light_cpu {

void GemmMicroKernel_AVX2FMA_F32(std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                                 float beta, const float *Bmat, std::size_t N,
                                 const float *Crow_base, std::size_t Cstride, float *Yrow_base,
                                 std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                                 const float *Apack);

void GemmMicroKernel_AVX2FMA_F64(std::size_t mr, std::size_t nb, std::size_t K, double alpha,
                                 double beta, const double *Bmat, std::size_t N,
                                 const double *Crow_base, std::size_t Cstride, double *Yrow_base,
                                 std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                                 const double *Apack);

// Converts ``n`` contiguous BFLOAT16 patterns to float32, eight at a time
// through an AVX2 zero-extend and 16-bit left shift, with an exact scalar tail.
// Used by the GEMM packing loops to widen while packing without a full-tensor
// conversion pass. Only requires AVX2 (no F16C), so it is always available in
// this translation unit.
void GemmConvertBFloat16ToFloat32_AVX2(const std::uint16_t *src, float *dst, std::size_t n);

void GemmPackTransposeFloat32_AVX2(const float *src, std::size_t src_stride, float *dst,
                                   std::size_t dst_rows, std::size_t dst_cols);

void GemmPackTransposeFloat64_AVX2(const double *src, std::size_t src_stride, double *dst,
                                   std::size_t dst_rows, std::size_t dst_cols);

// Transposes and widens a ``dst_rows x dst_cols`` BFLOAT16 panel from a
// column-major source view. Full 8 x 8 tiles use an in-register transpose.
void GemmPackTransposeBFloat16ToFloat32_AVX2(const std::uint16_t *src, std::size_t src_stride,
                                             float *dst, std::size_t dst_rows,
                                             std::size_t dst_cols);

// Narrows ``n`` contiguous float32 values to BFLOAT16 with round-to-nearest-even,
// canonical NaNs, and an exact scalar tail.
void GemmConvertFloat32ToBFloat16_AVX2(const float *src, std::uint16_t *dst, std::size_t n);

// Native AVX2 BFLOAT16 micro-kernel with float32 accumulation. B and the packed
// A rows remain BFLOAT16 until they enter the register file.
void GemmMicroKernel_AVX2BF16(std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                              float beta, const std::uint16_t *Bmat, std::size_t N,
                              const float *Crow_base, std::size_t Cstride, float *Yrow_base,
                              std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                              const std::uint16_t *Apack);

// Decodes ``n`` contiguous Float8 patterns to float32 (Roadmap PR09.5) eight at
// a time through an AVX2 ``vgatherdps`` from the caller-supplied exact 256-entry
// per-format decode table (``detail::BuildFloat8DecodeTable``), with an exact
// scalar tail. The table encodes the format, so this vectorized decode is
// format-agnostic and used by the GEMM packing loops to decode while packing
// without a full-tensor conversion pass. Only requires AVX2.
void GemmDecodeFloat8ToFloat32_AVX2(const float *table, const std::uint8_t *src, float *dst,
                                    std::size_t n);

#ifdef ONNX_LIGHT_CPU_HAVE_F16C
// Converts ``n`` contiguous FLOAT16 patterns to float32 with the F16C
// ``vcvtph2ps`` instruction, eight at a time, with an exact scalar tail. Widens
// while packing in the GEMM packing loops. Requires the F16C ISA extension, so
// it is only declared/defined when the translation unit is compiled with it.
void GemmConvertFloat16ToFloat32_F16C(const std::uint16_t *src, float *dst, std::size_t n);

// F16C counterpart of GemmPackTransposeBFloat16ToFloat32_AVX2.
void GemmPackTransposeFloat16ToFloat32_F16C(const std::uint16_t *src, std::size_t src_stride,
                                            float *dst, std::size_t dst_rows, std::size_t dst_cols);

// Narrows ``n`` contiguous float32 values to FLOAT16 with F16C
// round-to-nearest-even conversion, canonical NaNs, and an exact scalar tail.
void GemmConvertFloat32ToFloat16_F16C(const float *src, std::uint16_t *dst, std::size_t n);

// Native AVX2/F16C FLOAT16 micro-kernel with float32 accumulation. B and the
// packed A rows remain FLOAT16 until they enter the register file.
void GemmMicroKernel_AVX2F16C(std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                              float beta, const std::uint16_t *Bmat, std::size_t N,
                              const float *Crow_base, std::size_t Cstride, float *Yrow_base,
                              std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                              const std::uint16_t *Apack);

// Dedicated FLOAT16 skinny kernels. Eight values are widened with one F16C
// conversion before AVX2 FMA accumulation. The low-row-count path requires a
// non-transposed B; the single-column path requires a non-transposed A.
void GemmFloat16SkinnyM_F16C(bool trans_a, std::size_t M, std::size_t N, std::size_t K, float alpha,
                             const std::uint16_t *A, const std::uint16_t *B, float *Y);
void GemmFloat16SkinnyN_F16C(std::size_t M, std::size_t K, float alpha, const std::uint16_t *A,
                             const std::uint16_t *B, float *Y);
#endif

} // namespace onnx_light_cpu
