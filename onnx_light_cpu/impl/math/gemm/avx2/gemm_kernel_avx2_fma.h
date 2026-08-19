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
#endif

} // namespace onnx_light_cpu
