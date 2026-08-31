// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Declarations for the AVX-512 Gemm micro-kernels. These are only usable when
// ONNX_LIGHT_CPU_HAVE_AVX512 is defined: gemm_kernel_avx512.cc (the file that
// implements them) is only compiled into lib_onnx_light_cpu when CMake's
// check_cxx_compiler_flag confirms the compiler accepts -mavx512f (or
// /arch:AVX512 for MSVC) -- see CMakeLists.txt. Gate every reference to these
// functions behind ``#ifdef ONNX_LIGHT_CPU_HAVE_AVX512``.

#pragma once

#include "onnx_light_cpu/impl/math/gemm/gemm_common.h"

#include <cstddef>

namespace onnx_light_cpu {

// AVX-512 micro-kernel processing up to ``kGemmAVX512MR`` general rows, or the
// specialized 24-row float32 tile, by ``nb`` columns. ``Apack`` is a packed,
// contiguous ``mr x K`` row-major panel (see ``PackARowBlock`` in
// gemm_kernel.cc): ``Apack[r * K + k]`` is ``A(m + r, k0 + k)``, already
// resolved for ``trans_a``.
void GemmMicroKernel_AVX512_F32(std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                                float beta, const float *Bmat, std::size_t N,
                                const float *Crow_base, std::size_t Cstride, float *Yrow_base,
                                std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                                const float *Apack);

void GemmMicroKernel_AVX512_F32_StridedA(std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                                         float beta, const float *Bmat, std::size_t N,
                                         const float *Crow_base, std::size_t Cstride,
                                         float *Yrow_base, std::size_t Ystride, std::size_t n0,
                                         GemmAccumMode mode, const float *A, std::size_t AStride);

void GemmSkinnyM1Kernel_AVX512_F32(std::size_t K, float alpha, const float *A, const float *B,
                                   std::size_t N, float beta, const float *C, float *Y,
                                   std::size_t n0);

void GemmMicroKernel_AVX512_F64(std::size_t mr, std::size_t nb, std::size_t K, double alpha,
                                double beta, const double *Bmat, std::size_t N,
                                const double *Crow_base, std::size_t Cstride, double *Yrow_base,
                                std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                                const double *Apack);

void GemmMicroKernel_AVX512_F64_StridedA(std::size_t mr, std::size_t nb, std::size_t K,
                                         double alpha, double beta, const double *Bmat,
                                         std::size_t N, const double *Crow_base,
                                         std::size_t Cstride, double *Yrow_base,
                                         std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                                         const double *A, std::size_t AStride);

} // namespace onnx_light_cpu
