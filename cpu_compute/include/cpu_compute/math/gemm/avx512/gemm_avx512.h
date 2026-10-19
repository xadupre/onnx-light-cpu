// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Declarations for the AVX-512 Gemm micro-tiles. These are only usable when
// CPU_COMPUTE_HAVE_AVX512 is defined: gemm_avx512.cc (the file that
// implements them) is only compiled into cpu_compute when CMake's
// check_cxx_compiler_flag confirms the compiler accepts -mavx512f (or
// /arch:AVX512 for MSVC) -- see CMakeLists.txt. Gate every reference to these
// functions behind ``#ifdef CPU_COMPUTE_HAVE_AVX512``.

#pragma once

#include "cpu_compute/math/gemm/gemm_common.h"

#include <cstddef>

namespace cpu_compute {

// AVX-512 micro-tiles processing up to ``kGemmAVX512MR`` general rows, or the
// specialized 24-row float32 tile, by ``nb`` columns. ``Apack`` is a packed,
// contiguous ``mr x K`` row-major panel (see ``PackARowBlock`` in
// gemm.cc): ``Apack[r * K + k]`` is ``A(m + r, k0 + k)``, already
// resolved for ``trans_a``.
void GemmMicroTile_AVX512_F32(std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                              float beta, const float *Bmat, std::size_t N, const float *Crow_base,
                              std::size_t Cstride, float *Yrow_base, std::size_t Ystride,
                              std::size_t n0, GemmAccumMode mode, const float *Apack);

void GemmMicroTile_AVX512_F32_StridedA(std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                                       float beta, const float *Bmat, std::size_t N,
                                       const float *Crow_base, std::size_t Cstride,
                                       float *Yrow_base, std::size_t Ystride, std::size_t n0,
                                       GemmAccumMode mode, const float *A, std::size_t AStride);

void GemmSkinnyM1Routine_AVX512_F32(std::size_t K, float alpha, const float *A, const float *B,
                                    std::size_t N, float beta, const float *C, float *Y,
                                    std::size_t n0);
void GemmSkinnyM1Routine64_AVX512_F32(std::size_t K, float alpha, const float *A, const float *B,
                                      std::size_t N, float beta, const float *C, float *Y,
                                      std::size_t n0);
void GemmSkinnyM1Routine128_AVX512_F32(std::size_t K, float alpha, const float *A, const float *B,
                                       std::size_t N, float beta, const float *C, float *Y,
                                       std::size_t n0);
void GemmSkinnyN1Range_AVX512_F32(std::size_t K, float alpha, const float *A, const float *B,
                                  float beta, const float *C, float *Y, std::size_t begin,
                                  std::size_t end);

void GemmMicroTile_AVX512_F64(std::size_t mr, std::size_t nb, std::size_t K, double alpha,
                              double beta, const double *Bmat, std::size_t N,
                              const double *Crow_base, std::size_t Cstride, double *Yrow_base,
                              std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                              const double *Apack);

void GemmMicroTile_AVX512_F64_StridedA(std::size_t mr, std::size_t nb, std::size_t K, double alpha,
                                       double beta, const double *Bmat, std::size_t N,
                                       const double *Crow_base, std::size_t Cstride,
                                       double *Yrow_base, std::size_t Ystride, std::size_t n0,
                                       GemmAccumMode mode, const double *A, std::size_t AStride);

} // namespace cpu_compute
