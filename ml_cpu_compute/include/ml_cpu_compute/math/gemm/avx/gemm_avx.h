// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ml_cpu_compute/math/gemm/gemm_common.h"

namespace ml_cpu_compute {

#ifdef ML_CPU_COMPUTE_HAVE_AVX
void GemmMicroTile_AVX_F32(std::size_t mr, std::size_t nb, std::size_t K, float alpha, float beta,
                           const float *Bmat, std::size_t N, const float *Crow_base,
                           std::size_t Cstride, float *Yrow_base, std::size_t Ystride,
                           std::size_t n0, GemmAccumMode mode, const float *Apack);
void GemmMicroTile_AVX_F64(std::size_t mr, std::size_t nb, std::size_t K, double alpha, double beta,
                           const double *Bmat, std::size_t N, const double *Crow_base,
                           std::size_t Cstride, double *Yrow_base, std::size_t Ystride,
                           std::size_t n0, GemmAccumMode mode, const double *Apack);
#endif

} // namespace ml_cpu_compute
