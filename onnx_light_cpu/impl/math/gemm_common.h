// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Types and declarations shared between gemm_kernel.cc (compiled with the
// project's baseline ONNX_LIGHT_CPU_SIMD_FLAGS, AVX2 by default) and
// gemm_kernel_avx512.cc (compiled with an additional -mavx512f, see
// CMakeLists.txt), so a single binary can carry an AVX-512 Gemm micro-kernel
// alongside the AVX2/SSE2/scalar ones and pick whichever matches
// DetectSimdLevel() at runtime.

#pragma once

#include <cstddef>

namespace onnx_light_cpu {

// How a micro-kernel call should combine its ``alpha * acc`` result with the
// existing contents of ``Y``. K is processed in fixed-size chunks (see the
// comment above ``kGemmTileK`` in gemm_kernel.cc) so a single (row block,
// column panel) output tile is written by several micro-kernel calls, one per
// k-chunk, each contributing its partial dot product. Since
// ``alpha * sum_k(...) == sum_chunks(alpha * partial_sum_chunk)``, every chunk
// after the first simply adds its scaled partial result to ``Y`` instead of
// overwriting it.
enum class GemmAccumMode {
  kInitZero,   ///< First chunk, no bias: ``Y = alpha * acc``.
  kInitBias,   ///< First chunk, with bias: ``Y = alpha * acc + beta * C``.
  kAccumulate, ///< Later chunk: ``Y += alpha * acc``.
};

// Register-blocking factor on M: the number of output rows every micro-kernel
// flavor (scalar, SSE2, AVX, AVX-512) processes together. Shared so the packed
// ``A`` row-panel built in gemm_kernel.cc (``kGemmMR x kc`` elements) has a
// size every kernel flavor agrees on.
constexpr std::size_t kGemmMR = 4;

// Current cache-blocking dimensions shared by the driver and prepared plans.
constexpr std::size_t kGemmTileM = 64;
constexpr std::size_t kGemmTileN = 256;
constexpr std::size_t kGemmTileK = 256;

// Scalar micro-kernels: also the tail handler for every vectorized flavor
// (AVX-512, AVX, SSE2) and the fallback for non-x86 builds. Defined (with
// external linkage) in gemm_kernel.cc so gemm_kernel_avx512.cc can reuse them
// for its own column tail instead of duplicating the mode-driven combine
// logic. ``Apack`` is a packed, contiguous ``mr x K`` row-major panel (see
// ``PackARowBlock``): ``Apack[r * K + k]`` is ``A(m + r, k0 + k)``.
void GemmMicroKernel_Scalar_F32(std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                                float beta, const float *Bmat, std::size_t N,
                                const float *Crow_base, std::size_t Cstride, float *Yrow_base,
                                std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                                const float *Apack);

void GemmMicroKernel_Scalar_F64(std::size_t mr, std::size_t nb, std::size_t K, double alpha,
                                double beta, const double *Bmat, std::size_t N,
                                const double *Crow_base, std::size_t Cstride, double *Yrow_base,
                                std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                                const double *Apack);

} // namespace onnx_light_cpu
