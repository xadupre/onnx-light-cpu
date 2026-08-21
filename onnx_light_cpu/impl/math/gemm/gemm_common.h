// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Types and declarations shared between gemm_kernel.cc and the ISA-specific
// GEMM micro-kernels.

#pragma once

#include "onnx_light_cpu/impl/simd_level.h"

#include <cstddef>
#include <cstdint>

namespace onnx_light_cpu {

/// Cache and register blocking selected for a matrix multiplication plan.
struct GemmBlocking {
  std::size_t mc = 0;
  std::size_t nc = 0;
  std::size_t kc = 0;
  std::size_t mr = 0;
  std::size_t nr = 0;
};

/// Computational path selected for a matrix multiplication plan.
enum class GemmAlgorithm {
  kGeneral,
  kDirect,
  kSkinnyM,
  kSkinnyN,
  kSplitK,
};

enum class GemmMicroarchitecture {
  kGeneric,
  kIntelCore,
  kAmdZen,
};

namespace detail {

GemmAlgorithm SelectGemmAlgorithm(bool trans_a, bool trans_b, std::size_t m, std::size_t n,
                                  std::size_t k, std::size_t vector_lanes,
                                  std::size_t register_rows);

GemmBlocking SelectGemmBlocking(std::size_t element_size, std::size_t vector_lanes,
                                std::size_t register_rows);

GemmBlocking ConstrainGemmBlockingForTasks(GemmBlocking blocking, std::size_t m, std::size_t n,
                                           std::size_t thread_count);

/// Width of the column micro-panel the tile loops walk inside one packed B
/// panel. The returned value is a multiple of ``blocking.nr`` that keeps the
/// contiguous ``kc x column_block`` slice of B small enough to be reused from
/// cache by every row tile of the packed A panel.
std::size_t SelectGemmColumnBlock(const GemmBlocking &blocking, std::size_t element_size);

GemmMicroarchitecture DetectGemmMicroarchitecture();

std::size_t SelectGemmRegisterRowsForMicroarchitecture(SimdLevel level, bool has_fma,
                                                       GemmMicroarchitecture microarchitecture);

std::size_t SelectGemmRegisterRows(SimdLevel level, bool has_fma);

template <GemmAlgorithm Algorithm>
void GemmFloat32Planned(bool trans_a, bool trans_b, std::size_t M, std::size_t N, std::size_t K,
                        float alpha, const float *A, const float *B, float beta, const float *C,
                        float *Y, const GemmBlocking *blocking = nullptr);

template <GemmAlgorithm Algorithm>
void GemmFloat64Planned(bool trans_a, bool trans_b, std::size_t M, std::size_t N, std::size_t K,
                        double alpha, const double *A, const double *B, double beta,
                        const double *C, double *Y, const GemmBlocking *blocking = nullptr);

/// FP16/BF16 GEMM executed through the typed source path for one prepared
/// algorithm. ``A`` and ``B`` are raw 16-bit ``FLOAT16`` (``is_bfloat16 ==
/// false``) or ``BFLOAT16`` (``is_bfloat16 == true``) patterns. Native
/// non-transposed-B paths keep compact panels in the source format; fallback
/// paths convert while packing or reducing. No full-tensor widening buffer is
/// allocated. The float32 ``M x N`` product is written to ``Y`` for the
/// caller's narrowing epilogue.
template <GemmAlgorithm Algorithm>
void GemmHalfPlanned(bool is_bfloat16, bool trans_a, bool trans_b, std::size_t M, std::size_t N,
                     std::size_t K, float alpha, const std::uint16_t *A, const std::uint16_t *B,
                     float *Y, const GemmBlocking *blocking = nullptr,
                     const GemmBlocking *compact_blocking = nullptr);

} // namespace detail

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

// Maximum register-blocking factors on M. Runtime profiles select a value no
// larger than the matching ISA-specific maximum.
constexpr std::size_t kGemmMR = 4;
constexpr std::size_t kGemmAVX2MR = 6;
constexpr std::size_t kGemmAVX512MR = 8;
constexpr std::size_t kGemmIntelAVX2MR = 5;
constexpr std::size_t kGemmZenAVX2MR = 6;
constexpr std::size_t kGemmNeonMR = 6;
constexpr std::size_t kGemmSveMR = 4;
// AMX tiles are at most 16 rows tall, so the AMX-BF16 kernel register-blocks a
// full 16-row output tile (Roadmap PR07.6).
constexpr std::size_t kGemmAmxBf16MR = 16;

// Current cache-blocking dimensions shared by the driver and prepared plans.
constexpr std::size_t kGemmTileM = 64;
constexpr std::size_t kGemmTileN = 256;
constexpr std::size_t kGemmTileK = 256;
// Width, in bytes, of one contiguous column micro-panel of the packed B panel.
// The tile loops walk the packed panel one micro-panel at a time; four cache
// lines per k row measured best on AVX2 and keep the slice small enough to be
// reused from L2 by every row tile while the packed A panel also stays there.
constexpr std::size_t kGemmColumnPanelBytes = 256;
// A scalar FMA is a fraction of one element-wise work unit once the inner loop
// is vectorized and register-blocked.
constexpr double kGemmFmasPerParallelWorkUnit = 64.0;

// Scalar micro-kernels: also the tail handler for every vectorized flavor
// (AVX-512, AVX, SSE2) and the fallback for non-x86 builds. Defined (with
// external linkage) in gemm_kernel.cc so gemm_kernel_avx512.cc can reuse them
// for its own column tail instead of duplicating the mode-driven combine
// logic. ``Apack`` is a packed, contiguous ``mr x K`` row-major panel (see
// ``PackAPanel``): ``Apack[r * K + k]`` is ``A(m + r, k0 + k)``.
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

// Signature shared by every member of the native FLOAT16 GEMM micro-kernel
// family (Roadmap PR07.3). Unlike the FP32 micro-kernels, both operands stay in
// FLOAT16: ``Apack`` is a packed contiguous ``mr x K`` row-major panel of raw
// FLOAT16 patterns (``Apack[r * K + k]`` is ``A(m + r, k0 + k)``) and ``Bmat``
// is a row-major FLOAT16 matrix or compact panel read with row stride ``N``.
// Each element is converted to float on the fly and the dot products accumulate
// in float32, so the float32 result matches the widen-then-float32 reference;
// the epilogue combines ``alpha * acc`` with ``Y`` according to ``mode`` and
// writes float32.
using GemmFp16MicroKernel = void (*)(std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                                     float beta, const std::uint16_t *Bmat, std::size_t N,
                                     const float *Crow_base, std::size_t Cstride, float *Yrow_base,
                                     std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                                     const std::uint16_t *Apack);

// Portable scalar member of the FLOAT16 micro-kernel family. It is the tail
// handler for the vectorized AVX-512FP16 kernel and the reference the
// differential tests run everywhere (no AVX-512FP16 hardware required).
void GemmMicroKernel_ScalarFp16(std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                                float beta, const std::uint16_t *Bmat, std::size_t N,
                                const float *Crow_base, std::size_t Cstride, float *Yrow_base,
                                std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                                const std::uint16_t *Apack);

// Signature shared by every member of the native BFLOAT16 GEMM micro-kernel
// family (Roadmap PR07.4). It mirrors ``GemmFp16MicroKernel`` but consumes
// BFLOAT16 (raw ``std::uint16_t``) operands: ``Apack`` is a packed contiguous
// ``mr x K`` row-major panel (``Apack[r * K + k]`` is ``A(m + r, k0 + k)``) and
// ``Bmat`` is a row-major BFLOAT16 matrix or compact panel read with row stride
// ``N``. Products accumulate in float32 so the result matches the
// widen-then-float32 reference; the epilogue combines ``alpha * acc`` with
// ``Y`` according to ``mode`` and writes float32.
using GemmBf16MicroKernel = void (*)(std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                                     float beta, const std::uint16_t *Bmat, std::size_t N,
                                     const float *Crow_base, std::size_t Cstride, float *Yrow_base,
                                     std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                                     const std::uint16_t *Apack);

// Portable scalar member of the BFLOAT16 micro-kernel family. It is the tail
// handler for the vectorized AVX-512BF16 kernel and the reference the
// differential tests run everywhere (no AVX-512BF16 hardware required).
void GemmMicroKernel_ScalarBf16(std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                                float beta, const std::uint16_t *Bmat, std::size_t N,
                                const float *Crow_base, std::size_t Cstride, float *Yrow_base,
                                std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                                const std::uint16_t *Apack);

namespace detail {

// Native FLOAT16 general GEMM driver. It packs matching K chunks of ``A`` and
// compact ``kc x nb`` panels of non-transposed ``B``. Each B panel is copied
// once, shared read-only across parallel row panels, and passed to ``kernel``
// with row stride ``nb`` and ``n0 == 0``. Later K chunks accumulate into the
// float32 ``Y``. ``blocking`` may provide the prepared plan dimensions;
// otherwise dimensions suitable for 16-bit panels are selected locally.
void GemmFp16NativeGeneral(bool trans_a, std::size_t M, std::size_t N, std::size_t K, float alpha,
                           const std::uint16_t *A, const std::uint16_t *B, float *Y,
                           GemmFp16MicroKernel kernel, std::size_t mr,
                           const GemmBlocking *blocking = nullptr);

// Native BFLOAT16 general GEMM driver. It uses the same compact B-panel and
// K-chunk accumulation driver as :cpp:func:`GemmFp16NativeGeneral`.
void GemmBf16NativeGeneral(bool trans_a, std::size_t M, std::size_t N, std::size_t K, float alpha,
                           const std::uint16_t *A, const std::uint16_t *B, float *Y,
                           GemmBf16MicroKernel kernel, std::size_t mr,
                           const GemmBlocking *blocking = nullptr);

} // namespace detail

} // namespace onnx_light_cpu
