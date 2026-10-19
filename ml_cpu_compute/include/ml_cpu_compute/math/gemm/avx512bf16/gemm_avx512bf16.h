// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Declaration for the native AVX-512BF16 Gemm micro-tiles (Roadmap PR07.4).
// This routine is only usable when ML_CPU_COMPUTE_HAVE_AVX512BF16 is defined:
// gemm_avx512bf16.cc (the file that implements it) is only compiled into
// ml_cpu_compute when CMake's check_cxx_compiler_flag confirms the compiler
// accepts -mavx512bf16 -- see CMakeLists.txt. Gate every reference to this
// function behind ``#ifdef ML_CPU_COMPUTE_HAVE_AVX512BF16`` and only call it
// once ``CpuSupportsAvx512Bf16()`` reports the ISA at runtime.

#pragma once

#include "ml_cpu_compute/math/gemm/gemm_common.h"

#include <cstddef>
#include <cstdint>

namespace ml_cpu_compute {

// Native AVX-512BF16 member of the BFLOAT16 micro-tiles family. It consumes the
// BFLOAT16 operands directly -- ``Apack`` is a packed ``mr x K`` row-major
// BFLOAT16 panel and ``Bmat`` is the BFLOAT16 ``B`` matrix with row stride
// ``N`` -- reducing pairs of ``k`` iterations with the ``vdpbf16ps``
// (``_mm512_dpbf16_ps``) dot-product and accumulating in float32, so the result
// matches the widen-then-float32 reference within float32 tolerance. Column
// counts that are not a multiple of the 16-lane vector width finish through the
// shared scalar tail handler ``GemmMicroTile_ScalarBf16``.
void GemmMicroTile_AVX512BF16(std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                              float beta, const std::uint16_t *Bmat, std::size_t N,
                              const float *Crow_base, std::size_t Cstride, float *Yrow_base,
                              std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                              const std::uint16_t *Apack);

} // namespace ml_cpu_compute
