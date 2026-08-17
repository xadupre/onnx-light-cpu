// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Declaration for the native AVX-512FP16 Gemm micro-kernel (Roadmap PR07.3).
// This kernel is only usable when ONNX_LIGHT_CPU_HAVE_AVX512FP16 is defined:
// gemm_kernel_avx512fp16.cc (the file that implements it) is only compiled into
// lib_onnx_light_cpu when CMake's check_cxx_compiler_flag confirms the compiler
// accepts -mavx512fp16 -- see CMakeLists.txt. Gate every reference to this
// function behind ``#ifdef ONNX_LIGHT_CPU_HAVE_AVX512FP16`` and only call it
// once ``CpuSupportsAvx512Fp16()`` reports the ISA at runtime.

#pragma once

#include "onnx_light_cpu/impl/math/gemm/gemm_common.h"

#include <cstddef>
#include <cstdint>

namespace onnx_light_cpu {

// Native AVX-512FP16 member of the FLOAT16 micro-kernel family. It consumes the
// FLOAT16 operands directly -- ``Apack`` is a packed ``mr x K`` row-major
// FLOAT16 panel and ``Bmat`` is the FLOAT16 ``B`` matrix with row stride ``N``
// -- converting them with ``vcvtph2psx`` and accumulating in float32, so the
// result matches the widen-then-float32 reference within float32 tolerance.
// Column counts that are not a multiple of the 16-lane vector width finish
// through the shared scalar tail handler ``GemmMicroKernel_ScalarFp16``.
void GemmMicroKernel_AVX512FP16(std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                                float beta, const std::uint16_t *Bmat, std::size_t N,
                                const float *Crow_base, std::size_t Cstride, float *Yrow_base,
                                std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                                const std::uint16_t *Apack);

} // namespace onnx_light_cpu
