// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Declaration for the native AMX-BF16 Gemm micro-tiles (Roadmap PR07.6). This
// routine is only usable when ML_CPU_COMPUTE_HAVE_AMX_BF16 is defined:
// gemm_amx_bf16.cc (the file that implements it) is only compiled into
// ml_cpu_compute when CMake's check_cxx_compiler_flag confirms the compiler
// accepts ``-mamx-tile -mamx-bf16`` -- see CMakeLists.txt. Gate every reference
// to this function behind ``#ifdef ML_CPU_COMPUTE_HAVE_AMX_BF16`` and only call
// it once ``CpuSupportsAmxBf16()`` reports the ISA and
// ``AmxTileStateAvailable()`` confirms the OS enabled tile state at runtime.

#pragma once

#include "ml_cpu_compute/math/gemm/gemm_common.h"

#include <cstddef>
#include <cstdint>

namespace ml_cpu_compute {

// Native AMX-BF16 member of the BFLOAT16 micro-tiles family. It shares the
// ``GemmBf16MicroTile`` signature so the shared ``GemmBf16NativeGeneral``
// driver, BFLOAT16 ``A`` packing, and column-tail handling are reused: ``Apack``
// is a packed ``mr x K`` row-major BFLOAT16 panel and ``Bmat`` is a compact
// BFLOAT16 panel with row stride ``N``. Internally it configures three AMX tiles
// through the PR07.5 tile-state lifecycle and reduces the ``K`` dimension with
// the ``tdpbf16ps`` (``_tile_dpbf16ps``) tile dot-product, accumulating in
// float32, so the result matches the widen-then-float32 reference within
// float32 tolerance. Partial ``mr``/``K``/column blocks are zero-padded into the
// fixed 16x16 tiles. When AMX tile state is unavailable it falls back to the
// shared scalar BFLOAT16 micro-tiles.
void GemmMicroTile_AMXBF16(std::size_t mr, std::size_t nb, std::size_t K, float alpha, float beta,
                           const std::uint16_t *Bmat, std::size_t N, const float *Crow_base,
                           std::size_t Cstride, float *Yrow_base, std::size_t Ystride,
                           std::size_t n0, GemmAccumMode mode, const std::uint16_t *Apack);

} // namespace ml_cpu_compute
