// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Declaration for the native AVX-512BF16 register-resident BFLOAT16
// dot-product throughput kernel (Processor Profile PR03). This kernel is
// only usable when ONNX_LIGHT_CPU_HAVE_AVX512BF16 is defined: the file that
// implements it is only compiled into lib_onnx_light_cpu when CMake's
// check_cxx_compiler_flag confirms the compiler accepts -mavx512bf16 -- see
// CMakeLists.txt. Gate every reference behind ``#ifdef
// ONNX_LIGHT_CPU_HAVE_AVX512BF16`` and only call once
// ``CpuSupportsAvx512Bf16()`` reports the ISA at runtime.

#pragma once

#include <cstddef>

namespace onnx_light_cpu {

/// Independent dot-product accumulators (4 ZMM registers x 16 float32
/// lanes) advanced per pass by ``ComputeArithmeticAvx512Bf16Round``.
constexpr std::size_t kComputeAvx512Bf16AccumulatorCount = 4 * 16;
/// ``vdpbf16ps`` contracts 2 packed BFLOAT16 element pairs into each
/// float32 accumulator lane per instruction.
constexpr std::size_t kComputeAvx512Bf16DotProductLength = 2;

/// Runs ``passes`` rounds of the shared arithmetic-profile chain length of
/// native ``vdpbf16ps`` BFLOAT16 dot-product reductions over
/// register-resident accumulators and returns a checksum (the caller
/// consumes it outside the timed interval so the compiler cannot remove the
/// work).
double ComputeArithmeticAvx512Bf16Round(std::size_t passes, double seed);

} // namespace onnx_light_cpu
