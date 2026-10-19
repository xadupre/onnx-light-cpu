// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Declaration for the native AVX-512VNNI register-resident INT8 dot-product
// throughput kernel (Processor Profile PR03). This kernel is only usable
// when ONNX_LIGHT_CPU_HAVE_AVX512VNNI is defined: the file that implements
// it is only compiled into lib_onnx_light_cpu when CMake's
// check_cxx_compiler_flag confirms the compiler accepts -mavx512vnni -- see
// CMakeLists.txt. Gate every reference behind ``#ifdef
// ONNX_LIGHT_CPU_HAVE_AVX512VNNI`` and only call once
// ``CpuSupportsAvx512Vnni()`` reports the ISA at runtime.

#pragma once

#include <cstddef>

namespace onnx_light_cpu {

/// Independent dot-product accumulators (4 ZMM registers x 16 int32 lanes)
/// advanced per pass by ``ComputeArithmeticAvx512VnniRound``.
constexpr std::size_t kComputeAvx512VnniAccumulatorCount = 4 * 16;
/// ``vpdpbusd`` contracts 4 packed uint8 x int8 element pairs into each
/// int32 accumulator lane per instruction.
constexpr std::size_t kComputeAvx512VnniDotProductLength = 4;

/// Runs ``passes`` rounds of the shared arithmetic-profile chain length of
/// native ``vpdpbusd`` INT8 dot-product reductions over register-resident
/// accumulators and returns a checksum (the caller consumes it outside the
/// timed interval so the compiler cannot remove the work).
double ComputeArithmeticAvx512VnniRound(std::size_t passes, double seed);

} // namespace onnx_light_cpu
