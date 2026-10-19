// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Declaration for the native AVX-512FP16 register-resident FP16 arithmetic
// throughput routine (Processor Profile PR03). This routine is only usable
// when CPU_COMPUTE_HAVE_AVX512FP16 is defined: the file that implements it
// is only compiled into cpu_compute when CMake's
// check_cxx_compiler_flag confirms the compiler accepts -mavx512fp16 -- see
// CMakeLists.txt. Gate every reference behind ``#ifdef
// CPU_COMPUTE_HAVE_AVX512FP16`` and only call once
// ``CpuSupportsAvx512Fp16()`` reports the ISA at runtime.

#pragma once

#include <cstddef>

namespace cpu_compute {

/// Independent scalar-equivalent accumulators (4 ZMM registers x 32 float16
/// lanes) advanced per pass by ``ComputeArithmeticAvx512Fp16Round``.
constexpr std::size_t kComputeAvx512Fp16AccumulatorCount = 4 * 32;

/// Runs ``passes`` rounds of the shared arithmetic-profile FMA chain length
/// over register-resident, native AVX-512FP16 accumulators and returns a
/// checksum (the caller consumes it outside the timed interval so the
/// compiler cannot remove the work).
double ComputeArithmeticAvx512Fp16Round(std::size_t passes, double seed);

} // namespace cpu_compute
