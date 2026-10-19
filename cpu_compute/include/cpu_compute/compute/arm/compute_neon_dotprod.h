// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Declaration for the native Advanced SIMD dot-product (SDOT/UDOT)
// register-resident INT8 dot-product throughput routine (Processor Profile
// PR03). This routine is only usable when CPU_COMPUTE_HAVE_NEON_DOTPROD is
// defined: the file that implements it is only compiled into
// cpu_compute when CMake's check_cxx_source_compiles probe confirms
// the compiler accepts -march=armv8.2-a+dotprod -- see CMakeLists.txt. Gate
// every reference behind ``#ifdef CPU_COMPUTE_HAVE_NEON_DOTPROD`` and only
// call once ``CpuSupportsNeonDotProd()`` reports the ISA at runtime.

#pragma once

#include <cstddef>

namespace cpu_compute {

/// Independent dot-product accumulators (4 128-bit registers x 4 int32
/// lanes) advanced per pass by ``ComputeArithmeticNeonDotProdRound``.
constexpr std::size_t kComputeNeonDotProdAccumulatorCount = 4 * 4;
/// ``UDOT`` contracts 4 packed uint8 element pairs into each int32
/// accumulator lane per instruction.
constexpr std::size_t kComputeNeonDotProdDotProductLength = 4;

/// Runs ``passes`` rounds of the shared arithmetic-profile chain length of
/// native ``UDOT`` INT8 dot-product reductions over register-resident
/// accumulators and returns a checksum (the caller consumes it outside the
/// timed interval so the compiler cannot remove the work).
double ComputeArithmeticNeonDotProdRound(std::size_t passes, double seed);

} // namespace cpu_compute
