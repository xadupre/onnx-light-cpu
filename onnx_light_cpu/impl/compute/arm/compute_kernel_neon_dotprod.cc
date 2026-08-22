// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Native Advanced SIMD dot-product (UDOT) register-resident INT8
// dot-product throughput kernel (Processor Profile PR03). Compiled with an
// extra -march=armv8.2-a+dotprod (see the per-file COMPILE_OPTIONS override
// in CMakeLists.txt) even though the rest of onnx_light_cpu keeps the
// project's baseline architecture flags.
//
// Every operand and accumulator stays in a 128-bit NEON register for the
// whole timed chain of ``vdotq_u32`` reductions.

#include "onnx_light_cpu/impl/compute/arm/compute_kernel_neon_dotprod.h"

#include "onnx_light_cpu/impl/compute_arithmetic_profile.h"

#include <arm_neon.h>

namespace onnx_light_cpu {

namespace {

#if defined(_MSC_VER)
#define ONNX_LIGHT_CPU_NOINLINE __declspec(noinline)
#else
#define ONNX_LIGHT_CPU_NOINLINE __attribute__((noinline))
#endif

constexpr int kRegisters = 4;

} // namespace

ONNX_LIGHT_CPU_NOINLINE double ComputeArithmeticNeonDotProdRound(std::size_t passes, double seed) {
  uint32x4_t acc[kRegisters];
  uint8x16_t lhs[kRegisters];
  uint8x16_t rhs[kRegisters];
  for (int reg = 0; reg < kRegisters; ++reg) {
    const std::uint8_t base = static_cast<std::uint8_t>(1 + (static_cast<int>(seed) + reg) % 4);
    acc[reg] = vdupq_n_u32(0);
    lhs[reg] = vdupq_n_u8(base);
    rhs[reg] = vdupq_n_u8(static_cast<std::uint8_t>(1 + reg));
  }
  for (std::size_t pass = 0; pass < passes; ++pass) {
    for (std::size_t chain = 0; chain < kComputeChainLength; ++chain) {
      for (int reg = 0; reg < kRegisters; ++reg) {
        acc[reg] = vdotq_u32(acc[reg], lhs[reg], rhs[reg]);
      }
    }
  }
  double sum = 0.0;
  for (int reg = 0; reg < kRegisters; ++reg) {
    alignas(16) std::uint32_t lanes[4];
    vst1q_u32(lanes, acc[reg]);
    for (std::uint32_t lane : lanes) {
      sum += static_cast<double>(lane);
    }
  }
  return sum;
}

} // namespace onnx_light_cpu
