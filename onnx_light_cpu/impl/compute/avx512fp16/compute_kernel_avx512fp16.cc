// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Native AVX-512FP16 register-resident FP16 arithmetic throughput kernel
// (Processor Profile PR03). Compiled with an extra -mavx512fp16 (see the
// per-file COMPILE_OPTIONS override in CMakeLists.txt) even though the rest
// of onnx_light_cpu keeps the project's baseline ONNX_LIGHT_CPU_SIMD_FLAGS.
//
// Unlike the FLOAT16 GEMM micro-kernel (which widens to float32 before
// accumulating so results match the reference kernel), this benchmark
// exercises the native ``vfmadd*ph`` half-precision fused multiply-add
// directly: it measures the CPU's real FP16 arithmetic throughput, not a
// converting float32 path.

#include "onnx_light_cpu/impl/compute/avx512fp16/compute_kernel_avx512fp16.h"

#include "onnx_light_cpu/impl/compute_arithmetic_profile.h"

#include <immintrin.h>

namespace onnx_light_cpu {

namespace {

#if defined(_MSC_VER)
#define ONNX_LIGHT_CPU_NOINLINE __declspec(noinline)
#else
#define ONNX_LIGHT_CPU_NOINLINE __attribute__((noinline))
#endif

constexpr int kRegisters = 4;

} // namespace

ONNX_LIGHT_CPU_NOINLINE double ComputeArithmeticAvx512Fp16Round(std::size_t passes, double seed) {
  __m512h acc[kRegisters];
  __m512h mul[kRegisters];
  __m512h add[kRegisters];
  for (int reg = 0; reg < kRegisters; ++reg) {
    const _Float16 base = static_cast<_Float16>(seed + static_cast<double>(reg));
    acc[reg] = _mm512_set1_ph(base);
    // FP16's narrow exponent range (max ~65504) cannot absorb the same
    // slightly-above-one multiplier used by the wider float32/float64
    // kernels over the very large iteration counts this benchmark runs;
    // keep the multiplier exactly 1 (the hardware still executes a real
    // ``vfmadd*ph`` every iteration) and let the tiny additive term settle
    // to a stable value once FP16's ~3-digit precision absorbs it.
    mul[reg] = _mm512_set1_ph(static_cast<_Float16>(1.0));
    add[reg] = _mm512_set1_ph(static_cast<_Float16>(0.0009765625 * (reg + 1)));
  }
  for (std::size_t pass = 0; pass < passes; ++pass) {
    for (std::size_t chain = 0; chain < kComputeChainLength; ++chain) {
      for (int reg = 0; reg < kRegisters; ++reg) {
        acc[reg] = _mm512_fmadd_ph(acc[reg], mul[reg], add[reg]);
      }
    }
  }
  alignas(64) _Float16 lanes[32];
  double sum = 0.0;
  for (int reg = 0; reg < kRegisters; ++reg) {
    _mm512_store_ph(lanes, acc[reg]);
    for (_Float16 lane : lanes) {
      sum += static_cast<double>(lane);
    }
  }
  return sum;
}

} // namespace onnx_light_cpu
