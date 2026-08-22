// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Native AVX-512F register-resident FP32/FP64 arithmetic throughput kernels
// (Processor Profile PR03). Compiled with an extra -mavx512f (see the
// per-file COMPILE_OPTIONS override in CMakeLists.txt) even though the rest
// of onnx_light_cpu keeps the project's baseline ONNX_LIGHT_CPU_SIMD_FLAGS.
//
// Every accumulator and operand stays in a ZMM register for the whole timed
// chain: this is a compute kernel, not a memory-bandwidth benchmark.

#include "onnx_light_cpu/impl/compute/avx512/compute_kernel_avx512.h"

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

ONNX_LIGHT_CPU_NOINLINE double ComputeArithmeticAvx512Float32Round(std::size_t passes,
                                                                   double seed) {
  __m512 acc[kRegisters];
  __m512 mul[kRegisters];
  __m512 add[kRegisters];
  for (int reg = 0; reg < kRegisters; ++reg) {
    const float base = static_cast<float>(seed) + static_cast<float>(reg);
    acc[reg] = _mm512_set1_ps(base);
    mul[reg] = _mm512_set1_ps(1.0000001f + static_cast<float>(reg) * 1e-7f);
    add[reg] = _mm512_set1_ps(1.0e-6f * static_cast<float>(reg + 1));
  }
  for (std::size_t pass = 0; pass < passes; ++pass) {
    for (std::size_t chain = 0; chain < kComputeChainLength; ++chain) {
      for (int reg = 0; reg < kRegisters; ++reg) {
        acc[reg] = _mm512_fmadd_ps(acc[reg], mul[reg], add[reg]);
      }
    }
  }
  alignas(64) float lanes[16];
  double sum = 0.0;
  for (int reg = 0; reg < kRegisters; ++reg) {
    _mm512_store_ps(lanes, acc[reg]);
    for (float lane : lanes) {
      sum += lane;
    }
  }
  return sum;
}

ONNX_LIGHT_CPU_NOINLINE double ComputeArithmeticAvx512Float64Round(std::size_t passes,
                                                                   double seed) {
  __m512d acc[kRegisters];
  __m512d mul[kRegisters];
  __m512d add[kRegisters];
  for (int reg = 0; reg < kRegisters; ++reg) {
    const double base = seed + static_cast<double>(reg);
    acc[reg] = _mm512_set1_pd(base);
    mul[reg] = _mm512_set1_pd(1.0000001 + static_cast<double>(reg) * 1e-9);
    add[reg] = _mm512_set1_pd(1.0e-9 * static_cast<double>(reg + 1));
  }
  for (std::size_t pass = 0; pass < passes; ++pass) {
    for (std::size_t chain = 0; chain < kComputeChainLength; ++chain) {
      for (int reg = 0; reg < kRegisters; ++reg) {
        acc[reg] = _mm512_fmadd_pd(acc[reg], mul[reg], add[reg]);
      }
    }
  }
  alignas(64) double lanes[8];
  double sum = 0.0;
  for (int reg = 0; reg < kRegisters; ++reg) {
    _mm512_store_pd(lanes, acc[reg]);
    for (double lane : lanes) {
      sum += lane;
    }
  }
  return sum;
}

} // namespace onnx_light_cpu
