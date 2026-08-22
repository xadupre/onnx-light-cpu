// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Native AVX-512VNNI register-resident INT8 dot-product throughput kernel
// (Processor Profile PR03). Compiled with an extra -mavx512f;-mavx512vnni
// (see the per-file COMPILE_OPTIONS override in CMakeLists.txt) even though
// the rest of onnx_light_cpu keeps the project's baseline
// ONNX_LIGHT_CPU_SIMD_FLAGS.
//
// Every operand and accumulator stays in a ZMM register for the whole timed
// chain of ``vpdpbusd`` reductions.

#include "onnx_light_cpu/impl/compute/avx512vnni/compute_kernel_avx512vnni.h"

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

ONNX_LIGHT_CPU_NOINLINE double ComputeArithmeticAvx512VnniRound(std::size_t passes, double seed) {
  __m512i acc[kRegisters];
  __m512i lhs[kRegisters]; // unsigned uint8 operand
  __m512i rhs[kRegisters]; // signed int8 operand
  for (int reg = 0; reg < kRegisters; ++reg) {
    const std::int8_t base = static_cast<std::int8_t>(1 + (static_cast<int>(seed) + reg) % 4);
    acc[reg] = _mm512_setzero_si512();
    lhs[reg] = _mm512_set1_epi8(base);
    rhs[reg] = _mm512_set1_epi8(static_cast<std::int8_t>(1 + reg));
  }
  for (std::size_t pass = 0; pass < passes; ++pass) {
    for (std::size_t chain = 0; chain < kComputeChainLength; ++chain) {
      for (int reg = 0; reg < kRegisters; ++reg) {
        acc[reg] = _mm512_dpbusd_epi32(acc[reg], lhs[reg], rhs[reg]);
      }
    }
  }
  alignas(64) std::int32_t lanes[16];
  double sum = 0.0;
  for (int reg = 0; reg < kRegisters; ++reg) {
    _mm512_store_si512(reinterpret_cast<__m512i *>(lanes), acc[reg]);
    for (std::int32_t lane : lanes) {
      sum += static_cast<double>(lane);
    }
  }
  return sum;
}

} // namespace onnx_light_cpu
