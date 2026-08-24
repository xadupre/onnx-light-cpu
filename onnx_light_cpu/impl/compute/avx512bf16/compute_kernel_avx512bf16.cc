// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Native AVX-512BF16 register-resident BFLOAT16 dot-product throughput
// kernel (Processor Profile PR03). Compiled with an extra
// -mavx512f;-mavx512bw;-mavx512bf16 (see the per-file COMPILE_OPTIONS
// override in CMakeLists.txt) even though the rest of onnx_light_cpu keeps
// the project's baseline ONNX_LIGHT_CPU_SIMD_FLAGS.
//
// Every operand and accumulator stays in a ZMM register for the whole timed
// chain of ``vdpbf16ps`` reductions.

#include "onnx_light_cpu/impl/compute/avx512bf16/compute_kernel_avx512bf16.h"

#include "onnx_light_cpu/impl/compute_arithmetic_profile.h"

#include <bit>

#include <immintrin.h>

namespace onnx_light_cpu {

namespace {

#if defined(_MSC_VER)
#define ONNX_LIGHT_CPU_NOINLINE __declspec(noinline)
#else
#define ONNX_LIGHT_CPU_NOINLINE __attribute__((noinline))
#endif

constexpr int kRegisters = 4;

std::uint16_t Float32ToBf16(float value) {
  return static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value) >> 16);
}

// Packs two BFLOAT16 halves (each broadcast to every lane) into the
// ``vdpbf16ps`` pair layout, matching the AVX-512BF16 GEMM kernel's
// ``BroadcastAPair`` helper.
__m512bh BroadcastBf16Pair(float low, float high) {
  const std::uint32_t packed = static_cast<std::uint32_t>(Float32ToBf16(low)) |
                               (static_cast<std::uint32_t>(Float32ToBf16(high)) << 16);
  return reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int>(packed)));
}

} // namespace

ONNX_LIGHT_CPU_NOINLINE double ComputeArithmeticAvx512Bf16Round(std::size_t passes, double seed) {
  __m512 acc[kRegisters];
  __m512bh a_pair[kRegisters];
  __m512bh b_pair[kRegisters];
  for (int reg = 0; reg < kRegisters; ++reg) {
    acc[reg] = _mm512_set1_ps(static_cast<float>(seed) + static_cast<float>(reg));
    a_pair[reg] = BroadcastBf16Pair(1.0f + static_cast<float>(reg) * 0.01f, 0.5f);
    b_pair[reg] = BroadcastBf16Pair(0.25f, 0.125f + static_cast<float>(reg) * 0.01f);
  }
  for (std::size_t pass = 0; pass < passes; ++pass) {
    for (std::size_t chain = 0; chain < kComputeChainLength; ++chain) {
      for (int reg = 0; reg < kRegisters; ++reg) {
        acc[reg] = _mm512_dpbf16_ps(acc[reg], a_pair[reg], b_pair[reg]);
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

} // namespace onnx_light_cpu
