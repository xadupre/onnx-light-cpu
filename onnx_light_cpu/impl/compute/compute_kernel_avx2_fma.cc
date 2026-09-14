// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/compute/compute_kernel_avx2_fma.h"
#include "onnx_light_cpu/impl/compute_arithmetic_profile.h"

#include <immintrin.h>

#if defined(_MSC_VER)
#define ONNX_LIGHT_CPU_NOINLINE __declspec(noinline)
#else
#define ONNX_LIGHT_CPU_NOINLINE __attribute__((noinline))
#endif

namespace onnx_light_cpu {

ONNX_LIGHT_CPU_NOINLINE double ComputeArithmeticAvx2Float32Round(std::size_t passes, double seed) {
  __m256 acc[kComputeAvx2Registers];
  __m256 mul[kComputeAvx2Registers];
  __m256 add[kComputeAvx2Registers];
  for (int reg = 0; reg < kComputeAvx2Registers; ++reg) {
    acc[reg] = _mm256_set1_ps(static_cast<float>(seed) + static_cast<float>(reg));
    mul[reg] = _mm256_set1_ps(1.0000001f + static_cast<float>(reg) * 1e-7f);
    add[reg] = _mm256_set1_ps(1.0e-6f * static_cast<float>(reg + 1));
  }
  for (std::size_t pass = 0; pass < passes; ++pass) {
    for (std::size_t chain = 0; chain < kComputeChainLength; ++chain) {
      for (int reg = 0; reg < kComputeAvx2Registers; ++reg) {
        acc[reg] = _mm256_fmadd_ps(acc[reg], mul[reg], add[reg]);
      }
    }
  }
  alignas(32) float lanes[8];
  double sum = 0.0;
  for (int reg = 0; reg < kComputeAvx2Registers; ++reg) {
    _mm256_store_ps(lanes, acc[reg]);
    for (float lane : lanes) {
      sum += lane;
    }
  }
  return sum;
}

ONNX_LIGHT_CPU_NOINLINE double ComputeArithmeticAvx2Float64Round(std::size_t passes, double seed) {
  __m256d acc[kComputeAvx2Registers];
  __m256d mul[kComputeAvx2Registers];
  __m256d add[kComputeAvx2Registers];
  for (int reg = 0; reg < kComputeAvx2Registers; ++reg) {
    acc[reg] = _mm256_set1_pd(seed + static_cast<double>(reg));
    mul[reg] = _mm256_set1_pd(1.0000001 + static_cast<double>(reg) * 1e-9);
    add[reg] = _mm256_set1_pd(1.0e-9 * static_cast<double>(reg + 1));
  }
  for (std::size_t pass = 0; pass < passes; ++pass) {
    for (std::size_t chain = 0; chain < kComputeChainLength; ++chain) {
      for (int reg = 0; reg < kComputeAvx2Registers; ++reg) {
        acc[reg] = _mm256_fmadd_pd(acc[reg], mul[reg], add[reg]);
      }
    }
  }
  alignas(32) double lanes[4];
  double sum = 0.0;
  for (int reg = 0; reg < kComputeAvx2Registers; ++reg) {
    _mm256_store_pd(lanes, acc[reg]);
    for (double lane : lanes) {
      sum += lane;
    }
  }
  return sum;
}

} // namespace onnx_light_cpu
