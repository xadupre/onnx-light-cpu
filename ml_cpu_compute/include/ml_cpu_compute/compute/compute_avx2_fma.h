// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

namespace ml_cpu_compute {

#ifdef ML_CPU_COMPUTE_HAVE_AVX2_FMA
constexpr int kComputeAvx2Registers = 4;
constexpr std::size_t kComputeAvx2Float32AccumulatorCount = kComputeAvx2Registers * 8;
constexpr std::size_t kComputeAvx2Float64AccumulatorCount = kComputeAvx2Registers * 4;

double ComputeArithmeticAvx2Float32Round(std::size_t passes, double seed);
double ComputeArithmeticAvx2Float64Round(std::size_t passes, double seed);
#endif

} // namespace ml_cpu_compute
