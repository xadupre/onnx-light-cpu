// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

namespace onnx_light_cpu {

#ifdef ONNX_LIGHT_CPU_HAVE_AVX
constexpr int kComputeAvxRegisters = 4;
constexpr std::size_t kComputeAvxFloat32AccumulatorCount = kComputeAvxRegisters * 8;
constexpr std::size_t kComputeAvxFloat64AccumulatorCount = kComputeAvxRegisters * 4;

double ComputeArithmeticAvxFloat32Round(std::size_t passes, double seed);
double ComputeArithmeticAvxFloat64Round(std::size_t passes, double seed);
#endif

} // namespace onnx_light_cpu
