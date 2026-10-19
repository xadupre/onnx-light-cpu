// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

namespace ml_cpu_compute {

#ifdef ML_CPU_COMPUTE_HAVE_AVX
#define ML_CPU_COMPUTE_DECLARE_BINARY_AVX(STEM, T)                                                 \
  void STEM##_AVX(const T *left, const T *right, T *out, std::size_t count);                       \
  void STEM##Left_AVX(T left, const T *right, T *out, std::size_t count);                          \
  void STEM##Right_AVX(const T *left, T right, T *out, std::size_t count);

ML_CPU_COMPUTE_DECLARE_BINARY_AVX(BinaryAddFloat32, float)
ML_CPU_COMPUTE_DECLARE_BINARY_AVX(BinarySubFloat32, float)
ML_CPU_COMPUTE_DECLARE_BINARY_AVX(BinaryMulFloat32, float)
ML_CPU_COMPUTE_DECLARE_BINARY_AVX(BinaryDivFloat32, float)
ML_CPU_COMPUTE_DECLARE_BINARY_AVX(BinaryAddFloat64, double)
ML_CPU_COMPUTE_DECLARE_BINARY_AVX(BinarySubFloat64, double)
ML_CPU_COMPUTE_DECLARE_BINARY_AVX(BinaryMulFloat64, double)
ML_CPU_COMPUTE_DECLARE_BINARY_AVX(BinaryDivFloat64, double)
ML_CPU_COMPUTE_DECLARE_BINARY_AVX(BinaryPReluFloat32, float)

#undef ML_CPU_COMPUTE_DECLARE_BINARY_AVX
#endif

} // namespace ml_cpu_compute
