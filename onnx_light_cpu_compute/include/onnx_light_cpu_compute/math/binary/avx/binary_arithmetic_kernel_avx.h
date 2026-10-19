// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

namespace onnx_light_cpu {

#ifdef ONNX_LIGHT_CPU_HAVE_AVX
#define ONNX_LIGHT_CPU_DECLARE_BINARY_AVX(STEM, T)                                                 \
  void STEM##_AVX(const T *left, const T *right, T *out, std::size_t count);                       \
  void STEM##Left_AVX(T left, const T *right, T *out, std::size_t count);                          \
  void STEM##Right_AVX(const T *left, T right, T *out, std::size_t count);

ONNX_LIGHT_CPU_DECLARE_BINARY_AVX(BinaryAddFloat32, float)
ONNX_LIGHT_CPU_DECLARE_BINARY_AVX(BinarySubFloat32, float)
ONNX_LIGHT_CPU_DECLARE_BINARY_AVX(BinaryMulFloat32, float)
ONNX_LIGHT_CPU_DECLARE_BINARY_AVX(BinaryDivFloat32, float)
ONNX_LIGHT_CPU_DECLARE_BINARY_AVX(BinaryAddFloat64, double)
ONNX_LIGHT_CPU_DECLARE_BINARY_AVX(BinarySubFloat64, double)
ONNX_LIGHT_CPU_DECLARE_BINARY_AVX(BinaryMulFloat64, double)
ONNX_LIGHT_CPU_DECLARE_BINARY_AVX(BinaryDivFloat64, double)
ONNX_LIGHT_CPU_DECLARE_BINARY_AVX(BinaryPReluFloat32, float)

#undef ONNX_LIGHT_CPU_DECLARE_BINARY_AVX
#endif

} // namespace onnx_light_cpu
