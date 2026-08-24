// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

// The binary arithmetic kernels dispatch on the same runtime SIMD levels as
// the rest of the math kernel family.
#include "onnx_light_cpu/impl/simd_level.h"

namespace onnx_light_cpu {

/// Binary PR02 FP32/FP64 SIMD arithmetic kernels for ``Add``, ``Sub``, ``Mul``
/// and ``Div``. Each operator provides three entry points that mirror
/// ``BinaryBroadcastPlan::LoopFamily``:
///
/// * ``Contiguous``  -- ``out[i] = a[i] OP b[i]`` for two equal-length arrays.
/// * ``LeftScalar``  -- ``out[i] = a OP b[i]`` with ``a`` a single broadcast
///   scalar (the left operand of the ONNX op).
/// * ``RightScalar`` -- ``out[i] = a[i] OP b`` with ``b`` a single broadcast
///   scalar (the right operand of the ONNX op).
///
/// Operand order is preserved exactly (important for ``Sub``/``Div``), and
/// every path matches the portable scalar reference on ordinary values, NaN,
/// infinities, and signed zero. Each entry point dispatches at runtime to the
/// best available ISA (AVX-512/AVX2/SSE2 on x86, SVE/SVE2/NEON on AArch64)
/// with an exact scalar tail; unsupported ISAs stay on the portable scalar
/// loop without any global architecture flag.
#define ONNX_LIGHT_CPU_DECLARE_BINARY_ARITH(NAME, T)                                               \
  void NAME##Contiguous(const T *left, const T *right, T *out, std::size_t count);                 \
  void NAME##LeftScalar(T left, const T *right, T *out, std::size_t count);                        \
  void NAME##RightScalar(const T *left, T right, T *out, std::size_t count);

ONNX_LIGHT_CPU_DECLARE_BINARY_ARITH(BinaryAddFloat32, float)
ONNX_LIGHT_CPU_DECLARE_BINARY_ARITH(BinarySubFloat32, float)
ONNX_LIGHT_CPU_DECLARE_BINARY_ARITH(BinaryMulFloat32, float)
ONNX_LIGHT_CPU_DECLARE_BINARY_ARITH(BinaryDivFloat32, float)

ONNX_LIGHT_CPU_DECLARE_BINARY_ARITH(BinaryAddFloat64, double)
ONNX_LIGHT_CPU_DECLARE_BINARY_ARITH(BinarySubFloat64, double)
ONNX_LIGHT_CPU_DECLARE_BINARY_ARITH(BinaryMulFloat64, double)
ONNX_LIGHT_CPU_DECLARE_BINARY_ARITH(BinaryDivFloat64, double)

#undef ONNX_LIGHT_CPU_DECLARE_BINARY_ARITH

#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
// Raw AVX-512F implementations, defined in a translation unit compiled with
// ``-mavx512f``/``/arch:AVX512``. Declared here so the baseline dispatcher
// (compiled without that flag) can call them once ``DetectSimdLevel()``
// reports AVX-512 support.
#define ONNX_LIGHT_CPU_DECLARE_BINARY_ARITH_AVX512(NAME, T)                                        \
  void NAME##_AVX512(const T *left, const T *right, T *out, std::size_t count);                    \
  void NAME##Left_AVX512(T left, const T *right, T *out, std::size_t count);                       \
  void NAME##Right_AVX512(const T *left, T right, T *out, std::size_t count);

ONNX_LIGHT_CPU_DECLARE_BINARY_ARITH_AVX512(BinaryAddFloat32, float)
ONNX_LIGHT_CPU_DECLARE_BINARY_ARITH_AVX512(BinarySubFloat32, float)
ONNX_LIGHT_CPU_DECLARE_BINARY_ARITH_AVX512(BinaryMulFloat32, float)
ONNX_LIGHT_CPU_DECLARE_BINARY_ARITH_AVX512(BinaryDivFloat32, float)

ONNX_LIGHT_CPU_DECLARE_BINARY_ARITH_AVX512(BinaryAddFloat64, double)
ONNX_LIGHT_CPU_DECLARE_BINARY_ARITH_AVX512(BinarySubFloat64, double)
ONNX_LIGHT_CPU_DECLARE_BINARY_ARITH_AVX512(BinaryMulFloat64, double)
ONNX_LIGHT_CPU_DECLARE_BINARY_ARITH_AVX512(BinaryDivFloat64, double)

#undef ONNX_LIGHT_CPU_DECLARE_BINARY_ARITH_AVX512
#endif // ONNX_LIGHT_CPU_HAVE_AVX512

#ifdef ONNX_LIGHT_CPU_HAVE_SVE
// Raw SVE/SVE2 implementations (SVE2 shares the same baseline SVE FP32/FP64
// arithmetic instructions, so one kernel serves both levels), defined in a
// translation unit compiled with ``-march=armv8-a+sve``.
#define ONNX_LIGHT_CPU_DECLARE_BINARY_ARITH_SVE(NAME, T)                                           \
  void NAME##_SVE(const T *left, const T *right, T *out, std::size_t count);                       \
  void NAME##Left_SVE(T left, const T *right, T *out, std::size_t count);                          \
  void NAME##Right_SVE(const T *left, T right, T *out, std::size_t count);

ONNX_LIGHT_CPU_DECLARE_BINARY_ARITH_SVE(BinaryAddFloat32, float)
ONNX_LIGHT_CPU_DECLARE_BINARY_ARITH_SVE(BinarySubFloat32, float)
ONNX_LIGHT_CPU_DECLARE_BINARY_ARITH_SVE(BinaryMulFloat32, float)
ONNX_LIGHT_CPU_DECLARE_BINARY_ARITH_SVE(BinaryDivFloat32, float)

ONNX_LIGHT_CPU_DECLARE_BINARY_ARITH_SVE(BinaryAddFloat64, double)
ONNX_LIGHT_CPU_DECLARE_BINARY_ARITH_SVE(BinarySubFloat64, double)
ONNX_LIGHT_CPU_DECLARE_BINARY_ARITH_SVE(BinaryMulFloat64, double)
ONNX_LIGHT_CPU_DECLARE_BINARY_ARITH_SVE(BinaryDivFloat64, double)

#undef ONNX_LIGHT_CPU_DECLARE_BINARY_ARITH_SVE
#endif // ONNX_LIGHT_CPU_HAVE_SVE

} // namespace onnx_light_cpu
