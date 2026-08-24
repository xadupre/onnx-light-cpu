// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/binary/binary_arithmetic_kernel.h"

#include <arm_sve.h>

#include <cstddef>
#include <cstdint>

namespace onnx_light_cpu {

// ---------------------------------------------------------------------------
// SVE/SVE2 implementations, compiled in this dedicated translation unit with
// ``-march=armv8-a+sve`` so the rest of the library stays on the AArch64
// baseline and only reaches these once ``DetectArmSimdLevel()`` reports
// ``ArmSimdLevel::kSve``/``kSve2`` at runtime (SVE2 adds no new FP32/FP64
// arithmetic instructions over baseline SVE, so one kernel serves both
// levels). Every loop iteration is fully predicated with ``svwhilelt`` --
// there is no separate scalar tail, and the store never writes past
// ``count`` because inactive lanes are masked. The intrinsic names
// (``sv<op>_f32_x``/``sv<op>_f64_x``) are regular across Add/Sub/Mul/Div, so
// a single macro instantiates the whole type x operator matrix.
// ---------------------------------------------------------------------------
#define ONNX_LIGHT_CPU_BIN_SVE_F32(STEM, INTRIN)                                                   \
  void STEM##_SVE(const float *left, const float *right, float *out, std::size_t count) {          \
    const std::uint64_t total = static_cast<std::uint64_t>(count);                                 \
    for (std::uint64_t i = 0; i < total; i += svcntw()) {                                          \
      const svbool_t pg = svwhilelt_b32(i, total);                                                 \
      const svfloat32_t left_vec = svld1_f32(pg, left + i);                                        \
      const svfloat32_t right_vec = svld1_f32(pg, right + i);                                      \
      svst1_f32(pg, out + i, INTRIN(pg, left_vec, right_vec));                                     \
    }                                                                                              \
  }                                                                                                \
  void STEM##Left_SVE(float left, const float *right, float *out, std::size_t count) {             \
    const std::uint64_t total = static_cast<std::uint64_t>(count);                                 \
    const svfloat32_t left_vec = svdup_n_f32(left);                                                \
    for (std::uint64_t i = 0; i < total; i += svcntw()) {                                          \
      const svbool_t pg = svwhilelt_b32(i, total);                                                 \
      const svfloat32_t right_vec = svld1_f32(pg, right + i);                                      \
      svst1_f32(pg, out + i, INTRIN(pg, left_vec, right_vec));                                     \
    }                                                                                              \
  }                                                                                                \
  void STEM##Right_SVE(const float *left, float right, float *out, std::size_t count) {            \
    const std::uint64_t total = static_cast<std::uint64_t>(count);                                 \
    const svfloat32_t right_vec = svdup_n_f32(right);                                              \
    for (std::uint64_t i = 0; i < total; i += svcntw()) {                                          \
      const svbool_t pg = svwhilelt_b32(i, total);                                                 \
      const svfloat32_t left_vec = svld1_f32(pg, left + i);                                        \
      svst1_f32(pg, out + i, INTRIN(pg, left_vec, right_vec));                                     \
    }                                                                                              \
  }

#define ONNX_LIGHT_CPU_BIN_SVE_F64(STEM, INTRIN)                                                   \
  void STEM##_SVE(const double *left, const double *right, double *out, std::size_t count) {       \
    const std::uint64_t total = static_cast<std::uint64_t>(count);                                 \
    for (std::uint64_t i = 0; i < total; i += svcntd()) {                                          \
      const svbool_t pg = svwhilelt_b64(i, total);                                                 \
      const svfloat64_t left_vec = svld1_f64(pg, left + i);                                        \
      const svfloat64_t right_vec = svld1_f64(pg, right + i);                                      \
      svst1_f64(pg, out + i, INTRIN(pg, left_vec, right_vec));                                     \
    }                                                                                              \
  }                                                                                                \
  void STEM##Left_SVE(double left, const double *right, double *out, std::size_t count) {          \
    const std::uint64_t total = static_cast<std::uint64_t>(count);                                 \
    const svfloat64_t left_vec = svdup_n_f64(left);                                                \
    for (std::uint64_t i = 0; i < total; i += svcntd()) {                                          \
      const svbool_t pg = svwhilelt_b64(i, total);                                                 \
      const svfloat64_t right_vec = svld1_f64(pg, right + i);                                      \
      svst1_f64(pg, out + i, INTRIN(pg, left_vec, right_vec));                                     \
    }                                                                                              \
  }                                                                                                \
  void STEM##Right_SVE(const double *left, double right, double *out, std::size_t count) {         \
    const std::uint64_t total = static_cast<std::uint64_t>(count);                                 \
    const svfloat64_t right_vec = svdup_n_f64(right);                                              \
    for (std::uint64_t i = 0; i < total; i += svcntd()) {                                          \
      const svbool_t pg = svwhilelt_b64(i, total);                                                 \
      const svfloat64_t left_vec = svld1_f64(pg, left + i);                                        \
      svst1_f64(pg, out + i, INTRIN(pg, left_vec, right_vec));                                     \
    }                                                                                              \
  }

ONNX_LIGHT_CPU_BIN_SVE_F32(BinaryAddFloat32, svadd_f32_x)
ONNX_LIGHT_CPU_BIN_SVE_F32(BinarySubFloat32, svsub_f32_x)
ONNX_LIGHT_CPU_BIN_SVE_F32(BinaryMulFloat32, svmul_f32_x)
ONNX_LIGHT_CPU_BIN_SVE_F32(BinaryDivFloat32, svdiv_f32_x)
ONNX_LIGHT_CPU_BIN_SVE_F64(BinaryAddFloat64, svadd_f64_x)
ONNX_LIGHT_CPU_BIN_SVE_F64(BinarySubFloat64, svsub_f64_x)
ONNX_LIGHT_CPU_BIN_SVE_F64(BinaryMulFloat64, svmul_f64_x)
ONNX_LIGHT_CPU_BIN_SVE_F64(BinaryDivFloat64, svdiv_f64_x)

#undef ONNX_LIGHT_CPU_BIN_SVE_F32
#undef ONNX_LIGHT_CPU_BIN_SVE_F64

} // namespace onnx_light_cpu
