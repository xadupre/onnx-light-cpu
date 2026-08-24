// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/half_conversion.h"

#include "onnx_light_cpu/impl/math/gemm/avx2/gemm_kernel_avx2_fma.h"
#include "onnx_light_cpu/impl/simd_level.h"

namespace onnx_light_cpu::detail {

void ConvertFloat16ToFloat32(const std::uint16_t *src, float *dst, std::size_t count) {
#ifdef ONNX_LIGHT_CPU_HAVE_F16C
  if (CpuSupportsF16C()) {
    GemmConvertFloat16ToFloat32_F16C(src, dst, count);
    return;
  }
#endif
  for (std::size_t i = 0; i < count; ++i) {
    dst[i] = Float16BitsToFloat(src[i]);
  }
}

void ConvertBFloat16ToFloat32(const std::uint16_t *src, float *dst, std::size_t count) {
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
  if (DetectSimdLevel() >= SimdLevel::kAVX2) {
    GemmConvertBFloat16ToFloat32_AVX2(src, dst, count);
    return;
  }
#endif
  for (std::size_t i = 0; i < count; ++i) {
    dst[i] = Bfloat16BitsToFloat(src[i]);
  }
}

void ConvertFloat32ToFloat16(const float *src, std::uint16_t *dst, std::size_t count) {
#ifdef ONNX_LIGHT_CPU_HAVE_F16C
  if (CpuSupportsF16C()) {
    GemmConvertFloat32ToFloat16_F16C(src, dst, count);
    return;
  }
#endif
  for (std::size_t i = 0; i < count; ++i) {
    dst[i] = FloatToFloat16Bits(src[i]);
  }
}

void ConvertFloat32ToBFloat16(const float *src, std::uint16_t *dst, std::size_t count) {
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
  if (DetectSimdLevel() >= SimdLevel::kAVX2) {
    GemmConvertFloat32ToBFloat16_AVX2(src, dst, count);
    return;
  }
#endif
  for (std::size_t i = 0; i < count; ++i) {
    dst[i] = FloatToBFloat16Bits(src[i]);
  }
}

} // namespace onnx_light_cpu::detail
