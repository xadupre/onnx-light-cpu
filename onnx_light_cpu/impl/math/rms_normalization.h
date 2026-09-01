// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace onnx_light_cpu {

void RmsNormalizationFloat32(const float *input, const float *scale, float *output,
                             std::size_t rows, std::size_t width, float epsilon);

void RmsNormalizationFloat16(const std::uint16_t *input, const std::uint16_t *scale,
                             std::uint16_t *output, std::size_t rows, std::size_t width,
                             float epsilon);

void RmsNormalizationBFloat16(const std::uint16_t *input, const std::uint16_t *scale,
                              std::uint16_t *output, std::size_t rows, std::size_t width,
                              float epsilon);

#ifdef ONNX_LIGHT_CPU_HAVE_RMS_F16C
void RmsNormalizationFloat16_F16C(const std::uint16_t *input, const std::uint16_t *scale,
                                  std::uint16_t *output, std::size_t row_begin, std::size_t row_end,
                                  std::size_t width, float epsilon);
#endif

#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
void RmsNormalizationFloat32_AVX2(const float *input, const float *scale, float *output,
                                  std::size_t row_begin, std::size_t row_end, std::size_t width,
                                  float epsilon);
void RmsNormalizationBFloat16_AVX2(const std::uint16_t *input, const std::uint16_t *scale,
                                   std::uint16_t *output, std::size_t row_begin,
                                   std::size_t row_end, std::size_t width, float epsilon);
#endif

#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
void RmsNormalizationFloat32_AVX512(const float *input, const float *scale, float *output,
                                    std::size_t row_begin, std::size_t row_end, std::size_t width,
                                    float epsilon);
#endif

} // namespace onnx_light_cpu
