// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace onnx_light_cpu {

struct Float32NormalizationMoments {
  float mean;
  float variance;
};

float ComputeNormalizationMeanSquareFloat32(const float *input, std::size_t count);
Float32NormalizationMoments ComputeNormalizationMomentsFloat32(const float *input,
                                                               std::size_t count);
void ApplyNormalizationAffineFloat32(const float *input, const float *scale, const float *bias,
                                     float *output, std::size_t count, float center,
                                     float multiplier);
void ApplyRmsNormalizationFloat32(const float *input, const float *scale, float *output,
                                  std::size_t count, float multiplier);
void ApplyNormalizationScaleBiasFloat32(const float *input, float *output, std::size_t count,
                                        float multiplier, float offset);

// Half affine rounds only the scaled result, unlike RMSNormalization's intermediate rounding.
float ComputeNormalizationMeanSquareFloat16(const std::uint16_t *input, std::size_t count);
Float32NormalizationMoments ComputeNormalizationMomentsFloat16(const std::uint16_t *input,
                                                               std::size_t count);
void ApplyNormalizationAffineFloat16(const std::uint16_t *input, const std::uint16_t *scale,
                                     std::uint16_t *output, std::size_t count, float multiplier);
void ApplyNormalizationScaleBiasFloat16(const std::uint16_t *input, std::uint16_t *output,
                                        std::size_t count, float multiplier, float offset);
void ApplyLayerNormalizationFloat16(const std::uint16_t *input, const std::uint16_t *scale,
                                    const std::uint16_t *bias, std::uint16_t *output,
                                    std::size_t count, float center, float multiplier);
void ApplyGroupNormalizationFloat16(const std::uint16_t *input, std::uint16_t *output,
                                    std::size_t count, float center, float multiplier, float scale,
                                    float bias);
void LpNormalizationFloat16(const std::uint16_t *input, std::uint16_t *output, std::size_t vectors,
                            std::size_t width);
double ComputeNormalizationMeanSquareFloat64(const double *input, std::size_t count);
float ComputeNormalizationMeanSquareFloat64StashFloat32(const double *input, std::size_t count);
void ApplyNormalizationAffineFloat64(const double *input, const double *scale, double *output,
                                     std::size_t count, double multiplier);
void ApplyNormalizationAffineFloat64StashFloat32(const double *input, const double *scale,
                                                 double *output, std::size_t count,
                                                 float multiplier);
const char *NormalizationFloat16Path();
const char *NormalizationFloat64Path();

#ifdef ONNX_LIGHT_CPU_HAVE_RMS_F16C
float ComputeNormalizationMeanSquareFloat16_F16C(const std::uint16_t *input, std::size_t count);
void ApplyNormalizationAffineFloat16_F16C(const std::uint16_t *input, const std::uint16_t *scale,
                                          std::uint16_t *output, std::size_t count,
                                          float multiplier);
#endif

#ifdef ONNX_LIGHT_CPU_HAVE_AVX_F16C
Float32NormalizationMoments ComputeNormalizationMomentsFloat16_F16C(const std::uint16_t *input,
                                                                    std::size_t count);
void ApplyNormalizationScaleBiasFloat16_F16C(const std::uint16_t *input, std::uint16_t *output,
                                             std::size_t count, float multiplier, float offset);
void ApplyLayerNormalizationFloat16_F16C(const std::uint16_t *input, const std::uint16_t *scale,
                                         const std::uint16_t *bias, std::uint16_t *output,
                                         std::size_t count, float center, float multiplier);
void ApplyGroupNormalizationFloat16_F16C(const std::uint16_t *input, std::uint16_t *output,
                                         std::size_t count, float center, float multiplier,
                                         float scale, float bias);
void LpNormalizationFloat16_F16C(const std::uint16_t *input, std::uint16_t *output,
                                 std::size_t vectors, std::size_t width);
#endif

#ifdef ONNX_LIGHT_CPU_HAVE_AVX
double ComputeNormalizationMeanSquareFloat64_AVX(const double *input, std::size_t count);
float ComputeNormalizationMeanSquareFloat64StashFloat32_AVX(const double *input, std::size_t count);
void ApplyNormalizationAffineFloat64_AVX(const double *input, const double *scale, double *output,
                                         std::size_t count, double multiplier);
void ApplyNormalizationAffineFloat64StashFloat32_AVX(const double *input, const double *scale,
                                                     double *output, std::size_t count,
                                                     float multiplier);
#endif

#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
float ComputeResidualMeanSquareFloat32_AVX2(const float *input, const float *skip,
                                            const float *bias, float *residual, std::size_t count);
void ApplyNormalizationScaleBiasFloat32_AVX2(const float *input, float *output, std::size_t count,
                                             float multiplier, float offset);
float ComputeNormalizationMeanSquareFloat32_AVX2(const float *input, std::size_t count);
Float32NormalizationMoments ComputeNormalizationMomentsFloat32_AVX2(const float *input,
                                                                    std::size_t count);
void ApplyNormalizationAffineFloat32_AVX2(const float *input, const float *scale, const float *bias,
                                          float *output, std::size_t count, float center,
                                          float multiplier);
void ApplyRmsNormalizationFloat32_AVX2(const float *input, const float *scale, float *output,
                                       std::size_t count, float multiplier);
#endif

#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
void ApplyNormalizationScaleBiasFloat32_AVX512(const float *input, float *output, std::size_t count,
                                               float multiplier, float offset);
float ComputeNormalizationMeanSquareFloat32_AVX512(const float *input, std::size_t count);
Float32NormalizationMoments ComputeNormalizationMomentsFloat32_AVX512(const float *input,
                                                                      std::size_t count);
void ApplyNormalizationAffineFloat32_AVX512(const float *input, const float *scale,
                                            const float *bias, float *output, std::size_t count,
                                            float center, float multiplier);
void ApplyRmsNormalizationFloat32_AVX512(const float *input, const float *scale, float *output,
                                         std::size_t count, float multiplier);
#endif

} // namespace onnx_light_cpu
