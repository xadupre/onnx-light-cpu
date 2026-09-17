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
void ApplyNormalizationScaleBiasFloat32(const float *input, float *output, std::size_t count,
                                        float multiplier, float offset);

// Half affine rounds only the scaled result, unlike RMSNormalization's intermediate rounding.
float ComputeNormalizationMeanSquareFloat16(const std::uint16_t *input, std::size_t count);
void ApplyNormalizationAffineFloat16(const std::uint16_t *input, const std::uint16_t *scale,
                                     std::uint16_t *output, std::size_t count, float multiplier);
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
void ApplyNormalizationScaleBiasFloat32_AVX2(const float *input, float *output, std::size_t count,
                                             float multiplier, float offset);
float ComputeNormalizationMeanSquareFloat32_AVX2(const float *input, std::size_t count);
Float32NormalizationMoments ComputeNormalizationMomentsFloat32_AVX2(const float *input,
                                                                    std::size_t count);
void ApplyNormalizationAffineFloat32_AVX2(const float *input, const float *scale, const float *bias,
                                          float *output, std::size_t count, float center,
                                          float multiplier);
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
#endif

} // namespace onnx_light_cpu
