// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

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

#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
float ComputeNormalizationMeanSquareFloat32_AVX2(const float *input, std::size_t count);
Float32NormalizationMoments ComputeNormalizationMomentsFloat32_AVX2(const float *input,
                                                                    std::size_t count);
void ApplyNormalizationAffineFloat32_AVX2(const float *input, const float *scale, const float *bias,
                                          float *output, std::size_t count, float center,
                                          float multiplier);
#endif

#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
float ComputeNormalizationMeanSquareFloat32_AVX512(const float *input, std::size_t count);
Float32NormalizationMoments ComputeNormalizationMomentsFloat32_AVX512(const float *input,
                                                                      std::size_t count);
void ApplyNormalizationAffineFloat32_AVX512(const float *input, const float *scale,
                                            const float *bias, float *output, std::size_t count,
                                            float center, float multiplier);
#endif

} // namespace onnx_light_cpu
