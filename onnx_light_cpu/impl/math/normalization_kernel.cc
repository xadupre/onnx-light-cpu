// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/normalization_kernel.h"

#include "onnx_light_cpu/impl/simd_level.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace onnx_light_cpu {
namespace {

using MeanSquareFunction = float (*)(const float *, std::size_t);
using MomentsFunction = Float32NormalizationMoments (*)(const float *, std::size_t);
using AffineFunction = void (*)(const float *, const float *, const float *, float *, std::size_t,
                                float, float);

float CancellationFloor(float second_moment, std::size_t count) {
  const std::size_t accumulation_steps = count / 4 + (count % 4 != 0);
  const float accumulation_epsilon =
      std::numeric_limits<float>::epsilon() * static_cast<float>(accumulation_steps) * 8.0F;
  const float subtraction_epsilon = std::sqrt(std::numeric_limits<float>::epsilon()) * 4.0F;
  return second_moment * (accumulation_epsilon + subtraction_epsilon);
}

float CenteredVariance(const float *input, std::size_t count, float mean) {
  float sums[4] = {};
  for (std::size_t index = 0; index < count; ++index) {
    const float delta = input[index] - mean;
    sums[index & 3] += delta * delta;
  }
  return (sums[0] + sums[1] + sums[2] + sums[3]) / static_cast<float>(count);
}

float MeanSquareScalar(const float *input, std::size_t count) {
  float sums[4] = {};
  for (std::size_t index = 0; index < count; ++index) {
    sums[index & 3] += input[index] * input[index];
  }
  return (sums[0] + sums[1] + sums[2] + sums[3]) / static_cast<float>(count);
}

Float32NormalizationMoments MomentsScalar(const float *input, std::size_t count) {
  float sums[4] = {};
  float square_sums[4] = {};
  for (std::size_t index = 0; index < count; ++index) {
    const float value = input[index];
    sums[index & 3] += value;
    square_sums[index & 3] += value * value;
  }
  const float mean = (sums[0] + sums[1] + sums[2] + sums[3]) / static_cast<float>(count);
  const float second_moment = (square_sums[0] + square_sums[1] + square_sums[2] + square_sums[3]) /
                              static_cast<float>(count);
  return {mean, second_moment - mean * mean};
}

void AffineScalar(const float *input, const float *scale, const float *bias, float *output,
                  std::size_t count, float center, float multiplier) {
  for (std::size_t index = 0; index < count; ++index) {
    float value = (input[index] - center) * multiplier * scale[index];
    if (bias != nullptr) {
      value += bias[index];
    }
    output[index] = value;
  }
}

struct NormalizationDispatch {
  MeanSquareFunction mean_square;
  MomentsFunction moments;
  AffineFunction affine;
};

const NormalizationDispatch &GetNormalizationDispatch() {
  static const NormalizationDispatch dispatch = [] {
    const SimdLevel simd = DetectSimdLevel();
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
    if (simd == SimdLevel::kAVX512) {
      return NormalizationDispatch{&ComputeNormalizationMeanSquareFloat32_AVX512,
                                   &ComputeNormalizationMomentsFloat32_AVX512,
                                   &ApplyNormalizationAffineFloat32_AVX512};
    }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
    if (simd >= SimdLevel::kAVX2 && CpuSupportsFma()) {
      return NormalizationDispatch{&ComputeNormalizationMeanSquareFloat32_AVX2,
                                   &ComputeNormalizationMomentsFloat32_AVX2,
                                   &ApplyNormalizationAffineFloat32_AVX2};
    }
#endif
    return NormalizationDispatch{&MeanSquareScalar, &MomentsScalar, &AffineScalar};
  }();
  return dispatch;
}

} // namespace

float ComputeNormalizationMeanSquareFloat32(const float *input, std::size_t count) {
  if (count == 0) {
    throw std::invalid_argument("normalization reduction size must be positive.");
  }
  return GetNormalizationDispatch().mean_square(input, count);
}

Float32NormalizationMoments ComputeNormalizationMomentsFloat32(const float *input,
                                                               std::size_t count) {
  if (count == 0) {
    throw std::invalid_argument("normalization reduction size must be positive.");
  }
  Float32NormalizationMoments moments = GetNormalizationDispatch().moments(input, count);
  const float second_moment = moments.variance + moments.mean * moments.mean;
  if (!(moments.variance > CancellationFloor(second_moment, count))) {
    moments.variance = CenteredVariance(input, count, moments.mean);
  }
  return moments;
}

void ApplyNormalizationAffineFloat32(const float *input, const float *scale, const float *bias,
                                     float *output, std::size_t count, float center,
                                     float multiplier) {
  GetNormalizationDispatch().affine(input, scale, bias, output, count, center, multiplier);
}

} // namespace onnx_light_cpu
