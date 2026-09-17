// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/normalization_kernel.h"

#include "onnx_light_cpu/impl/math/half_conversion.h"
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
using ScaleBiasFunction = void (*)(const float *, float *, std::size_t, float, float);

// Match the SIMD path's separately rounded multiply and add on FMA-capable CPUs.
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
void ScaleBiasScalar(const float *input, float *output, std::size_t count, float multiplier,
                     float offset) {
#ifdef __clang__
#pragma clang fp contract(off)
#endif
  for (std::size_t index = 0; index < count; ++index) {
    output[index] = input[index] * multiplier + offset;
  }
}

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
  ScaleBiasFunction scale_bias;
};

const NormalizationDispatch &GetNormalizationDispatch() {
  static const NormalizationDispatch dispatch = [] {
    const SimdLevel simd = DetectSimdLevel();
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
    if (simd == SimdLevel::kAVX512) {
      return NormalizationDispatch{
          &ComputeNormalizationMeanSquareFloat32_AVX512, &ComputeNormalizationMomentsFloat32_AVX512,
          &ApplyNormalizationAffineFloat32_AVX512, &ApplyNormalizationScaleBiasFloat32_AVX512};
    }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
    if (simd >= SimdLevel::kAVX2 && CpuSupportsFma()) {
      return NormalizationDispatch{
          &ComputeNormalizationMeanSquareFloat32_AVX2, &ComputeNormalizationMomentsFloat32_AVX2,
          &ApplyNormalizationAffineFloat32_AVX2, &ApplyNormalizationScaleBiasFloat32_AVX2};
    }
#endif
    return NormalizationDispatch{&MeanSquareScalar, &MomentsScalar, &AffineScalar,
                                 &ScaleBiasScalar};
  }();
  return dispatch;
}

float MeanSquareFloat16Scalar(const std::uint16_t *input, std::size_t count) {
  float sums[4] = {};
  for (std::size_t index = 0; index < count; ++index) {
    const float value = detail::Float16BitsToFloat(input[index]);
    sums[index & 3] += value * value;
  }
  return (sums[0] + sums[1] + sums[2] + sums[3]) / static_cast<float>(count);
}

void AffineFloat16Scalar(const std::uint16_t *input, const std::uint16_t *scale,
                         std::uint16_t *output, std::size_t count, float multiplier) {
  for (std::size_t index = 0; index < count; ++index) {
    const float value = detail::Float16BitsToFloat(input[index]);
    const float weight = detail::Float16BitsToFloat(scale[index]);
    output[index] = detail::FloatToFloat16Bits(value * multiplier * weight);
  }
}

template <typename Stash> Stash MeanSquareFloat64Scalar(const double *input, std::size_t count) {
  Stash sums[4] = {};
  for (std::size_t index = 0; index < count; ++index) {
    const Stash value = static_cast<Stash>(input[index]);
    sums[index & 3] += value * value;
  }
  return (sums[0] + sums[1] + sums[2] + sums[3]) / static_cast<Stash>(count);
}

template <typename Stash>
void AffineFloat64Scalar(const double *input, const double *scale, double *output,
                         std::size_t count, Stash multiplier) {
  for (std::size_t index = 0; index < count; ++index) {
    const Stash value = static_cast<Stash>(input[index]);
    const Stash weight = static_cast<Stash>(scale[index]);
    output[index] = static_cast<double>(value * multiplier * weight);
  }
}

struct Float16Dispatch {
  float (*mean_square)(const std::uint16_t *, std::size_t);
  void (*affine)(const std::uint16_t *, const std::uint16_t *, std::uint16_t *, std::size_t, float);
  const char *path;
};

const Float16Dispatch &GetFloat16Dispatch() {
  static const Float16Dispatch dispatch = [] {
#ifdef ONNX_LIGHT_CPU_HAVE_RMS_F16C
    if (DetectSimdLevel() >= SimdLevel::kAVX && CpuSupportsF16C()) {
      return Float16Dispatch{&ComputeNormalizationMeanSquareFloat16_F16C,
                             &ApplyNormalizationAffineFloat16_F16C, "f16c"};
    }
#endif
    return Float16Dispatch{&MeanSquareFloat16Scalar, &AffineFloat16Scalar, "scalar"};
  }();
  return dispatch;
}

struct Float64Dispatch {
  double (*mean_square)(const double *, std::size_t);
  float (*mean_square_float)(const double *, std::size_t);
  void (*affine)(const double *, const double *, double *, std::size_t, double);
  void (*affine_float)(const double *, const double *, double *, std::size_t, float);
  const char *path;
};

const Float64Dispatch &GetFloat64Dispatch() {
  static const Float64Dispatch dispatch = [] {
#ifdef ONNX_LIGHT_CPU_HAVE_AVX
    if (DetectSimdLevel() >= SimdLevel::kAVX) {
      return Float64Dispatch{&ComputeNormalizationMeanSquareFloat64_AVX,
                             &ComputeNormalizationMeanSquareFloat64StashFloat32_AVX,
                             &ApplyNormalizationAffineFloat64_AVX,
                             &ApplyNormalizationAffineFloat64StashFloat32_AVX, "avx"};
    }
#endif
    return Float64Dispatch{&MeanSquareFloat64Scalar<double>, &MeanSquareFloat64Scalar<float>,
                           &AffineFloat64Scalar<double>, &AffineFloat64Scalar<float>, "scalar"};
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

void ApplyNormalizationScaleBiasFloat32(const float *input, float *output, std::size_t count,
                                        float multiplier, float offset) {
  GetNormalizationDispatch().scale_bias(input, output, count, multiplier, offset);
}

float ComputeNormalizationMeanSquareFloat16(const std::uint16_t *input, std::size_t count) {
  if (count == 0) {
    throw std::invalid_argument("normalization reduction size must be positive.");
  }
  return GetFloat16Dispatch().mean_square(input, count);
}

void ApplyNormalizationAffineFloat16(const std::uint16_t *input, const std::uint16_t *scale,
                                     std::uint16_t *output, std::size_t count, float multiplier) {
  GetFloat16Dispatch().affine(input, scale, output, count, multiplier);
}

double ComputeNormalizationMeanSquareFloat64(const double *input, std::size_t count) {
  if (count == 0) {
    throw std::invalid_argument("normalization reduction size must be positive.");
  }
  return GetFloat64Dispatch().mean_square(input, count);
}

float ComputeNormalizationMeanSquareFloat64StashFloat32(const double *input, std::size_t count) {
  if (count == 0) {
    throw std::invalid_argument("normalization reduction size must be positive.");
  }
  return GetFloat64Dispatch().mean_square_float(input, count);
}

void ApplyNormalizationAffineFloat64(const double *input, const double *scale, double *output,
                                     std::size_t count, double multiplier) {
  GetFloat64Dispatch().affine(input, scale, output, count, multiplier);
}

void ApplyNormalizationAffineFloat64StashFloat32(const double *input, const double *scale,
                                                 double *output, std::size_t count,
                                                 float multiplier) {
  GetFloat64Dispatch().affine_float(input, scale, output, count, multiplier);
}

const char *NormalizationFloat16Path() { return GetFloat16Dispatch().path; }

const char *NormalizationFloat64Path() { return GetFloat64Dispatch().path; }

} // namespace onnx_light_cpu
