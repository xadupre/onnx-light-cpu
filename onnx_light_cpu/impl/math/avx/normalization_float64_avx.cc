// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/normalization_kernel.h"

#include <immintrin.h>

namespace onnx_light_cpu {
namespace {

template <typename Stash> struct Vector;

template <> struct Vector<double> {
  using Type = __m256d;
  static Type Zero() { return _mm256_setzero_pd(); }
  static Type Broadcast(double value) { return _mm256_set1_pd(value); }
  static Type Load(const double *input) { return _mm256_loadu_pd(input); }
  static void Store(double *output, Type value) { _mm256_storeu_pd(output, value); }
  static Type Add(Type a, Type b) { return _mm256_add_pd(a, b); }
  static Type Multiply(Type a, Type b) { return _mm256_mul_pd(a, b); }
  static double Sum(Type value) {
    const __m128d halves =
        _mm_add_pd(_mm256_castpd256_pd128(value), _mm256_extractf128_pd(value, 1));
    return _mm_cvtsd_f64(_mm_add_sd(halves, _mm_unpackhi_pd(halves, halves)));
  }
};

template <> struct Vector<float> {
  using Type = __m128;
  static Type Zero() { return _mm_setzero_ps(); }
  static Type Broadcast(float value) { return _mm_set1_ps(value); }
  static Type Load(const double *input) { return _mm256_cvtpd_ps(_mm256_loadu_pd(input)); }
  static void Store(double *output, Type value) {
    _mm256_storeu_pd(output, _mm256_cvtps_pd(value));
  }
  static Type Add(Type a, Type b) { return _mm_add_ps(a, b); }
  static Type Multiply(Type a, Type b) { return _mm_mul_ps(a, b); }
  static float Sum(Type value) {
    const __m128 halves = _mm_add_ps(value, _mm_movehl_ps(value, value));
    return _mm_cvtss_f32(_mm_add_ss(halves, _mm_shuffle_ps(halves, halves, 1)));
  }
};

template <typename Stash> Stash MeanSquare(const double *input, std::size_t count) {
  using V = Vector<Stash>;
  typename V::Type sums[4] = {V::Zero(), V::Zero(), V::Zero(), V::Zero()};
  std::size_t index = 0;
  for (; index + 16 <= count; index += 16) {
    for (std::size_t lane = 0; lane < 4; ++lane) {
      const auto value = V::Load(input + index + lane * 4);
      sums[lane] = V::Add(sums[lane], V::Multiply(value, value));
    }
  }
  for (; index + 4 <= count; index += 4) {
    const auto value = V::Load(input + index);
    sums[0] = V::Add(sums[0], V::Multiply(value, value));
  }
  Stash sum = V::Sum(V::Add(V::Add(sums[0], sums[1]), V::Add(sums[2], sums[3])));
  for (; index < count; ++index) {
    const Stash value = static_cast<Stash>(input[index]);
    sum += value * value;
  }
  return sum / static_cast<Stash>(count);
}

template <typename Stash>
void Affine(const double *input, const double *scale, double *output, std::size_t count,
            Stash multiplier) {
  using V = Vector<Stash>;
  const auto inverse = V::Broadcast(multiplier);
  std::size_t index = 0;
  for (; index + 4 <= count; index += 4) {
    const auto value = V::Load(input + index);
    const auto weight = V::Load(scale + index);
    V::Store(output + index, V::Multiply(V::Multiply(value, inverse), weight));
  }
  for (; index < count; ++index) {
    const Stash value = static_cast<Stash>(input[index]);
    const Stash weight = static_cast<Stash>(scale[index]);
    output[index] = static_cast<double>(value * multiplier * weight);
  }
}

} // namespace

double ComputeNormalizationMeanSquareFloat64_AVX(const double *input, std::size_t count) {
  return MeanSquare<double>(input, count);
}

float ComputeNormalizationMeanSquareFloat64StashFloat32_AVX(const double *input,
                                                            std::size_t count) {
  return MeanSquare<float>(input, count);
}

void ApplyNormalizationAffineFloat64_AVX(const double *input, const double *scale, double *output,
                                         std::size_t count, double multiplier) {
  Affine(input, scale, output, count, multiplier);
}

void ApplyNormalizationAffineFloat64StashFloat32_AVX(const double *input, const double *scale,
                                                     double *output, std::size_t count,
                                                     float multiplier) {
  Affine(input, scale, output, count, multiplier);
}

} // namespace onnx_light_cpu
