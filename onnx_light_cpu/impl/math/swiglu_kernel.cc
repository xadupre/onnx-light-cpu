// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/math_kernels.h"

#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <algorithm>
#include <array>

namespace onnx_light_cpu {
namespace {

template <typename T, auto Exp>
void SwiGLUFloat(const T *gate, const T *value, T *output, std::size_t count, T alpha) {
  constexpr std::size_t kBlockSize = 256;
  std::array<T, kBlockSize> exponent;
  for (std::size_t offset = 0; offset < count; offset += kBlockSize) {
    const std::size_t block = std::min(kBlockSize, count - offset);
    for (std::size_t i = 0; i < block; ++i) {
      exponent[i] = -alpha * gate[offset + i];
    }
    Exp(exponent.data(), exponent.data(), block);
    for (std::size_t i = 0; i < block; ++i) {
      output[offset + i] = gate[offset + i] * value[offset + i] / (static_cast<T>(1) + exponent[i]);
    }
  }
}

template <auto Decode, auto Encode>
void SwiGLUHalf(const std::uint16_t *gate, const std::uint16_t *value, std::uint16_t *output,
                std::size_t count, float alpha) {
  constexpr std::size_t kBlockSize = 256;
  alignas(32) std::array<float, kBlockSize> gate_float;
  alignas(32) std::array<float, kBlockSize> value_float;
  alignas(32) std::array<float, kBlockSize> exponent;
  for (std::size_t offset = 0; offset < count; offset += kBlockSize) {
    const std::size_t block = std::min(kBlockSize, count - offset);
    Decode(gate + offset, gate_float.data(), block);
    Decode(value + offset, value_float.data(), block);
    for (std::size_t i = 0; i < block; ++i) {
      exponent[i] = -alpha * gate_float[i];
    }
    ExpFloat32(exponent.data(), exponent.data(), block);
    for (std::size_t i = 0; i < block; ++i) {
      exponent[i] = gate_float[i] * value_float[i] / (1.0f + exponent[i]);
    }
    Encode(exponent.data(), output + offset, block);
  }
}

} // namespace

void SwiGLUFloat32(const float *gate, const float *value, float *output, std::size_t count,
                   float alpha) {
  SwiGLUFloat<float, ExpFloat32>(gate, value, output, count, alpha);
}

void SwiGLUFloat64(const double *gate, const double *value, double *output, std::size_t count,
                   double alpha) {
  SwiGLUFloat<double, ExpFloat64>(gate, value, output, count, alpha);
}

void SwiGLUFloat16(const std::uint16_t *gate, const std::uint16_t *value, std::uint16_t *output,
                   std::size_t count, float alpha) {
  SwiGLUHalf<detail::ConvertFloat16ToFloat32, detail::ConvertFloat32ToFloat16>(gate, value, output,
                                                                               count, alpha);
}

void SwiGLUBFloat16(const std::uint16_t *gate, const std::uint16_t *value, std::uint16_t *output,
                    std::size_t count, float alpha) {
  SwiGLUHalf<detail::ConvertBFloat16ToFloat32, detail::ConvertFloat32ToBFloat16>(
      gate, value, output, count, alpha);
}

} // namespace onnx_light_cpu
