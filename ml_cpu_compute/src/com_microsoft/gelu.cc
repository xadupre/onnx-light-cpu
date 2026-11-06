// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "ml_cpu_compute/com_microsoft/gelu.h"
#include "ml_cpu_compute/checked_arithmetic.h"
#include "ml_cpu_compute/math/half_conversion.h"
#include "ml_cpu_compute/math/unary_execution_tuning.h"

#include <cmath>

namespace ml_cpu_compute {
namespace {

constexpr UnaryExecutionTuning kGeluExecutionTuning{128 * 1024, 32 * 1024, 32, false};

template <typename T> T Gelu(T x) {
  return (T(0.5) * x) * (T(1) + std::erf(x * std::sqrt(T(0.5))));
}

template <typename T> void Compute(const T *input, T *output, std::size_t count) {
  CheckedByteSize(count, sizeof(T), "Gelu", "tensor byte size");
  ExecuteUnaryRanges<T>(count, kGeluExecutionTuning, [=](std::int64_t begin, std::int64_t end) {
    for (std::int64_t i = begin; i < end; ++i) {
      output[i] = Gelu(input[i]);
    }
  });
}

template <bool BFloat16>
void ComputeHalf(const std::uint16_t *input, std::uint16_t *output, std::size_t count) {
  CheckedByteSize(count, sizeof(std::uint16_t), "Gelu", "tensor byte size");
  ExecuteUnaryRanges<std::uint16_t>(
      count, kGeluExecutionTuning, [=](std::int64_t begin, std::int64_t end) {
        for (std::int64_t i = begin; i < end; ++i) {
          const float x = BFloat16 ? detail::Bfloat16BitsToFloat(input[i])
                                   : detail::Float16BitsToFloat(input[i]);
          const float y = Gelu(x);
          output[i] = BFloat16 ? detail::FloatToBFloat16Bits(y) : detail::FloatToFloat16Bits(y);
        }
      });
}

} // namespace

void GeluFloat32(const float *input, float *output, std::size_t count) {
  Compute(input, output, count);
}
void GeluFloat64(const double *input, double *output, std::size_t count) {
  Compute(input, output, count);
}
void GeluFloat16(const std::uint16_t *input, std::uint16_t *output, std::size_t count) {
  ComputeHalf<false>(input, output, count);
}
void GeluBFloat16(const std::uint16_t *input, std::uint16_t *output, std::size_t count) {
  ComputeHalf<true>(input, output, count);
}

} // namespace ml_cpu_compute
