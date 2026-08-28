// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/rms_normalization.h"

#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/impl/simd_level.h"

#include <cmath>

namespace onnx_light_cpu {

void RmsNormalizationFloat16(const std::uint16_t *input, const std::uint16_t *scale,
                             std::uint16_t *output, std::size_t rows, std::size_t width,
                             float epsilon) {
#ifdef ONNX_LIGHT_CPU_HAVE_RMS_F16C
  if (CpuSupportsF16C()) {
    ExecuteRanges(static_cast<std::int64_t>(rows), static_cast<double>(width),
                  [=](std::int64_t begin, std::int64_t end) {
                    RmsNormalizationFloat16_F16C(input, scale, output,
                                                 static_cast<std::size_t>(begin),
                                                 static_cast<std::size_t>(end), width, epsilon);
                  });
    return;
  }
#endif
  ExecuteRanges(static_cast<std::int64_t>(rows), static_cast<double>(width),
                [=](std::int64_t begin, std::int64_t end) {
                  for (std::size_t row = static_cast<std::size_t>(begin);
                       row < static_cast<std::size_t>(end); ++row) {
                    const std::size_t offset = row * width;
                    float sum_squares = 0.0F;
                    for (std::size_t column = 0; column < width; ++column) {
                      const float value = detail::Float16BitsToFloat(input[offset + column]);
                      sum_squares += value * value;
                    }
                    const float inverse_rms =
                        1.0F / std::sqrt(sum_squares / static_cast<float>(width) + epsilon);
                    for (std::size_t column = 0; column < width; ++column) {
                      const float value = detail::Float16BitsToFloat(input[offset + column]);
                      const float weight = detail::Float16BitsToFloat(scale[column]);
                      const float normalized = detail::Float16BitsToFloat(
                          detail::FloatToFloat16Bits(value * inverse_rms));
                      output[offset + column] = detail::FloatToFloat16Bits(normalized * weight);
                    }
                  }
                });
}

} // namespace onnx_light_cpu
