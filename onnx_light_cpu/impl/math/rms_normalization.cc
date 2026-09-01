// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/rms_normalization.h"

#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/impl/simd_level.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace onnx_light_cpu {
namespace {

std::int64_t ToRowCount(std::size_t rows) {
  if (rows > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument("onnx_light_cpu::RMSNormalization: row count exceeds int64_t.");
  }
  return static_cast<std::int64_t>(rows);
}

} // namespace

void RmsNormalizationFloat32(const float *input, const float *scale, float *output,
                             std::size_t rows, std::size_t width, float epsilon) {
  const SimdLevel simd = DetectSimdLevel();
  ExecuteRanges(
      ToRowCount(rows), static_cast<double>(width) * 0.625,
      [=](std::int64_t begin, std::int64_t end) {
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
        if (simd == SimdLevel::kAVX512) {
          RmsNormalizationFloat32_AVX512(input, scale, output, static_cast<std::size_t>(begin),
                                         static_cast<std::size_t>(end), width, epsilon);
          return;
        }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
        if (simd >= SimdLevel::kAVX2 && CpuSupportsFma()) {
          RmsNormalizationFloat32_AVX2(input, scale, output, static_cast<std::size_t>(begin),
                                       static_cast<std::size_t>(end), width, epsilon);
          return;
        }
#endif
        for (std::size_t row = static_cast<std::size_t>(begin); row < static_cast<std::size_t>(end);
             ++row) {
          const std::size_t offset = row * width;
          float sums[4] = {};
          std::size_t column = 0;
          for (; column + 4 <= width; column += 4) {
            for (std::size_t lane = 0; lane < 4; ++lane) {
              const float value = input[offset + column + lane];
              sums[lane] += value * value;
            }
          }
          for (; column < width; ++column) {
            const float value = input[offset + column];
            sums[column & 3] += value * value;
          }
          const float inverse_rms =
              1.0F / std::sqrt((sums[0] + sums[1] + sums[2] + sums[3]) / static_cast<float>(width) +
                               epsilon);
          for (column = 0; column < width; ++column) {
            output[offset + column] = input[offset + column] * inverse_rms * scale[column];
          }
        }
      });
}

void RmsNormalizationFloat16(const std::uint16_t *input, const std::uint16_t *scale,
                             std::uint16_t *output, std::size_t rows, std::size_t width,
                             float epsilon) {
#ifdef ONNX_LIGHT_CPU_HAVE_RMS_F16C
  if (CpuSupportsF16C()) {
    ExecuteRanges(
        ToRowCount(rows), static_cast<double>(width), [=](std::int64_t begin, std::int64_t end) {
          RmsNormalizationFloat16_F16C(input, scale, output, static_cast<std::size_t>(begin),
                                       static_cast<std::size_t>(end), width, epsilon);
        });
    return;
  }
#endif
  ExecuteRanges(
      ToRowCount(rows), static_cast<double>(width), [=](std::int64_t begin, std::int64_t end) {
        for (std::size_t row = static_cast<std::size_t>(begin); row < static_cast<std::size_t>(end);
             ++row) {
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
            const float normalized =
                detail::Float16BitsToFloat(detail::FloatToFloat16Bits(value * inverse_rms));
            output[offset + column] = detail::FloatToFloat16Bits(normalized * weight);
          }
        }
      });
}

void RmsNormalizationBFloat16(const std::uint16_t *input, const std::uint16_t *scale,
                              std::uint16_t *output, std::size_t rows, std::size_t width,
                              float epsilon) {
  const SimdLevel simd = DetectSimdLevel();
  ExecuteRanges(
      ToRowCount(rows), static_cast<double>(width) * 0.625,
      [=](std::int64_t begin, std::int64_t end) {
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
        if (simd >= SimdLevel::kAVX2 && CpuSupportsFma()) {
          RmsNormalizationBFloat16_AVX2(input, scale, output, static_cast<std::size_t>(begin),
                                        static_cast<std::size_t>(end), width, epsilon);
          return;
        }
#endif
        for (std::size_t row = static_cast<std::size_t>(begin); row < static_cast<std::size_t>(end);
             ++row) {
          const std::size_t offset = row * width;
          float sums[4] = {};
          std::size_t column = 0;
          for (; column + 4 <= width; column += 4) {
            for (std::size_t lane = 0; lane < 4; ++lane) {
              const float value = detail::Bfloat16BitsToFloat(input[offset + column + lane]);
              sums[lane] += value * value;
            }
          }
          for (; column < width; ++column) {
            const float value = detail::Bfloat16BitsToFloat(input[offset + column]);
            sums[column & 3] += value * value;
          }
          const float inverse_rms =
              1.0F / std::sqrt((sums[0] + sums[1] + sums[2] + sums[3]) / static_cast<float>(width) +
                               epsilon);
          for (column = 0; column < width; ++column) {
            const float value = detail::Bfloat16BitsToFloat(input[offset + column]);
            const float weight = detail::Bfloat16BitsToFloat(scale[column]);
            const float normalized =
                detail::Bfloat16BitsToFloat(detail::FloatToBFloat16Bits(value * inverse_rms));
            output[offset + column] = detail::FloatToBFloat16Bits(normalized * weight);
          }
        }
      });
}

} // namespace onnx_light_cpu
