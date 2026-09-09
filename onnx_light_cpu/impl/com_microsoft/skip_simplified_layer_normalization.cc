// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/com_microsoft/skip_simplified_layer_normalization.h"

#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/normalization_kernel.h"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace onnx_light_cpu {

void SkipSimplifiedLayerNormalizationFloat32(const float *input, const float *skip,
                                             const float *gamma, const float *bias, float *output,
                                             float *input_skip_bias_sum, float *mean,
                                             float *inv_std_var, std::size_t rows,
                                             std::size_t width, std::size_t skip_rows,
                                             float epsilon) {
  ExecuteRanges(
      static_cast<std::int64_t>(rows), static_cast<double>(width),
      [&](std::int64_t begin, std::int64_t end) {
        for (std::size_t row = static_cast<std::size_t>(begin); row < static_cast<std::size_t>(end);
             ++row) {
          const float *input_row = input + row * width;
          const float *skip_row = skip + (row % skip_rows) * width;
          float *output_row = output + row * width;
          for (std::size_t i = 0; i < width; ++i) {
            const float residual = input_row[i] + skip_row[i];
            output_row[i] = bias == nullptr ? residual : residual + bias[i];
          }
          if (input_skip_bias_sum != nullptr) {
            std::memcpy(input_skip_bias_sum + row * width, output_row, width * sizeof(float));
          }
          const float inverse =
              1.0F / std::sqrt(ComputeNormalizationMeanSquareFloat32(output_row, width) + epsilon);
          if (mean != nullptr) {
            mean[row] = 0.0F;
          }
          if (inv_std_var != nullptr) {
            inv_std_var[row] = inverse;
          }
          ApplyNormalizationAffineFloat32(output_row, gamma, nullptr, output_row, width, 0.0F,
                                          inverse);
        }
      });
}

} // namespace onnx_light_cpu
