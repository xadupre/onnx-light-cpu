// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace onnx_light_cpu {

void RmsNormalizationFloat16(const std::uint16_t *input, const std::uint16_t *scale,
                             std::uint16_t *output, std::size_t rows, std::size_t width,
                             float epsilon);

#ifdef ONNX_LIGHT_CPU_HAVE_RMS_F16C
void RmsNormalizationFloat16_F16C(const std::uint16_t *input, const std::uint16_t *scale,
                                  std::uint16_t *output, std::size_t row_begin, std::size_t row_end,
                                  std::size_t width, float epsilon);
#endif

} // namespace onnx_light_cpu
