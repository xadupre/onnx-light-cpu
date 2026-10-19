// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace onnx_light_cpu {

/// Computes residual RMS normalization using the output buffer as the row workspace.
/// Requires positive width and skip_rows when rows is nonzero. Skip rows repeat per batch.
/// Writes optional residual, zero-mean, and inverse-RMS outputs when their pointers are non-null.
void SkipSimplifiedLayerNormalizationFloat32(const float *input, const float *skip,
                                             const float *gamma, const float *bias, float *output,
                                             float *input_skip_bias_sum, float *mean,
                                             float *inv_std_var, std::size_t rows,
                                             std::size_t width, std::size_t skip_rows,
                                             float epsilon);

bool TrySkipSimplifiedLayerNormalizationFloat16(
    const std::uint16_t *input, const std::uint16_t *skip, const std::uint16_t *gamma,
    const std::uint16_t *bias, std::uint16_t *output, std::uint16_t *input_skip_bias_sum,
    float *mean, float *inv_std_var, std::size_t rows, std::size_t width, std::size_t skip_rows,
    float epsilon);

#ifdef ONNX_LIGHT_CPU_HAVE_AVX_F16C
void SkipSimplifiedLayerNormalizationFloat16_F16C(
    const std::uint16_t *input, const std::uint16_t *skip, const std::uint16_t *gamma,
    const std::uint16_t *bias, std::uint16_t *output, std::uint16_t *input_skip_bias_sum,
    float *mean, float *inv_std_var, std::size_t row_begin, std::size_t row_end, std::size_t width,
    std::size_t skip_rows, float epsilon);
#endif

} // namespace onnx_light_cpu
