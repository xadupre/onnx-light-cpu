// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

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

} // namespace onnx_light_cpu
