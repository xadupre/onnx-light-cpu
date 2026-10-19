// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace ml_cpu_compute::detail {

void MoEMatVecFloat32(const float *input, const float *weights, float *output, std::size_t rows,
                      std::size_t columns);

void MoEMatVecFloat16(const std::uint16_t *input, const std::uint16_t *weights, float *output,
                      std::size_t rows, std::size_t columns);

void MoEMatVecBFloat16(const std::uint16_t *input, const std::uint16_t *weights, float *output,
                       std::size_t rows, std::size_t columns);

void MoEMatVecFloat32Float16(const float *input, const std::uint16_t *weights, float *output,
                             std::size_t rows, std::size_t columns);

void MoEMatVecFloat32BFloat16(const float *input, const std::uint16_t *weights, float *output,
                              std::size_t rows, std::size_t columns);

} // namespace ml_cpu_compute::detail
