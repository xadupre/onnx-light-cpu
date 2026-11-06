// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace ml_cpu_compute {

/// Exact (erf, not tanh) Gelu, with no scratch allocation.
void GeluFloat32(const float *input, float *output, std::size_t count);
void GeluFloat64(const double *input, double *output, std::size_t count);
void GeluFloat16(const std::uint16_t *input, std::uint16_t *output, std::size_t count);
void GeluBFloat16(const std::uint16_t *input, std::uint16_t *output, std::size_t count);

} // namespace ml_cpu_compute
