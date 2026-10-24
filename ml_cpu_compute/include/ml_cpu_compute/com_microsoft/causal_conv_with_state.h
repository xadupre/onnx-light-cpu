// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace ml_cpu_compute {

// Three-tap depthwise convolution with unit dilation, no activation and length >= 2.
// The caller validates tensor sizes and batch * channels; bias and past may be null.
// Output and present must not alias the inputs. Present retains the input's half bits.
void CausalConvWidth3Float16(const std::uint16_t *input, const std::uint16_t *weight,
                             const std::uint16_t *bias, const std::uint16_t *past,
                             std::uint16_t *output, std::uint16_t *present, std::size_t batch,
                             std::size_t channels, std::size_t length, bool channels_last);

} // namespace ml_cpu_compute
