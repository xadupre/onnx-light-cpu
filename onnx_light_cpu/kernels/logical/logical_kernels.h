// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

// ``NotBool`` dispatches on the runtime SIMD level shared across all kernel
// families, so pull in the neutral ``SimdLevel`` enum and ``DetectSimdLevel``
// detection entry point rather than the math kernel declarations.
#include "onnx_light_cpu/kernels/simd_level.h"

namespace onnx_light_cpu {

/// Computes the elementwise logical negation: out[i] = (input[i] == 0) for
/// ``bool``. ONNX ``bool`` tensors are stored as one byte per element, so the
/// input and output are the raw byte patterns (as ``uint8_t``): every zero byte
/// maps to ``1`` and every non-zero byte maps to ``0``, matching
/// ``numpy.logical_not``. Dispatches to the best available SIMD path at
/// runtime.
void NotBool(const uint8_t *input, uint8_t *output, std::size_t count);

} // namespace onnx_light_cpu
