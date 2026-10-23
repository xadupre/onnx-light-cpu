// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

namespace ml_cpu_compute {

// Rotate a vector prefix of one split-half head, returning the number of pairs
// written. Buffers need not be aligned; the caller handles the remaining pairs.
using RotaryEmbeddingSplitHalfFn = std::size_t (*)(const void *input, const void *cos,
                                                   const void *sin, void *output, std::size_t half);

// Returns nullptr when the requested type has no supported SIMD implementation.
RotaryEmbeddingSplitHalfFn SelectRotaryEmbeddingSplitHalf(bool float16);

} // namespace ml_cpu_compute
