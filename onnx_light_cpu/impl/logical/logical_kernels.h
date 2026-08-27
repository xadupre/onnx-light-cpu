// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

// ``NotBool`` dispatches on the runtime SIMD level shared across all kernel
// families, so pull in the neutral ``SimdLevel`` enum and ``DetectSimdLevel``
// detection entry point rather than the math kernel declarations.
#include "onnx_light_cpu/impl/simd_level.h"

namespace onnx_light_cpu {

/// Computes the elementwise logical negation: out[i] = (input[i] == 0) for
/// ``bool``. ONNX ``bool`` tensors are stored as one byte per element, so the
struct NotExecutionTuning {
  // Zero disables executor dispatch.
  std::size_t parallel_threshold_bytes = 2 * 1024 * 1024;
  std::size_t target_block_bytes = 256 * 1024;
  // Zero uses every participant made available by the session executor.
  std::size_t max_participants = 32;
  bool use_cost_model = true;
  // Zero leaves the participant count to the executor cost model.
  std::size_t preferred_participants = 0;
};

/// input and output are the raw byte patterns (as ``uint8_t``): every zero byte
/// maps to ``1`` and every non-zero byte maps to ``0``, matching
/// ``numpy.logical_not``. Dispatches to the best available SIMD path at
/// runtime.
void NotBool(const uint8_t *input, uint8_t *output, std::size_t count);
void NotBoolWithTuning(const uint8_t *input, uint8_t *output, std::size_t count,
                       const NotExecutionTuning &tuning);

#ifdef ONNX_LIGHT_CPU_HAVE_AVX512BW
void NotBool_AVX512(const uint8_t *input, uint8_t *output, std::size_t count);
#endif

} // namespace onnx_light_cpu
