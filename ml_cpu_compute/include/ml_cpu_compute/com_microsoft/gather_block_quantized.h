// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ml_cpu_compute {

// The dimensions between the gather axis (first) and quantization axis (last)
// are flattened into rows_per_gather. Packed data and zero points restart at
// each last-dimension row. Buffers must contain the implied element counts.
struct GatherBlockQuantizedParameters {
  int64_t rows = 0;
  int64_t rows_per_gather = 1;
  int64_t packed_columns = 0;
  int64_t index_count = 0;
  int64_t bits = 4;
  int64_t block_size = 128;
};

// Scales and output have the same type; uint16_t denotes IEEE FLOAT16 bits.
// Out-of-range indices produce zeros. Output must not overlap any input.
#define ML_CPU_DECLARE_GATHER_BLOCK_QUANTIZED(Scale, Index)                                        \
  void GatherBlockQuantized(const uint8_t *data, const Index *indices, const Scale *scales,        \
                            const uint8_t *zero_points, Scale *output,                             \
                            const GatherBlockQuantizedParameters &parameters)

ML_CPU_DECLARE_GATHER_BLOCK_QUANTIZED(float, int32_t);
ML_CPU_DECLARE_GATHER_BLOCK_QUANTIZED(float, int64_t);
ML_CPU_DECLARE_GATHER_BLOCK_QUANTIZED(uint16_t, int32_t);
ML_CPU_DECLARE_GATHER_BLOCK_QUANTIZED(uint16_t, int64_t);

#undef ML_CPU_DECLARE_GATHER_BLOCK_QUANTIZED

} // namespace ml_cpu_compute
