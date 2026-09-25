// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu/impl/data_type.h"

#include <cstddef>
#include <cstdint>

namespace onnx_light_cpu {

struct MatMulNBitsExecutionTuning {
  std::size_t parallel_threshold_outputs = 256;
  std::size_t target_block_outputs = 64;
  std::int64_t max_participants = 32;
};

inline constexpr MatMulNBitsExecutionTuning kDefaultMatMulNBitsExecutionTuning{};

// Scratch per active INT4 callback; packed weights are borrowed, never retained or copied.
inline constexpr std::size_t kMatMulNBitsInt4WorkspaceBytes = 6144;
const char *MatMulNBitsInt4Implementation();

bool MatMulNBitsAccuracy4Float32Available();
void MatMulNBitsAccuracy4Float32(const float *a, const std::int8_t *packed_weights,
                                 const float *block_scales, const float *bias, float *y,
                                 std::size_t rows, std::size_t k, std::size_t n,
                                 std::int64_t max_participants);

void MatMulNBits(const void *a, const std::uint8_t *b, const void *scales, const void *bias,
                 void *y, DataType data_type, std::size_t rows, std::size_t k, std::size_t n,
                 std::size_t bits, std::size_t block_size,
                 const MatMulNBitsExecutionTuning &tuning = kDefaultMatMulNBitsExecutionTuning);

void MatMulNBitsFloat32(
    const float *a, const std::uint8_t *b, const float *scales, const float *bias, float *y,
    std::size_t rows, std::size_t k, std::size_t n, std::size_t block_size,
    const MatMulNBitsExecutionTuning &tuning = kDefaultMatMulNBitsExecutionTuning);

} // namespace onnx_light_cpu
