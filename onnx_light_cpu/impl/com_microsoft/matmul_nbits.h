// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace onnx_light_cpu {

struct MatMulNBitsExecutionTuning {
  std::size_t parallel_threshold_outputs = 256;
  std::size_t target_block_outputs = 64;
  std::int64_t max_participants = 32;
};

inline constexpr MatMulNBitsExecutionTuning kDefaultMatMulNBitsExecutionTuning{};

void MatMulNBitsFloat32(
    const float *a, const std::uint8_t *b, const float *scales, const float *bias, float *y,
    std::size_t rows, std::size_t k, std::size_t n, std::size_t block_size,
    const MatMulNBitsExecutionTuning &tuning = kDefaultMatMulNBitsExecutionTuning);

} // namespace onnx_light_cpu
