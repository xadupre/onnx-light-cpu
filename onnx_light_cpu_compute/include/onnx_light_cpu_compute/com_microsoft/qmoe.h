// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu_compute/com_microsoft/moe.h"

namespace onnx_light_cpu {

struct QMoEAttributes : MoEAttributes {
  std::size_t expert_weight_bits = 4;
  std::size_t block_size = 0;
};

// Explicit raw layout: adjacent last-axis values pack low nibble first.
// CPU tensor quantization uses output_major, rowwise_scales and unsigned_int8.
// Producer block quantization uses row/block scales and signed INT8 instead.
// TensorRT-preprocessed buffers are not a raw layout.
struct QMoEWeight {
  const std::uint8_t *data = nullptr;
  const void *scales = nullptr;
  MoEDataType scale_type = MoEDataType::kFloat32;
  bool output_major = false;
  bool rowwise_scales = false;
  bool unsigned_int8 = false;
};

struct QMoEScratchSize {
  std::size_t value_count = 0;
  MoEScratchSize moe;
};

QMoEScratchSize ComputeQMoEScratchSize(std::size_t tokens, std::size_t hidden, std::size_t experts,
                                       std::size_t intermediate, const QMoEAttributes &attributes,
                                       bool has_fc3 = false);

void DequantizeQMoEWeights(QMoEWeight weight, float *output, std::size_t experts,
                           std::size_t inputs, std::size_t outputs,
                           const QMoEAttributes &attributes);

void ComputeQMoE(const void *input, const void *router, QMoEWeight fc1, const void *bias1,
                 QMoEWeight fc2, const void *bias2, const QMoEWeight *fc3, const void *bias3,
                 void *output, std::size_t tokens, std::size_t hidden, std::size_t experts,
                 std::size_t intermediate, MoEDataType data_type, const QMoEAttributes &attributes,
                 float *values, std::size_t value_count, MoEScratch scratch);

} // namespace onnx_light_cpu
