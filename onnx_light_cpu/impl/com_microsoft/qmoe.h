// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu/impl/com_microsoft/moe.h"

namespace onnx_light_cpu {

struct QMoEAttributes : MoEAttributes {
  std::size_t expert_weight_bits = 4;
  std::size_t block_size = 0;
};

// Quantization packs adjacent elements of the last stored axis, low nibble first.
// Producer tensor-level scales are per last-axis column; block scales are per
// stored row. CPU row-wise tensors instead use rowwise_scales and unsigned_int8.
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
