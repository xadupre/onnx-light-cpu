// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace ml_cpu_compute {

struct MoEAttributes {
  std::size_t k = 1;
  std::string activation = "relu";
  bool normalize = false;
  float alpha = 1.0f;
  float beta = 0.0f;
  float limit = std::numeric_limits<float>::infinity();
  bool topk_tie_break_low = false;
  bool router_probs_are_logits = true;
};

enum class MoEDataType {
  kFloat32,
  kFloat16,
  kBFloat16,
};

struct MoERoute {
  float weight = 0.0f;
  std::size_t expert = 0;
};

struct MoEScratchSize {
  std::size_t route_count = 0;
  std::size_t value_count = 0;
  std::size_t token_blocks = 0;
};

struct MoEScratch {
  MoERoute *routes = nullptr;
  std::size_t route_count = 0;
  float *values = nullptr;
  std::size_t value_count = 0;
  std::size_t token_blocks = 0;
};

MoEScratchSize ComputeMoEScratchSize(std::size_t tokens, std::size_t hidden, std::size_t experts,
                                     std::size_t intermediate, MoEDataType data_type,
                                     const MoEAttributes &attributes);

std::size_t ComputeMoEMaxScratchBytes(std::size_t tokens, std::size_t hidden, std::size_t experts,
                                      std::size_t fc1_width, std::size_t intermediate);

void ComputeMoE(const void *input, const void *router, const void *fc1, const void *bias1,
                const void *fc2, const void *bias2, void *output, std::size_t tokens,
                std::size_t hidden, std::size_t experts, std::size_t intermediate,
                MoEDataType data_type, const MoEAttributes &attributes, MoEScratch scratch);

} // namespace ml_cpu_compute
