// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace onnx_light_cpu {

struct MoEAttributes {
  std::size_t k = 1;
  std::string activation = "relu";
  bool normalize = false;
  float alpha = 1.0f;
  float beta = 0.0f;
  float limit = std::numeric_limits<float>::infinity();
};

void ComputeMoE(const void *input, const void *router, const void *fc1, const void *bias1,
                const void *fc2, const void *bias2, void *output, std::size_t tokens,
                std::size_t hidden, std::size_t experts, std::size_t intermediate, bool float16,
                const MoEAttributes &attributes);

} // namespace onnx_light_cpu
