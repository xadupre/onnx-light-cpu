// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

namespace onnx_light_cpu {

struct AttentionSoftmaxBlockResult {
  float maximum;
  float correction;
};

AttentionSoftmaxBlockResult AttentionSoftmaxBlockFloat32_AVX512(float *scores, std::size_t count,
                                                                float previous_maximum,
                                                                float &denominator);

} // namespace onnx_light_cpu
