// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace onnx_light_cpu {

struct AttentionSoftmaxBlockResult {
  float maximum;
  float correction;
};

bool AttentionApplyAdditiveMaskFloat32_AVX512(float *scores, const float *mask, std::size_t count);

bool AttentionApplyBooleanMaskFloat32_AVX512(float *scores, const std::uint8_t *mask,
                                             std::size_t count);

AttentionSoftmaxBlockResult AttentionSoftmaxBlockFloat32_AVX512(float *scores, std::size_t count,
                                                                float previous_maximum,
                                                                float &denominator);

} // namespace onnx_light_cpu
