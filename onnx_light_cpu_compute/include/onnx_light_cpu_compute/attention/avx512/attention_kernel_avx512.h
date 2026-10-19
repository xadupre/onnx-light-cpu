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

void AttentionScaleFloat32_AVX512(float *values, float factor, std::size_t count);

void AttentionNormalizeRowsFloat32_AVX512(float *values, const float *denominators,
                                          std::size_t rows, std::size_t columns,
                                          std::ptrdiff_t stride);

void AttentionSoftmaxRowsFloat32_AVX512(float *scores, float *denominators, std::size_t rows,
                                        std::size_t columns);

void AttentionPackRowsFloat32_AVX512(const float *source, float *destination, std::size_t rows,
                                     std::size_t dimension, std::ptrdiff_t stride);

void AttentionScatterRowsFloat32_AVX512(const float *source, float *destination, std::size_t rows,
                                        std::size_t dimension, std::ptrdiff_t stride);

void AttentionScoreQ8K128D64Float32_AVX512(const float *q, const float *k, float scale,
                                           float *scores);

void AttentionProbabilityValueQ8K128D64Float32_AVX512(const float *probabilities, const float *v,
                                                      float *output, std::ptrdiff_t output_stride);

} // namespace onnx_light_cpu
