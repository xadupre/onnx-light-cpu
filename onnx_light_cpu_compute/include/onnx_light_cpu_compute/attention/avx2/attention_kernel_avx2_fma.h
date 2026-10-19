// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace onnx_light_cpu {

/// AVX2+FMA counterpart of AttentionSoftmaxBlockResult (the AVX-512
/// primitive lives in impl/attention/avx512); kept as a separate type so
/// this header has no dependency on the AVX-512 one.
struct AttentionSoftmaxBlockResultAVX2 {
  float maximum;
  float correction;
};

/// Computes scaled Q . K for a block of strided key rows without leaving the
/// AVX2 translation unit between dot products.
void AttentionScoreBlockFloat32_AVX2_FMA(const float *q, const float *k, std::size_t rows,
                                         std::ptrdiff_t stride, std::size_t head_dim, float scale,
                                         float *scores);

/// Adds a block of weighted value rows, skipping zero weights (including
/// masked values that may contain NaNs), in the original key order.
void AttentionAccumulateBlockFloat32_AVX2_FMA(float *accumulator, const float *weights,
                                              const float *v, std::size_t rows,
                                              std::ptrdiff_t stride, std::size_t head_dim);

/// Computes values[d] *= factor for d in [0, count), vectorized
/// with AVX2. Used to rescale the running P @ V accumulator by the
/// online-softmax correction factor when a new block maximum is found.
void AttentionScaleFloat32_AVX2_FMA(float *values, float factor, std::size_t count);

/// AVX2+FMA counterpart of AttentionApplyAdditiveMaskFloat32_AVX512: adds
/// mask[i] to scores[i] for i in [0, count) and returns
/// whether at least one position remains unmasked, i.e. its resulting bias is
/// not -infinity. Finite biases, including -FLT_MAX, remain unmasked.
bool AttentionApplyAdditiveMaskFloat32_AVX2_FMA(float *scores, const float *mask,
                                                std::size_t count);

/// AVX2+FMA counterpart of AttentionApplyBooleanMaskFloat32_AVX512:
/// replaces every disallowed (mask[i] == 0) score with -infinity and
/// returns whether at least one position is allowed.
bool AttentionApplyBooleanMaskFloat32_AVX2_FMA(float *scores, const std::uint8_t *mask,
                                               std::size_t count);

/// AVX2+FMA counterpart of AttentionSoftmaxBlockFloat32_AVX512: given one
/// score block (already biased by any mask), computes the new running
/// online-softmax maximum and rescale correction against
/// previous_maximum, subtracts the new maximum from every score,
/// exponentiates in place (reusing ExpFloat32_AVX2_FMA), and folds the
/// block's exponential sum into denominator (denominator = denominator
/// * correction + block_sum). The caller is responsible for rescaling its
/// own P @ V accumulator by the returned correction and for using the
/// now-exponentiated (unnormalized probability) scores to accumulate
/// P @ V.
AttentionSoftmaxBlockResultAVX2 AttentionSoftmaxBlockFloat32_AVX2_FMA(float *scores,
                                                                      std::size_t count,
                                                                      float previous_maximum,
                                                                      float &denominator);

} // namespace onnx_light_cpu
