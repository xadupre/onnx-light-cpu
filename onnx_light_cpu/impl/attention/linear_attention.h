// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace onnx_light_cpu {

enum class LinearAttentionRule {
  kLinear,
  kGated,
  kDelta,
  kGatedDelta,
};

enum class LinearAttentionDecayLayout {
  kNone,
  kPerHead,
  kPerKeyDimension,
};

enum class LinearAttentionBetaLayout {
  kNone,
  kShared,
  kPerHead,
};

struct LinearAttentionParameters {
  std::size_t batch_size = 0;
  std::size_t sequence_length = 0;
  std::size_t query_heads = 0;
  std::size_t key_value_heads = 0;
  std::size_t key_head_size = 0;
  std::size_t value_head_size = 0;
  LinearAttentionRule rule = LinearAttentionRule::kGatedDelta;
  LinearAttentionDecayLayout decay_layout = LinearAttentionDecayLayout::kNone;
  LinearAttentionBetaLayout beta_layout = LinearAttentionBetaLayout::kNone;
  float scale = 1.0f;
};

/// Computes the recurrent LinearAttention update in float32.
///
/// ``state`` is seeded by the caller and updated in place. Work is split over
/// independent ``(batch, key-value head)`` pairs; the token recurrence within
/// each pair remains sequential.
void LinearAttentionFloat32(const LinearAttentionParameters &parameters, const float *query,
                            const float *key, const float *value, const float *decay,
                            const float *beta, float *state, float *output);

} // namespace onnx_light_cpu
