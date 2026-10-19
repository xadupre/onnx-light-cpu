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
  /// Number of query heads (``Hq``). May be smaller than, equal to, or a
  /// multiple of ``key_value_heads``, or ``key_value_heads`` may be a
  /// multiple of it instead (see ``key_value_heads`` below); the ratio
  /// between the larger and the smaller must be exact.
  std::size_t query_heads = 0;
  /// Number of key/value (state) heads (``Hkv``). ``state``, ``decay``, and
  /// ``beta`` are always indexed by this head count. Grouping between
  /// ``query_heads`` and ``key_value_heads`` may go in either direction:
  /// ``query_heads % key_value_heads == 0`` (standard grouped-query
  /// attention, one state head serves several query heads) or
  /// ``key_value_heads % query_heads == 0`` (inverse grouping, several state
  /// heads share one query head). The output head count is always
  /// ``max(query_heads, key_value_heads)``.
  std::size_t key_value_heads = 0;
  /// Number of heads actually stored in the ``key`` tensor (``Hk``). Zero
  /// (the default) means "same as ``key_value_heads``", preserving the
  /// original single-head-count behaviour used by every existing caller.
  /// When set and smaller than ``key_value_heads``, ``key_value_heads`` must
  /// be an exact multiple of it: consecutive groups of
  /// ``key_value_heads / key_heads`` state heads then share one physical key
  /// head, independently of the query/state grouping direction above.
  std::size_t key_heads = 0;
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
/// independent ``(batch, state head)`` pairs -- one per
/// ``parameters.key_value_heads`` -- the token recurrence within each pair
/// remains sequential. The caller must have already validated every head
/// count relationship documented on :cpp:struct:`LinearAttentionParameters`;
/// this function assumes (and does not re-check) that they divide evenly.
void LinearAttentionFloat32(const LinearAttentionParameters &parameters, const float *query,
                            const float *key, const float *value, const float *decay,
                            const float *beta, float *state, float *output);

} // namespace onnx_light_cpu
