// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/traditionalml/tree_ensemble.h"

#include <immintrin.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace onnx_light_cpu {

void EvaluateBalancedFloatRows_AVX512(const float *input, std::size_t features,
                                      const TreeEnsembleCompactFloatNode *nodes,
                                      const float *leaf_weights, const std::int64_t *tree_roots,
                                      std::size_t tree_count, std::size_t depth,
                                      std::size_t row_begin, std::size_t row_end, float base,
                                      float *output) {
  static_assert(offsetof(TreeEnsembleCompactFloatNode, split) == 0);
  static_assert(offsetof(TreeEnsembleCompactFloatNode, feature_id) == 4);
  constexpr std::size_t kLanes = 16;
  const auto *node_words = reinterpret_cast<const std::int32_t *>(nodes);
  const auto *node_splits = reinterpret_cast<const float *>(nodes);
  std::fill(output + row_begin, output + row_end, base);

  for (std::size_t tree = 0; tree < tree_count; ++tree) {
    const __m512i root = _mm512_set1_epi32(static_cast<std::int32_t>(tree_roots[tree]));
    std::size_t row = row_begin;
    for (; row + kLanes <= row_end; row += kLanes) {
      const __m512i input_indices =
          _mm512_setr_epi32(static_cast<std::int32_t>(row * features),
                            static_cast<std::int32_t>((row + 1) * features),
                            static_cast<std::int32_t>((row + 2) * features),
                            static_cast<std::int32_t>((row + 3) * features),
                            static_cast<std::int32_t>((row + 4) * features),
                            static_cast<std::int32_t>((row + 5) * features),
                            static_cast<std::int32_t>((row + 6) * features),
                            static_cast<std::int32_t>((row + 7) * features),
                            static_cast<std::int32_t>((row + 8) * features),
                            static_cast<std::int32_t>((row + 9) * features),
                            static_cast<std::int32_t>((row + 10) * features),
                            static_cast<std::int32_t>((row + 11) * features),
                            static_cast<std::int32_t>((row + 12) * features),
                            static_cast<std::int32_t>((row + 13) * features),
                            static_cast<std::int32_t>((row + 14) * features),
                            static_cast<std::int32_t>((row + 15) * features));
      __m512i node_indices = root;
      for (std::size_t level = 0; level < depth; ++level) {
        const __m512i word_indices = _mm512_slli_epi32(node_indices, 2);
        const __m512i feature_ids =
            _mm512_i32gather_epi32(word_indices, node_words + 1, sizeof(std::int32_t));
        const __m512 splits = _mm512_i32gather_ps(word_indices, node_splits, sizeof(std::int32_t));
        const __m512 values =
            _mm512_i32gather_ps(_mm512_add_epi32(input_indices, feature_ids), input, sizeof(float));
        const __mmask16 go_true = _mm512_cmp_ps_mask(values, splits, _CMP_LE_OQ);
        const __m512i child_words =
            _mm512_add_epi32(word_indices, _mm512_mask_blend_epi32(go_true, _mm512_set1_epi32(3),
                                                                   _mm512_set1_epi32(2)));
        node_indices = _mm512_i32gather_epi32(child_words, node_words, sizeof(std::int32_t));
      }
      const __m512 weights = _mm512_i32gather_ps(node_indices, leaf_weights, sizeof(float));
      _mm512_storeu_ps(output + row, _mm512_add_ps(_mm512_loadu_ps(output + row), weights));
    }
    for (; row < row_end; ++row) {
      std::size_t node = static_cast<std::size_t>(tree_roots[tree]);
      for (std::size_t level = 0; level < depth; ++level) {
        const TreeEnsembleCompactFloatNode &current = nodes[node];
        node = input[row * features + current.feature_id] <= current.split ? current.true_child
                                                                           : current.false_child;
      }
      output[row] += leaf_weights[node];
    }
  }
}

} // namespace onnx_light_cpu
