// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/traditionalml/tree_ensemble.h"

#include <immintrin.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace onnx_light_cpu {

void EvaluateBalancedFloatRows_AVX2(const float *input, std::size_t features,
                                    const TreeEnsembleCompactFloatNode *nodes,
                                    const float *leaf_weights, const std::int64_t *tree_roots,
                                    std::size_t tree_count, std::size_t depth,
                                    std::size_t row_begin, std::size_t row_end, float base,
                                    float *output) {
  static_assert(offsetof(TreeEnsembleCompactFloatNode, split) == 0);
  static_assert(offsetof(TreeEnsembleCompactFloatNode, feature_id) == 4);
  constexpr std::size_t kLanes = 8;
  const auto *node_words = reinterpret_cast<const std::int32_t *>(nodes);
  const auto *node_splits = reinterpret_cast<const float *>(nodes);
  std::fill(output + row_begin, output + row_end, base);

  for (std::size_t tree = 0; tree < tree_count; ++tree) {
    const __m256i root = _mm256_set1_epi32(static_cast<std::int32_t>(tree_roots[tree]));
    std::size_t row = row_begin;
    for (; row + kLanes <= row_end; row += kLanes) {
      const __m256i input_indices =
          _mm256_setr_epi32(static_cast<std::int32_t>(row * features),
                            static_cast<std::int32_t>((row + 1) * features),
                            static_cast<std::int32_t>((row + 2) * features),
                            static_cast<std::int32_t>((row + 3) * features),
                            static_cast<std::int32_t>((row + 4) * features),
                            static_cast<std::int32_t>((row + 5) * features),
                            static_cast<std::int32_t>((row + 6) * features),
                            static_cast<std::int32_t>((row + 7) * features));
      __m256i node_indices = root;
      for (std::size_t level = 0; level < depth; ++level) {
        const __m256i word_indices = _mm256_slli_epi32(node_indices, 2);
        const __m256i feature_ids =
            _mm256_i32gather_epi32(node_words + 1, word_indices, sizeof(std::int32_t));
        const __m256 splits = _mm256_i32gather_ps(node_splits, word_indices, sizeof(std::int32_t));
        const __m256 values =
            _mm256_i32gather_ps(input, _mm256_add_epi32(input_indices, feature_ids), sizeof(float));
        const __m256 go_true = _mm256_cmp_ps(values, splits, _CMP_LE_OQ);
        const __m256i child_words = _mm256_add_epi32(
            word_indices, _mm256_blendv_epi8(_mm256_set1_epi32(3), _mm256_set1_epi32(2),
                                             _mm256_castps_si256(go_true)));
        node_indices = _mm256_i32gather_epi32(node_words, child_words, sizeof(std::int32_t));
      }
      const __m256 weights = _mm256_i32gather_ps(leaf_weights, node_indices, sizeof(float));
      _mm256_storeu_ps(output + row, _mm256_add_ps(_mm256_loadu_ps(output + row), weights));
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
