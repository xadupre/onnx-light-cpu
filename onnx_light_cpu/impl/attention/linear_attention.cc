// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/attention/linear_attention.h"

#include "onnx_light_cpu/impl/execution.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace onnx_light_cpu {
namespace {

void ScaleState(float *state, std::size_t count, float scale) {
  for (std::size_t i = 0; i < count; ++i) {
    state[i] *= scale;
  }
}

void AddScaledVector(float *destination, const float *source, std::size_t count, float scale) {
  for (std::size_t i = 0; i < count; ++i) {
    destination[i] += scale * source[i];
  }
}

void Retrieve(const float *state, const float *key, std::size_t key_size, std::size_t value_size,
              float *retrieved) {
  std::fill(retrieved, retrieved + value_size, 0.0f);
  for (std::size_t i = 0; i < key_size; ++i) {
    AddScaledVector(retrieved, state + i * value_size, value_size, key[i]);
  }
}

void ReadOutput(const float *state, const float *query, std::size_t key_size,
                std::size_t value_size, float scale, float *output) {
  std::fill(output, output + value_size, 0.0f);
  for (std::size_t i = 0; i < key_size; ++i) {
    AddScaledVector(output, state + i * value_size, value_size, query[i]);
  }
  if (scale != 1.0f) {
    ScaleState(output, value_size, scale);
  }
}

void ProcessHead(const LinearAttentionParameters &p, std::size_t batch, std::size_t kv_head,
                 const float *query, const float *key, const float *value, const float *decay,
                 const float *beta, float *state, float *output, float *retrieved) {
  const std::size_t dk = p.key_head_size;
  const std::size_t dv = p.value_head_size;
  const std::size_t q_token_stride = p.query_heads * dk;
  const std::size_t k_token_stride = p.key_value_heads * dk;
  const std::size_t v_token_stride = p.key_value_heads * dv;
  const std::size_t output_token_stride = p.query_heads * dv;
  const std::size_t state_size = dk * dv;
  const std::size_t query_heads_per_group = p.query_heads / p.key_value_heads;
  const std::size_t query_head_begin = kv_head * query_heads_per_group;
  const std::size_t output_head_begin = query_head_begin;
  const std::size_t sequence_base = batch * p.sequence_length;
  float *head_state = state + (batch * p.key_value_heads + kv_head) * state_size;

  std::size_t decay_token_stride = 0;
  if (p.decay_layout == LinearAttentionDecayLayout::kPerKeyDimension) {
    decay_token_stride = p.key_value_heads * dk;
  } else if (p.decay_layout == LinearAttentionDecayLayout::kPerHead) {
    decay_token_stride = p.key_value_heads;
  }
  const std::size_t beta_token_stride =
      p.beta_layout == LinearAttentionBetaLayout::kPerHead ? p.key_value_heads : 1;

  for (std::size_t token = 0; token < p.sequence_length; ++token) {
    const std::size_t sequence_index = sequence_base + token;
    const float *token_key = key + sequence_index * k_token_stride + kv_head * dk;
    const float *token_value = value + sequence_index * v_token_stride + kv_head * dv;

    if (p.decay_layout == LinearAttentionDecayLayout::kPerHead) {
      const float gate = std::exp(decay[sequence_index * decay_token_stride + kv_head]);
      ScaleState(head_state, state_size, gate);
    } else if (p.decay_layout == LinearAttentionDecayLayout::kPerKeyDimension) {
      const float *token_decay = decay + sequence_index * decay_token_stride + kv_head * dk;
      for (std::size_t i = 0; i < dk; ++i) {
        ScaleState(head_state + i * dv, dv, std::exp(token_decay[i]));
      }
    }

    const bool use_beta = p.beta_layout != LinearAttentionBetaLayout::kNone;
    if (use_beta) {
      Retrieve(head_state, token_key, dk, dv, retrieved);
      const std::size_t beta_head =
          p.beta_layout == LinearAttentionBetaLayout::kPerHead ? kv_head : 0;
      const float update_rate = beta[sequence_index * beta_token_stride + beta_head];
      for (std::size_t j = 0; j < dv; ++j) {
        retrieved[j] = update_rate * (token_value[j] - retrieved[j]);
      }
    }

    const float *update = use_beta ? retrieved : token_value;
    for (std::size_t i = 0; i < dk; ++i) {
      AddScaledVector(head_state + i * dv, update, dv, token_key[i]);
    }

    const float *token_query = query + sequence_index * q_token_stride + query_head_begin * dk;
    float *token_output = output + sequence_index * output_token_stride + output_head_begin * dv;
    for (std::size_t q = 0; q < query_heads_per_group; ++q) {
      ReadOutput(head_state, token_query + q * dk, dk, dv, p.scale, token_output + q * dv);
    }
  }
}

} // namespace

void LinearAttentionFloat32(const LinearAttentionParameters &parameters, const float *query,
                            const float *key, const float *value, const float *decay,
                            const float *beta, float *state, float *output) {
  const std::int64_t tasks =
      static_cast<std::int64_t>(parameters.batch_size * parameters.key_value_heads);
  const double work_per_task =
      static_cast<double>(parameters.sequence_length) * parameters.key_head_size *
      parameters.value_head_size *
      (parameters.beta_layout == LinearAttentionBetaLayout::kNone ? 2.0 : 3.0);
  ExecuteRanges(tasks, work_per_task, [&](std::int64_t begin, std::int64_t end) {
    std::vector<float> retrieved(parameters.value_head_size);
    for (std::int64_t task = begin; task < end; ++task) {
      const std::size_t batch = static_cast<std::size_t>(task) / parameters.key_value_heads;
      const std::size_t head = static_cast<std::size_t>(task) % parameters.key_value_heads;
      ProcessHead(parameters, batch, head, query, key, value, decay, beta, state, output,
                  retrieved.data());
    }
  });
}

} // namespace onnx_light_cpu
