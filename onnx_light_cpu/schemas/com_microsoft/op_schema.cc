// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include <map>

namespace onnx_light_cpu {
namespace {

namespace schema_ns = ONNX_LIGHT_NAMESPACE::core::schema;
using schema_ns::AttributeParam;
using schema_ns::AttributeType;
using schema_ns::LightOpSchema;
using schema_ns::SchemaBuilder;
using schema_ns::TensorType;

LightOpSchema MakeCDistSchema() {
  return LightOpSchema(
      "CDist", kMicrosoftDomain, 1,
      "Computes the pairwise Euclidean or squared Euclidean distances between the rows of two "
      "rank-2 tensors.",
      {{"A", "A rank-2 tensor with shape (M, N).", "T"},
       {"B", "A rank-2 tensor with shape (K, N).", "T"}},
      {{"C", "The pairwise distances with shape (M, K).", "T"}},
      {{"T",
        {TensorType::kFloat, TensorType::kDouble},
        "Constrain inputs and output to float or double tensors."}},
      {AttributeParam{"metric", "Distance metric: sqeuclidean or euclidean.", AttributeType::STRING,
                      false, std::string("sqeuclidean")}},
      false, true);
}

LightOpSchema MakeBiasGeluSchema() {
  return LightOpSchema(
      "BiasGelu", kMicrosoftDomain, 1,
      "Adds a rank-1 bias to the last input dimension and applies the exact Gaussian Error "
      "Linear Unit.",
      {{"A", "The input tensor.", "T"},
       {"B", "The rank-1 bias whose length matches the last dimension of A.", "T"}},
      {{"C", "The output tensor, with the same shape and type as A.", "T"}},
      {{"T",
        {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble, TensorType::kBfloat16},
        "Constrain inputs and output to floating-point tensors."}},
      false, true);
}

LightOpSchema MakeGroupQueryAttentionSchema() {
  LightOpSchema schema(
      "GroupQueryAttention", kMicrosoftDomain, 1,
      "Grouped-query attention over rank-3 Q/K/V tensors, with an optional tensor KV cache and "
      "split-half (rotary_interleaved=0) rotary position embedding applied to Q/K at the "
      "position derived from seqlens_k (or an explicit position_ids).",
      {{"query", "Query tensor with shape (batch, sequence, num_heads * head_size).", "T"},
       {"key", "Key tensor with shape (batch, sequence, kv_num_heads * head_size).", "T"},
       {"value", "Value tensor with shape (batch, sequence, kv_num_heads * value_head_size).", "T"},
       {"past_key",
        "Optional KV cache key tensor with shape "
        "(batch, kv_num_heads, past_sequence_length, head_size).",
        "T_CACHE"},
       {"past_value",
        "Optional KV cache value tensor with shape "
        "(batch, kv_num_heads, past_sequence_length, value_head_size).",
        "T_CACHE"},
       {"seqlens_k", "INT32 tensor containing one total sequence length minus one per batch.", "M"},
       {"total_sequence_length", "INT32 scalar total sequence length.", "M"},
       {"cos_cache",
        "Optional rotary cosine cache with shape (max_sequence_length, head_size / 2); required "
        "when do_rotary is set.",
        "T"},
       {"sin_cache",
        "Optional rotary sine cache with shape (max_sequence_length, head_size / 2); required "
        "when do_rotary is set.",
        "T"},
       {"position_ids",
        "Optional INT64 tensor with shape (batch, sequence) giving the rotary position of the "
        "first token of each batch; only valid when do_rotary is set.",
        "I"},
       {"attention_bias",
        "Optional additive/boolean mask broadcastable to "
        "(batch, num_heads, sequence, total_sequence_length).",
        "T"}},
      {{"output", "Output tensor with the same shape and type as query.", "T"},
       {"present_key",
        "Optional present KV cache key tensor with shape "
        "(batch, kv_num_heads, past_sequence_length + sequence, head_size); the concatenation "
        "of past_key with the (rotated) current key.",
        "T_CACHE"},
       {"present_value",
        "Optional present KV cache value tensor with shape "
        "(batch, kv_num_heads, past_sequence_length + sequence, value_head_size); the "
        "concatenation of past_value with the current value.",
        "T_CACHE"}},
      {{"T",
        {TensorType::kFloat16, TensorType::kFloat, TensorType::kBfloat16},
        "Constrain query, key, value, output, and rotary cache tensors to floating-point "
        "tensors."},
       {"T_CACHE",
        {TensorType::kFloat16, TensorType::kFloat, TensorType::kBfloat16},
        "Constrain the supported non-quantized KV cache tensors."},
       {"M", {TensorType::kInt32}, "Constrain sequence-length inputs to INT32."},
       {"I", {TensorType::kInt64}, "Constrain position_ids to INT64."}},
      {AttributeParam{"num_heads", "Number of query heads.", AttributeType::INT, true},
       AttributeParam{"kv_num_heads", "Number of key/value heads.", AttributeType::INT, true},
       AttributeParam{"causal", "Whether to apply causal masking.", AttributeType::INT, false,
                      int64_t{1}},
       AttributeParam{"scale", "Optional attention-score scale.", AttributeType::FLOAT, false},
       AttributeParam{"softcap", "Optional attention-score softcap.", AttributeType::FLOAT, false,
                      0.0f},
       AttributeParam{"do_rotary", "Whether to apply rotary position embedding.",
                      AttributeType::INT, false, int64_t{0}},
       AttributeParam{"rotary_interleaved", "Rotary layout; only 0 (split-half) is supported.",
                      AttributeType::INT, false, int64_t{0}}},
      false, true);
  schema.set_min_output(1);
  schema.set_max_output(3);
  return schema;
}

LightOpSchema MakeLinearAttentionSchema() {
  LightOpSchema schema(
      "LinearAttention", kMicrosoftDomain, 1,
      "Recurrent linear attention with optional decay, delta updates, and persistent state.",
      {{"query", "Query tensor with shape (batch, sequence, q_num_heads * key_head_size).", "T"},
       {"key", "Key tensor with shape (batch, sequence, key_heads * key_head_size).", "T"},
       {"value", "Value tensor with shape (batch, sequence, kv_num_heads * value_head_size).", "T"},
       {"past_state",
        "Optional state tensor with shape "
        "(batch, kv_num_heads, key_head_size, value_head_size).",
        "T"},
       {"decay", "Optional log-space decay tensor used by gated update rules.", "T"},
       {"beta", "Optional update-rate tensor used by delta update rules.", "T"}},
      {{"output",
        "Output tensor with shape "
        "(batch, sequence, max(q_num_heads, kv_num_heads) * value_head_size).",
        "T"},
       {"present_state",
        "Updated state with shape "
        "(batch, kv_num_heads, key_head_size, value_head_size).",
        "T"}},
      {{"T", {TensorType::kFloat}, "Constrain the CPU implementation to FLOAT tensors."}},
      {AttributeParam{"q_num_heads", "Number of query heads.", AttributeType::INT, true},
       AttributeParam{"kv_num_heads", "Number of value/state heads.", AttributeType::INT, true},
       AttributeParam{"update_rule", "linear, gated, delta, or gated_delta.", AttributeType::STRING,
                      false, std::string("gated_delta")},
       AttributeParam{"scale", "Query readout scale; zero selects 1/sqrt(key_head_size).",
                      AttributeType::FLOAT, false, 0.0f},
       AttributeParam{"chunk_size", "CPU scheduling hint with no semantic effect.",
                      AttributeType::INT, false, int64_t{64}},
       AttributeParam{"state_window", "State window; the CPU implementation supports only zero.",
                      AttributeType::INT, false, int64_t{0}}},
      false, true);
  schema.set_min_output(2);
  schema.set_max_output(2);
  return schema;
}

} // namespace

std::vector<LightOpSchema> GetMicrosoftOpSchemasWithHistory(const std::string &op_type,
                                                            bool init_doc) {
  static const std::map<std::string, SchemaBuilder> builders = {
      {"BiasGelu", [] { return std::vector<LightOpSchema>{MakeBiasGeluSchema()}; }},
      {"CDist", [] { return std::vector<LightOpSchema>{MakeCDistSchema()}; }},
      {"GroupQueryAttention",
       [] { return std::vector<LightOpSchema>{MakeGroupQueryAttentionSchema()}; }},
      {"LinearAttention", [] { return std::vector<LightOpSchema>{MakeLinearAttentionSchema()}; }},
  };
  return schema_ns::CollectSchemasFromBuilders(builders, op_type, init_doc);
}

} // namespace onnx_light_cpu
