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
  return LightOpSchema(
      "GroupQueryAttention", kMicrosoftDomain, 1,
      "Grouped-query attention over rank-3 Q/K/V tensors.",
      {{"query", "Query tensor with shape (batch, sequence, num_heads * head_size).", "T"},
       {"key", "Key tensor with shape (batch, sequence, kv_num_heads * head_size).", "T"},
       {"value", "Value tensor with shape (batch, sequence, kv_num_heads * value_head_size).", "T"},
       {"past_key", "Optional KV cache key tensor.", "T_CACHE"},
       {"past_value", "Optional KV cache value tensor.", "T_CACHE"},
       {"seqlens_k", "INT32 tensor containing one total sequence length minus one per batch.", "M"},
       {"total_sequence_length", "INT32 scalar total sequence length.", "M"}},
      {{"output", "Output tensor with the same shape and type as query.", "T"}},
      {{"T",
        {TensorType::kFloat16, TensorType::kFloat, TensorType::kBfloat16},
        "Constrain query, key, value, and output to floating-point tensors."},
       {"T_CACHE",
        {TensorType::kFloat16, TensorType::kFloat, TensorType::kBfloat16},
        "Constrain the supported non-quantized KV cache tensors."},
       {"M", {TensorType::kInt32}, "Constrain sequence-length inputs to INT32."}},
      {AttributeParam{"num_heads", "Number of query heads.", AttributeType::INT, true},
       AttributeParam{"kv_num_heads", "Number of key/value heads.", AttributeType::INT, true},
       AttributeParam{"causal", "Whether to apply causal masking.", AttributeType::INT, false,
                      int64_t{1}},
       AttributeParam{"scale", "Optional attention-score scale.", AttributeType::FLOAT, false},
       AttributeParam{"softcap", "Optional attention-score softcap.", AttributeType::FLOAT, false,
                      0.0f}},
      false, true);
}

} // namespace

std::vector<LightOpSchema> GetMicrosoftOpSchemasWithHistory(const std::string &op_type,
                                                            bool init_doc) {
  static const std::map<std::string, SchemaBuilder> builders = {
      {"BiasGelu", [] { return std::vector<LightOpSchema>{MakeBiasGeluSchema()}; }},
      {"CDist", [] { return std::vector<LightOpSchema>{MakeCDistSchema()}; }},
      {"GroupQueryAttention",
       [] { return std::vector<LightOpSchema>{MakeGroupQueryAttentionSchema()}; }},
  };
  return schema_ns::CollectSchemasFromBuilders(builders, op_type, init_doc);
}

} // namespace onnx_light_cpu
