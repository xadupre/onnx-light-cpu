// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/schemas/ai_onnx/op_schema.h"

#include <map>

namespace onnx_light_cpu {
namespace {

namespace schema_ns = ONNX_LIGHT_NAMESPACE::core::schema;
using schema_ns::AttributeParam;
using schema_ns::AttributeType;
using schema_ns::LightOpSchema;
using schema_ns::SchemaBuilder;
using schema_ns::TensorType;

LightOpSchema MakeRotaryEmbeddingSchema() {
  LightOpSchema schema(
      "RotaryEmbedding", schema_ns::kOnnxDomain, 23,
      "Applies split-half or interleaved rotary embedding to the first rotary_embedding_dim "
      "elements of each head; preserves the remaining elements.",
      {{"X", "Rank-3 (batch, sequence, hidden) or rank-4 (batch, heads, sequence, head).", "T"},
       {"cos_cache", "Rank-2 indexed cache, or rank-3 per-token cache.", "T"},
       {"sin_cache", "Cache matching cos_cache.", "T"},
       {"position_ids", "Optional INT64 (batch, sequence) indices for rank-2 caches.", "M"}},
      {{"Y", "Rotated tensor with X's shape and type.", "T"}},
      {{"T",
        {TensorType::kFloat, TensorType::kFloat16, TensorType::kBfloat16},
        "Matching floating-point input and caches."},
       {"M", {TensorType::kInt64}, "Position indices."}},
      {AttributeParam{"num_heads", "Required for rank-3 input.", AttributeType::INT, false},
       AttributeParam{"rotary_embedding_dim", "Zero rotates the full head.", AttributeType::INT,
                      false, int64_t{0}},
       AttributeParam{"interleaved", "Interleaved layout if 1.", AttributeType::INT, false,
                      int64_t{0}}},
      false, true);
  return schema;
}

LightOpSchema MakeSimplifiedLayerNormalizationSchema() {
  LightOpSchema schema(
      "SimplifiedLayerNormalization", schema_ns::kOnnxDomain, 1,
      "Experimental ai.onnx operator compatibility adapter for ONNX Runtime. Normalizes X by "
      "sqrt(mean(X * X) + epsilon) over the suffix beginning at axis, then multiplies by Scale. "
      "Scale broadcasts right-aligned to the entire shape of X. X and Scale may independently "
      "use FLOAT, FLOAT16, DOUBLE, or BFLOAT16; Y uses Scale's type. stash_type selects "
      "arithmetic and optional saved-statistics precision: 1 uses FLOAT and 11 uses DOUBLE, "
      "independently of input types, scale layout, or whether statistics are requested. "
      "There is no intermediate low-precision rounding before scaling. "
      "The statistics shape is X[:axis] followed by ones for every normalized dimension, "
      "matching runtime behavior rather than upstream schema inference. This is not a "
      "standardized ONNX operator.",
      {{"X", "Input tensor of rank at least one.", "T"},
       {"Scale", "Scale tensor broadcastable to the entire shape of X.", "V"}},
      {{"Y", "Normalized and scaled tensor with X's shape and Scale's type.", "V"},
       {"inv_std_var", "Optional reciprocal standard deviation, with all reduced dimensions one.",
        "U"}},
      {{"T",
        {TensorType::kFloat, TensorType::kFloat16, TensorType::kDouble, TensorType::kBfloat16},
        "Input floating-point type."},
       {"V",
        {TensorType::kFloat, TensorType::kFloat16, TensorType::kDouble, TensorType::kBfloat16},
        "Independent Scale and Y floating-point type."},
       {"U",
        {TensorType::kFloat, TensorType::kDouble},
        "Arithmetic and saved-statistics type selected by stash_type."}},
      {AttributeParam{"axis", "First normalized axis; negative axes count from the end.",
                      AttributeType::INT, false, int64_t{-1}},
       AttributeParam{"epsilon", "Value added to the mean square; IEEE values are permitted.",
                      AttributeType::FLOAT, false, 1.0e-5f},
       AttributeParam{"stash_type",
                      "Arithmetic and saved-statistics type: 1 (FLOAT) or 11 (DOUBLE).",
                      AttributeType::INT, false, int64_t{1}}},
      false, true);
  schema.set_min_output(1);
  schema.set_max_output(2);
  return schema;
}

} // namespace

std::vector<LightOpSchema> GetExperimentalOpSchemasWithHistory(const std::string &op_type,
                                                               bool init_doc) {
  static const std::map<std::string, SchemaBuilder> builders = {
      {"RotaryEmbedding", [] { return std::vector<LightOpSchema>{MakeRotaryEmbeddingSchema()}; }},
      {"SimplifiedLayerNormalization",
       [] { return std::vector<LightOpSchema>{MakeSimplifiedLayerNormalizationSchema()}; }},
  };
  return schema_ns::CollectSchemasFromBuilders(builders, op_type, init_doc);
}

} // namespace onnx_light_cpu
