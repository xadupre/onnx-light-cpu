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

} // namespace

std::vector<LightOpSchema> GetMicrosoftOpSchemasWithHistory(const std::string &op_type,
                                                            bool init_doc) {
  static const std::map<std::string, SchemaBuilder> builders = {
      {"BiasGelu", [] { return std::vector<LightOpSchema>{MakeBiasGeluSchema()}; }},
      {"CDist", [] { return std::vector<LightOpSchema>{MakeCDistSchema()}; }},
  };
  return schema_ns::CollectSchemasFromBuilders(builders, op_type, init_doc);
}

} // namespace onnx_light_cpu
