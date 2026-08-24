// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/binary/binary_manifest.h"

#include "onnx_op/operator_sets_logical.h"
#include "onnx_op/operator_sets_math.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

namespace schema_ns = ONNX_LIGHT_NAMESPACE::core::schema;
namespace logical_ns = ONNX_LIGHT_NAMESPACE::onnx_op::logical;
namespace math_ns = ONNX_LIGHT_NAMESPACE::onnx_op::math;

schema_ns::TensorType ToTensorType(onnx_light_cpu::BinaryDataType type) {
  using DT = onnx_light_cpu::BinaryDataType;
  using TT = schema_ns::TensorType;
  switch (type) {
  case DT::FLOAT:
    return TT::kFloat;
  case DT::UINT8:
    return TT::kUint8;
  case DT::INT8:
    return TT::kInt8;
  case DT::UINT16:
    return TT::kUint16;
  case DT::INT16:
    return TT::kInt16;
  case DT::INT32:
    return TT::kInt32;
  case DT::INT64:
    return TT::kInt64;
  case DT::BOOL:
    return TT::kBool;
  case DT::FLOAT16:
    return TT::kFloat16;
  case DT::DOUBLE:
    return TT::kDouble;
  case DT::UINT32:
    return TT::kUint32;
  case DT::UINT64:
    return TT::kUint64;
  case DT::BFLOAT16:
    return TT::kBfloat16;
  default:
    throw std::invalid_argument("unsupported binary manifest type");
  }
}

std::vector<schema_ns::LightOpSchema> LoadSchemas(std::string_view op_type) {
  std::vector<schema_ns::LightOpSchema> schemas =
      math_ns::GetAllOnnxOpMathSchemasWithHistory(std::string(op_type), /*init_doc=*/false);
  if (schemas.empty()) {
    schemas = logical_ns::GetAllOnnxOpLogicalSchemasWithHistory(std::string(op_type),
                                                                /*init_doc=*/false);
  }
  return schemas;
}

const schema_ns::LightOpSchema &LatestSchema(std::string_view op_type) {
  static const auto cache = [] {
    std::unordered_map<std::string, std::vector<schema_ns::LightOpSchema>> map;
    for (const auto &entry : onnx_light_cpu::GetBinaryManifest()) {
      map.emplace(std::string(entry.op_type), LoadSchemas(entry.op_type));
    }
    return map;
  }();
  const auto it = cache.find(std::string(op_type));
  if (it == cache.end() || it->second.empty()) {
    throw std::invalid_argument("missing schema for op");
  }
  return *std::max_element(it->second.begin(), it->second.end(), [](const auto &a, const auto &b) {
    return a.since_version() < b.since_version();
  });
}

bool ConstraintAllows(const schema_ns::LightOpSchema &schema, const std::string &type_name,
                      schema_ns::TensorType candidate) {
  const auto it = std::find_if(schema.type_constraints().begin(), schema.type_constraints().end(),
                               [&](const schema_ns::TypeConstraintParam &param) {
                                 return param.type_param_str == type_name;
                               });
  if (it == schema.type_constraints().end()) {
    return false;
  }
  return std::find(it->allowed_type_strs.begin(), it->allowed_type_strs.end(), candidate) !=
         it->allowed_type_strs.end();
}

TEST(OnnxLightBinaryManifest, MatchesLatestLightSchemaVersionsAndTypeConstraints) {
  for (const auto &entry : onnx_light_cpu::GetBinaryManifest()) {
    const auto &schema = LatestSchema(entry.op_type);
    EXPECT_EQ(schema.domain(), schema_ns::kOnnxDomain) << entry.op_type;
    EXPECT_EQ(schema.since_version(), entry.since_version) << entry.op_type;
    ASSERT_GE(schema.inputs().size(), 2u) << entry.op_type;
    ASSERT_GE(schema.outputs().size(), 1u) << entry.op_type;
    for (const auto &signature : entry.signatures) {
      EXPECT_TRUE(ConstraintAllows(schema, schema.inputs()[0].type, ToTensorType(signature.left)))
          << entry.op_type;
      EXPECT_TRUE(ConstraintAllows(schema, schema.inputs()[1].type, ToTensorType(signature.right)))
          << entry.op_type;
      EXPECT_TRUE(
          ConstraintAllows(schema, schema.outputs()[0].type, ToTensorType(signature.output)))
          << entry.op_type;
    }
    if (entry.op_type == "BitShift") {
      const auto attr = std::find_if(
          schema.attributes().begin(), schema.attributes().end(),
          [](const schema_ns::AttributeParam &param) { return param.name == "direction"; });
      ASSERT_NE(attr, schema.attributes().end());
      EXPECT_TRUE(attr->required);
      EXPECT_EQ(attr->type, schema_ns::AttributeType::STRING);
    }
  }
}

} // namespace
