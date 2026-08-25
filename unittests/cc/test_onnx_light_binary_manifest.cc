// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/binary/binary_manifest.h"
#include "onnx_light_cpu/kernels/elementwise/binary_kernel.h"

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/tuning/kernel_tuning.h"
#include "onnx_op/operator_sets_logical.h"
#include "onnx_op/operator_sets_math.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

namespace schema_ns = ONNX_LIGHT_NAMESPACE::core::schema;
namespace logical_ns = ONNX_LIGHT_NAMESPACE::onnx_op::logical;
namespace math_ns = ONNX_LIGHT_NAMESPACE::onnx_op::math;
namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

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

TEST(OnnxLightBinaryManifest, RegistersValidatedTuningSchemaForEveryOperatorAndInputType) {
  EXPECT_NO_THROW(onnx_light_cpu::BinaryElementwiseKernel::RegisterTuningSchemas());
  EXPECT_NO_THROW(onnx_light_cpu::BinaryElementwiseKernel::RegisterTuningSchemas());

  for (const auto &entry : onnx_light_cpu::GetBinaryManifest()) {
    std::set<int32_t> element_types;
    for (const auto &signature : entry.signatures) {
      if (!element_types.insert(static_cast<int32_t>(signature.left)).second) {
        continue;
      }
      const rt_ns::KernelTuningKey key{
          "onnx_light_cpu",     std::string(entry.op_type),
          "broadcast_plan",     static_cast<int32_t>(signature.left),
          sym_ns::Device::kCPU, onnx_light_cpu::BinaryElementwiseKernel::kTuningAbi};
      const auto schema = rt_ns::GetKernelTuningRegistry().FindSchema(key);
      ASSERT_NE(schema, nullptr) << entry.op_type;
      const auto &parameters = schema->portable_defaults();
      EXPECT_EQ(parameters.Get<int64_t>("parallel.bulk_threshold_bytes"), 1024 * 1024);
      EXPECT_EQ(parameters.Get<int64_t>("parallel.block_threshold_bytes"), 1024 * 1024);
      EXPECT_EQ(parameters.Get<int64_t>("parallel.scalar_threshold_bytes"), 256 * 1024);
      EXPECT_EQ(parameters.Get<int64_t>("parallel.target_block_bytes"), 1024 * 1024);
      EXPECT_EQ(parameters.Get<int64_t>("parallel.max_participants"), 0);
      EXPECT_NO_THROW(schema->Validate(parameters));
    }
  }
}

struct InlineExecutor {
  std::int64_t dispatches = 0;
  std::int64_t blocks = 0;

  static void Run(void *context, std::int64_t num_blocks, void *task_context,
                  onnx_light_cpu::ExecutionBlockFn task) {
    auto &self = *static_cast<InlineExecutor *>(context);
    ++self.dispatches;
    self.blocks = num_blocks;
    for (std::int64_t block = 0; block < num_blocks; ++block) {
      task(task_context, block);
    }
  }
};

TEST(OnnxLightBinaryManifest, ConfiguredTuningControlsExecutionAndRejectsInvalidValues) {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("Add");
  node.add_input("left");
  node.add_input("right");
  node.add_output("output");
  const rt_ns::KernelContext context(rt_ns::OpsetId(std::string(), 14));
  onnx_light_cpu::BinaryElementwiseKernel kernel(node, context);
  onnx_light_cpu::BinaryElementwiseKernel::RegisterTuningSchemas();

  const auto key = kernel.TuningKey(static_cast<int32_t>(rt_ns::DataType::FLOAT));
  EXPECT_EQ(key.library, "onnx_light_cpu");
  EXPECT_EQ(key.kernel, "Add");
  EXPECT_EQ(key.implementation, "broadcast_plan");
  EXPECT_EQ(key.tuning_abi, onnx_light_cpu::BinaryElementwiseKernel::kTuningAbi);
  const auto schema = rt_ns::GetKernelTuningRegistry().FindSchema(key);
  ASSERT_NE(schema, nullptr);
  auto parameters = schema->portable_defaults();

  constexpr std::size_t count = 64;
  const rt_ns::Tensor left =
      rt_ns::Tensor::FromFloat("left", {count}, std::vector<float>(count, 1));
  const rt_ns::Tensor right =
      rt_ns::Tensor::FromFloat("right", {count}, std::vector<float>(count, 2));
  rt_ns::Tensor output = rt_ns::Tensor::FromFloat("output", {count}, std::vector<float>(count, 0));
  InlineExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 8, &InlineExecutor::Run};
  onnx_light_cpu::ExecutionExecutorScope scope(&view);

  kernel(left, right, output);
  EXPECT_EQ(executor.dispatches, 0);

  parameters.values["parallel.bulk_threshold_bytes"] = int64_t{0};
  parameters.values["parallel.block_threshold_bytes"] = int64_t{0};
  parameters.values["parallel.scalar_threshold_bytes"] = int64_t{0};
  parameters.values["parallel.target_block_bytes"] = int64_t{12};
  parameters.values["parallel.max_participants"] = int64_t{2};
  EXPECT_NO_THROW(schema->Validate(parameters));
  EXPECT_NO_THROW(kernel.Configure(parameters));
  kernel(left, right, output);
  EXPECT_EQ(executor.dispatches, 1);
  EXPECT_EQ(executor.blocks, 2);
  for (float value : std::span(output.AsFloat(), count)) {
    EXPECT_FLOAT_EQ(value, 3.0f);
  }

  parameters.values["parallel.target_block_bytes"] = int64_t{0};
  EXPECT_THROW(schema->Validate(parameters), std::invalid_argument);
  EXPECT_THROW(kernel.Configure(parameters), std::invalid_argument);
  parameters = schema->portable_defaults();
  parameters.values["parallel.bulk_threshold_bytes"] = int64_t{-1};
  EXPECT_THROW(schema->Validate(parameters), std::invalid_argument);
  EXPECT_THROW(kernel.Configure(parameters), std::invalid_argument);
  parameters = schema->portable_defaults();
  parameters.values["parallel.max_participants"] = int64_t{-1};
  EXPECT_THROW(schema->Validate(parameters), std::invalid_argument);
  EXPECT_THROW(kernel.Configure(parameters), std::invalid_argument);

  EXPECT_EQ(kernel.TuningKey(static_cast<int32_t>(rt_ns::DataType::STRING)).device,
            sym_ns::Device::kUndefined);
}

} // namespace
