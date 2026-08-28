// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/binary/binary_manifest.h"
#include "onnx_light_cpu/kernels/elementwise/binary_kernel.h"

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/tuning/kernel_tuning.h"
#include "onnx_core/runtime/tuning/kernel_tuning_cache.h"
#include "onnx_op/operator_sets_logical.h"
#include "onnx_op/operator_sets_math.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
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

schema_ns::TensorType ToTensorType(onnx_light_cpu::DataType type) {
  using DT = onnx_light_cpu::DataType;
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
      EXPECT_TRUE(rt_ns::GetKernelTuningRegistry().FindCalibrationFunction(key)) << entry.op_type;
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

TEST(OnnxLightBinaryManifest, CalibrationIsBoundedCorrectnessGatedAndReturnsCompleteProfile) {
  onnx_light_cpu::BinaryElementwiseKernel::RegisterTuningSchemas();
  const rt_ns::KernelTuningKey key{
      "onnx_light_cpu",     "Add",
      "broadcast_plan",     static_cast<int32_t>(rt_ns::DataType::FLOAT),
      sym_ns::Device::kCPU, onnx_light_cpu::BinaryElementwiseKernel::kTuningAbi};
  const auto schema = rt_ns::GetKernelTuningRegistry().FindSchema(key);
  ASSERT_NE(schema, nullptr);
  const auto calibrate = rt_ns::GetKernelTuningRegistry().FindCalibrationFunction(key);
  ASSERT_TRUE(calibrate);

  rt_ns::CalibrationOptions tiny_options;
  tiny_options.maximum_memory_bytes = 1;
  rt_ns::CalibrationReporter tiny_reporter(tiny_options);
  rt_ns::CpuExecutionDescriptor serial_execution;
  serial_execution.effective_threads = 1;
  const auto fallback = calibrate(key, serial_execution, tiny_options, tiny_reporter);
  EXPECT_EQ(fallback.values, schema->portable_defaults().values);
  EXPECT_EQ(tiny_reporter.benchmark_cases(), 0u);
  ASSERT_FALSE(tiny_reporter.diagnostics().empty());
  rt_ns::KernelCalibrationSelection selection;
  selection.library = "onnx_light_cpu";
  selection.kernels = {"Add"};
  selection.implementations = {"broadcast_plan"};
  selection.element_types = {static_cast<int32_t>(rt_ns::DataType::FLOAT)};
  selection.device = sym_ns::Device::kCPU;
  const auto published = rt_ns::CalibrateRegisteredKernels(selection, tiny_options);
  ASSERT_EQ(published.calibrated.size(), 1u);
  EXPECT_GT(published.published_generation, 0u);

  InlineExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 8, &InlineExecutor::Run};
  onnx_light_cpu::ExecutionExecutorScope scope(&view);
  rt_ns::CalibrationOptions options;
  options.maximum_duration_ms = 100;
  options.maximum_memory_bytes = uint64_t{64} << 20;
  rt_ns::CalibrationReporter reporter(options);
  rt_ns::CpuExecutionDescriptor execution;
  execution.effective_threads = 8;
  const auto calibrated = calibrate(key, execution, options, reporter);
  EXPECT_NO_THROW(schema->Validate(calibrated));
  EXPECT_EQ(calibrated.values.size(), 5u);
  EXPECT_GT(reporter.benchmark_cases(), 0u);
  EXPECT_LE(reporter.peak_memory_bytes(), options.maximum_memory_bytes);
  ASSERT_FALSE(reporter.diagnostics().empty());

  rt_ns::KernelTuningKey pow_key = key;
  pow_key.kernel = "Pow";
  const auto pow_schema = rt_ns::GetKernelTuningRegistry().FindSchema(pow_key);
  ASSERT_NE(pow_schema, nullptr);
  const auto calibrate_pow = rt_ns::GetKernelTuningRegistry().FindCalibrationFunction(pow_key);
  ASSERT_TRUE(calibrate_pow);
  rt_ns::CalibrationOptions constrained_pow_options;
  constrained_pow_options.maximum_memory_bytes = uint64_t{6} << 20;
  rt_ns::CalibrationReporter constrained_pow_reporter(constrained_pow_options);
  const auto constrained_pow =
      calibrate_pow(pow_key, execution, constrained_pow_options, constrained_pow_reporter);
  EXPECT_EQ(constrained_pow.values, pow_schema->portable_defaults().values);
  EXPECT_EQ(constrained_pow_reporter.benchmark_cases(), 0u);

  rt_ns::CalibrationOptions pow_options;
  pow_options.maximum_duration_ms = 250;
  pow_options.maximum_memory_bytes = uint64_t{64} << 20;
  rt_ns::CalibrationReporter pow_reporter(pow_options);
  const auto calibrated_pow = calibrate_pow(pow_key, execution, pow_options, pow_reporter);
  EXPECT_NO_THROW(pow_schema->Validate(calibrated_pow));
  EXPECT_GT(pow_reporter.benchmark_cases(), 0u);
  EXPECT_LE(pow_reporter.peak_memory_bytes(), pow_options.maximum_memory_bytes);

  for (const auto integer_type : {rt_ns::DataType::INT32, rt_ns::DataType::INT64}) {
    rt_ns::KernelTuningKey integer_pow_key = pow_key;
    integer_pow_key.element_type = static_cast<int32_t>(integer_type);
    const auto integer_pow_schema = rt_ns::GetKernelTuningRegistry().FindSchema(integer_pow_key);
    ASSERT_NE(integer_pow_schema, nullptr);
    const auto calibrate_integer_pow =
        rt_ns::GetKernelTuningRegistry().FindCalibrationFunction(integer_pow_key);
    ASSERT_TRUE(calibrate_integer_pow);
    rt_ns::CalibrationReporter integer_pow_reporter(pow_options);
    rt_ns::KernelTuningParameters integer_pow;
    EXPECT_NO_THROW(integer_pow = calibrate_integer_pow(integer_pow_key, execution, pow_options,
                                                        integer_pow_reporter));
    EXPECT_NO_THROW(integer_pow_schema->Validate(integer_pow));
    EXPECT_GT(integer_pow_reporter.benchmark_cases(), 0u);
    EXPECT_LE(integer_pow_reporter.peak_memory_bytes(), pow_options.maximum_memory_bytes);
  }
}

TEST(OnnxLightBinaryManifest, CalibratedProfilesPersistAndOldConfigurationsStayImmutable) {
  onnx_light_cpu::BinaryElementwiseKernel::RegisterTuningSchemas();
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("Add");
  node.add_input("left");
  node.add_input("right");
  node.add_output("output");
  const rt_ns::KernelContext context(rt_ns::OpsetId(std::string(), 14));
  onnx_light_cpu::BinaryElementwiseKernel old_kernel(node, context);
  onnx_light_cpu::BinaryElementwiseKernel new_kernel(node, context);
  const auto key = old_kernel.TuningKey(static_cast<int32_t>(rt_ns::DataType::FLOAT));
  const auto schema = rt_ns::GetKernelTuningRegistry().FindSchema(key);
  ASSERT_NE(schema, nullptr);

  rt_ns::CpuExecutionDescriptor execution;
  execution.processor = ONNX_LIGHT_NAMESPACE::core::platform::GetCpuDescriptor();
  execution.effective_threads = 8;
  const auto before = rt_ns::GetKernelTuningRegistry().Snapshot();
  const auto *portable = before.Resolve(key, execution);
  ASSERT_NE(portable, nullptr);
  old_kernel.Configure(*portable);

  auto calibrated = schema->portable_defaults();
  calibrated.values["parallel.bulk_threshold_bytes"] = int64_t{0};
  calibrated.values["parallel.block_threshold_bytes"] = int64_t{0};
  calibrated.values["parallel.scalar_threshold_bytes"] = int64_t{0};
  calibrated.values["parallel.target_block_bytes"] = int64_t{12};
  calibrated.values["parallel.max_participants"] = int64_t{2};
  rt_ns::GetKernelTuningRegistry().PublishCalibratedProfiles(
      std::span<const rt_ns::KernelTuningParameters>(&calibrated, 1), execution);
  const auto after = rt_ns::GetKernelTuningRegistry().Snapshot();
  new_kernel.Configure(*after.Resolve(key, execution));
  EXPECT_EQ(before.Resolve(key, execution)->Get<int64_t>("parallel.bulk_threshold_bytes"),
            1024 * 1024);
  EXPECT_EQ(after.Resolve(key, execution)->Get<int64_t>("parallel.bulk_threshold_bytes"), 0);

  constexpr std::size_t count = 64;
  const rt_ns::Tensor left =
      rt_ns::Tensor::FromFloat("left", {count}, std::vector<float>(count, 1));
  const rt_ns::Tensor right =
      rt_ns::Tensor::FromFloat("right", {count}, std::vector<float>(count, 2));
  rt_ns::Tensor output = rt_ns::Tensor::FromFloat("output", {count}, std::vector<float>(count, 0));
  InlineExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 8, &InlineExecutor::Run};
  onnx_light_cpu::ExecutionExecutorScope scope(&view);
  old_kernel(left, right, output);
  EXPECT_EQ(executor.dispatches, 0);
  new_kernel(left, right, output);
  EXPECT_EQ(executor.dispatches, 1);
  EXPECT_EQ(executor.blocks, 2);

  const std::filesystem::path cache_path =
      std::filesystem::temp_directory_path() / "onnx_light_cpu_binary_tuning_cache_test.txt";
  std::error_code error;
  std::filesystem::remove(cache_path, error);
  rt_ns::KernelTuningCacheOptions cache_options;
  cache_options.path = cache_path;
  cache_options.execution = execution;
  const std::array profiles = {calibrated};
  const auto update = rt_ns::UpdateKernelTuningCache(profiles, cache_options);
  EXPECT_EQ(update.status, rt_ns::KernelTuningCacheUpdateStatus::kUpdated);
  const auto inspection = rt_ns::InspectKernelTuningCache(cache_options);
  EXPECT_EQ(inspection.status, rt_ns::KernelTuningCacheLoadStatus::kLoaded);
  ASSERT_EQ(inspection.profiles.size(), 1u);
  EXPECT_EQ(inspection.profiles[0].parameters.key, key);
  std::filesystem::remove(cache_path, error);
}

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
