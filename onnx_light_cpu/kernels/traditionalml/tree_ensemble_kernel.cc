// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/traditionalml/tree_ensemble_kernel.h"

#include "onnx_light_cpu/impl/checked_arithmetic.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/tuning/kernel_tuning.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace onnx_light_cpu {
namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

using AttributeProto = ONNX_LIGHT_NAMESPACE::AttributeProto;
using NodeProto = ONNX_LIGHT_NAMESPACE::NodeProto;
using RuntimeDataType = rt_ns::DataType;
using rt_ns::RuntimeContext;
using rt_ns::Tensor;

constexpr const char *kTreeMajorBatchRows = "parallel.tree_major_batch_rows";
constexpr const char *kRowParallelThreshold = "parallel.row_threshold";
constexpr const char *kTreeParallelRowLimit = "parallel.tree_row_limit";
constexpr const char *kMaximumParticipants = "parallel.max_participants";
constexpr std::array<RuntimeDataType, 3> kSupportedTypes = {
    RuntimeDataType::FLOAT, RuntimeDataType::DOUBLE, RuntimeDataType::FLOAT16};

bool SupportsElementType(std::int32_t element_type) {
  return std::find(kSupportedTypes.begin(), kSupportedTypes.end(),
                   static_cast<RuntimeDataType>(element_type)) != kSupportedTypes.end();
}

rt_ns::KernelTuningKey MakeTuningKey(std::int32_t element_type) {
  return {"onnx_light_cpu", "TreeEnsemble",       "prepared_v5",
          element_type,     sym_ns::Device::kCPU, TreeEnsembleKernel::kTuningAbi};
}

void ValidateTuning(const rt_ns::KernelTuningParameters &parameters) {
  for (const char *name :
       {kTreeMajorBatchRows, kRowParallelThreshold, kTreeParallelRowLimit, kMaximumParticipants}) {
    const std::int64_t value = parameters.Get<std::int64_t>(name);
    if (value < 0 || static_cast<std::uint64_t>(value) >
                         static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
      throw std::invalid_argument(std::string("TreeEnsemble ") + name +
                                  " must be zero or a positive value representable by size_t.");
    }
  }
  if (parameters.Get<std::int64_t>(kTreeMajorBatchRows) == 0 ||
      parameters.Get<std::int64_t>(kRowParallelThreshold) == 0 ||
      parameters.Get<std::int64_t>(kTreeParallelRowLimit) == 0) {
    throw std::invalid_argument("TreeEnsemble thresholds and batch size must be positive.");
  }
  if (parameters.Get<std::int64_t>(kRowParallelThreshold) >=
      parameters.Get<std::int64_t>(kTreeParallelRowLimit)) {
    throw std::invalid_argument(
        "TreeEnsemble parallel.row_threshold must be smaller than parallel.tree_row_limit.");
  }
}

rt_ns::KernelTuningParameters MakeTuningDefaults(std::int32_t element_type) {
  return {MakeTuningKey(element_type),
          {{kTreeMajorBatchRows,
            static_cast<std::int64_t>(kDefaultTreeEnsembleExecutionTuning.tree_major_batch_rows)},
           {kRowParallelThreshold,
            static_cast<std::int64_t>(kDefaultTreeEnsembleExecutionTuning.row_parallel_threshold)},
           {kTreeParallelRowLimit,
            static_cast<std::int64_t>(kDefaultTreeEnsembleExecutionTuning.tree_parallel_row_limit)},
           {kMaximumParticipants,
            static_cast<std::int64_t>(kDefaultTreeEnsembleExecutionTuning.maximum_participants)}}};
}

Tensor GetTensorAttribute(const NodeProto &node, const char *name, bool required) {
  const AttributeProto *attribute = rt_ns::FindAttribute(node, name);
  if (attribute == nullptr) {
    if (required) {
      throw std::invalid_argument(std::string("onnx_light_cpu::TreeEnsemble: missing tensor "
                                              "attribute '") +
                                  name + "'.");
    }
    return {};
  }
  if (attribute->type() != AttributeProto::AttributeType::TENSOR) {
    throw std::invalid_argument(std::string("onnx_light_cpu::TreeEnsemble: attribute '") + name +
                                "' must be a tensor.");
  }
  Tensor tensor = rt_ns::TensorFromProto(attribute->t());
  if (tensor.shape.size() != 1) {
    throw std::invalid_argument(std::string("onnx_light_cpu::TreeEnsemble: attribute '") + name +
                                "' must have rank 1.");
  }
  return tensor;
}

std::vector<double> ReadValueTensor(const NodeProto &node, const char *name, bool required,
                                    std::int32_t expected_data_type) {
  const Tensor tensor = GetTensorAttribute(node, name, required);
  if (tensor.data_type == 0 && !required) {
    return {};
  }
  if (tensor.data_type != expected_data_type) {
    throw std::invalid_argument(std::string("onnx_light_cpu::TreeEnsemble: attribute '") + name +
                                "' must match the input element type.");
  }

  const std::size_t size = static_cast<std::size_t>(tensor.element_count());
  std::vector<double> values(size);
  switch (static_cast<RuntimeDataType>(tensor.data_type)) {
  case RuntimeDataType::FLOAT: {
    const float *source = tensor.AsFloat();
    std::transform(source, source + size, values.begin(),
                   [](float value) { return static_cast<double>(value); });
    return values;
  }
  case RuntimeDataType::DOUBLE: {
    const double *source = tensor.AsDouble();
    std::copy(source, source + size, values.begin());
    return values;
  }
  case RuntimeDataType::FLOAT16: {
    const auto *source = reinterpret_cast<const std::uint16_t *>(tensor.bytes());
    std::transform(source, source + size, values.begin(), [](std::uint16_t value) {
      return static_cast<double>(rt_ns::Float16BitsToFloat(value));
    });
    return values;
  }
  default:
    throw std::invalid_argument(
        "onnx_light_cpu::TreeEnsemble: only FLOAT, DOUBLE and FLOAT16 are supported.");
  }
}

std::vector<TreeBranchMode> ReadNodeModes(const NodeProto &node) {
  const Tensor tensor = GetTensorAttribute(node, "nodes_modes", true);
  if (static_cast<RuntimeDataType>(tensor.data_type) != RuntimeDataType::UINT8) {
    throw std::invalid_argument(
        "onnx_light_cpu::TreeEnsemble: 'nodes_modes' must have type UINT8.");
  }
  const std::size_t size = static_cast<std::size_t>(tensor.element_count());
  const auto *source = reinterpret_cast<const std::uint8_t *>(tensor.bytes());
  std::vector<TreeBranchMode> modes;
  modes.reserve(size);
  for (std::size_t index = 0; index < size; ++index) {
    if (source[index] > static_cast<std::uint8_t>(TreeBranchMode::kMember)) {
      throw std::invalid_argument(
          "onnx_light_cpu::TreeEnsemble: 'nodes_modes' contains an invalid mode.");
    }
    modes.push_back(static_cast<TreeBranchMode>(source[index]));
  }
  return modes;
}

void ValidateValueType(onnx_light_cpu::DataType data_type) {
  switch (data_type) {
  case onnx_light_cpu::DataType::FLOAT:
  case onnx_light_cpu::DataType::DOUBLE:
  case onnx_light_cpu::DataType::FLOAT16:
    return;
  default:
    throw std::invalid_argument(
        "onnx_light_cpu::TreeEnsemble: only FLOAT, DOUBLE and FLOAT16 inputs are supported.");
  }
}

TreeEnsembleAttributes BuildAttributes(const NodeProto &node, const Tensor &input) {
  TreeEnsembleAttributes attributes;
  const auto value_type = static_cast<onnx_light_cpu::DataType>(input.data_type);
  ValidateValueType(value_type);
  attributes.n_features = input.shape[1];
  attributes.n_targets = rt_ns::GetAttributeIntOrDefault(node, "n_targets", 1);
  attributes.value_type = value_type;
  attributes.aggregate =
      static_cast<TreeAggregate>(rt_ns::GetAttributeIntOrDefault(node, "aggregate_function", 1));
  attributes.post_transform =
      static_cast<TreePostTransform>(rt_ns::GetAttributeIntOrDefault(node, "post_transform", 0));
  attributes.tree_roots = rt_ns::GetAttributeIntsOrDefault(node, "tree_roots", {});
  attributes.nodes_featureids = rt_ns::GetAttributeIntsOrDefault(node, "nodes_featureids", {});
  attributes.nodes_splits = ReadValueTensor(node, "nodes_splits", true, input.data_type);
  attributes.nodes_modes = ReadNodeModes(node);
  attributes.nodes_truenodeids = rt_ns::GetAttributeIntsOrDefault(node, "nodes_truenodeids", {});
  attributes.nodes_falsenodeids = rt_ns::GetAttributeIntsOrDefault(node, "nodes_falsenodeids", {});
  attributes.nodes_trueleafs = rt_ns::GetAttributeIntsOrDefault(node, "nodes_trueleafs", {});
  attributes.nodes_falseleafs = rt_ns::GetAttributeIntsOrDefault(node, "nodes_falseleafs", {});
  attributes.nodes_missing_value_tracks_true =
      rt_ns::GetAttributeIntsOrDefault(node, "nodes_missing_value_tracks_true", {});
  attributes.nodes_hitrates = ReadValueTensor(node, "nodes_hitrates", false, input.data_type);
  attributes.membership_values = ReadValueTensor(node, "membership_values", false, input.data_type);
  attributes.leaf_targetids = rt_ns::GetAttributeIntsOrDefault(node, "leaf_targetids", {});
  attributes.leaf_weights = ReadValueTensor(node, "leaf_weights", true, input.data_type);
  attributes.base_values = ReadValueTensor(node, "base_values", false, input.data_type);
  return attributes;
}

std::vector<double> ReadInput(const Tensor &input) {
  const std::size_t size = static_cast<std::size_t>(input.element_count());
  std::vector<double> values(size);
  switch (static_cast<RuntimeDataType>(input.data_type)) {
  case RuntimeDataType::FLOAT: {
    const float *source = input.AsFloat();
    std::transform(source, source + size, values.begin(),
                   [](float value) { return static_cast<double>(value); });
    return values;
  }
  case RuntimeDataType::DOUBLE: {
    const double *source = input.AsDouble();
    std::copy(source, source + size, values.begin());
    return values;
  }
  case RuntimeDataType::FLOAT16: {
    const auto *source = reinterpret_cast<const std::uint16_t *>(input.bytes());
    std::transform(source, source + size, values.begin(), [](std::uint16_t value) {
      return static_cast<double>(rt_ns::Float16BitsToFloat(value));
    });
    return values;
  }
  default:
    throw std::invalid_argument(
        "onnx_light_cpu::TreeEnsemble: only FLOAT, DOUBLE and FLOAT16 inputs are supported.");
  }
}

} // namespace

TreeEnsembleKernel::TreeEnsembleKernel(const NodeProto &node, RuntimeContext &rt)
    : KernelBase(rt.kernel_ctx()) {
  set_node(node);
  rt_ns::RequireInputCount(node, 1);
  rt_ns::RequireOutputCount(node, 1);
  const Tensor &input = rt_ns::GetInput(node, 0, rt.tensors());
  if (input.shape.size() != 2) {
    throw std::invalid_argument("onnx_light_cpu::TreeEnsemble: input must have rank 2.");
  }
  attributes_ = BuildAttributes(node, input);
  input_data_type_ = input.data_type;
  feature_count_ = input.shape[1];
  BuildPlan();
}

void TreeEnsembleKernel::RegisterTuningSchemas() {
  static std::once_flag once;
  std::call_once(once, [] {
    for (RuntimeDataType type : kSupportedTypes) {
      rt_ns::RegisterKernelTuningSchema(rt_ns::KernelTuningSchema(
          MakeTuningDefaults(static_cast<std::int32_t>(type)), ValidateTuning));
    }
  });
}

rt_ns::KernelTuningKey TreeEnsembleKernel::TuningKey(std::int32_t element_type) const {
  return SupportsElementType(element_type) ? MakeTuningKey(element_type) : rt_ns::KernelTuningKey{};
}

void TreeEnsembleKernel::Configure(const rt_ns::KernelTuningParameters &parameters) {
  if (!SupportsElementType(parameters.key.element_type) ||
      parameters.key != MakeTuningKey(parameters.key.element_type) ||
      parameters.key.element_type != input_data_type_) {
    throw std::invalid_argument("TreeEnsemble tuning parameters have an incompatible key.");
  }
  ValidateTuning(parameters);
  const TreeEnsembleExecutionTuning tuning{
      static_cast<std::size_t>(parameters.Get<std::int64_t>(kTreeMajorBatchRows)),
      static_cast<std::size_t>(parameters.Get<std::int64_t>(kRowParallelThreshold)),
      static_cast<std::size_t>(parameters.Get<std::int64_t>(kTreeParallelRowLimit)),
      static_cast<std::size_t>(parameters.Get<std::int64_t>(kMaximumParticipants)),
  };
  if (tuning == tuning_) {
    return;
  }
  tuning_ = tuning;
  BuildPlan();
}

void TreeEnsembleKernel::BuildPlan() {
  plan_ = std::make_unique<TreeEnsemblePlan>(attributes_, tuning_);
  plan_->CompactRuntimeStorage();
}

void TreeEnsembleKernel::Run(RuntimeContext &rt) {
  rt.RecordKernelUsage(kName);
  const NodeProto &node = *node_;
  rt_ns::RequireInputCount(node, 1);
  rt_ns::RequireOutputCount(node, 1);
  const Tensor &input = rt_ns::GetInput(node, 0, rt.tensors());
  if (input.shape.size() != 2) {
    throw std::invalid_argument("onnx_light_cpu::TreeEnsemble: input must have rank 2.");
  }

  if (input.data_type != input_data_type_ || input.shape[1] != feature_count_) {
    throw std::invalid_argument(
        "onnx_light_cpu::TreeEnsemble: input type and feature count must remain constant.");
  }

  const std::int64_t rows = input.shape[0];
  const std::int64_t targets = plan_->attributes().n_targets;
  const std::size_t output_size =
      CheckedProduct({CheckedDimension(rows, "TreeEnsemble", "row count"),
                      CheckedDimension(targets, "TreeEnsemble", "target count")},
                     "TreeEnsemble", "output element count");
  const std::size_t output_bytes =
      CheckedByteSize(output_size, input.element_size(), "TreeEnsemble", "output byte size");
  Tensor output = rt.MakeOutputTensor(0, input.data_type, {rows, targets}, output_bytes);
  switch (static_cast<rt_ns::DataType>(input.data_type)) {
  case rt_ns::DataType::FLOAT:
    plan_->EvaluateInto(input.AsFloat(), static_cast<std::size_t>(input.element_count()),
                        static_cast<std::size_t>(rows), output.AsFloat());
    break;
  case rt_ns::DataType::DOUBLE:
    plan_->EvaluateInto(input.AsDouble(), static_cast<std::size_t>(input.element_count()),
                        static_cast<std::size_t>(rows), output.AsDouble());
    break;
  case rt_ns::DataType::FLOAT16: {
    std::vector<double> values = plan_->Evaluate(ReadInput(input), static_cast<std::size_t>(rows));
    auto *destination = reinterpret_cast<std::uint16_t *>(output.mutable_bytes());
    std::transform(values.begin(), values.end(), destination,
                   [](double value) { return rt_ns::FloatToFloat16Bits(value); });
    break;
  }
  default:
    throw std::invalid_argument(
        "onnx_light_cpu::TreeEnsemble: only FLOAT, DOUBLE and FLOAT16 inputs are supported.");
  }
  rt_ns::SetOutput(node, 0, std::move(output), rt);
}

void RegisterTreeEnsembleKernel() {
  TreeEnsembleKernel::RegisterTuningSchemas();
  rt_ns::NodeKernelFn factory = [](const NodeProto &node,
                                   RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    return std::make_unique<TreeEnsembleKernel>(node, rt);
  };
  KernelRegistration info;
  info.domain = "ai.onnx.ml";
  info.op_type = "TreeEnsemble";
  info.device = sym_ns::Device::kCPU;
  info.kernel_name = TreeEnsembleKernel::kName;
  info.types = {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE, rt_ns::DataType::FLOAT16};
  info.since_version = 5;
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
