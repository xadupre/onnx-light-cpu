// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/traditionalml/tree_ensemble_kernel.h"

#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/memory/simple_tensor.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
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
using rt_ns::DataType;
using rt_ns::RuntimeContext;
using rt_ns::Tensor;

const AttributeProto *FindAttribute(const NodeProto &node, const char *name) {
  for (int index = 0; index < node.attribute_size(); ++index) {
    if (node.attribute(index).name() == name) {
      return &node.attribute(index);
    }
  }
  return nullptr;
}

Tensor GetTensorAttribute(const NodeProto &node, const char *name, bool required) {
  const AttributeProto *attribute = FindAttribute(node, name);
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
  switch (static_cast<DataType>(tensor.data_type)) {
  case DataType::FLOAT: {
    const float *source = tensor.AsFloat();
    std::transform(source, source + size, values.begin(),
                   [](float value) { return static_cast<double>(value); });
    return values;
  }
  case DataType::DOUBLE: {
    const double *source = tensor.AsDouble();
    std::copy(source, source + size, values.begin());
    return values;
  }
  case DataType::FLOAT16: {
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
  if (static_cast<DataType>(tensor.data_type) != DataType::UINT8) {
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

TreeValueType ValueType(std::int32_t data_type) {
  switch (static_cast<DataType>(data_type)) {
  case DataType::FLOAT:
    return TreeValueType::kFloat32;
  case DataType::DOUBLE:
    return TreeValueType::kFloat64;
  case DataType::FLOAT16:
    return TreeValueType::kFloat16;
  default:
    throw std::invalid_argument(
        "onnx_light_cpu::TreeEnsemble: only FLOAT, DOUBLE and FLOAT16 inputs are supported.");
  }
}

TreeEnsembleAttributes BuildAttributes(const NodeProto &node, const Tensor &input) {
  TreeEnsembleAttributes attributes;
  attributes.n_features = input.shape[1];
  attributes.n_targets = rt_ns::GetAttributeIntOrDefault(node, "n_targets", 1);
  attributes.value_type = ValueType(input.data_type);
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
  switch (static_cast<DataType>(input.data_type)) {
  case DataType::FLOAT: {
    const float *source = input.AsFloat();
    std::transform(source, source + size, values.begin(),
                   [](float value) { return static_cast<double>(value); });
    return values;
  }
  case DataType::DOUBLE: {
    const double *source = input.AsDouble();
    std::copy(source, source + size, values.begin());
    return values;
  }
  case DataType::FLOAT16: {
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

TreeEnsembleKernel::TreeEnsembleKernel(const rt_ns::KernelContext &ctx) : KernelBase(ctx) {}

void TreeEnsembleKernel::Run(RuntimeContext &rt) {
  RecordKernelUsage(kName);
  const NodeProto &node = *node_;
  rt_ns::RequireInputCount(node, 1);
  rt_ns::RequireOutputCount(node, 1);
  const Tensor &input = rt_ns::GetInput(node, 0, rt.tensors());
  if (input.shape.size() != 2) {
    throw std::invalid_argument("onnx_light_cpu::TreeEnsemble: input must have rank 2.");
  }

  std::call_once(initialize_once_, [&]() {
    auto plan = std::make_unique<TreeEnsemblePlan>(BuildAttributes(node, input));
    input_data_type_ = input.data_type;
    feature_count_ = input.shape[1];
    plan->CompactRuntimeStorage();
    plan_ = std::move(plan);
  });
  if (input.data_type != input_data_type_ || input.shape[1] != feature_count_) {
    throw std::invalid_argument(
        "onnx_light_cpu::TreeEnsemble: input type and feature count must remain constant.");
  }

  const std::int64_t rows = input.shape[0];
  const std::int64_t targets = plan_->attributes().n_targets;
  const std::size_t output_size =
      static_cast<std::size_t>(rows) * static_cast<std::size_t>(targets);
  const std::size_t output_bytes = output_size * input.element_size();
  Tensor output = rt.MakeOutputTensor(0, input.data_type, {rows, targets}, output_bytes);
  switch (static_cast<DataType>(input.data_type)) {
  case DataType::FLOAT:
    plan_->EvaluateInto(input.AsFloat(), static_cast<std::size_t>(input.element_count()),
                        static_cast<std::size_t>(rows), output.AsFloat());
    break;
  case DataType::DOUBLE:
    plan_->EvaluateInto(input.AsDouble(), static_cast<std::size_t>(input.element_count()),
                        static_cast<std::size_t>(rows), output.AsDouble());
    break;
  case DataType::FLOAT16: {
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
  rt_ns::NodeKernelFn factory = [](const NodeProto &node,
                                   RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    auto kernel = std::make_unique<TreeEnsembleKernel>(rt.kernel_ctx());
    kernel->set_node(node);
    return kernel;
  };
  KernelRegistration info;
  info.domain = "ai.onnx.ml";
  info.op_type = "TreeEnsemble";
  info.device = sym_ns::Device::kCPU;
  info.kernel_name = TreeEnsembleKernel::kName;
  info.types = {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16};
  info.since_version = 5;
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
