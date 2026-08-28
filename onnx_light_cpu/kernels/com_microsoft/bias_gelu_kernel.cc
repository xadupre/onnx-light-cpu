// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/com_microsoft/bias_gelu_kernel.h"

#include "onnx_light_cpu/impl/com_microsoft/bias_gelu.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/tuning/kernel_tuning.h"
#include "onnx_core/symbolic/sym_tensor.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace onnx_light_cpu {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

using NodeProto = ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::DataType;
using rt_ns::NodeKernelFn;
using rt_ns::RuntimeContext;
using rt_ns::Tensor;

namespace {

constexpr const char *kParallelThresholdBytes = "parallel.threshold_bytes";
constexpr const char *kTargetBlockBytes = "parallel.target_block_bytes";
constexpr const char *kMaxParticipants = "parallel.max_participants";
constexpr const char *kPreferredParticipants = "parallel.preferred_participants";
constexpr const char *kCostModel = "parallel.cost_model";
constexpr std::array<DataType, 4> kSupportedTypes = {DataType::FLOAT, DataType::DOUBLE,
                                                     DataType::FLOAT16, DataType::BFLOAT16};

bool SupportsElementType(int32_t element_type) {
  for (DataType type : kSupportedTypes) {
    if (element_type == static_cast<int32_t>(type)) {
      return true;
    }
  }
  return false;
}

rt_ns::KernelTuningKey MakeTuningKey(int32_t element_type) {
  return {"onnx_light_cpu", "BiasGelu",           "row_dispatch",
          element_type,     sym_ns::Device::kCPU, BiasGeluKernel::kTuningAbi};
}

void ValidateTuning(const rt_ns::KernelTuningParameters &parameters) {
  for (const char *name :
       {kParallelThresholdBytes, kTargetBlockBytes, kMaxParticipants, kPreferredParticipants}) {
    const int64_t value = parameters.Get<int64_t>(name);
    if (value < 0 || static_cast<uint64_t>(value) > std::numeric_limits<std::size_t>::max()) {
      throw std::invalid_argument(std::string("BiasGelu ") + name +
                                  " must be zero or a positive value representable by size_t.");
    }
  }
  if (parameters.Get<int64_t>(kTargetBlockBytes) == 0) {
    throw std::invalid_argument("BiasGelu parallel.target_block_bytes must be positive.");
  }
  const int64_t cost_model = parameters.Get<int64_t>(kCostModel);
  if (cost_model != 0 && cost_model != 1) {
    throw std::invalid_argument("BiasGelu parallel.cost_model must be 0 or 1.");
  }
}

const BiasGeluExecutionTuning &DefaultTuning(int32_t element_type) {
  switch (static_cast<DataType>(element_type)) {
  case DataType::DOUBLE:
    return kDefaultBiasGeluFloat64ExecutionTuning;
  case DataType::FLOAT16:
  case DataType::BFLOAT16:
    return kDefaultBiasGeluHalfExecutionTuning;
  case DataType::FLOAT:
  default:
    return kDefaultBiasGeluFloat32ExecutionTuning;
  }
}

rt_ns::KernelTuningParameters MakeTuningDefaults(int32_t element_type) {
  const BiasGeluExecutionTuning &defaults = DefaultTuning(element_type);
  return {MakeTuningKey(element_type),
          {{kParallelThresholdBytes, static_cast<int64_t>(defaults.parallel_threshold_bytes)},
           {kTargetBlockBytes, static_cast<int64_t>(defaults.target_block_bytes)},
           {kMaxParticipants, static_cast<int64_t>(defaults.max_participants)},
           {kPreferredParticipants, static_cast<int64_t>(defaults.preferred_participants)},
           {kCostModel, defaults.use_cost_model ? int64_t{1} : int64_t{0}}}};
}

void ValidateTensors(const Tensor &a, const Tensor &b, const Tensor &output) {
  if (a.data_type != b.data_type || a.data_type != output.data_type) {
    throw std::invalid_argument("onnx_light_cpu::BiasGelu: A, B and output dtypes must match.");
  }
  if (a.shape.size() == 0) {
    throw std::invalid_argument("onnx_light_cpu::BiasGelu: A must have positive rank.");
  }
  if (b.shape.size() != 1) {
    throw std::invalid_argument("onnx_light_cpu::BiasGelu: B must be a rank-1 tensor.");
  }
  if (b.shape[0] != a.shape[a.shape.size() - 1]) {
    throw std::invalid_argument(
        "onnx_light_cpu::BiasGelu: B's length must match A's last dimension.");
  }
  if (output.shape != a.shape) {
    throw std::invalid_argument("onnx_light_cpu::BiasGelu: output shape must match A's shape.");
  }
}

} // namespace

BiasGeluKernel::BiasGeluKernel(const NodeProto &node, const rt_ns::KernelContext &ctx)
    : KernelBase(ctx) {
  set_node(node);
}

void BiasGeluKernel::RegisterTuningSchemas() {
  static std::once_flag once;
  std::call_once(once, [] {
    for (DataType type : kSupportedTypes) {
      rt_ns::RegisterKernelTuningSchema(rt_ns::KernelTuningSchema(
          MakeTuningDefaults(static_cast<int32_t>(type)), ValidateTuning));
    }
  });
}

rt_ns::KernelTuningKey BiasGeluKernel::TuningKey(int32_t element_type) const {
  return SupportsElementType(element_type) ? MakeTuningKey(element_type) : rt_ns::KernelTuningKey{};
}

void BiasGeluKernel::Configure(const rt_ns::KernelTuningParameters &parameters) {
  if (!SupportsElementType(parameters.key.element_type) ||
      parameters.key != MakeTuningKey(parameters.key.element_type)) {
    throw std::invalid_argument("BiasGelu tuning parameters have an incompatible key.");
  }
  ValidateTuning(parameters);
  tuning_ = {
      static_cast<std::size_t>(parameters.Get<int64_t>(kParallelThresholdBytes)),
      static_cast<std::size_t>(parameters.Get<int64_t>(kTargetBlockBytes)),
      static_cast<std::size_t>(parameters.Get<int64_t>(kMaxParticipants)),
      parameters.Get<int64_t>(kCostModel) != 0,
      static_cast<std::size_t>(parameters.Get<int64_t>(kPreferredParticipants)),
  };
  tuning_configured_ = true;
}

Tensor BiasGeluKernel::operator()(const Tensor &a, const Tensor &b, RuntimeContext *rt) const {
  const std::size_t bytes = static_cast<std::size_t>(a.element_count()) * a.element_size();
  Tensor output = rt != nullptr ? rt->MakeOutputTensor(0, a.data_type, a.shape, bytes)
                                : rt_ns::MakeOutputTensor(a.data_type, a.shape, bytes, nullptr);
  (*this)(a, b, output);
  return output;
}

void BiasGeluKernel::operator()(const Tensor &a, const Tensor &b, Tensor &output) const {
  ValidateTensors(a, b, output);
  const std::size_t inner = static_cast<std::size_t>(b.shape[0]);
  const std::size_t outer = inner == 0 ? 0 : static_cast<std::size_t>(a.element_count()) / inner;
  const BiasGeluExecutionTuning &tuning = tuning_configured_ ? tuning_ : DefaultTuning(a.data_type);
  switch (static_cast<DataType>(a.data_type)) {
  case DataType::FLOAT:
    BiasGeluFloat32WithTuning(a.AsFloat(), b.AsFloat(), output.AsFloat(), outer, inner, tuning);
    return;
  case DataType::DOUBLE:
    BiasGeluFloat64WithTuning(a.AsDouble(), b.AsDouble(), output.AsDouble(), outer, inner, tuning);
    return;
  case DataType::FLOAT16:
    BiasGeluFloat16WithTuning(reinterpret_cast<const std::uint16_t *>(a.bytes()),
                              reinterpret_cast<const std::uint16_t *>(b.bytes()),
                              reinterpret_cast<std::uint16_t *>(output.mutable_bytes()), outer,
                              inner, tuning);
    return;
  case DataType::BFLOAT16:
    BiasGeluBFloat16WithTuning(reinterpret_cast<const std::uint16_t *>(a.bytes()),
                               reinterpret_cast<const std::uint16_t *>(b.bytes()),
                               reinterpret_cast<std::uint16_t *>(output.mutable_bytes()), outer,
                               inner, tuning);
    return;
  default:
    throw std::invalid_argument("onnx_light_cpu::BiasGelu: unsupported data type " +
                                std::to_string(a.data_type) +
                                ", only FLOAT, DOUBLE, FLOAT16 and BFLOAT16 are supported.");
  }
}

void BiasGeluKernel::Run(RuntimeContext &rt) {
  RecordKernelUsage(kName);
  const NodeProto &node = *node_;
  rt_ns::RequireInputCount(node, 2);
  rt_ns::RequireOutputCount(node, 1);
  const Tensor &a = rt_ns::GetInput(node, 0, rt.tensors());
  const Tensor &b = rt_ns::GetInput(node, 1, rt.tensors());
  rt_ns::SetOutput(node, 0, (*this)(a, b, &rt), rt);
}

void RegisterBiasGeluKernel() {
  BiasGeluKernel::RegisterTuningSchemas();
  NodeKernelFn factory = [](const NodeProto &node,
                            RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    return std::make_unique<BiasGeluKernel>(node, rt.kernel_ctx());
  };
  KernelRegistration info;
  info.domain = kMicrosoftDomain;
  info.op_type = "BiasGelu";
  info.device = sym_ns::Device::kCPU;
  info.kernel_name = BiasGeluKernel::kName;
  info.types = {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16};
  info.since_version = 1;
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
