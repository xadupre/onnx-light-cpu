// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/elementwise/binary_kernel.h"

#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/tuning/kernel_tuning.h"
#include "onnx_core/symbolic/sym_tensor.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>

namespace onnx_light_cpu {
namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

constexpr const char *kBulkThresholdBytes = "parallel.bulk_threshold_bytes";
constexpr const char *kBlockThresholdBytes = "parallel.block_threshold_bytes";
constexpr const char *kScalarThresholdBytes = "parallel.scalar_threshold_bytes";
constexpr const char *kTargetBlockBytes = "parallel.target_block_bytes";

std::string KernelName(std::string_view op_type) {
  return std::string("onnx_light_cpu::") + std::string(op_type);
}

rt_ns::KernelTuningKey MakeTuningKey(std::string_view op_type, int32_t element_type) {
  return {"onnx_light_cpu", std::string(op_type), "broadcast_plan",
          element_type,     sym_ns::Device::kCPU, BinaryElementwiseKernel::kTuningAbi};
}

void ValidateTuning(const rt_ns::KernelTuningParameters &parameters) {
  for (const char *name :
       {kBulkThresholdBytes, kBlockThresholdBytes, kScalarThresholdBytes, kTargetBlockBytes}) {
    const int64_t value = parameters.Get<int64_t>(name);
    if (value < 0 || static_cast<uint64_t>(value) > std::numeric_limits<std::size_t>::max()) {
      throw std::invalid_argument(std::string("Binary ") + name +
                                  " must be zero or a positive value representable by size_t.");
    }
  }
  if (parameters.Get<int64_t>(kTargetBlockBytes) == 0) {
    throw std::invalid_argument("Binary parallel.target_block_bytes must be positive.");
  }
}

rt_ns::KernelTuningParameters MakeTuningDefaults(std::string_view op_type, int32_t element_type) {
  return {MakeTuningKey(op_type, element_type),
          {{kBulkThresholdBytes,
            static_cast<int64_t>(kDefaultBinaryExecutionTuning.bulk_parallel_threshold_bytes)},
           {kBlockThresholdBytes,
            static_cast<int64_t>(kDefaultBinaryExecutionTuning.block_parallel_threshold_bytes)},
           {kScalarThresholdBytes,
            static_cast<int64_t>(kDefaultBinaryExecutionTuning.scalar_parallel_threshold_bytes)},
           {kTargetBlockBytes,
            static_cast<int64_t>(kDefaultBinaryExecutionTuning.target_block_bytes)}}};
}

bool SupportsElementType(const BinaryManifestEntry &entry, int32_t element_type) {
  return std::any_of(entry.signatures.begin(), entry.signatures.end(),
                     [element_type](const BinaryTypeSignature &signature) {
                       return static_cast<int32_t>(signature.left) == element_type;
                     });
}

BinaryKernelDescriptor::Attributes
ParseBinaryAttributes(const ONNX_LIGHT_NAMESPACE::NodeProto &node) {
  BinaryKernelDescriptor::Attributes attributes;
  if (node.op_type() == "Mod") {
    attributes.mod_fmod = rt_ns::GetAttributeIntOrDefault(node, "fmod", 0);
  } else if (node.op_type() == "BitShift") {
    const std::string direction = rt_ns::GetRequiredAttributeString(node, "direction");
    if (direction == "RIGHT") {
      attributes.bitshift_direction = BinaryKernelDescriptor::Attributes::BitShiftDirection::kRight;
    } else if (direction != "LEFT") {
      throw std::invalid_argument(
          "onnx_light_cpu::BinaryElementwiseKernel: invalid BitShift direction.");
    }
  }
  return attributes;
}

} // namespace

BinaryElementwiseKernel::BinaryElementwiseKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                                                 const rt_ns::KernelContext &ctx)
    : rt_ns::KernelBase(ctx),
      descriptor_(node.op_type(), ctx.opset.version, ParseBinaryAttributes(node)) {
  set_node(node);
}

void BinaryElementwiseKernel::RegisterTuningSchemas() {
  static std::once_flag once;
  std::call_once(once, [] {
    for (const BinaryManifestEntry &entry : GetBinaryManifest()) {
      std::set<int32_t> registered_types;
      for (const BinaryTypeSignature &signature : entry.signatures) {
        const int32_t element_type = static_cast<int32_t>(signature.left);
        if (registered_types.insert(element_type).second) {
          rt_ns::RegisterKernelTuningSchema(rt_ns::KernelTuningSchema(
              MakeTuningDefaults(entry.op_type, element_type), ValidateTuning));
        }
      }
    }
  });
}

rt_ns::KernelTuningKey BinaryElementwiseKernel::TuningKey(int32_t element_type) const {
  return SupportsElementType(descriptor_.manifest_entry(), element_type)
             ? MakeTuningKey(descriptor_.op_type(), element_type)
             : rt_ns::KernelTuningKey{};
}

void BinaryElementwiseKernel::Configure(const rt_ns::KernelTuningParameters &parameters) {
  if (!SupportsElementType(descriptor_.manifest_entry(), parameters.key.element_type) ||
      parameters.key != MakeTuningKey(descriptor_.op_type(), parameters.key.element_type)) {
    throw std::invalid_argument("Binary tuning parameters have an incompatible key.");
  }
  ValidateTuning(parameters);
  tuning_ = {
      static_cast<std::size_t>(parameters.Get<int64_t>(kBulkThresholdBytes)),
      static_cast<std::size_t>(parameters.Get<int64_t>(kBlockThresholdBytes)),
      static_cast<std::size_t>(parameters.Get<int64_t>(kScalarThresholdBytes)),
      static_cast<std::size_t>(parameters.Get<int64_t>(kTargetBlockBytes)),
  };
}

rt_ns::Tensor BinaryElementwiseKernel::operator()(const rt_ns::Tensor &left,
                                                  const rt_ns::Tensor &right,
                                                  rt_ns::RuntimeContext *rt) const {
  const auto output_type = static_cast<rt_ns::DataType>(descriptor_.ResolveOutputType(
      static_cast<BinaryDataType>(left.data_type), static_cast<BinaryDataType>(right.data_type)));
  const std::shared_ptr<const BinaryBroadcastPlan> plan =
      plan_cache_.GetOrCreate(descriptor_, static_cast<BinaryDataType>(left.data_type),
                              static_cast<BinaryDataType>(right.data_type),
                              static_cast<BinaryDataType>(output_type), left.shape, right.shape);
  const rt_ns::Shape output_shape(
      std::vector<std::int64_t>(plan->output_shape().begin(), plan->output_shape().end()));
  const std::size_t out_bytes =
      plan->inner_loop_elements() * plan->outer_block_count() * plan->adapter().output_size;
  rt_ns::Tensor output =
      rt != nullptr ? rt->MakeOutputTensor(0, output_type, output_shape, out_bytes)
                    : rt_ns::MakeOutputTensor(output_type, output_shape, out_bytes, nullptr);
  (*this)(left, right, output);
  return output;
}

void BinaryElementwiseKernel::operator()(const rt_ns::Tensor &left, const rt_ns::Tensor &right,
                                         rt_ns::Tensor &output) const {
  const auto output_type = static_cast<rt_ns::DataType>(descriptor_.ResolveOutputType(
      static_cast<BinaryDataType>(left.data_type), static_cast<BinaryDataType>(right.data_type)));
  const std::shared_ptr<const BinaryBroadcastPlan> plan =
      plan_cache_.GetOrCreate(descriptor_, static_cast<BinaryDataType>(left.data_type),
                              static_cast<BinaryDataType>(right.data_type),
                              static_cast<BinaryDataType>(output_type), left.shape, right.shape);
  const rt_ns::Shape output_shape(
      std::vector<std::int64_t>(plan->output_shape().begin(), plan->output_shape().end()));
  if (output.data_type != static_cast<int32_t>(output_type) || output.shape != output_shape) {
    throw std::invalid_argument(
        "onnx_light_cpu::BinaryElementwiseKernel: output tensor metadata mismatch.");
  }
  plan->Execute(left.bytes(), right.bytes(), output.mutable_bytes(), tuning_);
}

void BinaryElementwiseKernel::Run(rt_ns::RuntimeContext &rt) {
  const auto &node = *node_;
  RecordKernelUsage(KernelName(descriptor_.op_type()));
  rt_ns::RequireInputCount(node, 2);
  rt_ns::RequireOutputCount(node, 1);
  const rt_ns::Tensor &left = rt_ns::GetInput(node, 0, rt.tensors());
  const rt_ns::Tensor &right = rt_ns::GetInput(node, 1, rt.tensors());
  rt_ns::SetOutput(node, 0, (*this)(left, right, &rt), rt);
}

void RegisterBinaryKernels() {
  BinaryElementwiseKernel::RegisterTuningSchemas();
  for (const BinaryManifestEntry &entry : GetBinaryManifest()) {
    rt_ns::NodeKernelFn factory =
        [](const ONNX_LIGHT_NAMESPACE::NodeProto &node,
           rt_ns::RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
      return std::make_unique<BinaryElementwiseKernel>(node, rt.kernel_ctx());
    };
    KernelRegistration info;
    info.domain = "";
    info.op_type = std::string(entry.op_type);
    info.device = sym_ns::Device::kCPU;
    info.kernel_name = KernelName(entry.op_type);
    for (const BinaryTypeSignature &signature : entry.signatures) {
      const auto type = static_cast<rt_ns::DataType>(signature.left);
      if (std::find(info.types.begin(), info.types.end(), type) == info.types.end()) {
        info.types.push_back(type);
      }
    }
    info.since_version = entry.since_version;
    RegisterKernel(std::move(info), std::move(factory));
  }
}

} // namespace onnx_light_cpu
