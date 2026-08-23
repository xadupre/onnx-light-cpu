// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/elementwise/binary_kernel.h"

#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/symbolic/sym_tensor.h"

#include <algorithm>
#include <memory>
#include <string>

namespace onnx_light_cpu {
namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

std::string KernelName(std::string_view op_type) {
  return std::string("onnx_light_cpu::") + std::string(op_type);
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
  plan->Execute(left.bytes(), right.bytes(), output.mutable_bytes());
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
