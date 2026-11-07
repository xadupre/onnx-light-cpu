// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/com_microsoft/gelu_kernel.h"
#include "ml_cpu_compute/checked_arithmetic.h"
#include "ml_cpu_compute/com_microsoft/gelu.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_light_cpu/kernels/execution.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include <memory>
#include <stdexcept>
#include <utility>

namespace onnx_light_cpu {
namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
using rt_ns::DataType;

MicrosoftGeluKernel::MicrosoftGeluKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                                         const rt_ns::KernelContext &ctx)
    : KernelBase(ctx) {
  if (node.attribute_size() != 0) {
    throw std::invalid_argument("com.microsoft::Gelu does not accept attributes.");
  }
  set_node(node);
}

void MicrosoftGeluKernel::Run(rt_ns::RuntimeContext &rt) {
  const RuntimeExecutionScope execution_scope;
  rt.RecordKernelUsage(kName);
  rt_ns::RequireInputCount(*node_, 1);
  rt_ns::RequireOutputCount(*node_, 1);
  const auto &x = rt_ns::GetInput(*node_, 0, rt.tensors());
  const auto type = static_cast<DataType>(x.data_type);
  if (type != DataType::FLOAT && type != DataType::DOUBLE && type != DataType::FLOAT16 &&
      type != DataType::BFLOAT16) {
    throw std::invalid_argument("com.microsoft::Gelu requires a floating-point tensor.");
  }
  const auto count = static_cast<std::size_t>(x.shape.product(0, x.shape.size(), "Gelu X"));
  const auto bytes = ml_cpu_compute::CheckedByteSize(count, x.element_size(), "Gelu", "output");
  auto y = rt.MakeOutputTensor(0, x.data_type, x.shape, bytes);
  switch (type) {
  case DataType::FLOAT:
    ml_cpu_compute::GeluFloat32(x.AsFloat(), y.AsFloat(), count);
    break;
  case DataType::DOUBLE:
    ml_cpu_compute::GeluFloat64(x.AsDouble(), y.AsDouble(), count);
    break;
  case DataType::FLOAT16:
    ml_cpu_compute::GeluFloat16(reinterpret_cast<const std::uint16_t *>(x.bytes()),
                                reinterpret_cast<std::uint16_t *>(y.mutable_bytes()), count);
    break;
  case DataType::BFLOAT16:
    ml_cpu_compute::GeluBFloat16(reinterpret_cast<const std::uint16_t *>(x.bytes()),
                                 reinterpret_cast<std::uint16_t *>(y.mutable_bytes()), count);
    break;
  default:
    throw std::logic_error("com.microsoft::Gelu encountered an unexpected DataType.");
  }
  rt_ns::SetOutput(*node_, 0, std::move(y), rt);
}

void RegisterMicrosoftGeluKernel() {
  rt_ns::NodeKernelFn factory =
      [](const ONNX_LIGHT_NAMESPACE::NodeProto &node, rt_ns::RuntimeContext &,
         const rt_ns::KernelContext &ctx) -> std::unique_ptr<rt_ns::KernelBase> {
    return std::make_unique<MicrosoftGeluKernel>(node, ctx);
  };
  KernelRegistration info;
  info.domain = kMicrosoftDomain;
  info.op_type = "Gelu";
  info.device = ONNX_LIGHT_NAMESPACE::core::symbolic::Device::kCPU;
  info.kernel_name = MicrosoftGeluKernel::kName;
  info.types = {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16};
  info.since_version = 1;
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
