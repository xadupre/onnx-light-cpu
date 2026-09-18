// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/math/tanh_kernel.h"

#include "onnx_light_cpu/impl/math/math_kernels.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/tensor_buffer_validation.h"

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/symbolic/sym_tensor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

namespace onnx_light_cpu {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
using ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::DataType;
using rt_ns::RuntimeContext;
using rt_ns::Tensor;

Tensor TanhKernel::operator()(const Tensor &x, RuntimeContext *rt) const {
  const std::size_t size_bytes = tensor_validation::ValidateBufferCapacity(x, kName, "input");
  Tensor output = rt != nullptr
                      ? rt->MakeOutputTensor(0, x.data_type, x.shape, size_bytes)
                      : rt_ns::MakeOutputTensor(x.data_type, x.shape, size_bytes, nullptr);
  (*this)(x, output);
  return output;
}

void TanhKernel::operator()(const Tensor &x, Tensor &output) const {
  if (output.data_type != x.data_type || output.shape != x.shape) {
    throw std::invalid_argument(
        "onnx_light_cpu::Tanh: output dtype and shape must match the input.");
  }
  tensor_validation::ValidateBufferCapacity(x, kName, "input");
  tensor_validation::ValidateBufferCapacity(output, kName, "output");
  const std::size_t count = static_cast<std::size_t>(x.element_count());
  switch (static_cast<DataType>(x.data_type)) {
  case DataType::FLOAT:
    TanhFloat32(x.AsFloat(), output.AsFloat(), count);
    return;
  case DataType::FLOAT16:
    TanhFloat16(reinterpret_cast<const std::uint16_t *>(x.bytes()),
                reinterpret_cast<std::uint16_t *>(output.mutable_bytes()), count);
    return;
  case DataType::BFLOAT16:
    TanhBFloat16(reinterpret_cast<const std::uint16_t *>(x.bytes()),
                 reinterpret_cast<std::uint16_t *>(output.mutable_bytes()), count);
    return;
  default:
    throw std::invalid_argument(
        "onnx_light_cpu::Tanh: only FLOAT, FLOAT16 and BFLOAT16 are supported.");
  }
}

void TanhKernel::Run(RuntimeContext &rt) {
  rt.RecordKernelUsage(kName);
  const NodeProto &node = *node_;
  rt_ns::RequireInputCount(node, 1);
  rt_ns::RequireOutputCount(node, 1);
  const Tensor &x = rt_ns::GetInput(node, 0, rt.tensors());
  rt_ns::SetOutput(node, 0, (*this)(x, &rt), rt);
}

void RegisterTanhKernel() {
  rt_ns::NodeKernelFn factory = [](const NodeProto &node,
                                   RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    auto kernel = std::make_unique<TanhKernel>(rt.kernel_ctx());
    kernel->set_node(node);
    return kernel;
  };
  KernelRegistration info;
  info.domain = "";
  info.op_type = "Tanh";
  info.device = ONNX_LIGHT_NAMESPACE::core::symbolic::Device::kCPU;
  info.kernel_name = TanhKernel::kName;
  info.types = {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16};
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
