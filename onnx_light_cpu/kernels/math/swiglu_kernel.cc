// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/math/swiglu_kernel.h"

#include "onnx_light_cpu/impl/math/math_kernels.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/symbolic/sym_tensor.h"

#include <cstdint>
#include <memory>
#include <stdexcept>

namespace onnx_light_cpu {
namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

using NodeProto = ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::DataType;
using rt_ns::RuntimeContext;
using rt_ns::Tensor;

void ValidateTensors(const Tensor &gate, const Tensor &value, const Tensor &output) {
  if (gate.data_type != value.data_type || gate.data_type != output.data_type) {
    throw std::invalid_argument("onnx_light_cpu::SwiGLU: input and output dtypes must match.");
  }
  if (gate.shape != value.shape) {
    throw std::invalid_argument(
        "onnx_light_cpu::SwiGLU: inputs must have identical shapes; broadcasting is unsupported.");
  }
  if (output.shape != gate.shape || output.size_bytes() != gate.size_bytes()) {
    throw std::invalid_argument("onnx_light_cpu::SwiGLU: output tensor metadata mismatch.");
  }
}

} // namespace

Tensor SwiGLUKernel::operator()(const Tensor &gate, const Tensor &value, float alpha,
                                RuntimeContext *rt) const {
  if (gate.shape != value.shape) {
    throw std::invalid_argument(
        "onnx_light_cpu::SwiGLU: inputs must have identical shapes; broadcasting is unsupported.");
  }
  const std::size_t bytes = gate.size_bytes();
  Tensor output = rt != nullptr
                      ? rt->MakeOutputTensor(0, gate.data_type, gate.shape, bytes)
                      : rt_ns::MakeOutputTensor(gate.data_type, gate.shape, bytes, nullptr);
  (*this)(gate, value, alpha, output);
  return output;
}

void SwiGLUKernel::operator()(const Tensor &gate, const Tensor &value, float alpha,
                              Tensor &output) const {
  ValidateTensors(gate, value, output);
  const std::size_t count = static_cast<std::size_t>(gate.element_count());
  switch (static_cast<DataType>(gate.data_type)) {
  case DataType::FLOAT:
    SwiGLUFloat32(gate.AsFloat(), value.AsFloat(), output.AsFloat(), count, alpha);
    return;
  case DataType::DOUBLE:
    SwiGLUFloat64(gate.AsDouble(), value.AsDouble(), output.AsDouble(), count,
                  static_cast<double>(alpha));
    return;
  case DataType::FLOAT16:
    SwiGLUFloat16(reinterpret_cast<const std::uint16_t *>(gate.bytes()),
                  reinterpret_cast<const std::uint16_t *>(value.bytes()),
                  reinterpret_cast<std::uint16_t *>(output.mutable_bytes()), count, alpha);
    return;
  case DataType::BFLOAT16:
    SwiGLUBFloat16(reinterpret_cast<const std::uint16_t *>(gate.bytes()),
                   reinterpret_cast<const std::uint16_t *>(value.bytes()),
                   reinterpret_cast<std::uint16_t *>(output.mutable_bytes()), count, alpha);
    return;
  default:
    throw std::invalid_argument(
        "onnx_light_cpu::SwiGLU: only FLOAT, DOUBLE, FLOAT16 and BFLOAT16 are supported.");
  }
}

void SwiGLUKernel::Run(RuntimeContext &rt) {
  RecordKernelUsage(kName);
  const NodeProto &node = *node_;
  rt_ns::RequireInputCount(node, 2);
  rt_ns::RequireOutputCount(node, 1);
  const float alpha = rt_ns::GetAttributeFloatOrDefault(node, "alpha", 1.0f);
  const Tensor &gate = rt_ns::GetInput(node, 0, rt.tensors());
  const Tensor &value = rt_ns::GetInput(node, 1, rt.tensors());
  rt_ns::SetOutput(node, 0, (*this)(gate, value, alpha, &rt), rt);
}

void RegisterSwiGLUKernel() {
  rt_ns::NodeKernelFn factory = [](const NodeProto &node,
                                   RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    auto kernel = std::make_unique<SwiGLUKernel>(rt.kernel_ctx());
    kernel->set_node(node);
    return kernel;
  };
  KernelRegistration info;
  info.domain = "";
  info.op_type = "SwiGLU";
  info.device = sym_ns::Device::kCPU;
  info.kernel_name = SwiGLUKernel::kName;
  info.types = {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16};
  info.since_version = 28;
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
