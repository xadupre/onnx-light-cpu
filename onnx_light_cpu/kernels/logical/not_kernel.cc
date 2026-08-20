// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/logical/not_kernel.h"

#include "onnx_light_cpu/impl/logical/logical_kernels.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/kernels/parallel_for.h"
#include "onnx_core/symbolic/sym_tensor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
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

Tensor NotKernel::operator()(const Tensor &x, RuntimeContext *rt) const {
  const std::size_t y_n_bytes = static_cast<std::size_t>(x.element_count()) * x.element_size();
  Tensor y = rt != nullptr ? rt->MakeOutputTensor(0, x.data_type, x.shape, y_n_bytes)
                           : rt_ns::MakeOutputTensor(x.data_type, x.shape, y_n_bytes, nullptr);
  (*this)(x, y);
  return y;
}

void NotKernel::operator()(const Tensor &x, Tensor &output) const {
  if (output.data_type != x.data_type) {
    throw std::invalid_argument("onnx_light_cpu::NotKernel: output dtype must match input dtype.");
  }
  if (output.shape != x.shape) {
    throw std::invalid_argument("onnx_light_cpu::NotKernel: output shape must match input shape.");
  }
  if (output.size_bytes() != x.size_bytes()) {
    throw std::invalid_argument("onnx_light_cpu::NotKernel: output buffer size mismatch.");
  }
  if (static_cast<DataType>(x.data_type) != DataType::BOOL) {
    throw std::invalid_argument("onnx_light_cpu::NotKernel: unsupported data type " +
                                std::to_string(x.data_type) + ", only BOOL is supported.");
  }
  const std::int64_t n = x.element_count();
  const std::uint8_t *px = x.AsBool();
  std::uint8_t *py = output.AsBool();
  // When onnx-light has installed a session ``CpuExecutor`` on the calling
  // thread, ``NotBool`` can split this range through it without an
  // onnx-light-cpu scheduler.
  NotBool(px, py, static_cast<std::size_t>(n));
}

void NotKernel::Run(RuntimeContext &rt) {
  RecordKernelUsage(kName);
  const NodeProto &node = *node_;
  rt_ns::RequireInputCount(node, 1);
  rt_ns::RequireOutputCount(node, 1);
  const Tensor &x = rt_ns::GetInput(node, 0, rt.tensors());
  rt_ns::SetOutput(node, 0, (*this)(x, &rt), rt);
}

void RegisterNotKernel() {
  NodeKernelFn factory = [](const NodeProto &node,
                            RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    auto kernel = std::make_unique<NotKernel>(rt.kernel_ctx());
    kernel->set_node(node);
    return kernel;
  };
  // Empty domain -> normalised to the default ONNX domain, overriding the
  // built-in Not entry with the SIMD-accelerated kernel for the CPU device.
  rt_ns::RegisterKernelFn("", "Not", sym_ns::Device::kCPU, std::move(factory));
}

} // namespace onnx_light_cpu
