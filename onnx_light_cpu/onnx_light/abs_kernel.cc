// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/onnx_light/abs_kernel.h"

#include "onnx_light_cpu/cpu_kernels.h"

#include "onnx_core/runtime/cast_helper.h"
#include "onnx_core/runtime/kernel_dispatch_table.h"
#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/symbolic/sym_tensor.h"

#include <cmath>
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

rt_ns::Tensor AbsKernel::operator()(const Tensor &x, RuntimeContext *rt) const {
  const std::size_t y_n_bytes = static_cast<std::size_t>(x.element_count()) * x.element_size();
  Tensor y = rt_ns::MakeOutputTensor(x.data_type, x.shape, y_n_bytes,
                                     rt != nullptr ? rt->allocator() : nullptr);
  (*this)(x, y);
  return y;
}

void AbsKernel::operator()(const Tensor &x, Tensor &output) const {
  if (output.data_type != x.data_type) {
    throw std::invalid_argument("onnx_light_cpu::AbsKernel: output dtype must match input dtype.");
  }
  if (output.shape != x.shape) {
    throw std::invalid_argument("onnx_light_cpu::AbsKernel: output shape must match input shape.");
  }
  if (output.size_bytes() != x.size_bytes()) {
    throw std::invalid_argument("onnx_light_cpu::AbsKernel: output buffer size mismatch.");
  }
  const std::int64_t n = x.element_count();
  const std::size_t count = static_cast<std::size_t>(n);
  switch (static_cast<DataType>(x.data_type)) {
  case DataType::FLOAT:
    AbsFloat32(x.AsFloat(), output.AsFloat(), count);
    return;
  case DataType::DOUBLE:
    AbsFloat64(x.AsDouble(), output.AsDouble(), count);
    return;
  case DataType::INT32:
    AbsInt32(x.AsInt32(), output.AsInt32(), count);
    return;
  case DataType::INT64:
    AbsInt64(x.AsInt64(), output.AsInt64(), count);
    return;
  case DataType::FLOAT16: {
    const std::uint16_t *px = reinterpret_cast<const std::uint16_t *>(x.bytes());
    std::uint16_t *py = reinterpret_cast<std::uint16_t *>(output.mutable_bytes());
    AbsFloat16(px, py, count);
    return;
  }
  case DataType::BFLOAT16: {
    const std::uint16_t *px = reinterpret_cast<const std::uint16_t *>(x.bytes());
    std::uint16_t *py = reinterpret_cast<std::uint16_t *>(output.mutable_bytes());
    for (std::int64_t i = 0; i < n; ++i) {
      py[i] = rt_ns::FloatToBfloat16Bits(std::fabs(rt_ns::Bfloat16BitsToFloat(px[i])));
    }
    return;
  }
  case DataType::INT8:
    AbsInt8(x.AsInt8(), output.AsInt8(), count);
    return;
  case DataType::INT16: {
    const std::int16_t *px = x.AsInt16();
    std::int16_t *py = output.AsInt16();
    for (std::int64_t i = 0; i < n; ++i) {
      const std::int32_t v = static_cast<std::int32_t>(px[i]);
      py[i] = static_cast<std::int16_t>(v < 0 ? -v : v);
    }
    return;
  }
  default:
    throw std::invalid_argument(
        "onnx_light_cpu::AbsKernel: unsupported data type " + std::to_string(x.data_type) +
        ", only FLOAT, DOUBLE, FLOAT16, BFLOAT16, INT8, INT16, INT32 and INT64 are supported.");
  }
}

void AbsKernel::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  rt_ns::RequireInputCount(node, 1);
  rt_ns::RequireOutputCount(node, 1);
  const Tensor &x = rt_ns::GetInput(node, 0, rt.tensors());
  rt_ns::SetOutput(node, 0, (*this)(x, &rt), rt);
}

void RegisterKernels() {
  NodeKernelFn factory = [](const NodeProto &node,
                            RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    auto kernel = std::make_unique<AbsKernel>(rt.kernel_ctx());
    kernel->set_node(node);
    return kernel;
  };
  // Empty domain -> normalised to the default ONNX domain, overriding the
  // built-in Abs entry with the SIMD-accelerated kernel for the CPU device.
  rt_ns::RegisterKernelFn("", "Abs", sym_ns::Device::kCPU, std::move(factory));
}

} // namespace onnx_light_cpu
