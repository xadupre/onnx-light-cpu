// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/math/abs_kernel.h"

#include "onnx_light_cpu/impl/math/math_kernels.h"
#include "onnx_light_cpu/kernels/kernel_registry.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/kernels/parallel_for.h"
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

namespace {

// Runs the elementwise SIMD kernel ``Fn`` over ``[0, n)``. ``Fn`` (defined in
// ``onnx_light_cpu/impl/math/math_kernels.h``) already parallelizes internally
// via onnx-light-cpu's own thread pool (see ``onnx_light_cpu/impl/parallel_for.h``),
// so it is called directly on the whole range here; wrapping it in another
// ``ParallelFor`` (onnx-light's core thread pool) would nest two independent,
// non-cooperating thread pools and oversubscribe the CPU instead of speeding
// things up.
template <typename T, void (*Fn)(const T *, T *, std::size_t)>
void RunParallel(const T *input, T *output, std::int64_t n) {
  Fn(input, output, static_cast<std::size_t>(n));
}

} // namespace

rt_ns::Tensor AbsKernel::operator()(const Tensor &x, RuntimeContext *rt) const {
  const std::size_t y_n_bytes = static_cast<std::size_t>(x.element_count()) * x.element_size();
  Tensor y = rt != nullptr ? rt->MakeOutputTensor(0, x.data_type, x.shape, y_n_bytes)
                           : rt_ns::MakeOutputTensor(x.data_type, x.shape, y_n_bytes, nullptr);
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
  switch (static_cast<DataType>(x.data_type)) {
  case DataType::FLOAT:
    RunParallel<float, AbsFloat32>(x.AsFloat(), output.AsFloat(), n);
    return;
  case DataType::DOUBLE:
    RunParallel<double, AbsFloat64>(x.AsDouble(), output.AsDouble(), n);
    return;
  case DataType::INT32:
    RunParallel<std::int32_t, AbsInt32>(x.AsInt32(), output.AsInt32(), n);
    return;
  case DataType::INT64:
    RunParallel<std::int64_t, AbsInt64>(x.AsInt64(), output.AsInt64(), n);
    return;
  case DataType::FLOAT16: {
    const std::uint16_t *px = reinterpret_cast<const std::uint16_t *>(x.bytes());
    std::uint16_t *py = reinterpret_cast<std::uint16_t *>(output.mutable_bytes());
    RunParallel<std::uint16_t, AbsFloat16>(px, py, n);
    return;
  }
  case DataType::BFLOAT16: {
    const std::uint16_t *px = reinterpret_cast<const std::uint16_t *>(x.bytes());
    std::uint16_t *py = reinterpret_cast<std::uint16_t *>(output.mutable_bytes());
    rt_ns::ParallelFor(n, [px, py](std::int64_t begin, std::int64_t end) {
      for (std::int64_t i = begin; i < end; ++i) {
        py[i] = rt_ns::FloatToBfloat16Bits(std::fabs(rt_ns::Bfloat16BitsToFloat(px[i])));
      }
    });
    return;
  }
  case DataType::INT8:
    RunParallel<std::int8_t, AbsInt8>(x.AsInt8(), output.AsInt8(), n);
    return;
  case DataType::INT16: {
    const std::int16_t *px = x.AsInt16();
    std::int16_t *py = output.AsInt16();
    rt_ns::ParallelFor(n, [px, py](std::int64_t begin, std::int64_t end) {
      for (std::int64_t i = begin; i < end; ++i) {
        const std::int32_t v = static_cast<std::int32_t>(px[i]);
        py[i] = static_cast<std::int16_t>(v < 0 ? -v : v);
      }
    });
    return;
  }
  default:
    throw std::invalid_argument(
        "onnx_light_cpu::AbsKernel: unsupported data type " + std::to_string(x.data_type) +
        ", only FLOAT, DOUBLE, FLOAT16, BFLOAT16, INT8, INT16, INT32 and INT64 are supported.");
  }
}

void AbsKernel::Run(RuntimeContext &rt) {
  RecordKernelUsage(kName);
  const NodeProto &node = *node_;
  rt_ns::RequireInputCount(node, 1);
  rt_ns::RequireOutputCount(node, 1);
  const Tensor &x = rt_ns::GetInput(node, 0, rt.tensors());
  rt_ns::SetOutput(node, 0, (*this)(x, &rt), rt);
}

void RegisterAbsKernel() {
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

ONNX_LIGHT_CPU_REGISTER_KERNEL("Abs", AbsKernel::kName, RegisterAbsKernel)

} // namespace onnx_light_cpu
