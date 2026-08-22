// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/math/exp_log_kernel.h"

#include "onnx_light_cpu/impl/math/math_kernels.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
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

using Float32Fn = void (*)(const float *, float *, std::size_t);
using Float64Fn = void (*)(const double *, double *, std::size_t);
using Float16Fn = void (*)(const std::uint16_t *, std::uint16_t *, std::size_t);

// Shared implementation for the elementwise unary float math kernels
// (``Exp``/``Log``). ``scalar`` is used for the ``bfloat16`` fallback and to
// name the kernel in error messages.
void ComputeUnary(const Tensor &x, Tensor &output, const char *kernel_name, Float32Fn f32,
                  Float64Fn f64, Float16Fn f16, float (*scalar)(float)) {
  if (output.data_type != x.data_type) {
    throw std::invalid_argument(std::string("onnx_light_cpu::") + kernel_name +
                                ": output dtype must match input dtype.");
  }
  if (output.shape != x.shape) {
    throw std::invalid_argument(std::string("onnx_light_cpu::") + kernel_name +
                                ": output shape must match input shape.");
  }
  if (output.size_bytes() != x.size_bytes()) {
    throw std::invalid_argument(std::string("onnx_light_cpu::") + kernel_name +
                                ": output buffer size mismatch.");
  }
  const std::int64_t n = x.element_count();
  switch (static_cast<DataType>(x.data_type)) {
  case DataType::FLOAT: {
    const float *px = x.AsFloat();
    float *py = output.AsFloat();
    // When onnx-light has installed a session ``CpuExecutor`` on the calling
    // thread, the SIMD implementation can split this range through it without
    // another scheduler.
    f32(px, py, static_cast<std::size_t>(n));
    return;
  }
  case DataType::DOUBLE: {
    const double *px = x.AsDouble();
    double *py = output.AsDouble();
    f64(px, py, static_cast<std::size_t>(n));
    return;
  }
  case DataType::FLOAT16: {
    const std::uint16_t *px = reinterpret_cast<const std::uint16_t *>(x.bytes());
    std::uint16_t *py = reinterpret_cast<std::uint16_t *>(output.mutable_bytes());
    f16(px, py, static_cast<std::size_t>(n));
    return;
  }
  case DataType::BFLOAT16: {
    const std::uint16_t *px = reinterpret_cast<const std::uint16_t *>(x.bytes());
    std::uint16_t *py = reinterpret_cast<std::uint16_t *>(output.mutable_bytes());
    rt_ns::ParallelFor(n, [px, py, scalar](std::int64_t begin, std::int64_t end) {
      for (std::int64_t i = begin; i < end; ++i) {
        py[i] = rt_ns::FloatToBfloat16Bits(scalar(rt_ns::Bfloat16BitsToFloat(px[i])));
      }
    });
    return;
  }
  default:
    throw std::invalid_argument(std::string("onnx_light_cpu::") + kernel_name +
                                ": unsupported data type " + std::to_string(x.data_type) +
                                ", only FLOAT, DOUBLE, FLOAT16 and BFLOAT16 are supported.");
  }
}

Tensor MakeLike(const Tensor &x, RuntimeContext *rt) {
  const std::size_t y_n_bytes = static_cast<std::size_t>(x.element_count()) * x.element_size();
  return rt != nullptr ? rt->MakeOutputTensor(0, x.data_type, x.shape, y_n_bytes)
                       : rt_ns::MakeOutputTensor(x.data_type, x.shape, y_n_bytes, nullptr);
}

float ScalarExp(float v) { return std::exp(v); }
float ScalarLog(float v) { return std::log(v); }

} // namespace

Tensor ExpKernel::operator()(const Tensor &x, RuntimeContext *rt) const {
  Tensor y = MakeLike(x, rt);
  (*this)(x, y);
  return y;
}

void ExpKernel::operator()(const Tensor &x, Tensor &output) const {
  ComputeUnary(x, output, "ExpKernel", &ExpFloat32, &ExpFloat64, &ExpFloat16, &ScalarExp);
}

void ExpKernel::Run(RuntimeContext &rt) {
  RecordKernelUsage(kName);
  const NodeProto &node = *node_;
  rt_ns::RequireInputCount(node, 1);
  rt_ns::RequireOutputCount(node, 1);
  const Tensor &x = rt_ns::GetInput(node, 0, rt.tensors());
  rt_ns::SetOutput(node, 0, (*this)(x, &rt), rt);
}

Tensor LogKernel::operator()(const Tensor &x, RuntimeContext *rt) const {
  Tensor y = MakeLike(x, rt);
  (*this)(x, y);
  return y;
}

void LogKernel::operator()(const Tensor &x, Tensor &output) const {
  ComputeUnary(x, output, "LogKernel", &LogFloat32, &LogFloat64, &LogFloat16, &ScalarLog);
}

void LogKernel::Run(RuntimeContext &rt) {
  RecordKernelUsage(kName);
  const NodeProto &node = *node_;
  rt_ns::RequireInputCount(node, 1);
  rt_ns::RequireOutputCount(node, 1);
  const Tensor &x = rt_ns::GetInput(node, 0, rt.tensors());
  rt_ns::SetOutput(node, 0, (*this)(x, &rt), rt);
}

void RegisterExpKernel() {
  NodeKernelFn factory = [](const NodeProto &node,
                            RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    auto kernel = std::make_unique<ExpKernel>(rt.kernel_ctx());
    kernel->set_node(node);
    return kernel;
  };
  // Empty domain -> normalised to the default ONNX domain, overriding the
  // built-in Exp entry with the SIMD-accelerated kernel for the CPU device.
  KernelRegistration info;
  info.domain = "";
  info.op_type = "Exp";
  info.device = sym_ns::Device::kCPU;
  info.kernel_name = ExpKernel::kName;
  info.types = {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16};
  RegisterKernel(std::move(info), std::move(factory));
}

void RegisterLogKernel() {
  NodeKernelFn factory = [](const NodeProto &node,
                            RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    auto kernel = std::make_unique<LogKernel>(rt.kernel_ctx());
    kernel->set_node(node);
    return kernel;
  };
  // Empty domain -> normalised to the default ONNX domain, overriding the
  // built-in Log entry with the SIMD-accelerated kernel for the CPU device.
  KernelRegistration info;
  info.domain = "";
  info.op_type = "Log";
  info.device = sym_ns::Device::kCPU;
  info.kernel_name = LogKernel::kName;
  info.types = {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16};
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
