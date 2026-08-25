// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/logical/not_kernel.h"

#include "onnx_light_cpu/impl/logical/logical_kernels.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/kernels/parallel_for.h"
#include "onnx_core/runtime/tuning/kernel_tuning.h"
#include "onnx_core/symbolic/sym_tensor.h"

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

rt_ns::KernelTuningKey MakeTuningKey() {
  return {"onnx_light_cpu",     "Not",
          "simd_dispatch",      static_cast<int32_t>(DataType::BOOL),
          sym_ns::Device::kCPU, NotKernel::kTuningAbi};
}

void ValidateTuning(const rt_ns::KernelTuningParameters &parameters) {
  for (const char *name : {kParallelThresholdBytes, kTargetBlockBytes}) {
    const int64_t value = parameters.Get<int64_t>(name);
    if (value < 0 || static_cast<uint64_t>(value) > std::numeric_limits<std::size_t>::max()) {
      throw std::invalid_argument(std::string("Not ") + name +
                                  " must be zero or a positive value representable by size_t.");
    }
  }
  if (parameters.Get<int64_t>(kTargetBlockBytes) == 0) {
    throw std::invalid_argument("Not parallel.target_block_bytes must be positive.");
  }
}

rt_ns::KernelTuningParameters MakeTuningDefaults() {
  const NotExecutionTuning defaults;
  return {MakeTuningKey(),
          {{kParallelThresholdBytes, static_cast<int64_t>(defaults.parallel_threshold_bytes)},
           {kTargetBlockBytes, static_cast<int64_t>(defaults.target_block_bytes)}}};
}

} // namespace

NotKernel::NotKernel(const NodeProto &node, const rt_ns::KernelContext &ctx) : KernelBase(ctx) {
  set_node(node);
}

void NotKernel::RegisterTuningSchemas() {
  static std::once_flag once;
  std::call_once(once, [] {
    rt_ns::RegisterKernelTuningSchema(
        rt_ns::KernelTuningSchema(MakeTuningDefaults(), ValidateTuning));
  });
}

rt_ns::KernelTuningKey NotKernel::TuningKey(int32_t element_type) const {
  return element_type == static_cast<int32_t>(DataType::BOOL) ? MakeTuningKey()
                                                              : rt_ns::KernelTuningKey{};
}

void NotKernel::Configure(const rt_ns::KernelTuningParameters &parameters) {
  if (parameters.key != MakeTuningKey()) {
    throw std::invalid_argument("Not tuning parameters have an incompatible key.");
  }
  ValidateTuning(parameters);
  tuning_ = {
      static_cast<std::size_t>(parameters.Get<int64_t>(kParallelThresholdBytes)),
      static_cast<std::size_t>(parameters.Get<int64_t>(kTargetBlockBytes)),
  };
}

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
  NotBoolWithTuning(px, py, static_cast<std::size_t>(n), tuning_);
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
  NotKernel::RegisterTuningSchemas();
  NodeKernelFn factory = [](const NodeProto &node,
                            RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    return std::make_unique<NotKernel>(node, rt.kernel_ctx());
  };
  // Empty domain -> normalised to the default ONNX domain, overriding the
  // built-in Not entry with the SIMD-accelerated kernel for the CPU device.
  KernelRegistration info;
  info.domain = "";
  info.op_type = "Not";
  info.device = sym_ns::Device::kCPU;
  info.kernel_name = NotKernel::kName;
  info.types = {DataType::BOOL};
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
