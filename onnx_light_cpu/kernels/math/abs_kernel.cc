// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/math/abs_kernel.h"

#include "onnx_light_cpu/impl/math/math_kernels.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/tuning/kernel_tuning.h"
#include "onnx_core/symbolic/sym_tensor.h"

#include <array>
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
constexpr const char *kMaxParticipants = "parallel.max_participants";
constexpr const char *kPreferredParticipants = "parallel.preferred_participants";
constexpr const char *kCostModel = "parallel.cost_model";
constexpr const char *kStreamingStoreThresholdBytes = "memory.streaming_store_threshold_bytes";
constexpr std::array<DataType, 8> kSupportedTypes = {
    DataType::FLOAT,   DataType::DOUBLE,   DataType::INT32, DataType::INT64,
    DataType::FLOAT16, DataType::BFLOAT16, DataType::INT8,  DataType::INT16};

bool SupportsElementType(int32_t element_type) {
  for (DataType type : kSupportedTypes) {
    if (element_type == static_cast<int32_t>(type)) {
      return true;
    }
  }
  return false;
}

rt_ns::KernelTuningKey MakeTuningKey(int32_t element_type) {
  return {"onnx_light_cpu",     "Abs", "simd_dispatch", element_type, sym_ns::Device::kCPU,
          AbsKernel::kTuningAbi};
}

void ValidateTuning(const rt_ns::KernelTuningParameters &parameters) {
  for (const char *name : {kParallelThresholdBytes, kTargetBlockBytes, kMaxParticipants,
                           kPreferredParticipants, kStreamingStoreThresholdBytes}) {
    const int64_t value = parameters.Get<int64_t>(name);
    if (value < 0 || static_cast<uint64_t>(value) > std::numeric_limits<std::size_t>::max()) {
      throw std::invalid_argument(std::string("Abs ") + name +
                                  " must be zero or a positive value representable by size_t.");
    }
  }
  if (parameters.Get<int64_t>(kTargetBlockBytes) == 0) {
    throw std::invalid_argument("Abs parallel.target_block_bytes must be positive.");
  }
  const int64_t cost_model = parameters.Get<int64_t>(kCostModel);
  if (cost_model != 0 && cost_model != 1) {
    throw std::invalid_argument("Abs parallel.cost_model must be 0 or 1.");
  }
}

const UnaryExecutionTuning &DefaultTuning(int32_t element_type) {
  switch (static_cast<DataType>(element_type)) {
  case DataType::FLOAT:
    return kDefaultAbsFloat32ExecutionTuning;
  case DataType::DOUBLE:
  case DataType::INT64:
    return kDefaultAbs64ExecutionTuning;
  case DataType::FLOAT16:
  case DataType::BFLOAT16:
  case DataType::INT16:
    return kDefaultAbs16ExecutionTuning;
  case DataType::INT8:
    return kDefaultAbs8ExecutionTuning;
  default:
    return kDefaultAbs32ExecutionTuning;
  }
}

rt_ns::KernelTuningParameters MakeTuningDefaults(int32_t element_type) {
  const UnaryExecutionTuning &defaults = DefaultTuning(element_type);
  return {MakeTuningKey(element_type),
          {{kParallelThresholdBytes, static_cast<int64_t>(defaults.parallel_threshold_bytes)},
           {kTargetBlockBytes, static_cast<int64_t>(defaults.target_block_bytes)},
           {kMaxParticipants, static_cast<int64_t>(defaults.max_participants)},
           {kPreferredParticipants, static_cast<int64_t>(defaults.preferred_participants)},
           {kStreamingStoreThresholdBytes,
            static_cast<int64_t>(defaults.streaming_store_threshold_bytes)},
           {kCostModel, defaults.use_cost_model ? int64_t{1} : int64_t{0}}}};
}

} // namespace

AbsKernel::AbsKernel(const NodeProto &node, const rt_ns::KernelContext &ctx) : KernelBase(ctx) {
  set_node(node);
}

void AbsKernel::RegisterTuningSchemas() {
  static std::once_flag once;
  std::call_once(once, [] {
    for (DataType type : kSupportedTypes) {
      rt_ns::RegisterKernelTuningSchema(rt_ns::KernelTuningSchema(
          MakeTuningDefaults(static_cast<int32_t>(type)), ValidateTuning));
    }
  });
}

rt_ns::KernelTuningKey AbsKernel::TuningKey(int32_t element_type) const {
  return SupportsElementType(element_type) ? MakeTuningKey(element_type) : rt_ns::KernelTuningKey{};
}

void AbsKernel::Configure(const rt_ns::KernelTuningParameters &parameters) {
  if (!SupportsElementType(parameters.key.element_type) ||
      parameters.key != MakeTuningKey(parameters.key.element_type)) {
    throw std::invalid_argument("Abs tuning parameters have an incompatible key.");
  }
  ValidateTuning(parameters);
  tuning_ = {
      static_cast<std::size_t>(parameters.Get<int64_t>(kParallelThresholdBytes)),
      static_cast<std::size_t>(parameters.Get<int64_t>(kTargetBlockBytes)),
      static_cast<std::size_t>(parameters.Get<int64_t>(kMaxParticipants)),
      parameters.Get<int64_t>(kCostModel) != 0,
      static_cast<std::size_t>(parameters.Get<int64_t>(kPreferredParticipants)),
      static_cast<std::size_t>(parameters.Get<int64_t>(kStreamingStoreThresholdBytes)),
  };
  tuning_configured_ = true;
}

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
  const UnaryExecutionTuning &tuning = tuning_configured_ ? tuning_ : DefaultTuning(x.data_type);
  switch (static_cast<DataType>(x.data_type)) {
  case DataType::FLOAT:
    AbsFloat32WithTuning(x.AsFloat(), output.AsFloat(), static_cast<std::size_t>(n), tuning);
    return;
  case DataType::DOUBLE:
    AbsFloat64WithTuning(x.AsDouble(), output.AsDouble(), static_cast<std::size_t>(n), tuning);
    return;
  case DataType::INT32:
    AbsInt32WithTuning(x.AsInt32(), output.AsInt32(), static_cast<std::size_t>(n), tuning);
    return;
  case DataType::INT64:
    AbsInt64WithTuning(x.AsInt64(), output.AsInt64(), static_cast<std::size_t>(n), tuning);
    return;
  case DataType::FLOAT16: {
    const std::uint16_t *px = reinterpret_cast<const std::uint16_t *>(x.bytes());
    std::uint16_t *py = reinterpret_cast<std::uint16_t *>(output.mutable_bytes());
    AbsFloat16WithTuning(px, py, static_cast<std::size_t>(n), tuning);
    return;
  }
  case DataType::BFLOAT16: {
    const std::uint16_t *px = reinterpret_cast<const std::uint16_t *>(x.bytes());
    std::uint16_t *py = reinterpret_cast<std::uint16_t *>(output.mutable_bytes());
    AbsFloat16WithTuning(px, py, static_cast<std::size_t>(n), tuning);
    return;
  }
  case DataType::INT8:
    AbsInt8WithTuning(x.AsInt8(), output.AsInt8(), static_cast<std::size_t>(n), tuning);
    return;
  case DataType::INT16: {
    AbsInt16WithTuning(x.AsInt16(), output.AsInt16(), static_cast<std::size_t>(n), tuning);
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
  AbsKernel::RegisterTuningSchemas();
  NodeKernelFn factory = [](const NodeProto &node,
                            RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    return std::make_unique<AbsKernel>(node, rt.kernel_ctx());
  };
  // Empty domain -> normalised to the default ONNX domain, overriding the
  // built-in Abs entry with the SIMD-accelerated kernel for the CPU device.
  KernelRegistration info;
  info.domain = "";
  info.op_type = "Abs";
  info.device = sym_ns::Device::kCPU;
  info.kernel_name = AbsKernel::kName;
  info.types = {DataType::FLOAT,   DataType::DOUBLE,   DataType::INT32, DataType::INT64,
                DataType::FLOAT16, DataType::BFLOAT16, DataType::INT8,  DataType::INT16};
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
