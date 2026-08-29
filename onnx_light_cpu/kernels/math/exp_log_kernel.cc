// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/math/exp_log_kernel.h"

#include "onnx_light_cpu/impl/math/math_kernels.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"

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
#include <string_view>

namespace onnx_light_cpu {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

using NodeProto = ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::DataType;
using rt_ns::NodeKernelFn;
using rt_ns::RuntimeContext;
using rt_ns::Tensor;

namespace {

using Float32Fn = void (*)(const float *, float *, std::size_t, const UnaryExecutionTuning &);
using Float64Fn = void (*)(const double *, double *, std::size_t, const UnaryExecutionTuning &);
using Float16Fn = void (*)(const std::uint16_t *, std::uint16_t *, std::size_t,
                           const UnaryExecutionTuning &);

constexpr const char *kParallelThresholdBytes = "parallel.threshold_bytes";
constexpr const char *kTargetBlockBytes = "parallel.target_block_bytes";
constexpr const char *kMaxParticipants = "parallel.max_participants";
constexpr const char *kCostModel = "parallel.cost_model";
constexpr std::array<DataType, 4> kSupportedTypes = {DataType::FLOAT, DataType::DOUBLE,
                                                     DataType::FLOAT16, DataType::BFLOAT16};

bool SupportsElementType(int32_t element_type) {
  for (DataType type : kSupportedTypes) {
    if (element_type == static_cast<int32_t>(type)) {
      return true;
    }
  }
  return false;
}

const UnaryExecutionTuning &DefaultTuning(std::string_view op_type, int32_t element_type) {
  if (op_type == "Log" && element_type == static_cast<int32_t>(DataType::FLOAT16)) {
    return kDefaultLogFloat16ExecutionTuning;
  }
  if (op_type == "Log" && element_type == static_cast<int32_t>(DataType::BFLOAT16)) {
    return kDefaultLogBFloat16ExecutionTuning;
  }
  return element_type == static_cast<int32_t>(DataType::FLOAT16) ||
                 element_type == static_cast<int32_t>(DataType::BFLOAT16)
             ? kDefaultExpLogHalfExecutionTuning
             : kDefaultExpLogExecutionTuning;
}

rt_ns::KernelTuningKey MakeTuningKey(const char *op_type, int32_t element_type,
                                     std::uint32_t tuning_abi) {
  return {"onnx_light_cpu",     op_type,   "simd_dispatch", element_type,
          sym_ns::Device::kCPU, tuning_abi};
}

void ValidateTuning(const rt_ns::KernelTuningParameters &parameters) {
  for (const char *name : {kParallelThresholdBytes, kTargetBlockBytes, kMaxParticipants}) {
    const int64_t value = parameters.Get<int64_t>(name);
    if (value < 0 || static_cast<uint64_t>(value) > std::numeric_limits<std::size_t>::max()) {
      throw std::invalid_argument(std::string("Unary ") + name +
                                  " must be zero or a positive value representable by size_t.");
    }
  }
  if (parameters.Get<int64_t>(kTargetBlockBytes) == 0) {
    throw std::invalid_argument("Unary parallel.target_block_bytes must be positive.");
  }
  const int64_t cost_model = parameters.Get<int64_t>(kCostModel);
  if (cost_model != 0 && cost_model != 1) {
    throw std::invalid_argument("Unary parallel.cost_model must be 0 or 1.");
  }
}

rt_ns::KernelTuningParameters MakeTuningDefaults(const char *op_type, int32_t element_type,
                                                 std::uint32_t tuning_abi) {
  const UnaryExecutionTuning &defaults = DefaultTuning(op_type, element_type);
  return {MakeTuningKey(op_type, element_type, tuning_abi),
          {{kParallelThresholdBytes, static_cast<int64_t>(defaults.parallel_threshold_bytes)},
           {kTargetBlockBytes, static_cast<int64_t>(defaults.target_block_bytes)},
           {kMaxParticipants, static_cast<int64_t>(defaults.max_participants)},
           {kCostModel, defaults.use_cost_model ? int64_t{1} : int64_t{0}}}};
}

UnaryExecutionTuning ReadTuning(const rt_ns::KernelTuningParameters &parameters) {
  return {
      static_cast<std::size_t>(parameters.Get<int64_t>(kParallelThresholdBytes)),
      static_cast<std::size_t>(parameters.Get<int64_t>(kTargetBlockBytes)),
      static_cast<std::size_t>(parameters.Get<int64_t>(kMaxParticipants)),
      parameters.Get<int64_t>(kCostModel) != 0,
  };
}

// Shared implementation for the elementwise unary float math kernels
// (``Exp``/``Log``). ``scalar`` is used for the ``bfloat16`` fallback and to
// name the kernel in error messages.
void ComputeUnary(const Tensor &x, Tensor &output, const char *kernel_name, Float32Fn f32,
                  Float64Fn f64, Float16Fn f16, Float16Fn bf16,
                  const UnaryExecutionTuning &tuning) {
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
    f32(px, py, static_cast<std::size_t>(n), tuning);
    return;
  }
  case DataType::DOUBLE: {
    const double *px = x.AsDouble();
    double *py = output.AsDouble();
    f64(px, py, static_cast<std::size_t>(n), tuning);
    return;
  }
  case DataType::FLOAT16: {
    const std::uint16_t *px = reinterpret_cast<const std::uint16_t *>(x.bytes());
    std::uint16_t *py = reinterpret_cast<std::uint16_t *>(output.mutable_bytes());
    f16(px, py, static_cast<std::size_t>(n), tuning);
    return;
  }
  case DataType::BFLOAT16: {
    const std::uint16_t *px = reinterpret_cast<const std::uint16_t *>(x.bytes());
    std::uint16_t *py = reinterpret_cast<std::uint16_t *>(output.mutable_bytes());
    bf16(px, py, static_cast<std::size_t>(n), tuning);
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

} // namespace

struct ExpKernel::Tuning {
  UnaryExecutionTuning value;
};

struct LogKernel::Tuning {
  UnaryExecutionTuning value;
};

ExpKernel::ExpKernel(const NodeProto &node, const rt_ns::KernelContext &ctx) : KernelBase(ctx) {
  set_node(node);
}

ExpKernel::~ExpKernel() = default;

void ExpKernel::RegisterTuningSchemas() {
  static std::once_flag once;
  std::call_once(once, [] {
    for (DataType type : kSupportedTypes) {
      rt_ns::RegisterKernelTuningSchema(rt_ns::KernelTuningSchema(
          MakeTuningDefaults("Exp", static_cast<int32_t>(type), kTuningAbi), ValidateTuning));
    }
  });
}

rt_ns::KernelTuningKey ExpKernel::TuningKey(int32_t element_type) const {
  return SupportsElementType(element_type) ? MakeTuningKey("Exp", element_type, kTuningAbi)
                                           : rt_ns::KernelTuningKey{};
}

void ExpKernel::Configure(const rt_ns::KernelTuningParameters &parameters) {
  if (!SupportsElementType(parameters.key.element_type) ||
      parameters.key != MakeTuningKey("Exp", parameters.key.element_type, kTuningAbi)) {
    throw std::invalid_argument("Exp tuning parameters have an incompatible key.");
  }
  ValidateTuning(parameters);
  tuning_ = std::make_unique<Tuning>(Tuning{ReadTuning(parameters)});
  tuning_configured_ = true;
}

Tensor ExpKernel::operator()(const Tensor &x, RuntimeContext *rt) const {
  Tensor y = MakeLike(x, rt);
  (*this)(x, y);
  return y;
}

void ExpKernel::operator()(const Tensor &x, Tensor &output) const {
  const UnaryExecutionTuning &tuning =
      tuning_configured_ ? tuning_->value : DefaultTuning("Exp", x.data_type);
  ComputeUnary(x, output, "ExpKernel", &ExpFloat32WithTuning, &ExpFloat64WithTuning,
               &ExpFloat16WithTuning, &ExpBFloat16WithTuning, tuning);
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

LogKernel::LogKernel(const NodeProto &node, const rt_ns::KernelContext &ctx) : KernelBase(ctx) {
  set_node(node);
}

LogKernel::~LogKernel() = default;

void LogKernel::RegisterTuningSchemas() {
  static std::once_flag once;
  std::call_once(once, [] {
    for (DataType type : kSupportedTypes) {
      rt_ns::RegisterKernelTuningSchema(rt_ns::KernelTuningSchema(
          MakeTuningDefaults("Log", static_cast<int32_t>(type), kTuningAbi), ValidateTuning));
    }
  });
}

rt_ns::KernelTuningKey LogKernel::TuningKey(int32_t element_type) const {
  return SupportsElementType(element_type) ? MakeTuningKey("Log", element_type, kTuningAbi)
                                           : rt_ns::KernelTuningKey{};
}

void LogKernel::Configure(const rt_ns::KernelTuningParameters &parameters) {
  if (!SupportsElementType(parameters.key.element_type) ||
      parameters.key != MakeTuningKey("Log", parameters.key.element_type, kTuningAbi)) {
    throw std::invalid_argument("Log tuning parameters have an incompatible key.");
  }
  ValidateTuning(parameters);
  tuning_ = std::make_unique<Tuning>(Tuning{ReadTuning(parameters)});
  tuning_configured_ = true;
}

void LogKernel::operator()(const Tensor &x, Tensor &output) const {
  const UnaryExecutionTuning &tuning =
      tuning_configured_ ? tuning_->value : DefaultTuning("Log", x.data_type);
  ComputeUnary(x, output, "LogKernel", &LogFloat32WithTuning, &LogFloat64WithTuning,
               &LogFloat16WithTuning, &LogBFloat16WithTuning, tuning);
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
  ExpKernel::RegisterTuningSchemas();
  NodeKernelFn factory = [](const NodeProto &node,
                            RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    return std::make_unique<ExpKernel>(node, rt.kernel_ctx());
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
  LogKernel::RegisterTuningSchemas();
  NodeKernelFn factory = [](const NodeProto &node,
                            RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    return std::make_unique<LogKernel>(node, rt.kernel_ctx());
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
