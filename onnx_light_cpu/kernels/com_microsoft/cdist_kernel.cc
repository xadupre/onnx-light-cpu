// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/com_microsoft/cdist_kernel.h"

#include "onnx_light_cpu/impl/com_microsoft/cdist.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

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
constexpr std::array<DataType, 2> kSupportedTypes = {DataType::FLOAT, DataType::DOUBLE};

bool SupportsElementType(int32_t element_type) {
  for (DataType type : kSupportedTypes) {
    if (element_type == static_cast<int32_t>(type)) {
      return true;
    }
  }
  return false;
}

rt_ns::KernelTuningKey MakeTuningKey(int32_t element_type) {
  return {"onnx_light_cpu",       "CDist", "row_dispatch", element_type, sym_ns::Device::kCPU,
          CDistKernel::kTuningAbi};
}

void ValidateTuning(const rt_ns::KernelTuningParameters &parameters) {
  for (const char *name :
       {kParallelThresholdBytes, kTargetBlockBytes, kMaxParticipants, kPreferredParticipants}) {
    const int64_t value = parameters.Get<int64_t>(name);
    if (value < 0 || static_cast<uint64_t>(value) > std::numeric_limits<std::size_t>::max()) {
      throw std::invalid_argument(std::string("CDist ") + name +
                                  " must be zero or a positive value representable by size_t.");
    }
  }
  if (parameters.Get<int64_t>(kTargetBlockBytes) == 0) {
    throw std::invalid_argument("CDist parallel.target_block_bytes must be positive.");
  }
  const int64_t cost_model = parameters.Get<int64_t>(kCostModel);
  if (cost_model != 0 && cost_model != 1) {
    throw std::invalid_argument("CDist parallel.cost_model must be 0 or 1.");
  }
}

const CDistExecutionTuning &DefaultTuning(int32_t element_type) {
  switch (static_cast<DataType>(element_type)) {
  case DataType::DOUBLE:
    return kDefaultCDistFloat64ExecutionTuning;
  case DataType::FLOAT:
  default:
    return kDefaultCDistFloat32ExecutionTuning;
  }
}

rt_ns::KernelTuningParameters MakeTuningDefaults(int32_t element_type) {
  const CDistExecutionTuning &defaults = DefaultTuning(element_type);
  return {MakeTuningKey(element_type),
          {{kParallelThresholdBytes, static_cast<int64_t>(defaults.parallel_threshold_bytes)},
           {kTargetBlockBytes, static_cast<int64_t>(defaults.target_block_bytes)},
           {kMaxParticipants, static_cast<int64_t>(defaults.max_participants)},
           {kPreferredParticipants, static_cast<int64_t>(defaults.preferred_participants)},
           {kCostModel, defaults.use_cost_model ? int64_t{1} : int64_t{0}}}};
}

CDistMetric ParseMetric(const std::string &metric) {
  if (metric == "sqeuclidean") {
    return CDistMetric::kSqEuclidean;
  }
  if (metric == "euclidean") {
    return CDistMetric::kEuclidean;
  }
  throw std::invalid_argument(
      "onnx_light_cpu::CDist: metric must be \"sqeuclidean\" or \"euclidean\", got \"" + metric +
      "\".");
}

void ValidateTensors(const Tensor &a, const Tensor &b, const Tensor &output) {
  if (a.data_type != b.data_type || a.data_type != output.data_type) {
    throw std::invalid_argument("onnx_light_cpu::CDist: A, B and output dtypes must match.");
  }
  if (a.shape.size() != 2 || b.shape.size() != 2) {
    throw std::invalid_argument("onnx_light_cpu::CDist: A and B must be rank-2 tensors.");
  }
  if (a.shape[1] != b.shape[1]) {
    throw std::invalid_argument(
        "onnx_light_cpu::CDist: A and B must share the same feature dimension (N).");
  }
  if (output.shape.size() != 2 || output.shape[0] != a.shape[0] || output.shape[1] != b.shape[0]) {
    throw std::invalid_argument("onnx_light_cpu::CDist: output shape must be (M, K).");
  }
}

} // namespace

CDistKernel::CDistKernel(const NodeProto &node, const rt_ns::KernelContext &ctx) : KernelBase(ctx) {
  set_node(node);
}

void CDistKernel::RegisterTuningSchemas() {
  static std::once_flag once;
  std::call_once(once, [] {
    for (DataType type : kSupportedTypes) {
      rt_ns::RegisterKernelTuningSchema(rt_ns::KernelTuningSchema(
          MakeTuningDefaults(static_cast<int32_t>(type)), ValidateTuning));
    }
  });
}

rt_ns::KernelTuningKey CDistKernel::TuningKey(int32_t element_type) const {
  return SupportsElementType(element_type) ? MakeTuningKey(element_type) : rt_ns::KernelTuningKey{};
}

void CDistKernel::Configure(const rt_ns::KernelTuningParameters &parameters) {
  if (!SupportsElementType(parameters.key.element_type) ||
      parameters.key != MakeTuningKey(parameters.key.element_type)) {
    throw std::invalid_argument("CDist tuning parameters have an incompatible key.");
  }
  ValidateTuning(parameters);
  tuning_ = {
      static_cast<std::size_t>(parameters.Get<int64_t>(kParallelThresholdBytes)),
      static_cast<std::size_t>(parameters.Get<int64_t>(kTargetBlockBytes)),
      static_cast<std::size_t>(parameters.Get<int64_t>(kMaxParticipants)),
      parameters.Get<int64_t>(kCostModel) != 0,
      static_cast<std::size_t>(parameters.Get<int64_t>(kPreferredParticipants)),
  };
  tuning_configured_ = true;
}

Tensor CDistKernel::operator()(const Tensor &a, const Tensor &b, const std::string &metric,
                               RuntimeContext *rt) const {
  if (a.shape.size() != 2 || b.shape.size() != 2) {
    throw std::invalid_argument("onnx_light_cpu::CDist: A and B must be rank-2 tensors.");
  }
  const rt_ns::Shape output_shape{a.shape[0], b.shape[0]};
  const std::size_t bytes = static_cast<std::size_t>(a.shape[0]) *
                            static_cast<std::size_t>(b.shape[0]) * a.element_size();
  Tensor output = rt != nullptr
                      ? rt->MakeOutputTensor(0, a.data_type, output_shape, bytes)
                      : rt_ns::MakeOutputTensor(a.data_type, output_shape, bytes, nullptr);
  (*this)(a, b, metric, output);
  return output;
}

void CDistKernel::operator()(const Tensor &a, const Tensor &b, const std::string &metric,
                             Tensor &output) const {
  ValidateTensors(a, b, output);
  const CDistMetric parsed_metric = ParseMetric(metric);
  const std::size_t m = static_cast<std::size_t>(a.shape[0]);
  const std::size_t k = static_cast<std::size_t>(b.shape[0]);
  const std::size_t n = static_cast<std::size_t>(a.shape[1]);
  const CDistExecutionTuning &tuning = tuning_configured_ ? tuning_ : DefaultTuning(a.data_type);
  switch (static_cast<DataType>(a.data_type)) {
  case DataType::FLOAT:
    CDistFloat32WithTuning(a.AsFloat(), b.AsFloat(), output.AsFloat(), m, k, n, parsed_metric,
                           tuning);
    return;
  case DataType::DOUBLE:
    CDistFloat64WithTuning(a.AsDouble(), b.AsDouble(), output.AsDouble(), m, k, n, parsed_metric,
                           tuning);
    return;
  default:
    throw std::invalid_argument("onnx_light_cpu::CDist: unsupported data type " +
                                std::to_string(a.data_type) +
                                ", only FLOAT and DOUBLE are supported.");
  }
}

void CDistKernel::Run(RuntimeContext &rt) {
  RecordKernelUsage(kName);
  const NodeProto &node = *node_;
  rt_ns::RequireInputCount(node, 2);
  rt_ns::RequireOutputCount(node, 1);
  const Tensor &a = rt_ns::GetInput(node, 0, rt.tensors());
  const Tensor &b = rt_ns::GetInput(node, 1, rt.tensors());
  const std::string metric = rt_ns::GetAttributeStringOrDefault(node, "metric", "sqeuclidean");
  rt_ns::SetOutput(node, 0, (*this)(a, b, metric, &rt), rt);
}

void RegisterCDistKernel() {
  CDistKernel::RegisterTuningSchemas();
  NodeKernelFn factory = [](const NodeProto &node,
                            RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    return std::make_unique<CDistKernel>(node, rt.kernel_ctx());
  };
  KernelRegistration info;
  info.domain = kMicrosoftDomain;
  info.op_type = "CDist";
  info.device = sym_ns::Device::kCPU;
  info.kernel_name = CDistKernel::kName;
  info.types = {DataType::FLOAT, DataType::DOUBLE};
  info.since_version = 1;
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
