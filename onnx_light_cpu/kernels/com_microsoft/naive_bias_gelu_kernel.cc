// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/com_microsoft/naive_bias_gelu_kernel.h"

#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/symbolic/sym_tensor.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace onnx_light_cpu {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

using NodeProto = ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::DataType;
using rt_ns::RuntimeContext;
using rt_ns::Tensor;

namespace {

void ValidateTensors(const Tensor &a, const Tensor &b, const Tensor &output) {
  if (a.data_type != b.data_type || a.data_type != output.data_type) {
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveBiasGelu: A, B and output dtypes must match.");
  }
  if (a.shape.empty()) {
    throw std::invalid_argument("onnx_light_cpu::NaiveBiasGelu: A must have positive rank.");
  }
  if (b.shape.size() != 1) {
    throw std::invalid_argument("onnx_light_cpu::NaiveBiasGelu: B must be a rank-1 tensor.");
  }
  if (b.shape[0] != a.shape.back()) {
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveBiasGelu: B's length must match A's last dimension.");
  }
  if (output.shape != a.shape) {
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveBiasGelu: output shape must match A's shape.");
  }
}

template <typename T> T Gelu(T value) {
  return T(0.5) * value * (T(1) + std::erf(value / std::sqrt(T(2))));
}

template <typename T>
void ComputeTyped(const T *a, const T *bias, T *output, std::size_t count, std::size_t inner) {
  for (std::size_t index = 0; index < count; ++index) {
    output[index] = Gelu(a[index] + bias[index % inner]);
  }
}

template <bool BFloat16>
void ComputeHalf(const std::uint16_t *a, const std::uint16_t *bias, std::uint16_t *output,
                 std::size_t count, std::size_t inner) {
  for (std::size_t index = 0; index < count; ++index) {
    const float av =
        BFloat16 ? detail::Bfloat16BitsToFloat(a[index]) : detail::Float16BitsToFloat(a[index]);
    const float bv = BFloat16 ? detail::Bfloat16BitsToFloat(bias[index % inner])
                              : detail::Float16BitsToFloat(bias[index % inner]);
    const float value = Gelu(av + bv);
    output[index] =
        BFloat16 ? detail::FloatToBFloat16Bits(value) : detail::FloatToFloat16Bits(value);
  }
}

} // namespace

NaiveBiasGeluKernel::NaiveBiasGeluKernel(const NodeProto &node, const rt_ns::KernelContext &ctx)
    : KernelBase(ctx) {
  set_node(node);
}

Tensor NaiveBiasGeluKernel::operator()(const Tensor &a, const Tensor &b, RuntimeContext *rt) const {
  const std::size_t bytes = static_cast<std::size_t>(a.element_count()) * a.element_size();
  Tensor output = rt != nullptr ? rt->MakeOutputTensor(0, a.data_type, a.shape, bytes)
                                : rt_ns::MakeOutputTensor(a.data_type, a.shape, bytes, nullptr);
  (*this)(a, b, output);
  return output;
}

void NaiveBiasGeluKernel::operator()(const Tensor &a, const Tensor &b, Tensor &output) const {
  ValidateTensors(a, b, output);
  const std::size_t count = static_cast<std::size_t>(a.element_count());
  const std::size_t inner = static_cast<std::size_t>(b.shape[0]);
  if (count == 0) {
    return;
  }
  if (inner == 0) {
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveBiasGelu: B's length must be positive for non-empty A.");
  }
  switch (static_cast<DataType>(a.data_type)) {
  case DataType::FLOAT:
    ComputeTyped(a.AsFloat(), b.AsFloat(), output.AsFloat(), count, inner);
    return;
  case DataType::DOUBLE:
    ComputeTyped(a.AsDouble(), b.AsDouble(), output.AsDouble(), count, inner);
    return;
  case DataType::FLOAT16:
    ComputeHalf<false>(reinterpret_cast<const std::uint16_t *>(a.bytes()),
                       reinterpret_cast<const std::uint16_t *>(b.bytes()),
                       reinterpret_cast<std::uint16_t *>(output.mutable_bytes()), count, inner);
    return;
  case DataType::BFLOAT16:
    ComputeHalf<true>(reinterpret_cast<const std::uint16_t *>(a.bytes()),
                      reinterpret_cast<const std::uint16_t *>(b.bytes()),
                      reinterpret_cast<std::uint16_t *>(output.mutable_bytes()), count, inner);
    return;
  default:
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveBiasGelu: only FLOAT, DOUBLE, FLOAT16 and BFLOAT16 are supported.");
  }
}

void NaiveBiasGeluKernel::Run(RuntimeContext &rt) {
  RecordKernelUsage(kName);
  const NodeProto &node = *node_;
  rt_ns::RequireInputCount(node, 2);
  rt_ns::RequireOutputCount(node, 1);
  const Tensor &a = rt_ns::GetInput(node, 0, rt.tensors());
  const Tensor &b = rt_ns::GetInput(node, 1, rt.tensors());
  rt_ns::SetOutput(node, 0, (*this)(a, b, &rt), rt);
}

void RegisterNaiveBiasGeluKernel() {
  rt_ns::NodeKernelFn factory = [](const NodeProto &node,
                                   RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    return std::make_unique<NaiveBiasGeluKernel>(node, rt.kernel_ctx());
  };
  KernelRegistration info;
  info.domain = kMicrosoftDomain;
  info.op_type = "BiasGelu";
  info.device = sym_ns::Device::kCPU;
  info.kernel_name = NaiveBiasGeluKernel::kName;
  info.types = {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16};
  info.since_version = 1;
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
