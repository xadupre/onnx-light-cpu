// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/math/sigmoid_softmax_kernel.h"

#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/symbolic/sym_tensor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace onnx_light_cpu {

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

using NodeProto = ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::DataType;
using rt_ns::RuntimeContext;
using rt_ns::Tensor;

void ValidateOutput(const Tensor &x, const Tensor &output, const char *kernel_name) {
  if (output.data_type != x.data_type || output.shape != x.shape ||
      output.size_bytes() != x.size_bytes()) {
    throw std::invalid_argument(std::string(kernel_name) +
                                ": output dtype, shape, and size must match the input.");
  }
}

Tensor MakeLike(const Tensor &x, RuntimeContext *rt) {
  return rt != nullptr ? rt->MakeOutputTensor(0, x.data_type, x.shape, x.size_bytes())
                       : rt_ns::MakeOutputTensor(x.data_type, x.shape, x.size_bytes(), nullptr);
}

template <typename T> T StableSigmoid(T value) {
  if (value >= T{0}) {
    const T exponent = std::exp(-value);
    return T{1} / (T{1} + exponent);
  }
  const T exponent = std::exp(value);
  return exponent / (T{1} + exponent);
}

template <typename T> void SigmoidScalar(const T *input, T *output, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    output[i] = StableSigmoid(input[i]);
  }
}

template <typename Decode, typename Encode>
void SigmoidHalf(const std::uint16_t *input, std::uint16_t *output, std::size_t count,
                 Decode decode, Encode encode) {
  for (std::size_t i = 0; i < count; ++i) {
    output[i] = encode(StableSigmoid(decode(input[i])));
  }
}

template <typename T>
void Softmax(const T *input, T *output, std::int64_t outer, std::int64_t axis_dim,
             std::int64_t inner) {
  for (std::int64_t o = 0; o < outer; ++o) {
    for (std::int64_t i = 0; i < inner; ++i) {
      T maximum = -std::numeric_limits<T>::infinity();
      for (std::int64_t a = 0; a < axis_dim; ++a) {
        const std::size_t offset = static_cast<std::size_t>((o * axis_dim + a) * inner + i);
        maximum = std::max(maximum, input[offset]);
      }
      T sum = T{0};
      for (std::int64_t a = 0; a < axis_dim; ++a) {
        const std::size_t offset = static_cast<std::size_t>((o * axis_dim + a) * inner + i);
        output[offset] = std::exp(input[offset] - maximum);
        sum += output[offset];
      }
      for (std::int64_t a = 0; a < axis_dim; ++a) {
        const std::size_t offset = static_cast<std::size_t>((o * axis_dim + a) * inner + i);
        output[offset] /= sum;
      }
    }
  }
}

template <typename Decode, typename Encode>
void SoftmaxHalf(const std::uint16_t *input, std::uint16_t *output, std::int64_t outer,
                 std::int64_t axis_dim, std::int64_t inner, Decode decode, Encode encode) {
  for (std::int64_t o = 0; o < outer; ++o) {
    for (std::int64_t i = 0; i < inner; ++i) {
      float maximum = -std::numeric_limits<float>::infinity();
      for (std::int64_t a = 0; a < axis_dim; ++a) {
        const std::size_t offset = static_cast<std::size_t>((o * axis_dim + a) * inner + i);
        maximum = std::max(maximum, decode(input[offset]));
      }
      float sum = 0.0F;
      for (std::int64_t a = 0; a < axis_dim; ++a) {
        const std::size_t offset = static_cast<std::size_t>((o * axis_dim + a) * inner + i);
        sum += std::exp(decode(input[offset]) - maximum);
      }
      for (std::int64_t a = 0; a < axis_dim; ++a) {
        const std::size_t offset = static_cast<std::size_t>((o * axis_dim + a) * inner + i);
        output[offset] = encode(std::exp(decode(input[offset]) - maximum) / sum);
      }
    }
  }
}

std::int64_t ResolveAxis(std::int64_t axis, std::int64_t rank) {
  const std::int64_t resolved = axis < 0 ? axis + rank : axis;
  if (resolved < 0 || resolved >= rank) {
    throw std::invalid_argument("onnx_light_cpu::Softmax: axis is out of range.");
  }
  return resolved;
}

} // namespace

Tensor SigmoidKernel::operator()(const Tensor &x, RuntimeContext *rt) const {
  Tensor output = MakeLike(x, rt);
  (*this)(x, output);
  return output;
}

void SigmoidKernel::operator()(const Tensor &x, Tensor &output) const {
  ValidateOutput(x, output, "onnx_light_cpu::Sigmoid");
  const std::size_t count = static_cast<std::size_t>(x.element_count());
  switch (static_cast<DataType>(x.data_type)) {
  case DataType::FLOAT:
    SigmoidScalar(x.AsFloat(), output.AsFloat(), count);
    return;
  case DataType::DOUBLE:
    SigmoidScalar(x.AsDouble(), output.AsDouble(), count);
    return;
  case DataType::FLOAT16:
    SigmoidHalf(reinterpret_cast<const std::uint16_t *>(x.bytes()),
                reinterpret_cast<std::uint16_t *>(output.mutable_bytes()), count,
                detail::Float16BitsToFloat, detail::FloatToFloat16Bits);
    return;
  case DataType::BFLOAT16:
    SigmoidHalf(reinterpret_cast<const std::uint16_t *>(x.bytes()),
                reinterpret_cast<std::uint16_t *>(output.mutable_bytes()), count,
                detail::Bfloat16BitsToFloat, detail::FloatToBFloat16Bits);
    return;
  default:
    throw std::invalid_argument(
        "onnx_light_cpu::Sigmoid: only FLOAT, DOUBLE, FLOAT16 and BFLOAT16 are supported.");
  }
}

void SigmoidKernel::Run(RuntimeContext &rt) {
  RecordKernelUsage(kName);
  const NodeProto &node = *node_;
  rt_ns::RequireInputCount(node, 1);
  rt_ns::RequireOutputCount(node, 1);
  const Tensor &x = rt_ns::GetInput(node, 0, rt.tensors());
  rt_ns::SetOutput(node, 0, (*this)(x, &rt), rt);
}

Tensor SoftmaxKernel::operator()(const Tensor &x, std::int64_t axis, RuntimeContext *rt) const {
  Tensor output = MakeLike(x, rt);
  (*this)(x, axis, output);
  return output;
}

void SoftmaxKernel::operator()(const Tensor &x, std::int64_t axis, Tensor &output) const {
  ValidateOutput(x, output, "onnx_light_cpu::Softmax");
  const std::int64_t rank = static_cast<std::int64_t>(x.shape.size());
  if (rank == 0) {
    throw std::invalid_argument("onnx_light_cpu::Softmax: input rank must be at least one.");
  }
  const std::int64_t resolved_axis = ResolveAxis(axis, rank);
  std::int64_t outer = 1;
  for (std::int64_t dimension = 0; dimension < resolved_axis; ++dimension) {
    outer *= x.shape[static_cast<std::size_t>(dimension)];
  }
  std::int64_t axis_dim = x.shape[static_cast<std::size_t>(resolved_axis)];
  std::int64_t inner = 1;
  for (std::int64_t dimension = resolved_axis + 1; dimension < rank; ++dimension) {
    inner *= x.shape[static_cast<std::size_t>(dimension)];
  }
  if (ctx_.opset.version < 13) {
    axis_dim *= inner;
    inner = 1;
  }

  switch (static_cast<DataType>(x.data_type)) {
  case DataType::FLOAT:
    Softmax(x.AsFloat(), output.AsFloat(), outer, axis_dim, inner);
    return;
  case DataType::DOUBLE:
    Softmax(x.AsDouble(), output.AsDouble(), outer, axis_dim, inner);
    return;
  case DataType::FLOAT16:
    SoftmaxHalf(reinterpret_cast<const std::uint16_t *>(x.bytes()),
                reinterpret_cast<std::uint16_t *>(output.mutable_bytes()), outer, axis_dim, inner,
                detail::Float16BitsToFloat, detail::FloatToFloat16Bits);
    return;
  case DataType::BFLOAT16:
    SoftmaxHalf(reinterpret_cast<const std::uint16_t *>(x.bytes()),
                reinterpret_cast<std::uint16_t *>(output.mutable_bytes()), outer, axis_dim, inner,
                detail::Bfloat16BitsToFloat, detail::FloatToBFloat16Bits);
    return;
  default:
    throw std::invalid_argument(
        "onnx_light_cpu::Softmax: only FLOAT, DOUBLE, FLOAT16 and BFLOAT16 are supported.");
  }
}

void SoftmaxKernel::Run(RuntimeContext &rt) {
  RecordKernelUsage(kName);
  const NodeProto &node = *node_;
  rt_ns::RequireInputCount(node, 1);
  rt_ns::RequireOutputCount(node, 1);
  const std::int64_t default_axis = ctx_.opset.version < 13 ? 1 : -1;
  const std::int64_t axis = rt_ns::GetAttributeIntOrDefault(node, "axis", default_axis);
  const Tensor &x = rt_ns::GetInput(node, 0, rt.tensors());
  rt_ns::SetOutput(node, 0, (*this)(x, axis, &rt), rt);
}

void RegisterSigmoidKernel() {
  rt_ns::NodeKernelFn factory = [](const NodeProto &node,
                                   RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    auto kernel = std::make_unique<SigmoidKernel>(rt.kernel_ctx());
    kernel->set_node(node);
    return kernel;
  };
  KernelRegistration info;
  info.domain = "";
  info.op_type = "Sigmoid";
  info.device = sym_ns::Device::kCPU;
  info.kernel_name = SigmoidKernel::kName;
  info.types = {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16};
  RegisterKernel(std::move(info), std::move(factory));
}

void RegisterSoftmaxKernel() {
  rt_ns::NodeKernelFn factory = [](const NodeProto &node,
                                   RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    auto kernel = std::make_unique<SoftmaxKernel>(rt.kernel_ctx());
    kernel->set_node(node);
    return kernel;
  };
  KernelRegistration info;
  info.domain = "";
  info.op_type = "Softmax";
  info.device = sym_ns::Device::kCPU;
  info.kernel_name = SoftmaxKernel::kName;
  info.types = {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16};
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
