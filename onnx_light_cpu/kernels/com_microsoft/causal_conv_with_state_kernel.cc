// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/com_microsoft/causal_conv_with_state_kernel.h"

#include "onnx_light_cpu/impl/checked_arithmetic.h"
#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/symbolic/sym_tensor.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace onnx_light_cpu {
namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;
using rt_ns::DataType;
using rt_ns::RuntimeContext;
using rt_ns::Tensor;
using NodeProto = ONNX_LIGHT_NAMESPACE::NodeProto;

template <typename T> float Read(const T *data, std::size_t index, DataType dtype) {
  if constexpr (sizeof(T) == sizeof(float)) {
    return data[index];
  } else {
    return dtype == DataType::FLOAT16 ? detail::Float16BitsToFloat(data[index])
                                      : detail::Bfloat16BitsToFloat(data[index]);
  }
}

template <typename T> void Write(T *data, std::size_t index, float value, DataType dtype) {
  if constexpr (sizeof(T) == sizeof(float)) {
    data[index] = value;
  } else {
    data[index] = dtype == DataType::FLOAT16 ? detail::FloatToFloat16Bits(value)
                                             : detail::FloatToBFloat16Bits(value);
  }
}

template <typename T>
void Compute(const Tensor &input, const Tensor &weight, const Tensor *bias, const Tensor *past,
             Tensor &output, Tensor &present, std::size_t batch, std::size_t channels,
             std::size_t length, std::size_t width, std::size_t pad, std::size_t dilation,
             bool channels_last, bool silu) {
  const auto *x = reinterpret_cast<const T *>(input.bytes());
  const auto *w = reinterpret_cast<const T *>(weight.bytes());
  const auto *b = bias ? reinterpret_cast<const T *>(bias->bytes()) : nullptr;
  const auto *p = past ? reinterpret_cast<const T *>(past->bytes()) : nullptr;
  auto *y = reinterpret_cast<T *>(output.mutable_bytes());
  auto *s = reinterpret_cast<T *>(present.mutable_bytes());
  const DataType dtype = static_cast<DataType>(input.data_type);
  ExecuteRanges(
      CheckedIndexMultiply(static_cast<std::int64_t>(batch), static_cast<std::int64_t>(channels),
                           "CausalConvWithState", "batch channels"),
      1, [&](std::int64_t first, std::int64_t last) {
        for (auto task = first; task < last; ++task) {
          const std::size_t n = static_cast<std::size_t>(task);
          const std::size_t row = n / channels;
          const std::size_t channel = n % channels;
          const auto input_at = [&](std::size_t t) {
            return channels_last ? (row * length + t) * channels + channel : n * length + t;
          };
          const auto state_at = [&](std::size_t t) {
            return channels_last ? (row * pad + t) * channels + channel : n * pad + t;
          };
          for (std::size_t t = 0; t < length; ++t) {
            float sum = b ? Read(b, channel, dtype) : 0.0f;
            for (std::size_t k = 0; k < width; ++k) {
              const std::size_t lag = (width - 1 - k) * dilation;
              float value = 0.0f;
              if (lag <= t) {
                value = Read(x, input_at(t - lag), dtype);
              } else if (lag - t <= pad && p != nullptr) {
                value = Read(p, state_at(pad - (lag - t)), dtype);
              }
              sum += Read(w, channel * width + k, dtype) * value;
            }
            Write(y, input_at(t), silu ? sum / (1.0f + std::exp(-sum)) : sum, dtype);
          }
          for (std::size_t t = 0; t < pad; ++t) {
            const std::size_t source = t + length;
            if (source >= pad) {
              s[state_at(t)] = x[input_at(source - pad)];
            } else if (p != nullptr) {
              s[state_at(t)] = p[state_at(source)];
            } else {
              Write(s, state_at(t), 0.0f, dtype);
            }
          }
        }
      });
}

bool HasInput(const NodeProto &node, int index) {
  return node.input_size() > index && !node.input(index).empty();
}

} // namespace

std::pair<Tensor, Tensor>
CausalConvWithStateKernel::operator()(const Tensor &input, const Tensor &weight, const Tensor *bias,
                                      const Tensor *past_state, const std::string &activation,
                                      bool channels_last, std::int64_t dilation,
                                      RuntimeContext *rt) const {
  constexpr const char *name = "CausalConvWithState";
  const DataType dtype = static_cast<DataType>(input.data_type);
  if (dtype != DataType::FLOAT && dtype != DataType::FLOAT16 && dtype != DataType::BFLOAT16) {
    throw std::invalid_argument(
        "CausalConvWithState: only FLOAT, FLOAT16 and BFLOAT16 are supported.");
  }
  if (activation != "none" && activation != "silu" && activation != "swish") {
    throw std::invalid_argument("CausalConvWithState: unsupported activation.");
  }
  if (dilation < 1 || dilation > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("CausalConvWithState: dilation must be in [1, INT_MAX].");
  }
  if ((!channels_last && input.shape.size() != 3) || (channels_last && input.shape.size() < 3) ||
      weight.shape.size() != 3 || weight.shape[1] != 1 || weight.shape[2] < 1) {
    throw std::invalid_argument("CausalConvWithState: invalid input or depthwise weight shape.");
  }
  std::size_t channels = 1;
  if (channels_last) {
    for (std::size_t i = 2; i < input.shape.size(); ++i) {
      channels = CheckedMultiply(channels, CheckedDimension(input.shape[i], name, "channels"), name,
                                 "channels");
    }
  } else {
    channels = CheckedDimension(input.shape[1], name, "channels");
  }
  const std::size_t batch = CheckedDimension(input.shape[0], name, "batch");
  const std::size_t length = CheckedDimension(input.shape[channels_last ? 1 : 2], name, "sequence");
  const std::size_t width = CheckedDimension(weight.shape[2], name, "kernel width");
  const std::size_t pad =
      CheckedMultiply(width - 1, static_cast<std::size_t>(dilation), name, "state length");
  if (channels > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
      weight.shape[0] != static_cast<std::int64_t>(channels) || pad > INT32_MAX ||
      (bias != nullptr &&
       (bias->shape.size() != 1 || bias->shape[0] != static_cast<std::int64_t>(channels)))) {
    throw std::invalid_argument(
        "CausalConvWithState: weight channels, bias or state length invalid.");
  }
  const rt_ns::Shape state_shape = [&] {
    rt_ns::Shape shape = input.shape;
    shape[channels_last ? 1 : 2] = static_cast<std::int64_t>(pad);
    return shape;
  }();
  if ((bias && bias->data_type != input.data_type) || weight.data_type != input.data_type ||
      (past_state &&
       (past_state->data_type != input.data_type || past_state->shape != state_shape))) {
    throw std::invalid_argument("CausalConvWithState: tensor types or past_state shape mismatch.");
  }
  const std::size_t elements = CheckedProduct({batch, channels, length}, name, "output");
  const std::size_t state_elements = CheckedProduct({batch, channels, pad}, name, "state");
  const std::size_t bytes = CheckedByteSize(elements, input.element_size(), name, "output");
  const std::size_t state_bytes =
      CheckedByteSize(state_elements, input.element_size(), name, "state");
  Tensor output = rt ? rt->MakeOutputTensor(0, input.data_type, input.shape, bytes)
                     : rt_ns::MakeOutputTensor(input.data_type, input.shape, bytes, nullptr);
  Tensor present = rt ? rt->MakeOutputTensor(1, input.data_type, state_shape, state_bytes)
                      : rt_ns::MakeOutputTensor(input.data_type, state_shape, state_bytes, nullptr);
  if (batch != 0 && channels != 0) {
    if (dtype == DataType::FLOAT) {
      Compute<float>(input, weight, bias, past_state, output, present, batch, channels, length,
                     width, pad, static_cast<std::size_t>(dilation), channels_last,
                     activation != "none");
    } else {
      Compute<std::uint16_t>(input, weight, bias, past_state, output, present, batch, channels,
                             length, width, pad, static_cast<std::size_t>(dilation), channels_last,
                             activation != "none");
    }
  }
  return {std::move(output), std::move(present)};
}

void CausalConvWithStateKernel::Run(RuntimeContext &rt) {
  rt.RecordKernelUsage(kName);
  const NodeProto &node = *node_;
  if (node.input_size() < 2 || node.input_size() > 4 || node.input(0).empty() ||
      node.input(1).empty()) {
    throw std::invalid_argument("CausalConvWithState: expected two to four inputs.");
  }
  rt_ns::RequireOutputCount(node, 2);
  if (node.output(0).empty() || node.output(1).empty() ||
      rt_ns::GetAttributeIntOrDefault(node, "ndim", 1) != 1 ||
      rt_ns::GetAttributeIntOrDefault(node, "state_window", 0) != 0 ||
      (rt_ns::GetAttributeIntOrDefault(node, "channels_last", 0) != 0 &&
       rt_ns::GetAttributeIntOrDefault(node, "channels_last", 0) != 1)) {
    throw std::invalid_argument(
        "CausalConvWithState: requires two outputs, ndim=1 and state_window=0.");
  }
  const Tensor &input = rt_ns::GetInput(node, 0, rt.tensors());
  const Tensor &weight = rt_ns::GetInput(node, 1, rt.tensors());
  const Tensor *bias = HasInput(node, 2) ? &rt_ns::GetInput(node, 2, rt.tensors()) : nullptr;
  const Tensor *past = HasInput(node, 3) ? &rt_ns::GetInput(node, 3, rt.tensors()) : nullptr;
  auto [output, state] = (*this)(input, weight, bias, past,
                                 rt_ns::GetAttributeStringOrDefault(node, "activation", "none"),
                                 rt_ns::GetAttributeIntOrDefault(node, "channels_last", 0) == 1,
                                 rt_ns::GetAttributeIntOrDefault(node, "dilation", 1), &rt);
  rt_ns::SetOutput(node, 0, std::move(output), rt);
  rt_ns::SetOutput(node, 1, std::move(state), rt);
}

void RegisterCausalConvWithStateKernel() {
  rt_ns::NodeKernelFn factory =
      [](const NodeProto &node, RuntimeContext &,
         const rt_ns::KernelContext &ctx) -> std::unique_ptr<rt_ns::KernelBase> {
    auto kernel = std::make_unique<CausalConvWithStateKernel>(ctx);
    kernel->set_node(node);
    return kernel;
  };
  KernelRegistration info;
  info.domain = kMicrosoftDomain;
  info.op_type = "CausalConvWithState";
  info.device = sym_ns::Device::kCPU;
  info.kernel_name = CausalConvWithStateKernel::kName;
  info.types = {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16};
  info.since_version = 1;
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
